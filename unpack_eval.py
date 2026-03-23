#!/usr/bin/env python3
"""
Unpack + Eval: 紧凑二进制 → 重建模型 → 保存 → 测PPL

解码流程 (参数化位域, 支持任意 mantissa_bits):
  1. 读 block_meta: pattern_idx(3bit) + base(5bit) → 重建4个指数
  2. 读 row_data: 按 compute_bit_layout 解码
  3. 重建: sign × (1.0 + mant_int × step) × 2^exponent
  4. 按 mask 位置放回4列, 其余补0

比特布局 (MSB→LSB):
  pos_D0(2) | pos_D1(2) | sign_D0(1) | exp_D0(2) | mant_D0(M) |
                          sign_D1(1) | exp_D1(2) | mant_D1(M)
  total = 10 + 2M bits

Usage:
  python unpack_eval.py \
      --packed_path /path/to/packed_model.pt \
      --base_model /path/to/tokenizer \
      --model_structure /path/to/pruned_model \
      --save_dir /path/to/unpacked \
      --dataset wikitext2
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import transformers

from datautils import get_loaders
from modelutils import find_layers


# ======================================================================
# 模型加载
# ======================================================================

def get_opt(model_path):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


def find_layers_fn(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for n, child in module.named_children():
        child_name = name + '.' + n if name != '' else n
        res.update(find_layers_fn(child, layers=layers, name=child_name))
    return res


# ======================================================================
# 参数化位域 (与 pack.py 完全一致)
# ======================================================================

def compute_bit_layout(mantissa_bits):
    M = mantissa_bits
    total_bits = 10 + 2 * M
    return {
        'mantissa_bits': M,
        'total_bits': total_bits,
        'shift_pos_d0':  8 + 2 * M,
        'shift_pos_d1':  6 + 2 * M,
        'shift_sign_d0': 5 + 2 * M,
        'shift_exp_d0':  3 + 2 * M,
        'shift_mant_d0': 3 + M,
        'shift_sign_d1': 2 + M,
        'shift_exp_d1':  M,
        'shift_mant_d1': 0,
        'mask_mant': (1 << M) - 1,
    }


# ======================================================================
# 解码工具
# ======================================================================

def decode_signed5(raw):
    raw = int(raw) & 0x1F
    return raw - 32 if raw >= 16 else raw


def reconstruct_exponents(base, gaps):
    exps = [base]
    for g in gaps:
        exps.append(exps[-1] + g)
    return exps


# ======================================================================
# 单 tensor 解码 (参数化位域)
# ======================================================================

def unpack_tensor(packed_data, device):
    rows, cols = packed_data['shape']
    num_groups = cols // 4
    mantissa_bits = packed_data['mantissa_bits']
    row_block_size = packed_data['row_block_size']
    rbs = row_block_size if row_block_size > 0 else rows
    num_rb = (rows + rbs - 1) // rbs if row_block_size > 0 else 1

    layout = compute_bit_layout(mantissa_bits)
    step = 2.0 ** (-mantissa_bits)
    mant_mask = layout['mask_mant']

    d0_patterns = packed_data['d0_patterns']
    d1_patterns = packed_data['d1_patterns']
    block_meta = packed_data['block_meta']
    row_data = packed_data['row_data'].to(device).long()

    W = torch.zeros(rows, num_groups, 4, dtype=torch.float32, device=device)

    for g in range(num_groups):
        for rb in range(num_rb):
            rs = rb * rbs if row_block_size > 0 else 0
            re = min(rs + rbs, rows) if row_block_size > 0 else rows
            br = re - rs
            if br == 0:
                continue

            # ---- 解码块元数据 → 4个指数 ----
            d0_byte = block_meta[g, rb, 0].item()
            d1_byte = block_meta[g, rb, 1].item()

            d0_pat_idx = (d0_byte >> 5) & 0x7
            d0_base = decode_signed5(d0_byte & 0x1F)
            d0_exps = reconstruct_exponents(d0_base, d0_patterns[d0_pat_idx])

            d1_pat_idx = (d1_byte >> 5) & 0x7
            d1_base = decode_signed5(d1_byte & 0x1F)
            d1_exps = reconstruct_exponents(d1_base, d1_patterns[d1_pat_idx])

            d0_scales = torch.tensor([2.0 ** e for e in d0_exps],
                                     dtype=torch.float32, device=device)
            d1_scales = torch.tensor([2.0 ** e for e in d1_exps],
                                     dtype=torch.float32, device=device)

            # ---- 参数化解码行数据 ----
            packed = row_data[rs:re, g]
            L = layout

            pos_d0  = (packed >> L['shift_pos_d0'])  & 0x3
            pos_d1  = (packed >> L['shift_pos_d1'])  & 0x3

            sign_d0 = (packed >> L['shift_sign_d0']) & 0x1
            expk_d0 = (packed >> L['shift_exp_d0'])  & 0x3
            mant_d0 = (packed >> L['shift_mant_d0']) & mant_mask

            sign_d1 = (packed >> L['shift_sign_d1']) & 0x1
            expk_d1 = (packed >> L['shift_exp_d1'])  & 0x3
            mant_d1 = (packed >> L['shift_mant_d1']) & mant_mask

            # ---- 重建 D0 ----
            mant_float_d0 = 1.0 + mant_d0.float() * step
            scale_d0 = d0_scales[expk_d0]
            val_d0 = mant_float_d0 * scale_d0
            val_d0 = torch.where(sign_d0 == 1, -val_d0, val_d0)

            # ---- 重建 D1 ----
            mant_float_d1 = 1.0 + mant_d1.float() * step
            scale_d1 = d1_scales[expk_d1]
            val_d1 = mant_float_d1 * scale_d1
            val_d1 = torch.where(sign_d1 == 1, -val_d1, val_d1)

            # ---- 填入位置 ----
            block = torch.zeros(br, 4, dtype=torch.float32, device=device)
            ridx = torch.arange(br, device=device)
            block[ridx, pos_d0] = val_d0
            block[ridx, pos_d1] = val_d1
            W[rs:re, g, :] = block

    return W.reshape(rows, cols)


# ======================================================================
# PPL 评估
# ======================================================================

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i, end=' ', flush=True)
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    print()
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * model.seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'PPL: {ppl.item():.3f}')
    model.config.use_cache = use_cache
    return ppl.item()


# ======================================================================
# 主函数
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Unpack + Eval')
    parser.add_argument('--packed_path', type=str, required=True)
    parser.add_argument('--model_structure', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('Unpack + Eval')
    print(f'  Packed:          {args.packed_path}')
    print(f'  Model structure: {args.model_structure}')
    print(f'  Base model:      {args.base_model}')
    print(f'  Dataset:         {args.dataset}')
    print(f'  Save dir:        {args.save_dir if args.save_dir else "(不保存)"}')
    print('=' * 70)

    # ---- 加载压缩数据 ----
    print('\nLoading packed data...')
    packed = torch.load(args.packed_path, map_location='cpu')
    config = packed['config']
    packed_tensors = packed['packed_tensors']
    non_linear_state = packed['non_linear_state']

    mantissa_bits = config['mantissa_bits']
    layout = compute_bit_layout(mantissa_bits)

    print(f'  Config: mantissa_bits={mantissa_bits}, '
          f'row_block_size={config["row_block_size"]}')
    print(f'  Bit layout: total={layout["total_bits"]} bits/pair')
    print(f'  Packed tensors: {len(packed_tensors)}')
    print(f'  Non-linear params: {len(non_linear_state)}')

    # ---- 加载模型结构 ----
    print('\nLoading model structure...')
    model = get_opt(args.model_structure)
    model.eval()
    layers = model.model.decoder.layers

    # ---- 解码并替换权重 ----
    print('\nUnpacking weights...')
    total_time = time.time()

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        subset = find_layers_fn(layer)
        for name in subset:
            full_name = f"layer{layer_idx}.{name}"
            if full_name not in packed_tensors:
                continue

            pt = packed_tensors[full_name]
            pt['mantissa_bits'] = config['mantissa_bits']
            pt['row_block_size'] = config['row_block_size']

            t0 = time.time()
            W_recon = unpack_tensor(pt, device)
            t1 = time.time()

            target_dtype = subset[name].weight.data.dtype
            if isinstance(subset[name], transformers.Conv1D):
                W_recon = W_recon.t()
            subset[name].weight.data = W_recon.reshape(
                subset[name].weight.shape).to(target_dtype).cpu()

            print(f'  {full_name}: {pt["shape"]} → {t1-t0:.2f}s')

    # ---- 加载非量化参数 ----
    current_state = model.state_dict()
    loaded_count = 0
    for key, val in non_linear_state.items():
        if key in current_state:
            current_state[key] = val
            loaded_count += 1
    model.load_state_dict(current_state)
    print(f'\n  Loaded {loaded_count} non-linear parameters')

    elapsed = time.time() - total_time
    print(f'  Total unpack time: {elapsed:.1f}s')

    # ---- 保存 ----
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        print(f'\n  Saved reconstructed model to: {args.save_dir}')

    # ---- 测 PPL ----
    print(f'\n{"=" * 70}')
    print(f'PPL Evaluation ({args.dataset})')
    print(f'{"=" * 70}')

    _, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=args.base_model)

    ppl = opt_eval(model, testloader, device)

    print(f'\n{"=" * 70}')
    print(f'Result')
    print(f'{"=" * 70}')
    print(f'  Dataset:        {args.dataset}')
    print(f'  Mantissa bits:  {mantissa_bits}')
    print(f'  Row block size: {config["row_block_size"]}')
    print(f'  Bits/pair:      {layout["total_bits"]}')
    print(f'  PPL:            {ppl:.3f}')


if __name__ == '__main__':
    main()

# python unpack_eval.py  --packed_path /home/hej/model/float/sparsegpt/compressed_full_10bits/mantissa_10/packed/packed_model.pt  --model_structure /home/hej/model/float/stage1_pruned  --base_model /home/hej/model/float/opt-6.7b  --dataset wikitext2python unpack_eval.py  --packed_path /home/hej/model/float/sparsegpt/compressed_full_10bits/mantissa_3/packed/packed_model.pt  --model_structure /home/hej/model/float/stage1_pruned  --base_model /home/hej/model/float/opt-6.7b  --dataset wikitext2