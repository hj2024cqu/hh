"""
SmoothQuant-style per-channel scaling for 2:4 pruned OPT model
===============================================================
核心改动vs原版：
  - 只加载模型一次，所有操作在单次逐层forward中完成
  - 收集激活统计和PPL评估共用同一个forward pass
  - 支持多alpha对比（不重新forward，只重新算scale）

用法:
  python opt_smooth_scale.py /path/to/pruned_model wikitext2 \
      --base_model /path/to/opt-6.7b --alpha 0.5 \
      --save /path/to/output
"""

import time
import math
import copy
import torch
import torch.nn as nn
from datautils import get_loaders
from modelutils import find_layers
from transformers import OPTForCausalLM

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_opt(model_path):
    def skip(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def prepare_inps(model, dataloader, dev, nsamples):
    """准备第0层的输入（只需做一次）"""
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
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
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

    model.config.use_cache = use_cache
    return inps, cache['attention_mask']


@torch.no_grad()
def collect_act_stats(model, inps, attention_mask, dev, nsamples):
    """逐层收集激活统计，同时推进 inps → outs（与PPL eval共享结构）"""
    print('\nCollecting activation statistics...')
    layers = model.model.decoder.layers
    outs = torch.zeros_like(inps)
    all_act_stats = {}

    for layer_idx in range(len(layers)):
        print(f'  Layer {layer_idx}...', end=' ', flush=True)
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)

        act_maxes = {}
        for name in subset:
            act_maxes[name] = torch.zeros(
                subset[name].weight.shape[1], dtype=torch.float32, device=dev
            )

        def make_hook(name):
            def hook_fn(_, inp, out):
                x = inp[0].data
                if len(x.shape) == 3:
                    x = x.reshape(-1, x.shape[-1])
                ch_max = x.abs().max(dim=0)[0].float()
                act_maxes[name] = torch.max(act_maxes[name], ch_max)
            return hook_fn

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_hook(name)))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        all_act_stats[layer_idx] = {
            name: act_maxes[name].cpu() for name in subset
        }

        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    print('Done.')
    return all_act_stats, inps  # inps 现在是最后一层的输出


def compute_smooth_scales(model, all_act_stats, alpha=0.5):
    """计算 SmoothQuant 缩放因子: s_j = act_max_j^α / w_max_j^(1-α)"""
    layers = model.model.decoder.layers
    all_scales = {}

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        subset = find_layers(layer)
        layer_scales = {}

        for name in subset:
            W = subset[name].weight.data.float()
            act_max = all_act_stats[layer_idx][name].float()

            # per-channel weight max (沿 output dim，只看非零)
            w_abs = W.abs()
            w_max = w_abs.max(dim=0)[0]

            act_max = act_max.clamp(min=1e-8)
            w_max = w_max.clamp(min=1e-8)

            s = (act_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
            layer_scales[name] = s

        all_scales[layer_idx] = layer_scales
    return all_scales


def apply_smooth_scales(model, all_scales):
    """
    应用缩放: W[:, j] *= s_j, 把 s_j^{-1} 吸收进前一层 LayerNorm 或 Linear

    OPT decoder layer 结构:
      self_attn_layer_norm → q_proj, k_proj, v_proj
      (attention internals) → out_proj
      final_layer_norm → fc1
      fc1 (+ activation) → fc2
    """
    layers = model.model.decoder.layers
    total_applied = 0

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        scales = all_scales[layer_idx]

        # === 1. self_attn_layer_norm → q_proj, k_proj, v_proj ===
        attn_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
        attn_scales_list = [scales[n] for n in attn_names if n in scales]

        if attn_scales_list:
            # SmoothQuant 原文: 取 max 保证所有分支都不会被过度缩放
            combined_s = torch.stack(attn_scales_list).max(dim=0)[0]

            # 吸收 s^{-1} 进 LayerNorm
            ln = layer.self_attn_layer_norm
            device = ln.weight.device
            combined_s_dev = combined_s.to(device)
            ln.weight.data.div_(combined_s_dev)
            if ln.bias is not None:
                ln.bias.data.div_(combined_s_dev)

            # W[:, j] *= s_j
            for name in attn_names:
                if name in scales:
                    parts = name.split('.')
                    linear = layer
                    for p in parts:
                        linear = getattr(linear, p)
                    linear.weight.data = (
                        linear.weight.data.float() * combined_s_dev.unsqueeze(0)
                    ).to(linear.weight.dtype)
                    total_applied += 1

            print(f'  Layer {layer_idx} attn: s range=[{combined_s.min():.4f}, {combined_s.max():.4f}], '
                  f'mean={combined_s.mean():.4f}')

        # === 2. out_proj: 输入来自 attention 内部，无 LayerNorm 可吸收 ===
        # SmoothQuant 原文也跳过 out_proj，因为没有好的吸收点
        # 如果要做，需要改 attention 的 value projection 输出，比较复杂，先跳过

        # === 3. final_layer_norm → fc1 ===
        if 'fc1' in scales:
            s_fc1 = scales['fc1']
            ln = layer.final_layer_norm
            device = ln.weight.device
            s_fc1_dev = s_fc1.to(device)

            ln.weight.data.div_(s_fc1_dev)
            if ln.bias is not None:
                ln.bias.data.div_(s_fc1_dev)

            layer.fc1.weight.data = (
                layer.fc1.weight.data.float() * s_fc1_dev.unsqueeze(0)
            ).to(layer.fc1.weight.dtype)

            total_applied += 1
            print(f'  Layer {layer_idx} fc1:  s range=[{s_fc1.min():.4f}, {s_fc1.max():.4f}], '
                  f'mean={s_fc1.mean():.4f}')

        # === 4. fc1 → fc2: 吸收到 fc1 的输出侧 ===
        if 'fc2' in scales:
            s_fc2 = scales['fc2']
            device = layer.fc1.weight.device
            s_fc2_dev = s_fc2.to(device)

            # fc1.weight[j, :] /= s_j (fc1 的第 j 个输出行)
            layer.fc1.weight.data = (
                layer.fc1.weight.data.float() / s_fc2_dev.unsqueeze(1)
            ).to(layer.fc1.weight.dtype)
            if layer.fc1.bias is not None:
                layer.fc1.bias.data = (
                    layer.fc1.bias.data.float() / s_fc2_dev
                ).to(layer.fc1.bias.dtype)

            # fc2.weight[:, j] *= s_j
            layer.fc2.weight.data = (
                layer.fc2.weight.data.float() * s_fc2_dev.unsqueeze(0)
            ).to(layer.fc2.weight.dtype)

            total_applied += 1
            print(f'  Layer {layer_idx} fc2:  s range=[{s_fc2.min():.4f}, {s_fc2.max():.4f}], '
                  f'mean={s_fc2.mean():.4f}')

    print(f'\nTotal scaled layers: {total_applied}')


def check_sparsity_preserved(model):
    """检查2:4稀疏是否保持"""
    layers = model.model.decoder.layers
    total_groups = 0
    valid_24 = 0

    for layer in layers:
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            if W.shape[1] % 4 != 0:
                continue
            rows, cols = W.shape
            W_grouped = W.reshape(rows, cols // 4, 4)
            nonzero_count = (W_grouped != 0).sum(dim=2)
            total_groups += nonzero_count.numel()
            valid_24 += (nonzero_count == 2).sum().item()

    rate = valid_24 / max(total_groups, 1)
    print(f'2:4 Sparsity check: {valid_24}/{total_groups} groups valid ({rate:.2%})')
    return rate


def analyze_exp_concentration(model, tag=""):
    """分析权重指数集中度（per-group 内指数 range）"""
    layers = model.model.decoder.layers
    all_ranges = []

    for layer in layers:
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data.float()
            if W.shape[1] % 4 != 0:
                continue
            rows, cols = W.shape
            W_grouped = W.reshape(rows, cols // 4, 4)
            nonzero_mask = W_grouped != 0

            abs_vals = W_grouped.abs().clamp(min=1e-38)
            exps = torch.floor(torch.log2(abs_vals))
            exp_max = exps.clone()
            exp_min = exps.clone()
            exp_max[~nonzero_mask] = -999
            exp_min[~nonzero_mask] = 999

            g_max = exp_max.max(dim=2)[0]
            g_min = exp_min.min(dim=2)[0]

            valid_mask = (g_max > -999) & (g_min < 999)
            if valid_mask.any():
                ranges = (g_max - g_min)[valid_mask]
                all_ranges.append(ranges)

    if all_ranges:
        all_ranges = torch.cat(all_ranges)
        print(f'[{tag}指数范围统计] '
              f'mean={all_ranges.mean():.2f}, median={all_ranges.median():.2f}, '
              f'max={all_ranges.max():.0f}, '
              f'<=3: {(all_ranges <= 3).float().mean():.2%}, '
              f'<=4: {(all_ranges <= 4).float().mean():.2%}')


@torch.no_grad()
def opt_eval(model, testenc, dev):
    """评估 PPL"""
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
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'PPL: {ppl.item():.3f}')

    model.config.use_cache = use_cache
    return ppl.item()


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to 2:4 pruned model')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--base_model', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='SmoothQuant alpha: 0=weight only, 1=act only, 0.5=balanced')
    parser.add_argument('--search_alpha', action='store_true',
                        help='Search over multiple alphas (0.25, 0.5, 0.75) and pick best')
    parser.add_argument('--save', type=str, default='', help='Save scaled model')
    args = parser.parse_args()

    DEV = torch.device('cuda:0')

    print('='*70)
    print('SmoothQuant-style Scaling for 2:4 Pruned Model')
    print(f'Alpha: {args.alpha}' + (' (+ search)' if args.search_alpha else ''))
    print('='*70)

    # === 加载模型（只加载一次）===
    print('\nLoading model...')
    model = get_opt(args.model)
    model.eval()

    print('Loading data...')
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=args.base_model
    )

    # === 缩放前分析 ===
    print('\n' + '='*70)
    print('Before Scaling')
    print('='*70)
    check_sparsity_preserved(model)
    analyze_exp_concentration(model, tag="缩放前")

    # === 收集激活统计（单次 forward pass）===
    print('\n' + '='*70)
    print('Collecting Activation Statistics')
    print('='*70)
    inps, attention_mask = prepare_inps(model, dataloader, DEV, args.nsamples)
    act_stats, _ = collect_act_stats(model, inps, attention_mask, DEV, args.nsamples)
    del inps
    torch.cuda.empty_cache()

    # === 搜索 alpha 或使用指定 alpha ===
    if args.search_alpha:
        alphas = [0.25, 0.5, 0.75]
    else:
        alphas = [args.alpha]

    best_alpha = args.alpha
    best_ppl = float('inf')

    for alpha in alphas:
        print(f'\n{"="*70}')
        print(f'Testing alpha={alpha}')
        print(f'{"="*70}')

        # 重新加载干净模型（避免上一次 scale 的残留影响）
        # 第一个 alpha 直接用已加载的 model，后续 alpha 需要重新加载
        if alpha != alphas[0]:
            del model
            torch.cuda.empty_cache()
            model = get_opt(args.model)
            model.eval()

        # 计算并应用缩放
        all_scales = compute_smooth_scales(model, act_stats, alpha=alpha)
        apply_smooth_scales(model, all_scales)

        # 验证
        check_sparsity_preserved(model)
        analyze_exp_concentration(model, tag=f"alpha={alpha} 缩放后")

        # 测 PPL
        ppl = opt_eval(model, testloader, DEV)

        if ppl < best_ppl:
            best_ppl = ppl
            best_alpha = alpha

        print(f'\nalpha={alpha}: PPL={ppl:.3f}')

    # === 最终结果 ===
    if len(alphas) > 1:
        print(f'\n{"="*70}')
        print(f'Best alpha: {best_alpha} (PPL={best_ppl:.3f})')
        print(f'{"="*70}')

        # 用最优 alpha 重新缩放并保存
        del model
        torch.cuda.empty_cache()
        model = get_opt(args.model)
        model.eval()

        all_scales = compute_smooth_scales(model, act_stats, alpha=best_alpha)
        apply_smooth_scales(model, all_scales)
        args.alpha = best_alpha

    # === 保存 ===
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)

        import json
        info = {
            'alpha': args.alpha,
            'ppl': best_ppl,
            'source_model': args.model,
        }
        with open(os.path.join(args.save, 'smooth_scale_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

        print(f'\nSaved to {args.save}')
        print(f'Next: use as input to opt_shared_exp_swap.py for quantization')