#!/usr/bin/env python3
"""
Benchmark: 共享指数量化 vs torchao baselines (OPT-6.7B, 4090 24GB)
===================================================================

测量三种配置的 PPL / 吞吐 (tok/s) / 显存:

  A) FP16 dense baseline         — 原始模型直接跑
  B) torchao Int4 + 2:4 sparse   — MarlinSparseLayout (工业级 kernel)
  C) 你的 shared-exp 方案         — unpack → FP16 权重 → 2:4 sparse kernel
                                   (理论上界: 假设 dequant 零开销)

前置条件:
  pip install torchao transformers accelerate

使用:
  # 跑全部三组
  python benchmark.py --model /path/to/opt-6.7b \
      --pruned_model /path/to/stage1_pruned \
      --packed_path /path/to/packed_model.pt \
      --base_model /path/to/opt-6.7b

  # 只跑 torchao baseline
  python benchmark.py --model /path/to/opt-6.7b --skip_custom

  # 只跑自定义方案
  python benchmark.py --pruned_model /path/to/stage1_pruned \
      --packed_path /path/to/packed_model.pt \
      --base_model /path/to/opt-6.7b --skip_torchao
"""

import os
import gc
import time
import json
import argparse
import torch
import torch.nn as nn


# ======================================================================
# 工具函数
# ======================================================================

def get_gpu_memory_mb():
    """当前 GPU 已分配显存 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_gpu():
    """清理 GPU 显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_opt(model_path):
    """快速加载 OPT, 跳过不必要的初始化"""
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.seqlen = model.config.max_position_embeddings
    return model


# ======================================================================
# 测量 PPL (与你的 opt_eval 一致)
# ======================================================================

@torch.no_grad()
def measure_ppl(model, tokenizer, dataset='wikitext2', seqlen=2048,
                device='cuda', max_samples=None):
    """
    逐层 PPL 测量, 内存友好 (适合 4090 24GB)
    """
    from datasets import load_dataset

    if dataset == 'wikitext2':
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n\n'.join(data['text'])
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids

    nsamples = input_ids.numel() // seqlen
    if max_samples:
        nsamples = min(nsamples, max_samples)

    nlls = []
    for i in range(nsamples):
        batch = input_ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        nlls.append(outputs.loss.float() * seqlen)

        if (i + 1) % 10 == 0:
            print(f'    PPL progress: {i+1}/{nsamples}')

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()


# ======================================================================
# 测量 Generate 吞吐
# ======================================================================

@torch.no_grad()
def measure_throughput(model, tokenizer, device='cuda',
                       prompt_len=128, gen_len=128,
                       warmup=3, repeats=10):
    """
    测量 prefill 延时 + decode tok/s

    返回 dict:
      prefill_ms:   首 token 延时 (ms)
      decode_tok_s: decode 阶段 tok/s
      total_tok_s:  端到端 tok/s (含 prefill)
      peak_mem_mb:  峰值显存 (MB)
    """
    # 构造固定长度 prompt
    dummy_ids = torch.randint(1, tokenizer.vocab_size,
                              (1, prompt_len), device=device)

    # Warmup
    for _ in range(warmup):
        _ = model.generate(dummy_ids, max_new_tokens=8,
                           do_sample=False, use_cache=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    prefill_times = []
    decode_times = []
    total_times = []

    for r in range(repeats):
        torch.cuda.synchronize()

        # --- Prefill: 生成第一个 token ---
        t0 = time.perf_counter()
        out1 = model.generate(dummy_ids, max_new_tokens=1,
                              do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0
        prefill_times.append(t_prefill)

        # --- 完整 generation ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_full = model.generate(dummy_ids, max_new_tokens=gen_len,
                                  do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t0
        total_times.append(t_total)

        actual_gen = out_full.shape[1] - prompt_len
        t_decode = t_total - t_prefill  # 近似: decode = total - prefill
        decode_times.append((t_decode, actual_gen - 1))

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    # 取中位数
    prefill_times.sort()
    total_times.sort()
    mid = repeats // 2

    med_prefill = prefill_times[mid]
    med_total = total_times[mid]

    decode_toks = [g for _, g in decode_times]
    decode_secs = [t for t, _ in decode_times]
    med_decode_tok_s = sorted(
        [tok / max(sec, 1e-6) for tok, sec in zip(decode_toks, decode_secs)]
    )[mid]

    med_total_tok_s = gen_len / med_total

    return {
        'prefill_ms': round(med_prefill * 1000, 2),
        'decode_tok_s': round(med_decode_tok_s, 2),
        'total_tok_s': round(med_total_tok_s, 2),
        'peak_mem_mb': round(peak_mem, 1),
        'prompt_len': prompt_len,
        'gen_len': gen_len,
        'repeats': repeats,
    }


# ======================================================================
# 配置 A: FP16 Dense Baseline
# ======================================================================

def benchmark_fp16_dense(model_path, tokenizer, device):
    print('\n' + '=' * 60)
    print('Config A: FP16 Dense Baseline')
    print('=' * 60)

    reset_gpu()
    model = load_opt(model_path).to(device).eval()

    print('  Measuring throughput...')
    speed = measure_throughput(model, tokenizer, device)
    print(f'    Prefill: {speed["prefill_ms"]} ms')
    print(f'    Decode:  {speed["decode_tok_s"]} tok/s')
    print(f'    Total:   {speed["total_tok_s"]} tok/s')
    print(f'    Memory:  {speed["peak_mem_mb"]} MB')

    print('  Measuring PPL...')
    ppl = measure_ppl(model, tokenizer, device=device, max_samples=40)
    print(f'    PPL: {ppl:.3f}')

    result = {**speed, 'ppl': round(ppl, 3), 'config': 'FP16 Dense'}

    del model
    reset_gpu()
    return result


# ======================================================================
# 配置 B: torchao Int4 + 2:4 Sparse (MarlinSparseLayout)
# ======================================================================

def benchmark_torchao_int4_sparse(pruned_model_path, tokenizer, device):
    """
    需要 2:4 剪枝后的模型。
    torchao MarlinSparseLayout 要求权重已是 2:4 sparse。
    """
    print('\n' + '=' * 60)
    print('Config B: torchao Int4 + 2:4 Sparse (Marlin)')
    print('=' * 60)

    reset_gpu()

    try:
        from torchao.quantization import quantize_, Int4WeightOnlyConfig
        from torchao.dtypes import MarlinSparseLayout
    except ImportError:
        print('  [SKIP] torchao not installed. Run: pip install torchao')
        return None

    model = load_opt(pruned_model_path).to(device).eval()

    print('  Applying Int4WeightOnly + MarlinSparseLayout...')
    try:
        quant_config = Int4WeightOnlyConfig(layout=MarlinSparseLayout())
        quantize_(model, quant_config)
    except Exception as e:
        print(f'  [ERROR] MarlinSparseLayout failed: {e}')
        print('  Falling back to Int4WeightOnly (dense)...')
        try:
            del model
            reset_gpu()
            model = load_opt(pruned_model_path).to(device).eval()
            quantize_(model, Int4WeightOnlyConfig(group_size=128))
        except Exception as e2:
            print(f'  [ERROR] Int4WeightOnly also failed: {e2}')
            return None

    print('  Compiling with torch.compile...')
    try:
        model = torch.compile(model, mode='max-autotune')
    except Exception as e:
        print(f'  [WARN] torch.compile failed: {e}, continuing without')

    print('  Measuring throughput...')
    speed = measure_throughput(model, tokenizer, device)
    print(f'    Prefill: {speed["prefill_ms"]} ms')
    print(f'    Decode:  {speed["decode_tok_s"]} tok/s')
    print(f'    Total:   {speed["total_tok_s"]} tok/s')
    print(f'    Memory:  {speed["peak_mem_mb"]} MB')

    print('  Measuring PPL...')
    ppl = measure_ppl(model, tokenizer, device=device, max_samples=40)
    print(f'    PPL: {ppl:.3f}')

    result = {**speed, 'ppl': round(ppl, 3), 'config': 'torchao Int4+2:4 Sparse'}

    del model
    reset_gpu()
    return result


# ======================================================================
# 配置 C: 你的 Shared-Exp 方案 (unpack → FP16 → 2:4 sparse kernel)
# ======================================================================

def benchmark_custom_shared_exp(pruned_model_path, packed_path,
                                 tokenizer, device):
    """
    理论上界 benchmark:
      1. 加载 packed_model.pt
      2. 解码回 FP16 权重 (模拟零开销 dequant)
      3. 灌回模型的 nn.Linear
      4. 用 torchao 的 2:4 sparse kernel 加速
      5. 测 throughput + PPL
    """
    print('\n' + '=' * 60)
    print('Config C: Custom Shared-Exp (理论上界)')
    print('=' * 60)

    if not os.path.exists(packed_path):
        print(f'  [SKIP] packed_model.pt not found: {packed_path}')
        return None

    reset_gpu()

    # ---- 加载 packed 数据 ----
    print('  Loading packed_model.pt...')
    packed = torch.load(packed_path, map_location='cpu')
    config = packed['config']
    packed_tensors = packed['packed_tensors']
    non_linear_state = packed['non_linear_state']
    mantissa_bits = config['mantissa_bits']
    row_block_size = config['row_block_size']

    print(f'    Mantissa bits: {mantissa_bits}')
    print(f'    Row block size: {row_block_size}')
    print(f'    Packed tensors: {len(packed_tensors)}')

    # ---- 加载模型骨架 ----
    model = load_opt(pruned_model_path)

    # ---- 解码 packed → FP16 权重并灌回 ----
    print('  Unpacking to FP16 weights...')
    from pack import compute_bit_layout, reconstruct_exponents

    layout = compute_bit_layout(mantissa_bits)
    M = mantissa_bits
    step = 2.0 ** (-M)

    layers = model.model.decoder.layers
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        subset = find_layers_simple(layer)

        for name in subset:
            full_name = f"layer{layer_idx}.{name}"
            if full_name not in packed_tensors:
                continue

            pt = packed_tensors[full_name]
            row_data = pt['row_data'].long()
            block_meta = pt['block_meta']
            d0_pats = pt['d0_patterns']
            d1_pats = pt['d1_patterns']
            rows, cols = pt['shape']
            num_groups = cols // 4
            rbs = row_block_size if row_block_size > 0 else rows
            num_rb = (rows + rbs - 1) // rbs if row_block_size > 0 else 1

            W_recon = torch.zeros(rows, cols, dtype=torch.float16)

            for g in range(num_groups):
                for rb in range(num_rb):
                    rs = rb * rbs if row_block_size > 0 else 0
                    re = min(rs + rbs, rows) if row_block_size > 0 else rows

                    # 解码 block_meta → exponents
                    for slot_idx, pats in [(0, d0_pats), (1, d1_pats)]:
                        byte_val = block_meta[g, rb, slot_idx].item()
                        pat_idx = (byte_val >> 5) & 0x7
                        base_raw = byte_val & 0x1F
                        base = base_raw if base_raw < 16 else base_raw - 32
                        exps = reconstruct_exponents(base, pats[pat_idx])

                        # 解码行数据
                        chunk = row_data[rs:re, g]
                        L = layout

                        if slot_idx == 0:
                            pos = ((chunk >> L['shift_pos_d0']) & 0x3).int()
                            sign = ((chunk >> L['shift_sign_d0']) & 0x1).int()
                            exp_k = ((chunk >> L['shift_exp_d0']) & 0x3).int()
                            mant = ((chunk >> L['shift_mant_d0']) & L['mask_mant']).int()
                        else:
                            pos = ((chunk >> L['shift_pos_d1']) & 0x3).int()
                            sign = ((chunk >> L['shift_sign_d1']) & 0x1).int()
                            exp_k = ((chunk >> L['shift_exp_d1']) & 0x3).int()
                            mant = ((chunk >> L['shift_mant_d1']) & L['mask_mant']).int()

                        for r_local in range(re - rs):
                            p = pos[r_local].item()
                            s = sign[r_local].item()
                            k = exp_k[r_local].item()
                            m = mant[r_local].item()

                            if k < len(exps):
                                scale = 2.0 ** exps[k]
                                mantissa = 1.0 + m * step
                                val = scale * mantissa
                                if s:
                                    val = -val
                                W_recon[rs + r_local, g * 4 + p] = val

            # 灌回权重
            subset[name].weight.data = W_recon.to(subset[name].weight.dtype)

    model = model.to(device).eval()

    # ---- 可选: 应用 2:4 sparse kernel ----
    try:
        from torchao.sparsity import sparsify_
        from torchao.sparsity.sparse_api import SemiSparseWeightConfig
        print('  Applying torchao 2:4 sparse kernel...')
        sparsify_(model, SemiSparseWeightConfig())
    except Exception as e:
        print(f'  [WARN] sparsify_ failed: {e}, running dense FP16')

    # ---- Benchmark ----
    print('  Measuring throughput...')
    speed = measure_throughput(model, tokenizer, device)
    print(f'    Prefill: {speed["prefill_ms"]} ms')
    print(f'    Decode:  {speed["decode_tok_s"]} tok/s')
    print(f'    Total:   {speed["total_tok_s"]} tok/s')
    print(f'    Memory:  {speed["peak_mem_mb"]} MB')

    print('  Measuring PPL...')
    ppl = measure_ppl(model, tokenizer, device=device, max_samples=40)
    print(f'    PPL: {ppl:.3f}')

    # 计算模型大小
    packed_size_mb = os.path.getsize(packed_path) / 1024 / 1024

    result = {
        **speed,
        'ppl': round(ppl, 3),
        'config': f'Custom SharedExp M={mantissa_bits} RBS={row_block_size}',
        'packed_size_mb': round(packed_size_mb, 1),
        'mantissa_bits': mantissa_bits,
        'row_block_size': row_block_size,
    }

    del model
    reset_gpu()
    return result


# ======================================================================
# 辅助
# ======================================================================

def find_layers_simple(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for n, child in module.named_children():
        child_name = name + '.' + n if name != '' else n
        res.update(find_layers_simple(child, layers=layers, name=child_name))
    return res


# ======================================================================
# 主函数
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark: Shared-Exp vs torchao baselines')
    parser.add_argument('--model', type=str, default='',
                        help='Dense FP16 model path (for baseline A)')
    parser.add_argument('--pruned_model', type=str, default='',
                        help='2:4 pruned model path (for B and C)')
    parser.add_argument('--packed_path', type=str, default='',
                        help='packed_model.pt path (for config C)')
    parser.add_argument('--base_model', type=str, default='',
                        help='Tokenizer model path')
    parser.add_argument('--prompt_len', type=int, default=128)
    parser.add_argument('--gen_len', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--ppl_samples', type=int, default=40,
                        help='PPL 样本数 (减少以节省时间)')
    parser.add_argument('--skip_torchao', action='store_true')
    parser.add_argument('--skip_custom', action='store_true')
    parser.add_argument('--skip_dense', action='store_true')
    parser.add_argument('--output', type=str, default='benchmark_results.json')
    args = parser.parse_args()

    device = torch.device('cuda:0')

    # 确定 tokenizer 来源
    tokenizer_path = args.base_model or args.model or args.pruned_model
    if not tokenizer_path:
        print('ERROR: 需要至少一个模型路径')
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print('=' * 70)
    print('Benchmark: Shared-Exp Quant vs torchao Baselines')
    print('=' * 70)
    print(f'Device:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM:        {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    print(f'Prompt len:  {args.prompt_len}')
    print(f'Gen len:     {args.gen_len}')
    print(f'Repeats:     {args.repeats}')
    print()

    results = {}

    # ---- A: FP16 Dense ----
    if not args.skip_dense and args.model:
        results['fp16_dense'] = benchmark_fp16_dense(
            args.model, tokenizer, device)

    # ---- B: torchao Int4 + Sparse ----
    if not args.skip_torchao and args.pruned_model:
        results['torchao_int4_sparse'] = benchmark_torchao_int4_sparse(
            args.pruned_model, tokenizer, device)

    # ---- C: Custom Shared-Exp ----
    if not args.skip_custom and args.pruned_model and args.packed_path:
        results['custom_shared_exp'] = benchmark_custom_shared_exp(
            args.pruned_model, args.packed_path, tokenizer, device)

    # ---- 汇总 ----
    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    print(f'{"Config":<35} {"PPL":>8} {"Decode tok/s":>13} {"Mem MB":>8}')
    print('-' * 70)
    for name, r in results.items():
        if r is None:
            continue
        ppl_str = f'{r["ppl"]:.2f}' if 'ppl' in r else 'N/A'
        print(f'{r["config"]:<35} {ppl_str:>8} {r["decode_tok_s"]:>13.1f} '
              f'{r["peak_mem_mb"]:>8.0f}')

    # ---- 保存 ----
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()