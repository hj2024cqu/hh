#!/usr/bin/env python3
"""
Pack: 2:4稀疏模型 → 紧凑二进制格式 (支持任意mantissa_bits, v1/v2中间产物)

比特布局 (MSB→LSB, 参数化):
  pos_D0(2) | pos_D1(2) | sign_D0(1) | exp_D0(2) | mant_D0(M) |
                          sign_D1(1) | exp_D1(2) | mant_D1(M)
  total = 10 + 2M bits

Usage:
  python pack.py --model /path/to/pruned \
      --precomputed_dir /path/to/test_output \
      --row_block_size 128 --mantissa_bits 3 \
      --output_dir /path/to/packed
"""

import os
import re
import json
import time
import argparse
import torch
import torch.nn as nn


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


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for n, child in module.named_children():
        child_name = name + '.' + n if name != '' else n
        res.update(find_layers(child, layers=layers, name=child_name))
    return res


# ======================================================================
# 参数化位域
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
# 编码工具
# ======================================================================

def encode_signed5(val):
    val = max(-16, min(15, int(val)))
    return val & 0x1F


def get_top8_patterns(pattern_info, slot):
    key = f'{slot}_patterns'
    if key not in pattern_info:
        return [[1, 1, 1]] * 8
    top8 = []
    for pat_str in list(pattern_info[key].keys())[:8]:
        gaps = [int(x) for x in pat_str.split('-')]
        top8.append(gaps)
    while len(top8) < 8:
        top8.append(top8[-1] if top8 else [1, 1, 1])
    return top8[:8]


def match_pattern(gaps, pattern_list):
    best_idx, best_dist = 0, float('inf')
    for i, pat in enumerate(pattern_list):
        dist = sum(abs(a - b) for a, b in zip(gaps, pat))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx, best_dist


def reconstruct_exponents(base, gaps):
    exps = [base]
    for g in gaps:
        exps.append(exps[-1] + g)
    return exps


def quantize_values_batch(vals, exponents, mantissa_bits, device):
    N = vals.shape[0]
    if N == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, empty, torch.zeros(0, device=device), torch.zeros(0, device=device)

    abs_vals = torch.abs(vals).clamp(min=1e-38)
    signs = (vals < 0).long()

    exp_t = torch.tensor(exponents, dtype=torch.float32, device=device)
    scales = 2.0 ** exp_t

    mantissas = abs_vals.unsqueeze(1) / scales.unsqueeze(0)
    clamped = torch.clamp(mantissas, 1.0, 2.0 - 1e-7)

    step = 2.0 ** (-mantissa_bits)
    max_mant = (1 << mantissa_bits) - 1

    mant_int = torch.round((clamped - 1.0) / step).long()
    mant_int = torch.clamp(mant_int, 0, max_mant)

    recon_mant = 1.0 + mant_int.float() * step
    recon = recon_mant * scales.unsqueeze(0)
    errors = torch.abs(abs_vals.unsqueeze(1) - recon)

    best_k = torch.argmin(errors, dim=1)
    idx = torch.arange(N, device=device)

    return (signs,
            best_k,
            mant_int[idx, best_k],
            torch.where(vals >= 0, recon[idx, best_k], -recon[idx, best_k]),
            errors[idx, best_k])


# ======================================================================
# 加载中间产物 (自动 v1 JSON / v2 .pt)
# ======================================================================

def load_all_codebooks(precomputed_dir):
    """返回 (format, data)"""
    pt_path = os.path.join(precomputed_dir, 'codebook_tables.pt')
    json_path = os.path.join(precomputed_dir, 'codebook_tables.json')

    if os.path.exists(pt_path):
        data = torch.load(pt_path, map_location='cpu')
        print(f'  Codebook format: v2 (.pt), {len(data)} tensors')
        return 'v2', data
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f'  Codebook format: v1 (JSON), {len(data)} tensors')
        return 'v1', data
    else:
        raise FileNotFoundError(
            f"No codebook_tables.pt or .json in {precomputed_dir}")


def load_all_masks(precomputed_dir):
    mask_path = os.path.join(precomputed_dir, 'sparse_masks.pt')
    data = torch.load(mask_path, map_location='cpu')
    # 检测格式
    sample = next(iter(data.values()))
    fmt = 'v2' if sample.dim() == 2 else 'v1'
    print(f'  Mask format: {fmt}, {len(data)} tensors')
    return fmt, data


def get_codebook_exps(cb_format, cb_data, tensor_name, g, rb):
    """获取 (g, rb) 的 D0/D1 指数"""
    if cb_format == 'v2':
        t = cb_data[tensor_name]
        d0 = t[g, rb, 0, :].tolist()
        d1 = t[g, rb, 1, :].tolist()
        return d0, d1
    else:
        td = cb_data[tensor_name]
        key = f"g{g}_rb{rb}"
        if key not in td:
            return [-8, -7, -6, -5], [-10, -9, -8, -7]
        return sorted(td[key]['d0']), sorted(td[key]['d1'])


def get_mask_positions(mask_fmt, mask_data, tensor_name, device):
    """返回 pos0[rows, groups], pos1[rows, groups] (long tensor on device)"""
    m = mask_data[tensor_name]
    if mask_fmt == 'v2':
        pos0 = ((m >> 2) & 0x3).long().to(device)
        pos1 = (m & 0x3).long().to(device)
    else:
        pos0 = m[:, :, 0].long().to(device)
        pos1 = m[:, :, 1].long().to(device)
    return pos0, pos1


# ======================================================================
# 单 tensor 编码
# ======================================================================

def pack_tensor(tensor_name, W, cb_format, cb_all, pos0_all, pos1_all,
                d0_pats_8, d1_pats_8, row_block_size, mantissa_bits, device):
    layout = compute_bit_layout(mantissa_bits)
    W = W.float().to(device)
    rows, cols = W.shape
    num_groups = cols // 4
    rbs = row_block_size if row_block_size > 0 else rows
    num_rb = (rows + rbs - 1) // rbs if row_block_size > 0 else 1

    W_g = W.reshape(rows, num_groups, 4)

    block_meta = torch.zeros(num_groups, num_rb, 2, dtype=torch.uint8)
    block_d0_exps = {}
    block_d1_exps = {}
    pat_mismatch = 0
    base_clamp = 0

    for g in range(num_groups):
        for rb in range(num_rb):
            d0_orig, d1_orig = get_codebook_exps(
                cb_format, cb_all, tensor_name, g, rb)

            for slot, exps_orig, pats_8, store in [
                ('d0', d0_orig, d0_pats_8, block_d0_exps),
                ('d1', d1_orig, d1_pats_8, block_d1_exps),
            ]:
                base = exps_orig[0]
                gaps = [exps_orig[i + 1] - exps_orig[i] for i in range(3)]
                pat_idx, dist = match_pattern(gaps, pats_8)
                if dist > 0:
                    pat_mismatch += 1
                exps_final = reconstruct_exponents(base, pats_8[pat_idx])
                store[(g, rb)] = exps_final

                if base < -16 or base > 15:
                    base_clamp += 1

                byte_val = ((pat_idx & 0x7) << 5) | encode_signed5(base)
                idx_slot = 0 if slot == 'd0' else 1
                block_meta[g, rb, idx_slot] = byte_val

    # 行数据 (用 int64 做中间计算, 防止大 mantissa_bits 溢出)
    row_data = torch.zeros(rows, num_groups, dtype=torch.int64, device=device)
    total_swap = 0

    for g in range(num_groups):
        for rb in range(num_rb):
            rs = rb * rbs if row_block_size > 0 else 0
            re = min(rs + rbs, rows) if row_block_size > 0 else rows
            br = re - rs
            if br == 0:
                continue

            d0e = block_d0_exps.get((g, rb), [-8, -7, -6, -5])
            d1e = block_d1_exps.get((g, rb), [-10, -9, -8, -7])

            p0 = pos0_all[rs:re, g]
            p1 = pos1_all[rs:re, g]

            ridx = torch.arange(br, device=device)
            val0 = W_g[rs:re, g, :][ridx, p0]
            val1 = W_g[rs:re, g, :][ridx, p1]

            s0n, k0n, m0n, _, err0n = quantize_values_batch(val0, d0e, mantissa_bits, device)
            s1n, k1n, m1n, _, err1n = quantize_values_batch(val1, d1e, mantissa_bits, device)

            s0s, k0s, m0s, _, err0s = quantize_values_batch(val0, d1e, mantissa_bits, device)
            s1s, k1s, m1s, _, err1s = quantize_values_batch(val1, d0e, mantissa_bits, device)

            do_swap = (err0s + err1s) < (err0n + err1n)
            total_swap += do_swap.sum().item()

            fp_d0 = torch.where(do_swap, p1, p0)
            fp_d1 = torch.where(do_swap, p0, p1)

            sd0 = torch.where(do_swap, s1s, s0n)
            kd0 = torch.where(do_swap, k1s, k0n)
            md0 = torch.where(do_swap, m1s, m0n)

            sd1 = torch.where(do_swap, s0s, s1n)
            kd1 = torch.where(do_swap, k0s, k1n)
            md1 = torch.where(do_swap, m0s, m1n)

            L = layout
            packed = (
                (fp_d0 << L['shift_pos_d0']) |
                (fp_d1 << L['shift_pos_d1']) |
                (sd0   << L['shift_sign_d0']) |
                (kd0   << L['shift_exp_d0']) |
                (md0   << L['shift_mant_d0']) |
                (sd1   << L['shift_sign_d1']) |
                (kd1   << L['shift_exp_d1']) |
                (md1   << L['shift_mant_d1'])
            )
            row_data[rs:re, g] = packed

    # 选存储 dtype
    if layout['total_bits'] <= 16:
        row_data_final = row_data.to(torch.int16).cpu()
    elif layout['total_bits'] <= 32:
        row_data_final = row_data.to(torch.int32).cpu()
    else:
        row_data_final = row_data.cpu()

    return {
        'shape': [rows, cols],
        'mantissa_bits': mantissa_bits,
        'total_bits_per_pair': layout['total_bits'],
        'd0_patterns': d0_pats_8,
        'd1_patterns': d1_pats_8,
        'block_meta': block_meta,
        'row_data': row_data_final,
        'stats': {
            'pattern_mismatch': pat_mismatch,
            'base_clamp': base_clamp,
            'total_swap': total_swap,
            'total_pairs': rows * num_groups,
        }
    }


# ======================================================================
# 主函数
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Pack: 2:4 → 紧凑编码')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--precomputed_dir', type=str, required=True)
    parser.add_argument('--row_block_size', type=int, default=128)
    parser.add_argument('--mantissa_bits', type=int, default=3)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    layout = compute_bit_layout(args.mantissa_bits)

    print('=' * 70)
    print('Pack: 2:4 Sparse → Compact Binary')
    print(f'  Model:          {args.model}')
    print(f'  Precomputed:    {args.precomputed_dir}')
    print(f'  Row block size: {args.row_block_size}')
    print(f'  Mantissa bits:  {args.mantissa_bits}')
    print(f'  Bits/value:     1+2+{args.mantissa_bits} = {1+2+args.mantissa_bits}')
    print(f'  Bits/pair:      {layout["total_bits"]}')
    print(f'  Storage dtype:  int16' if layout['total_bits'] <= 16
          else f'  Storage dtype:  int32' if layout['total_bits'] <= 32
          else f'  Storage dtype:  int64')
    print(f'  Output:         {args.output_dir}')
    print('=' * 70)

    # ---- 加载预计算数据 ----
    print('\nLoading precomputed data...')
    cb_format, cb_all = load_all_codebooks(args.precomputed_dir)
    mask_fmt, mask_all = load_all_masks(args.precomputed_dir)

    with open(os.path.join(args.precomputed_dir, 'pattern_analysis.json'), 'r') as f:
        all_patterns = json.load(f)
    print(f'  Patterns: {len(all_patterns)} tensors')

    # ---- 加载模型 ----
    print('\nLoading model...')
    model = get_opt(args.model)
    model.eval()
    layers = model.model.decoder.layers

    full_state = model.state_dict()
    linear_weight_keys = set()

    packed_tensors = {}
    total_original = 0
    total_compressed = 0
    total_time = time.time()

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        subset = find_layers(layer)

        for name in subset:
            full_name = f"layer{layer_idx}.{name}"

            W = subset[name].weight.data.clone()
            import transformers
            if isinstance(subset[name], transformers.Conv1D):
                W = W.t()
            rows, cols = W.shape

            if cols % 4 != 0 or full_name not in all_patterns:
                print(f'  [SKIP] {full_name}')
                continue

            if full_name not in (cb_all if cb_format == 'v2' else cb_all):
                print(f'  [SKIP] {full_name}: no codebook')
                continue

            if full_name not in mask_all:
                print(f'  [SKIP] {full_name}: no mask')
                continue

            for k in full_state.keys():
                if name + '.weight' in k and f'layers.{layer_idx}.' in k:
                    linear_weight_keys.add(k)

            pos0, pos1 = get_mask_positions(mask_fmt, mask_all, full_name, device)

            d0_pats = get_top8_patterns(all_patterns[full_name], 'd0')
            d1_pats = get_top8_patterns(all_patterns[full_name], 'd1')

            t0 = time.time()
            packed = pack_tensor(
                full_name, W, cb_format, cb_all, pos0, pos1,
                d0_pats, d1_pats, args.row_block_size, args.mantissa_bits, device)
            t1 = time.time()

            packed_tensors[full_name] = packed
            st = packed['stats']

            orig_bytes = rows * cols * 2
            rd_bytes_per_elem = 2 if layout['total_bits'] <= 16 else (4 if layout['total_bits'] <= 32 else 8)
            comp_bytes = packed['row_data'].numel() * rd_bytes_per_elem + packed['block_meta'].numel()
            total_original += orig_bytes
            total_compressed += comp_bytes

            print(f'  {full_name}: {rows}×{cols}, '
                  f'swap={st["total_swap"]}/{st["total_pairs"]} '
                  f'({st["total_swap"]/max(st["total_pairs"],1):.1%}), '
                  f'pat_mismatch={st["pattern_mismatch"]}, '
                  f'time={t1-t0:.1f}s')

    non_linear_state = {k: v for k, v in full_state.items() if k not in linear_weight_keys}

    save_dict = {
        'config': {
            'mantissa_bits': args.mantissa_bits,
            'row_block_size': args.row_block_size,
            'model_path': args.model,
            'total_bits_per_pair': layout['total_bits'],
        },
        'packed_tensors': packed_tensors,
        'non_linear_state': non_linear_state,
    }

    save_path = os.path.join(args.output_dir, 'packed_model.pt')
    torch.save(save_dict, save_path)
    file_size = os.path.getsize(save_path) / 1024 / 1024

    elapsed = time.time() - total_time

    print(f'\n{"=" * 70}')
    print(f'Summary')
    print(f'{"=" * 70}')
    print(f'  Packed tensors:   {len(packed_tensors)}')
    print(f'  Original (FP16):  {total_original / 1024 / 1024:.1f} MB')
    print(f'  Compressed:       {total_compressed / 1024 / 1024:.1f} MB')
    print(f'  Saved file:       {file_size:.1f} MB')
    print(f'  Saved to:         {save_path}')
    print(f'  Total time:       {elapsed:.1f}s')


if __name__ == '__main__':
    main()