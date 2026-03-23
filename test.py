#!/usr/bin/env python3
"""
分析 2:4 剪枝 + 共享指数量化模型 (带 checkpoint)
=========================================================
功能:
  1. 稀疏性检查 (2:4 结构验证)
  2. 生成 mask 表和指数选取表 (.pt, int8)
  3. 指数选取模式频率统计 (JSON)
  4. 每 N 层自动 checkpoint, 支持断点续跑

输出格式 (兼容 pack.py v2):
  - codebook_tables.pt : dict{ tensor_name → int8 tensor [num_groups, num_rb, 2, 4] }
  - sparse_masks.pt    : dict{ tensor_name → int8 tensor [rows, num_groups, 2] }
  - pattern_analysis.json / sparsity_check.json / summary.json

加速手段 (零精度损失):
  - C(n,4) 枚举: 预生成 combo 索引张量, GPU batch min/sum 替代 Python 循环
  - log2 / Counter: 一次性批量计算所有 (group, row_block)
  - combo 索引缓存: 同一 n_cand 只生成一次

使用方法:
python test.py /home/hej/model/float/stage1_pruned --mantissa_bits 4 --row_block_size 128 \
    --output_dir /home/hej/model/float/sparsegpt/compressed_128_4bits/mantissa_4bit \
    --checkpoint_every 4
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
from collections import Counter
from itertools import combinations


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
# Combo 索引缓存
# ======================================================================

_combo_cache = {}


def get_combo_tensor(n_cand, device):
    key = (n_cand, device)
    if key not in _combo_cache:
        combos = list(combinations(range(n_cand), 4))
        _combo_cache[key] = torch.tensor(combos, dtype=torch.long, device=device)
    return _combo_cache[key]


# ======================================================================
# 核心: 暴力枚举最优4指数
# ======================================================================

def select_top4_bruteforce_fast(vals, valid_mask, mantissa_bits=4,
                                skip_mantissa_quant=False, device='cpu'):
    v = vals[valid_mask]
    if len(v) == 0:
        return [-8, -7, -6, -5]

    abs_v = torch.abs(v).clamp(min=1e-38)
    exps = torch.floor(torch.log2(abs_v)).int()
    exp_list = exps.cpu().tolist()
    counts = Counter(exp_list)

    if len(counts) <= 4:
        result = sorted(counts.keys())
        while len(result) < 4:
            result.append(result[-1] + 1)
        return sorted(result[:4])

    unique_exps = sorted(counts.keys())
    if len(unique_exps) > 12:
        top_exps = [e for e, _ in counts.most_common(12)]
        unique_exps = sorted(top_exps)

    n_cand = len(unique_exps)
    cand_exp = torch.tensor(unique_exps, dtype=torch.float32, device=device)
    cand_scales = 2.0 ** cand_exp

    mantissas = abs_v.unsqueeze(0) / cand_scales.unsqueeze(1)
    clamped = torch.clamp(mantissas, 1.0, 2.0 - 1e-7)

    if skip_mantissa_quant:
        qm = clamped
    else:
        step = 2.0 ** (-mantissa_bits)
        qm = torch.round(clamped / step) * step
        qm = torch.clamp(qm, 1.0, 2.0 - step)

    recon = qm * cand_scales.unsqueeze(1)
    per_exp_errors = torch.abs(abs_v.unsqueeze(0) - recon)

    combo_idx = get_combo_tensor(n_cand, device)
    combo_errors = per_exp_errors[combo_idx]
    min_errors, _ = combo_errors.min(dim=1)
    total_costs = min_errors.sum(dim=1)
    best_idx = total_costs.argmin().item()
    best_combo = combo_idx[best_idx].cpu().tolist()

    result = sorted([unique_exps[i] for i in best_combo])
    return result


# ======================================================================
# 模式分类
# ======================================================================

def classify_pattern(exp_list):
    s = sorted(exp_list)
    gaps = [s[i + 1] - s[i] for i in range(3)]
    pattern_str = "-".join(str(g) for g in gaps)
    return pattern_str, s[0], s


# ======================================================================
# 批量预计算: D0/D1 值 + 指数
# ======================================================================

def precompute_d0_d1(W_g, nz_g, valid_row_mask, device):
    rows, num_groups, _ = W_g.shape

    pos_t = torch.arange(4, device=device).view(1, 1, 4).expand(rows, num_groups, 4)
    masked_pos = torch.where(
        nz_g, pos_t,
        torch.tensor(99, dtype=torch.long, device=device)
    )
    sorted_p, _ = torch.sort(masked_pos, dim=2)
    pos0 = sorted_p[:, :, 0].clamp(0, 3)
    pos1 = sorted_p[:, :, 1].clamp(0, 3)

    ridx = torch.arange(rows, device=device).view(-1, 1).expand(rows, num_groups)
    gidx = torch.arange(num_groups, device=device).view(1, -1).expand(rows, num_groups)
    v0 = W_g[ridx, gidx, pos0]
    v1 = W_g[ridx, gidx, pos1]

    need_sort = torch.abs(v0) < torch.abs(v1)
    val_d0 = torch.where(need_sort, v1, v0)
    val_d1 = torch.where(need_sort, v0, v1)
    pos_d0 = torch.where(need_sort, pos1, pos0)
    pos_d1 = torch.where(need_sort, pos0, pos1)

    exp_d0_all = torch.floor(torch.log2(torch.abs(val_d0).clamp(min=1e-38))).int()
    exp_d1_all = torch.floor(torch.log2(torch.abs(val_d1).clamp(min=1e-38))).int()

    return val_d0, val_d1, pos_d0, pos_d1, exp_d0_all, exp_d1_all


# ======================================================================
# 快速 Counter
# ======================================================================

def fast_counter(exp_slice):
    if len(exp_slice) == 0:
        return Counter()
    uniq, cnts = torch.unique(exp_slice, return_counts=True)
    return Counter(dict(zip(uniq.cpu().tolist(), cnts.cpu().tolist())))


# ======================================================================
# select_top4: 利用预计算指数
# ======================================================================

def select_top4_with_precomputed(vals_slice, exp_int_slice, valid_slice,
                                  mantissa_bits, skip_mantissa_quant, device):
    v = vals_slice[valid_slice]
    if len(v) == 0:
        return [-8, -7, -6, -5]

    abs_v = torch.abs(v).clamp(min=1e-38)
    e_valid = exp_int_slice[valid_slice]

    counts = fast_counter(e_valid)

    if len(counts) <= 4:
        result = sorted(counts.keys())
        while len(result) < 4:
            result.append(result[-1] + 1)
        return sorted(result[:4])

    unique_exps = sorted(counts.keys())
    if len(unique_exps) > 12:
        top_exps = [e for e, _ in counts.most_common(12)]
        unique_exps = sorted(top_exps)

    n_cand = len(unique_exps)
    cand_exp = torch.tensor(unique_exps, dtype=torch.float32, device=device)
    cand_scales = 2.0 ** cand_exp

    mantissas = abs_v.unsqueeze(0) / cand_scales.unsqueeze(1)
    clamped = torch.clamp(mantissas, 1.0, 2.0 - 1e-7)

    if skip_mantissa_quant:
        qm = clamped
    else:
        step = 2.0 ** (-mantissa_bits)
        qm = torch.round(clamped / step) * step
        qm = torch.clamp(qm, 1.0, 2.0 - step)

    recon = qm * cand_scales.unsqueeze(1)
    per_exp_errors = torch.abs(abs_v.unsqueeze(0) - recon)

    combo_idx = get_combo_tensor(n_cand, device)
    combo_errors = per_exp_errors[combo_idx]
    min_errors, _ = combo_errors.min(dim=1)
    total_costs = min_errors.sum(dim=1)
    best_idx = total_costs.argmin().item()
    best_combo = combo_idx[best_idx].cpu().tolist()

    result = sorted([unique_exps[i] for i in best_combo])
    return result


# ======================================================================
# 单 tensor 分析
# ======================================================================

def analyze_tensor(tensor_name, W, mantissa_bits=4, row_block_size=-1,
                   skip_mantissa_quant=False, device='cpu'):
    W = W.float().to(device)
    rows, cols = W.shape
    rbs = row_block_size

    # =================== 1. 稀疏性检查 ===================
    total_elements = rows * cols
    nonzero_elements = (W != 0).sum().item()
    sparsity_ratio = 1.0 - nonzero_elements / total_elements

    if cols % 4 != 0:
        sparsity_info = {
            "shape": [rows, cols],
            "total_elements": total_elements,
            "nonzero_elements": nonzero_elements,
            "sparsity_ratio": round(sparsity_ratio, 6),
            "is_2_4": False,
            "reason": "cols not divisible by 4"
        }
        return sparsity_info, None, None, None

    num_groups = cols // 4
    if rbs > 0:
        num_rb = (rows + rbs - 1) // rbs
    else:
        num_rb = 1
        rbs = rows

    W_g = W.reshape(rows, num_groups, 4)
    nz_g = (W_g != 0)
    nz_cnt = nz_g.sum(dim=2)
    valid_row_mask = (nz_cnt == 2)
    total_group_slots = rows * num_groups
    valid_count = valid_row_mask.sum().item()
    invalid_count = total_group_slots - valid_count

    nz_distribution = {}
    for k in range(5):
        cnt = (nz_cnt == k).sum().item()
        if cnt > 0:
            nz_distribution[str(k)] = cnt

    sparsity_info = {
        "shape": [rows, cols],
        "total_elements": total_elements,
        "nonzero_elements": nonzero_elements,
        "sparsity_ratio": round(sparsity_ratio, 6),
        "num_groups": num_groups,
        "num_row_blocks": num_rb,
        "row_block_size": row_block_size,
        "effective_rbs": rbs,
        "is_2_4": invalid_count == 0,
        "valid_2_4_groups": valid_count,
        "invalid_groups": invalid_count,
        "total_group_slots": total_group_slots,
        "total_codebook_entries": num_groups * num_rb,
        "nz_count_distribution": nz_distribution
    }

    # =================== 2. 批量预计算 D0/D1 ===================
    val_d0, val_d1, pos_d0, pos_d1, exp_d0_all, exp_d1_all = \
        precompute_d0_d1(W_g, nz_g, valid_row_mask, device)

    # mask: [rows, num_groups, 2], int8 — 直接兼容 pack.py v1 格式
    mask_tensor = torch.stack([pos_d0, pos_d1], dim=2).to(torch.int8).cpu()

    # =================== 3. 指数选取 ===================
    # codebook tensor: [num_groups, num_rb, 2, 4], int8 — 兼容 pack.py v2 格式
    codebook_tensor = torch.zeros(num_groups, num_rb, 2, 4, dtype=torch.int8)

    d0_patterns_counter = Counter()
    d1_patterns_counter = Counter()
    d0_pattern_examples = {}
    d1_pattern_examples = {}
    d0_base_counter = Counter()
    d1_base_counter = Counter()
    d0_exp_global = Counter()
    d1_exp_global = Counter()

    t0 = time.time()
    total_blocks = num_groups * num_rb
    processed = 0

    for g in range(num_groups):
        for rb in range(num_rb):
            if row_block_size > 0:
                rs = rb * row_block_size
                re = min(rs + row_block_size, rows)
            else:
                rs = 0
                re = rows

            vm = valid_row_mask[rs:re, g]
            vd0 = val_d0[rs:re, g]
            vd1 = val_d1[rs:re, g]
            ed0_pre = exp_d0_all[rs:re, g]
            ed1_pre = exp_d1_all[rs:re, g]

            exp_d0 = select_top4_with_precomputed(
                vd0, ed0_pre, vm, mantissa_bits, skip_mantissa_quant, device)
            exp_d1 = select_top4_with_precomputed(
                vd1, ed1_pre, vm, mantissa_bits, skip_mantissa_quant, device)

            if exp_d0 == exp_d1:
                exp_d1 = [e - 1 for e in exp_d1]

            # 写入 codebook tensor (int8, 范围 -128~127, 指数一般在此范围内)
            codebook_tensor[g, rb, 0, :] = torch.tensor(sorted(exp_d0), dtype=torch.int8)
            codebook_tensor[g, rb, 1, :] = torch.tensor(sorted(exp_d1), dtype=torch.int8)

            # 指数分布统计
            e0_valid = ed0_pre[vm]
            e1_valid = ed1_pre[vm]
            if len(e0_valid) > 0:
                d0_exp_global.update(fast_counter(e0_valid))
                d1_exp_global.update(fast_counter(e1_valid))

            pat_d0, base_d0, _ = classify_pattern(exp_d0)
            pat_d1, base_d1, _ = classify_pattern(exp_d1)

            d0_patterns_counter[pat_d0] += 1
            d1_patterns_counter[pat_d1] += 1
            d0_base_counter[base_d0] += 1
            d1_base_counter[base_d1] += 1

            if pat_d0 not in d0_pattern_examples:
                d0_pattern_examples[pat_d0] = []
            if len(d0_pattern_examples[pat_d0]) < 5:
                d0_pattern_examples[pat_d0].append({
                    "group": g, "row_block": rb,
                    "row_range": [rs, re],
                    "exponents": exp_d0, "base": base_d0
                })

            if pat_d1 not in d1_pattern_examples:
                d1_pattern_examples[pat_d1] = []
            if len(d1_pattern_examples[pat_d1]) < 5:
                d1_pattern_examples[pat_d1].append({
                    "group": g, "row_block": rb,
                    "row_range": [rs, re],
                    "exponents": exp_d1, "base": base_d1
                })

            processed += 1

        if (g + 1) % 256 == 0 or g == num_groups - 1:
            elapsed = time.time() - t0
            speed = processed / max(elapsed, 0.01)
            eta = (total_blocks - processed) / max(speed, 0.01)
            print(f"      group {g + 1}/{num_groups}, "
                  f"{processed}/{total_blocks} ({processed / total_blocks * 100:.0f}%), "
                  f"{elapsed:.1f}s, ETA {eta:.0f}s")

    # =================== 4. 整理 ===================
    total_cb = num_groups * num_rb

    pattern_info = {
        "num_groups": num_groups,
        "num_row_blocks": num_rb,
        "row_block_size": row_block_size,
        "total_codebook_entries": total_cb,
        "d0_patterns": {},
        "d1_patterns": {},
        "d0_exp_distribution": {str(k): v for k, v in sorted(d0_exp_global.items())},
        "d1_exp_distribution": {str(k): v for k, v in sorted(d1_exp_global.items())},
        "d0_base_distribution": {str(k): v for k, v in sorted(d0_base_counter.items())},
        "d1_base_distribution": {str(k): v for k, v in sorted(d1_base_counter.items())},
    }

    for pat, cnt in d0_patterns_counter.most_common():
        pattern_info["d0_patterns"][pat] = {
            "count": cnt,
            "frequency": round(cnt / total_cb, 6),
            "pct": f"{cnt / total_cb:.2%}",
            "examples": d0_pattern_examples.get(pat, [])
        }
    for pat, cnt in d1_patterns_counter.most_common():
        pattern_info["d1_patterns"][pat] = {
            "count": cnt,
            "frequency": round(cnt / total_cb, 6),
            "pct": f"{cnt / total_cb:.2%}",
            "examples": d1_pattern_examples.get(pat, [])
        }

    return sparsity_info, codebook_tensor, pattern_info, mask_tensor


# ======================================================================
# Checkpoint 存/读
# ======================================================================

CHECKPOINT_FILE = 'checkpoint_state.pt'


def save_checkpoint(output_dir, completed_layers, all_sparsity, all_patterns,
                    all_codebooks_pt, all_masks_pt):
    """
    增量保存 checkpoint:
      - codebook_tables.pt : dict{ tensor_name → int8 [G, RB, 2, 4] }
      - sparse_masks.pt    : dict{ tensor_name → int8 [rows, G, 2] }
      - sparsity_check.json
      - pattern_analysis.json
      - checkpoint_state.pt : 记录已完成的层号列表
    """
    # codebook .pt
    cb_path = os.path.join(output_dir, 'codebook_tables.pt')
    torch.save(all_codebooks_pt, cb_path)

    # mask .pt
    mask_path = os.path.join(output_dir, 'sparse_masks.pt')
    torch.save(all_masks_pt, mask_path)

    # sparsity json
    sp_path = os.path.join(output_dir, 'sparsity_check.json')
    with open(sp_path, 'w', encoding='utf-8') as f:
        json.dump(all_sparsity, f, indent=2, ensure_ascii=False)

    # pattern json
    pat_path = os.path.join(output_dir, 'pattern_analysis.json')
    with open(pat_path, 'w', encoding='utf-8') as f:
        json.dump(all_patterns, f, indent=2, ensure_ascii=False)

    # checkpoint meta
    ckpt_path = os.path.join(output_dir, CHECKPOINT_FILE)
    torch.save({'completed_layers': sorted(completed_layers)}, ckpt_path)

    print(f'  [Checkpoint] 已保存, 完成层: {sorted(completed_layers)}')


def load_checkpoint(output_dir):
    """
    尝试加载已有 checkpoint, 返回已完成的层号集合和已有数据。
    如果不存在则返回空。
    """
    ckpt_path = os.path.join(output_dir, CHECKPOINT_FILE)
    if not os.path.exists(ckpt_path):
        return set(), {}, {}, {}, {}

    meta = torch.load(ckpt_path, map_location='cpu')
    completed = set(meta.get('completed_layers', []))

    all_sparsity = {}
    sp_path = os.path.join(output_dir, 'sparsity_check.json')
    if os.path.exists(sp_path):
        with open(sp_path, 'r', encoding='utf-8') as f:
            all_sparsity = json.load(f)

    all_patterns = {}
    pat_path = os.path.join(output_dir, 'pattern_analysis.json')
    if os.path.exists(pat_path):
        with open(pat_path, 'r', encoding='utf-8') as f:
            all_patterns = json.load(f)

    all_codebooks_pt = {}
    cb_path = os.path.join(output_dir, 'codebook_tables.pt')
    if os.path.exists(cb_path):
        all_codebooks_pt = torch.load(cb_path, map_location='cpu')

    all_masks_pt = {}
    mask_path = os.path.join(output_dir, 'sparse_masks.pt')
    if os.path.exists(mask_path):
        all_masks_pt = torch.load(mask_path, map_location='cpu')

    print(f'  [Checkpoint] 已恢复, 完成层: {sorted(completed)}')
    return completed, all_sparsity, all_patterns, all_codebooks_pt, all_masks_pt


# ======================================================================
# 主函数
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='分析 2:4 剪枝模型 (加速版, 带 checkpoint)')
    parser.add_argument('model', type=str)
    parser.add_argument('--mantissa_bits', type=int, default=4)
    parser.add_argument('--row_block_size', type=int, default=-1,
                        help='-1=整列共享, 正数=行分块大小 (128/256/1024 等)')
    parser.add_argument('--skip_mantissa_quant', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./analysis_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--layers', type=int, nargs='*', default=None)
    parser.add_argument('--checkpoint_every', type=int, default=4,
                        help='每处理 N 层保存一次 checkpoint (默认 4)')
    parser.add_argument('--resume', action='store_true',
                        help='从 output_dir 中已有的 checkpoint 恢复')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    rbs_str = "整列共享" if args.row_block_size == -1 else f"{args.row_block_size} 行/块"

    print('=' * 70)
    print('2:4 稀疏 + 共享指数量化 模型分析 (加速版, 带 checkpoint)')
    print('=' * 70)
    print(f'模型路径:        {args.model}')
    print(f'尾数位数:        {args.mantissa_bits}')
    print(f'row_block_size:  {args.row_block_size} ({rbs_str})')
    print(f'跳过尾数量化:    {args.skip_mantissa_quant}')
    print(f'计算设备:        {device}')
    print(f'输出目录:        {args.output_dir}')
    print(f'checkpoint间隔:  每 {args.checkpoint_every} 层')
    if args.layers is not None:
        print(f'只分析层:        {args.layers}')
    print()
    print(f'输出格式 (兼容 pack.py v2):')
    print(f'  codebook_tables.pt : int8 [num_groups, num_rb, 2, 4]')
    print(f'  sparse_masks.pt    : int8 [rows, num_groups, 2]')
    print()

    # 预热 combo 缓存
    for n in range(5, 13):
        get_combo_tensor(n, device)
    print(f'Combo 缓存已预热 (n=5..12, 设备={device})')

    # ---- 尝试恢复 checkpoint ----
    completed_layers = set()
    all_sparsity = {}
    all_patterns = {}
    all_codebooks_pt = {}   # tensor_name → int8 tensor [G, RB, 2, 4]
    all_masks_pt = {}       # tensor_name → int8 tensor [rows, G, 2]

    if args.resume:
        completed_layers, all_sparsity, all_patterns, all_codebooks_pt, all_masks_pt = \
            load_checkpoint(args.output_dir)

    print('\n加载模型...')
    model = get_opt(args.model)
    model.eval()
    layers_list = model.model.decoder.layers
    num_layers = len(layers_list)
    print(f'模型层数: {num_layers}')

    total_time = time.time()
    layers_since_ckpt = 0  # 自上次 checkpoint 以来处理的层数

    for layer_idx in range(num_layers):
        if args.layers is not None and layer_idx not in args.layers:
            continue

        # 跳过已完成的层
        if layer_idx in completed_layers:
            print(f'\n  [跳过] Layer {layer_idx} (已在 checkpoint 中)')
            continue

        print(f'\n{"=" * 60}')
        print(f'Layer {layer_idx}')
        print(f'{"=" * 60}')

        layer = layers_list[layer_idx]
        subset = find_layers(layer)

        for name in subset:
            full_name = f"layer{layer_idx}.{name}"
            print(f'\n  {full_name}')

            W = subset[name].weight.data.clone()
            import transformers
            if isinstance(subset[name], transformers.Conv1D):
                W = W.t()

            t_tensor = time.time()
            sp_info, cb_tensor, pat_info, mask_t = analyze_tensor(
                full_name, W,
                mantissa_bits=args.mantissa_bits,
                row_block_size=args.row_block_size,
                skip_mantissa_quant=args.skip_mantissa_quant,
                device=device
            )
            t_tensor = time.time() - t_tensor

            all_sparsity[full_name] = sp_info
            print(f"    shape: {sp_info['shape']}, "
                  f"稀疏率: {sp_info['sparsity_ratio']:.4f}, "
                  f"2:4有效: {sp_info['is_2_4']}, "
                  f"耗时: {t_tensor:.1f}s")
            if 'num_row_blocks' in sp_info:
                print(f"    groups: {sp_info['num_groups']}, "
                      f"row_blocks: {sp_info['num_row_blocks']}, "
                      f"码表条目: {sp_info['total_codebook_entries']}")

            if pat_info is not None:
                all_patterns[full_name] = pat_info

                # 存 codebook 和 mask (.pt, int8)
                all_codebooks_pt[full_name] = cb_tensor.to(torch.int8).cpu()
                all_masks_pt[full_name] = mask_t.to(torch.int8).cpu()

                print(f"    D0 模式 (top 5):")
                for pat, info in list(pat_info['d0_patterns'].items())[:5]:
                    ex = info['examples'][0]
                    print(f"      gaps={pat}: {info['count']} ({info['pct']})"
                          f"  例: {ex['exponents']} "
                          f"(g={ex['group']}, rb={ex['row_block']})")
                print(f"    D1 模式 (top 5):")
                for pat, info in list(pat_info['d1_patterns'].items())[:5]:
                    ex = info['examples'][0]
                    print(f"      gaps={pat}: {info['count']} ({info['pct']})"
                          f"  例: {ex['exponents']} "
                          f"(g={ex['group']}, rb={ex['row_block']})")

        # 标记该层完成
        completed_layers.add(layer_idx)
        layers_since_ckpt += 1

        # 定期 checkpoint
        if layers_since_ckpt >= args.checkpoint_every:
            save_checkpoint(args.output_dir, completed_layers,
                            all_sparsity, all_patterns,
                            all_codebooks_pt, all_masks_pt)
            layers_since_ckpt = 0

    # =================== 最终保存 ===================
    elapsed = time.time() - total_time
    print(f'\n\n总用时: {elapsed:.1f}s')

    print(f'\n{"=" * 60}')
    print('保存最终结果')
    print(f'{"=" * 60}')

    # 最终 checkpoint (确保最后几层也被保存)
    save_checkpoint(args.output_dir, completed_layers,
                    all_sparsity, all_patterns,
                    all_codebooks_pt, all_masks_pt)

    # 打印文件大小
    for fname in ['codebook_tables.pt', 'sparse_masks.pt',
                   'sparsity_check.json', 'pattern_analysis.json']:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath):
            fsize = os.path.getsize(fpath) / 1024 / 1024
            print(f'  {fname}: {fsize:.1f} MB')

    # =================== 汇总 ===================
    print(f'\n{"=" * 60}')
    print('汇总报告')
    print(f'{"=" * 60}')

    global_d0_patterns = Counter()
    global_d1_patterns = Counter()
    global_d0_total = 0
    global_d1_total = 0

    for tname, pat_info in all_patterns.items():
        for pat, info in pat_info['d0_patterns'].items():
            global_d0_patterns[pat] += info['count']
            global_d0_total += info['count']
        for pat, info in pat_info['d1_patterns'].items():
            global_d1_patterns[pat] += info['count']
            global_d1_total += info['count']

    print(f'\n全局 D0 (大值槽) 指数gap模式:')
    print(f'  {"gap模式":<15} {"频数":>10} {"频率":>10}  {"含义"}')
    print(f'  {"-" * 65}')
    for pat, cnt in global_d0_patterns.most_common(15):
        gaps = [int(x) for x in pat.split('-')]
        offsets = [0]
        for gv in gaps:
            offsets.append(offsets[-1] + gv)
        desc = "连续 [b,b+1,b+2,b+3]" if gaps == [1, 1, 1] else \
            f"[b,b+{offsets[1]},b+{offsets[2]},b+{offsets[3]}]"
        print(f'  {pat:<15} {cnt:>10} {cnt / max(global_d0_total, 1):>9.2%}  {desc}')

    print(f'\n全局 D1 (小值槽) 指数gap模式:')
    print(f'  {"gap模式":<15} {"频数":>10} {"频率":>10}  {"含义"}')
    print(f'  {"-" * 65}')
    for pat, cnt in global_d1_patterns.most_common(15):
        gaps = [int(x) for x in pat.split('-')]
        offsets = [0]
        for gv in gaps:
            offsets.append(offsets[-1] + gv)
        desc = "连续 [b,b+1,b+2,b+3]" if gaps == [1, 1, 1] else \
            f"[b,b+{offsets[1]},b+{offsets[2]},b+{offsets[3]}]"
        print(f'  {pat:<15} {cnt:>10} {cnt / max(global_d1_total, 1):>9.2%}  {desc}')

    global_d0_base = Counter()
    global_d1_base = Counter()
    for tname, pat_info in all_patterns.items():
        for k, v in pat_info.get('d0_base_distribution', {}).items():
            global_d0_base[int(k)] += v
        for k, v in pat_info.get('d1_base_distribution', {}).items():
            global_d1_base[int(k)] += v

    print(f'\n全局 D0 base分布:')
    total_d0b = sum(global_d0_base.values())
    for base, cnt in sorted(global_d0_base.items()):
        print(f'  base={base:>4}: {cnt:>8} ({cnt / max(total_d0b, 1):.2%})')
    print(f'\n全局 D1 base分布:')
    total_d1b = sum(global_d1_base.values())
    for base, cnt in sorted(global_d1_base.items()):
        print(f'  base={base:>4}: {cnt:>8} ({cnt / max(total_d1b, 1):.2%})')

    summary = {
        "config": {
            "model": args.model,
            "mantissa_bits": args.mantissa_bits,
            "row_block_size": args.row_block_size,
            "row_block_size_desc": rbs_str,
            "skip_mantissa_quant": args.skip_mantissa_quant,
            "analyzed_layers": args.layers if args.layers else "all",
            "sharing_direction": {
                "column": "每4列 → 1个 column_group",
                "row": f"每 {rbs_str} → 1个 row_block",
                "codebook_key": "(column_group, row_block)"
            }
        },
        "global_d0_patterns": {
            pat: {"count": cnt, "frequency": round(cnt / max(global_d0_total, 1), 6),
                  "pct": f"{cnt / max(global_d0_total, 1):.2%}"}
            for pat, cnt in global_d0_patterns.most_common()
        },
        "global_d1_patterns": {
            pat: {"count": cnt, "frequency": round(cnt / max(global_d1_total, 1), 6),
                  "pct": f"{cnt / max(global_d1_total, 1):.2%}"}
            for pat, cnt in global_d1_patterns.most_common()
        },
        "global_d0_base_distribution": {str(k): v for k, v in sorted(global_d0_base.items())},
        "global_d1_base_distribution": {str(k): v for k, v in sorted(global_d1_base.items())},
        "total_time_seconds": round(elapsed, 1)
    }

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n  汇总报告 → {summary_path}')
    print('\n完成!')


if __name__ == '__main__':
    main()