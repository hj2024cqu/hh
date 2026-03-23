#!/usr/bin/env python3
"""
中间产物格式转换: v1 → v2
=========================================================
v1 格式 (test.py --save_codebooks_per_group --save_masks 生成):
  codebook_tables.json   {"tensor_name": {"g0_rb0": {"d0":[...], "d1":[...]}, ...}}
  sparse_masks.pt        {"tensor_name": int8[rows, groups, 2]}

v2 格式 (pack_v2.py 需要):
  codebook_tables.pt     {"tensor_name": int8[groups, num_rb, 2, 4]}
  sparse_masks.pt        {"tensor_name": uint8[rows, groups]}  (pos0<<2 | pos1)

转换是无损的, 转换后可直接被 pack_v2.py 读取。

Usage:
  python convert_v1_to_v2.py \
      --input_dir  /path/to/v1_output \
      --output_dir /path/to/v2_output \
      --row_block_size 128

  # 原地转换 (覆盖):
  python convert_v1_to_v2.py \
      --input_dir /path/to/output \
      --output_dir /path/to/output \
      --row_block_size 128
"""

import os
import re
import json
import time
import argparse
import torch


def convert_codebook(all_codebooks_json, sparsity_info=None):
    """
    JSON dict → int8 tensor dict

    输入: {"tensor_name": {"g{g}_rb{rb}": {"d0": [4 ints], "d1": [4 ints]}, ...}}
    输出: {"tensor_name": int8[num_groups, num_rb, 2, 4]}

    自动从 key 名推断 num_groups 和 num_rb,
    或从 sparsity_check.json 读取 (更可靠)。
    """
    result = {}

    for tensor_name, cb_dict in all_codebooks_json.items():
        # 解析所有 (g, rb) key
        max_g = -1
        max_rb = -1
        entries = {}

        for key_str, val in cb_dict.items():
            # 解析 "g123_rb45"
            m = re.match(r'g(\d+)_rb(\d+)', key_str)
            if not m:
                print(f'  [WARN] 跳过无法解析的 key: {key_str}')
                continue
            g = int(m.group(1))
            rb = int(m.group(2))
            max_g = max(max_g, g)
            max_rb = max(max_rb, rb)
            entries[(g, rb)] = val

        # 从 sparsity_info 获取精确维度 (如有)
        if sparsity_info and tensor_name in sparsity_info:
            sp = sparsity_info[tensor_name]
            num_groups = sp.get('num_groups', max_g + 1)
            num_rb = sp.get('num_row_blocks', max_rb + 1)
        else:
            num_groups = max_g + 1
            num_rb = max_rb + 1

        # 填充 tensor
        cb_tensor = torch.zeros(num_groups, num_rb, 2, 4, dtype=torch.int8)
        filled = 0

        for (g, rb), val in entries.items():
            d0 = sorted(val['d0'])
            d1 = sorted(val['d1'])
            cb_tensor[g, rb, 0, :] = torch.tensor(d0, dtype=torch.int8)
            cb_tensor[g, rb, 1, :] = torch.tensor(d1, dtype=torch.int8)
            filled += 1

        total_slots = num_groups * num_rb
        result[tensor_name] = cb_tensor

        print(f'  {tensor_name}: [{num_groups}, {num_rb}, 2, 4] int8, '
              f'filled {filled}/{total_slots} '
              f'({filled * 8 / 1024:.1f} KB)')

    return result


def convert_mask(all_masks_v1):
    """
    int8[rows, groups, 2] → uint8[rows, groups]  (pos0<<2 | pos1)
    """
    result = {}

    for tensor_name, mask_v1 in all_masks_v1.items():
        if mask_v1.dim() == 2:
            # 已经是 v2 格式
            print(f'  {tensor_name}: 已经是 v2 格式 {mask_v1.shape}, 跳过')
            result[tensor_name] = mask_v1
            continue

        if mask_v1.dim() != 3 or mask_v1.shape[2] != 2:
            print(f'  [WARN] {tensor_name}: 未知 shape {mask_v1.shape}, 跳过')
            continue

        pos0 = mask_v1[:, :, 0].to(torch.uint8)  # [rows, groups]
        pos1 = mask_v1[:, :, 1].to(torch.uint8)
        packed = (pos0 << 2) | (pos1 & 0x3)

        v1_bytes = mask_v1.numel()       # int8, 每元素1字节
        v2_bytes = packed.numel()         # uint8, 每元素1字节
        result[tensor_name] = packed

        print(f'  {tensor_name}: {list(mask_v1.shape)} → {list(packed.shape)}, '
              f'{v1_bytes/1024:.1f} KB → {v2_bytes/1024:.1f} KB '
              f'(-{(1 - v2_bytes/v1_bytes)*100:.0f}%)')

    return result


def main():
    parser = argparse.ArgumentParser(
        description='中间产物格式转换: v1 (JSON+int8) → v2 (pt tensor+uint8)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='v1 输出目录 (含 codebook_tables.json, sparse_masks.pt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='v2 输出目录 (可与 input_dir 相同以原地覆盖)')
    parser.add_argument('--row_block_size', type=int, default=-1,
                        help='row_block_size (用于校验, -1=自动推断)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    is_inplace = os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir)

    print('=' * 70)
    print('中间产物格式转换: v1 → v2')
    print('=' * 70)
    print(f'  输入目录:  {args.input_dir}')
    print(f'  输出目录:  {args.output_dir}')
    print(f'  原地覆盖:  {is_inplace}')
    print()

    t0 = time.time()

    # ---- 加载 sparsity_check.json (可选, 用于精确维度) ----
    sparsity_info = None
    sp_path = os.path.join(args.input_dir, 'sparsity_check.json')
    if os.path.exists(sp_path):
        with open(sp_path, 'r') as f:
            sparsity_info = json.load(f)
        print(f'已加载 sparsity_check.json ({len(sparsity_info)} tensors)')

    # ---- 转换 codebook ----
    json_cb_path = os.path.join(args.input_dir, 'codebook_tables.json')
    pt_cb_path = os.path.join(args.output_dir, 'codebook_tables.pt')

    if os.path.exists(json_cb_path):
        json_size = os.path.getsize(json_cb_path) / 1024 / 1024
        print(f'\n转换 codebook: {json_cb_path} ({json_size:.1f} MB)')

        with open(json_cb_path, 'r') as f:
            all_cb_json = json.load(f)
        print(f'  包含 {len(all_cb_json)} 个 tensor')

        all_cb_v2 = convert_codebook(all_cb_json, sparsity_info)
        torch.save(all_cb_v2, pt_cb_path)

        pt_size = os.path.getsize(pt_cb_path) / 1024 / 1024
        print(f'\n  保存 → {pt_cb_path} ({pt_size:.2f} MB)')
        print(f'  压缩比: {json_size:.1f} MB → {pt_size:.2f} MB '
              f'({json_size/max(pt_size,0.001):.0f}x)')

        # 原地模式: 备份旧 JSON → .json.bak, 然后保留
        if is_inplace:
            bak_path = json_cb_path + '.bak'
            if not os.path.exists(bak_path):
                os.rename(json_cb_path, bak_path)
                print(f'  原始 JSON 备份 → {bak_path}')
            else:
                print(f'  原始 JSON 备份已存在, 不覆盖')
    else:
        # 检查是否已经是 .pt 格式
        existing_pt = os.path.join(args.input_dir, 'codebook_tables.pt')
        if os.path.exists(existing_pt):
            print(f'\ncodebook 已经是 .pt 格式, 跳过')
            if args.input_dir != args.output_dir:
                import shutil
                shutil.copy2(existing_pt, pt_cb_path)
                print(f'  复制 → {pt_cb_path}')
        else:
            print(f'\n[WARN] 未找到 codebook_tables.json 或 .pt')

    # ---- 转换 mask ----
    mask_in_path = os.path.join(args.input_dir, 'sparse_masks.pt')
    mask_out_path = os.path.join(args.output_dir, 'sparse_masks.pt')

    if os.path.exists(mask_in_path):
        mask_size_before = os.path.getsize(mask_in_path) / 1024 / 1024
        print(f'\n转换 mask: {mask_in_path} ({mask_size_before:.1f} MB)')

        all_masks_v1 = torch.load(mask_in_path, map_location='cpu')
        print(f'  包含 {len(all_masks_v1)} 个 tensor')

        # 检查是否已经是 v2
        sample = next(iter(all_masks_v1.values()))
        if sample.dim() == 2:
            print(f'  已经是 v2 格式 (dim=2), 跳过转换')
            if args.input_dir != args.output_dir:
                import shutil
                shutil.copy2(mask_in_path, mask_out_path)
        else:
            # 备份原始
            if is_inplace:
                bak_path = mask_in_path + '.bak'
                if not os.path.exists(bak_path):
                    os.rename(mask_in_path, bak_path)
                    print(f'  原始备份 → {bak_path}')

            all_masks_v2 = convert_mask(all_masks_v1)
            torch.save(all_masks_v2, mask_out_path)

            mask_size_after = os.path.getsize(mask_out_path) / 1024 / 1024
            print(f'\n  保存 → {mask_out_path} ({mask_size_after:.2f} MB)')
            print(f'  压缩比: {mask_size_before:.1f} MB → {mask_size_after:.2f} MB '
                  f'(-{(1 - mask_size_after/max(mask_size_before,0.001))*100:.0f}%)')
    else:
        print(f'\n[WARN] 未找到 sparse_masks.pt')

    # ---- 复制其他文件 (非原地时) ----
    if not is_inplace:
        for fname in ['sparsity_check.json', 'pattern_analysis.json', 'summary.json']:
            src = os.path.join(args.input_dir, fname)
            dst = os.path.join(args.output_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
                print(f'\n  复制 {fname} → {dst}')

    elapsed = time.time() - t0

    # ---- 验证 ----
    print(f'\n{"=" * 70}')
    print('验证')
    print(f'{"=" * 70}')

    if os.path.exists(pt_cb_path):
        cb_check = torch.load(pt_cb_path, map_location='cpu')
        for name, t in list(cb_check.items())[:2]:
            print(f'  codebook [{name}]: shape={list(t.shape)}, '
                  f'dtype={t.dtype}, range=[{t.min().item()}, {t.max().item()}]')
            # 抽样检查
            g0_rb0_d0 = t[0, 0, 0, :].tolist()
            g0_rb0_d1 = t[0, 0, 1, :].tolist()
            print(f'    g=0,rb=0: D0={g0_rb0_d0}, D1={g0_rb0_d1}')

    if os.path.exists(mask_out_path):
        mask_check = torch.load(mask_out_path, map_location='cpu')
        for name, t in list(mask_check.items())[:2]:
            pos0 = (t >> 2) & 0x3
            pos1 = t & 0x3
            print(f'  mask [{name}]: shape={list(t.shape)}, dtype={t.dtype}')
            print(f'    pos0 range: [{pos0.min().item()}, {pos0.max().item()}], '
                  f'pos1 range: [{pos1.min().item()}, {pos1.max().item()}]')
            # 验证: pos0 != pos1 (2:4结构中两个非零位置不同)
            same = (pos0 == pos1).sum().item()
            total = pos0.numel()
            if same > 0:
                print(f'    [WARN] pos0==pos1 的条目: {same}/{total} '
                      f'({same/total:.2%}) — 可能有无效行')

    # ---- 与 v1 JSON 交叉验证 (抽样) ----
    if os.path.exists(json_cb_path) or os.path.exists(json_cb_path + '.bak'):
        json_src = json_cb_path if os.path.exists(json_cb_path) else json_cb_path + '.bak'
        print(f'\n  交叉验证 (抽样 vs {os.path.basename(json_src)}):')
        with open(json_src, 'r') as f:
            cb_json_check = json.load(f)

        cb_pt_check = torch.load(pt_cb_path, map_location='cpu')
        mismatch = 0
        checked = 0

        for tname in list(cb_json_check.keys())[:3]:  # 检查前3个tensor
            if tname not in cb_pt_check:
                continue
            json_dict = cb_json_check[tname]
            pt_tensor = cb_pt_check[tname]

            # 抽样10个 block
            keys = list(json_dict.keys())[:10]
            for key_str in keys:
                m = re.match(r'g(\d+)_rb(\d+)', key_str)
                if not m:
                    continue
                g, rb = int(m.group(1)), int(m.group(2))
                d0_json = sorted(json_dict[key_str]['d0'])
                d1_json = sorted(json_dict[key_str]['d1'])
                d0_pt = pt_tensor[g, rb, 0, :].tolist()
                d1_pt = pt_tensor[g, rb, 1, :].tolist()

                if d0_json != d0_pt or d1_json != d1_pt:
                    mismatch += 1
                    print(f'    MISMATCH {tname} {key_str}: '
                          f'D0 json={d0_json} pt={d0_pt}, '
                          f'D1 json={d1_json} pt={d1_pt}')
                checked += 1

        if mismatch == 0:
            print(f'    OK: {checked} 条抽样全部匹配')
        else:
            print(f'    [ERROR] {mismatch}/{checked} 条不匹配!')

    print(f'\n总用时: {elapsed:.1f}s')
    print('完成!')


if __name__ == '__main__':
    main()