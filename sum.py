#!/usr/bin/env python3
"""
分析不连续指数的模式分布
=========================

将所有不连续block归类为8种已知模式(7种不连续 + 连续),
超出已知模式的归入"其他"。

用法:
  python analyze_patterns.py /path/to/non_continuous_exponents.json [更多json...]

对比实验:
  python analyze_patterns.py \
    /path/to/baseline/non_continuous_exponents.json \
    /path/to/hessian/non_continuous_exponents.json
"""
import sys
import os
import json
from collections import defaultdict, OrderedDict

# ========== 8种已知模式 ==========
# offset tuple → (编号, 名称, 描述)
KNOWN_PATTERNS = OrderedDict([
    ((0, 1, 2, 3), ("P0", "连续",           "base ~ base+3")),
    ((0, 2, 3, 4), ("P1", "跳1后连续",       "缺base+1")),
    ((0, 1, 1, 2), ("P2", "重复+1",          "base+1出现两次")),
    ((0, 3, 4, 5), ("P3", "跳2后连续",       "缺base+1,+2")),
    ((0, 1, 2, 4), ("P4", "尾部跳1",         "缺base+3")),
    ((0, 1, 3, 4), ("P5", "中间跳1",         "缺base+2")),
    ((0, 1, 5, 6), ("P6", "中间跳3",         "缺base+2,+3,+4")),
    ((0, 1, 4, 5), ("P7", "中间跳2",         "缺base+2,+3")),
])

def offsets_from_exps(selected_exps):
    """将绝对指数列表转为相对offset元组"""
    s = sorted(selected_exps)
    base = s[0]
    return tuple(e - base for e in s)

def classify_pattern(offsets):
    """返回 (编号, 名称) 或 None"""
    info = KNOWN_PATTERNS.get(offsets)
    if info:
        return info[0], info[1]
    return None, None

def analyze_single_json(filepath):
    """分析一个JSON文件, 返回统计结果"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    config = data.get('config', {})
    records = data.get('records', [])
    
    # 按 D0/D1 分别统计
    results = {'D0': defaultdict(int), 'D1': defaultdict(int)}
    # 未知模式收集
    unknown = {'D0': defaultdict(int), 'D1': defaultdict(int)}
    # 按层统计
    per_layer = defaultdict(lambda: {'D0': 0, 'D1': 0})
    # 按tensor类型统计
    per_tensor_type = defaultdict(lambda: {'D0': 0, 'D1': 0})
    
    for r in records:
        slot = r['slot']   # 'D0' or 'D1'
        exps = r['selected_exps']
        offsets = offsets_from_exps(exps)
        
        pid, pname = classify_pattern(offsets)
        if pid is not None:
            results[slot][pid] += 1
        else:
            results[slot]['其他'] += 1
            # 记录未知模式的具体offset
            offset_str = str(offsets)
            unknown[slot][offset_str] += 1
        
        # 按层
        layer_idx = r.get('layer', -1)
        per_layer[layer_idx][slot] += 1
        
        # 按tensor类型 (提取 q_proj / k_proj / v_proj / out_proj / fc1 / fc2)
        tensor_name = r.get('tensor', '')
        # 取最后一个 . 后的部分
        short_name = tensor_name.split('.')[-1] if tensor_name else 'unknown'
        per_tensor_type[short_name][slot] += 1
    
    return config, results, unknown, per_layer, per_tensor_type, len(records)


def print_pattern_table(results, total_records, tag=""):
    """打印模式分布表"""
    print(f"\n{'─'*70}")
    print(f"  模式分布  (总不连续block: {total_records})")
    print(f"{'─'*70}")
    
    # 表头
    header = f"  {'编号':<6} {'模式名':<14} {'offset':<20} {'D0':>8} {'D1':>8} {'合计':>8} {'占比':>8}"
    print(header)
    print(f"  {'─'*68}")
    
    grand_d0 = 0
    grand_d1 = 0
    
    for offsets, (pid, pname, pdesc) in KNOWN_PATTERNS.items():
        if pid == 'P0':
            continue  # 连续模式不在JSON里, 跳过
        d0_count = results['D0'].get(pid, 0)
        d1_count = results['D1'].get(pid, 0)
        total = d0_count + d1_count
        pct = total / max(total_records, 1)
        offset_str = f"({','.join(str(x) for x in offsets)})"
        print(f"  {pid:<6} {pname:<14} {offset_str:<20} {d0_count:>8} {d1_count:>8} {total:>8} {pct:>7.1%}")
        grand_d0 += d0_count
        grand_d1 += d1_count
    
    # 其他
    d0_other = results['D0'].get('其他', 0)
    d1_other = results['D1'].get('其他', 0)
    total_other = d0_other + d1_other
    pct_other = total_other / max(total_records, 1)
    print(f"  {'其他':<6} {'未知模式':<14} {'—':<20} {d0_other:>8} {d1_other:>8} {total_other:>8} {pct_other:>7.1%}")
    
    grand_d0 += d0_other
    grand_d1 += d1_other
    grand_total = grand_d0 + grand_d1
    
    print(f"  {'─'*68}")
    print(f"  {'合计':<6} {'':<14} {'':<20} {grand_d0:>8} {grand_d1:>8} {grand_total:>8} {'100.0%':>8}")


def print_unknown_details(unknown):
    """打印未知模式的具体分布"""
    has_unknown = any(unknown[s] for s in ['D0', 'D1'])
    if not has_unknown:
        return
    
    print(f"\n  未知模式详情:")
    for slot in ['D0', 'D1']:
        if unknown[slot]:
            sorted_items = sorted(unknown[slot].items(), key=lambda x: -x[1])
            top5 = sorted_items[:5]
            print(f"    {slot}: ", end="")
            parts = [f"{off}×{cnt}" for off, cnt in top5]
            remaining = sum(c for _, c in sorted_items[5:])
            msg = ", ".join(parts)
            if remaining > 0:
                msg += f", ...其余{remaining}个"
            print(msg)


def print_per_tensor_summary(per_tensor_type, total_records):
    """按tensor类型打印不连续占比"""
    print(f"\n  按tensor类型:")
    print(f"    {'类型':<14} {'D0':>8} {'D1':>8} {'合计':>8} {'占比':>8}")
    print(f"    {'─'*46}")
    
    sorted_types = sorted(per_tensor_type.items(), key=lambda x: -(x[1]['D0'] + x[1]['D1']))
    for tname, counts in sorted_types:
        d0 = counts['D0']
        d1 = counts['D1']
        total = d0 + d1
        pct = total / max(total_records, 1)
        print(f"    {tname:<14} {d0:>8} {d1:>8} {total:>8} {pct:>7.1%}")


def print_per_layer_summary(per_layer, total_records, num_layers_to_show=5):
    """按层打印不连续分布(只显示最多和最少的几层)"""
    if not per_layer:
        return
    
    sorted_layers = sorted(per_layer.items(), key=lambda x: -(x[1]['D0'] + x[1]['D1']))
    
    print(f"\n  按层分布 (Top-{num_layers_to_show} / Bottom-{num_layers_to_show}):")
    print(f"    {'Layer':<10} {'D0':>8} {'D1':>8} {'合计':>8}")
    print(f"    {'─'*38}")
    
    # Top
    for layer_idx, counts in sorted_layers[:num_layers_to_show]:
        d0 = counts['D0']
        d1 = counts['D1']
        print(f"    Layer {layer_idx:<5} {d0:>8} {d1:>8} {d0+d1:>8}")
    
    if len(sorted_layers) > num_layers_to_show * 2:
        print(f"    {'...':<10}")
    
    # Bottom
    for layer_idx, counts in sorted_layers[-num_layers_to_show:]:
        if (layer_idx, counts) not in sorted_layers[:num_layers_to_show]:
            d0 = counts['D0']
            d1 = counts['D1']
            print(f"    Layer {layer_idx:<5} {d0:>8} {d1:>8} {d0+d1:>8}")


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_patterns.py <json_file> [json_file ...]")
        print()
        print("示例:")
        print("  python analyze_patterns.py results/non_continuous_exponents.json")
        print("  python analyze_patterns.py baseline/non_continuous_exponents.json hessian/non_continuous_exponents.json")
        sys.exit(1)
    
    all_results = []
    
    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            continue
        
        config, results, unknown, per_layer, per_tensor_type, total = analyze_single_json(filepath)
        all_results.append((filepath, config, results, unknown, per_layer, per_tensor_type, total))
        
        # 提取标识
        parent_dir = os.path.basename(os.path.dirname(filepath))
        rb = config.get('row_block_size', '?')
        mb = config.get('mantissa_bits', '?')
        hw = config.get('hessian_weighted', False)
        
        tag_parts = [parent_dir, f"rb={rb}", f"mb={mb}"]
        if hw:
            tag_parts.append("Hessian加权")
        tag = " | ".join(tag_parts)
        
        print(f"\n{'═'*70}")
        print(f"  {tag}")
        print(f"  文件: {filepath}")
        print(f"  总不连续block数: {total}")
        print(f"{'═'*70}")
        
        print_pattern_table(results, total)
        print_unknown_details(unknown)
        print_per_tensor_summary(per_tensor_type, total)
        print_per_layer_summary(per_layer, total)
    
    # 多文件对比
    if len(all_results) > 1:
        print(f"\n\n{'═'*70}")
        print(f"  对比总结")
        print(f"{'═'*70}")
        
        print(f"\n  {'配置':<40} {'不连续总数':>12} {'P1(跳1)':>10} {'P4(尾跳1)':>10} {'P5(中跳1)':>10}")
        print(f"  {'─'*82}")
        
        for filepath, config, results, _, _, _, total in all_results:
            parent = os.path.basename(os.path.dirname(filepath))
            hw = "✓" if config.get('hessian_weighted', False) else "✗"
            label = f"{parent} (H={hw})"
            
            p1 = results['D0'].get('P1', 0) + results['D1'].get('P1', 0)
            p4 = results['D0'].get('P4', 0) + results['D1'].get('P4', 0)
            p5 = results['D0'].get('P5', 0) + results['D1'].get('P5', 0)
            
            print(f"  {label:<40} {total:>12} {p1:>10} {p4:>10} {p5:>10}")


if __name__ == '__main__':
    main()