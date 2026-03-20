"""
2:4剪枝后的列交换优化 + 共享指数量化
单卡向量化版 - 暴力最优指数选择 + D0/D1差异化
=========================================================
共享指数方向：列方向（column-wise）
  - 每4列为一个column_group
  - 每个 (column_group, row_block) 共享一个4指数码表
  - D0（大值槽）和D1（小值槽）独立选择最优码表

指数选择策略：
  枚举所有可能的4指数组合（从候选指数中选4个）
  选总绝对量化误差最小的组合
  D0/D1独立枚举 → 天然差异化

使用方法：
    python opt_shared_exp_swap_fast.py ./pruned_model wikitext2 \
        --base_model /path/to/tokenizer --mantissa_bits 4 \
        --skip_mantissa_quant --row_block_size 16 --print_err
"""

import time
import math
from collections import Counter
from itertools import combinations
import torch
import torch.nn as nn
import transformers

from datautils import get_loaders
from modelutils import find_layers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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


class SharedExpQuantizerFast:
    """
    共享指数量化器（列方向共享）
    指数选择：暴力枚举最优4指数组合（最小化总绝对量化误差）
    码表大小：4个指数 = 2bit索引
    尾数范围：[1.0, 2.0)，无overlap
    """

    def __init__(self, block_size=128, mantissa_bits=4,
                 row_block_size=-1, skip_mantissa_quant=False,
                 debug_layer_name=None, print_err=False):
        self.block_size = block_size
        self.mantissa_bits = mantissa_bits
        self.row_block_size = row_block_size
        self.skip_mantissa_quant = skip_mantissa_quant
        self.debug_layer_name = debug_layer_name
        self.current_layer_name = None
        self.overflow_weight = 1.5
        self.debug_count = 0
        self.print_err = print_err

    def set_current_layer(self, layer_name):
        self.current_layer_name = layer_name

    # ------------------------------------------------------------------
    # 核心：暴力枚举最优4指数组合
    # ------------------------------------------------------------------

    def _select_top4_bruteforce(self, vals, valid_mask, device):
        """
        暴力枚举所有C(n,4)种4指数组合，选总量化误差最小的。
        
        vals: [N] 该槽位的值
        valid_mask: [N] bool
        返回: sorted list of 4 ints
        """
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
        
        # 候选指数：出现过的所有指数
        # 如果太多（>12），只取频率top12以控制搜索量 C(12,4)=495
        unique_exps = sorted(counts.keys())
        if len(unique_exps) > 12:
            top_exps = [e for e, _ in counts.most_common(12)]
            unique_exps = sorted(top_exps)
        
        # 预计算：abs_v在GPU上, unique_exps构成的scales
        # 对每个候选指数e，计算量化后的重建值和误差
        n_cand = len(unique_exps)
        cand_exp = torch.tensor(unique_exps, dtype=torch.float32, device=device)
        cand_scales = 2.0 ** cand_exp  # [n_cand]
        
        # mantissas[i, j] = abs_v[j] / scale[i]
        mantissas = abs_v.unsqueeze(0) / cand_scales.unsqueeze(1)  # [n_cand, M]
        
        # 有效范围 [1.0, 2.0)
        clamped = torch.clamp(mantissas, 1.0, 2.0 - 1e-7)
        
        if self.skip_mantissa_quant:
            qm = clamped
        else:
            step = 2.0 ** (-self.mantissa_bits)
            qm = torch.round(clamped / step) * step
            qm = torch.clamp(qm, 1.0, 2.0 - step)
        
        # 每个候选指数对每个元素的量化误差
        recon = qm * cand_scales.unsqueeze(1)  # [n_cand, M]
        per_exp_errors = torch.abs(abs_v.unsqueeze(0) - recon)  # [n_cand, M]
        
        # 枚举所有C(n_cand, 4)种组合
        best_cost = float('inf')
        best_combo = None
        
        for combo_indices in combinations(range(n_cand), 4):
            # 这4个指数对每个元素的最小误差
            combo_errors = per_exp_errors[list(combo_indices)]  # [4, M]
            min_errors, _ = combo_errors.min(dim=0)             # [M]
            total_cost = min_errors.sum().item()
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_combo = combo_indices
        
        result = sorted([unique_exps[i] for i in best_combo])
        return result

    # ------------------------------------------------------------------
    # 为一个block选择D0/D1码表
    # ------------------------------------------------------------------

    def _select_shared_exp_from_block(self, W_block, valid_mask, device):
        R, C = W_block.shape
        nz_mask = W_block != 0
        pos_t = torch.arange(C, device=device).unsqueeze(0).expand(R, C)
        masked = torch.where(nz_mask, pos_t,
                             torch.tensor(99, dtype=torch.long, device=device))
        sorted_p, _ = torch.sort(masked, dim=1)
        pos0 = sorted_p[:, 0].clamp(0, C - 1)
        pos1 = sorted_p[:, 1].clamp(0, C - 1)

        ridx = torch.arange(R, device=device)
        v0 = W_block[ridx, pos0]
        v1 = W_block[ridx, pos1]

        need_sort = torch.abs(v0) < torch.abs(v1)
        val_d0 = torch.where(need_sort, v1, v0)
        val_d1 = torch.where(need_sort, v0, v1)

        # D0和D1各自独立暴力搜索最优4指数
        exp_d0 = self._select_top4_bruteforce(val_d0, valid_mask, device)
        exp_d1 = self._select_top4_bruteforce(val_d1, valid_mask, device)

        # 保底差异化：如果完全相同，D1下移1
        if exp_d0 == exp_d1:
            exp_d1 = [e - 1 for e in exp_d1]

        if self.debug_count < 3 and self.current_layer_name:
            self.debug_count += 1
            vd0 = val_d0[valid_mask]
            vd1 = val_d1[valid_mask]
            if len(vd0) > 0:
                e0 = torch.floor(torch.log2(torch.abs(vd0).clamp(min=1e-38))).int()
                e1 = torch.floor(torch.log2(torch.abs(vd1).clamp(min=1e-38))).int()
                e0_counts = Counter(e0.cpu().tolist())
                e1_counts = Counter(e1.cpu().tolist())
                total_d0 = len(vd0)
                total_d1 = len(vd1)
                
                print(f"      [DEBUG] valid={len(vd0)}")
                print(f"        D0(large): range=[{e0.min().item()},{e0.max().item()}], "
                      f"selected={exp_d0}")
                print(f"          top exps: "
                      f"{', '.join(f'{e}:{c/total_d0:.1%}' for e,c in e0_counts.most_common(8))}")
                print(f"        D1(small): range=[{e1.min().item()},{e1.max().item()}], "
                      f"selected={exp_d1}")
                print(f"          top exps: "
                      f"{', '.join(f'{e}:{c/total_d1:.1%}' for e,c in e1_counts.most_common(8))}")
                union = sorted(set(exp_d0) | set(exp_d1))
                print(f"        Union coverage: {union} ({len(union)} binades)")

        return exp_d0, exp_d1

    # ------------------------------------------------------------------
    # 统一量化
    # ------------------------------------------------------------------

    def _quantize_column_unified(self, w, per_row_exp, nz_mask_col, device):
        active = nz_mask_col & (w != 0)
        total_count = active.sum().item()

        q = torch.zeros_like(w)
        overflow = torch.zeros_like(w, dtype=torch.bool)
        flag = torch.zeros_like(w, dtype=torch.int8)

        if total_count == 0:
            return q, overflow, flag, 0, 0, None

        w_a = w[active]
        exp_a = per_row_exp[active]

        signs = torch.sign(w_a)
        abs_w = torch.abs(w_a)

        scales = 2.0 ** exp_a
        mantissas = abs_w.unsqueeze(1) / scales

        valid = (mantissas >= 1.0) & (mantissas < 2.0)
        clamped = torch.clamp(mantissas, 1.0, 2.0 - 1e-7)

        if self.skip_mantissa_quant:
            qm = clamped
        else:
            step = 2.0 ** (-self.mantissa_bits)
            qm = torch.round(clamped / step) * step
            qm = torch.clamp(qm, 1.0, 2.0 - step)

        recon = qm * scales
        errors = torch.abs(abs_w.unsqueeze(1) - recon)

        best_k = torch.argmin(errors, dim=1)
        idx = torch.arange(len(best_k), device=device)

        best_valid = valid[idx, best_k]
        best_q = signs * qm[idx, best_k] * scales[idx, best_k]

        q[active] = best_q
        overflow[active] = ~best_valid
        flag[active] = best_k.to(torch.int8)

        ovf_cnt = (~best_valid).sum().item()
        
        overflow_exp_info = None
        if self.print_err and ovf_cnt > 0:
            ovf_mask = ~best_valid
            ovf_abs = abs_w[ovf_mask]
            ovf_true_exp = torch.floor(torch.log2(ovf_abs.clamp(min=1e-38))).int()
            ovf_selected_exp = exp_a[ovf_mask]
            overflow_exp_info = {
                'true_exps': ovf_true_exp.cpu().tolist(),
                'codebook_exps': ovf_selected_exp[0].cpu().tolist() if ovf_cnt > 0 else [],
            }
        
        return q, overflow, flag, ovf_cnt, total_count, overflow_exp_info

    # ------------------------------------------------------------------
    # 量化误差（用于交换决策）
    # ------------------------------------------------------------------

    def _compute_quant_error(self, val, shared_exp_list, device):
        abs_val = torch.abs(val).clamp(min=1e-38)
        shared_exp = torch.tensor(shared_exp_list, dtype=torch.float32, device=device)
        scales = (2.0 ** shared_exp).view(-1, 1)
        mantissas = abs_val.unsqueeze(0) / scales
        clamped = torch.clamp(mantissas, 1.0, 2.0 - 1e-7)
        if self.skip_mantissa_quant:
            qm = clamped
        else:
            step = 2.0 ** (-self.mantissa_bits)
            qm = torch.round(clamped / step) * step
            qm = torch.clamp(qm, 1.0, 2.0 - step)
        recon = qm * scales
        errors = torch.abs(abs_val.unsqueeze(0) - recon)
        min_err, _ = errors.min(dim=0)
        return min_err

    # ------------------------------------------------------------------
    # 交换决策
    # ------------------------------------------------------------------

    def _recompute_swap_for_group(self, W_group, exp_d0, exp_d1, device):
        R, C = W_group.shape
        nz_mask = W_group != 0
        nz_count = nz_mask.sum(dim=1)
        valid_g = (nz_count == 2)

        pos_t = torch.arange(C, device=device).unsqueeze(0).expand(R, C)
        masked = torch.where(nz_mask, pos_t,
                             torch.tensor(99, dtype=torch.long, device=device))
        sorted_p, _ = torch.sort(masked, dim=1)
        pos0 = sorted_p[:, 0].clamp(0, C - 1)
        pos1 = sorted_p[:, 1].clamp(0, C - 1)

        ridx = torch.arange(R, device=device)
        val0 = W_group[ridx, pos0]
        val1 = W_group[ridx, pos1]

        need_sort = torch.abs(val0) < torch.abs(val1)
        val_d0 = torch.where(need_sort, val1, val0)
        val_d1 = torch.where(need_sort, val0, val1)
        pos_d0 = torch.where(need_sort, pos1, pos0)
        pos_d1 = torch.where(need_sort, pos0, pos1)

        err_no = self._compute_quant_error(val_d0, exp_d0, device) + \
                 self._compute_quant_error(val_d1, exp_d1, device)
        err_sw = self._compute_quant_error(val_d0, exp_d1, device) + \
                 self._compute_quant_error(val_d1, exp_d0, device)
        need_swap = (err_sw < err_no) & valid_g

        final_s0 = torch.where(need_swap, pos_d1, pos_d0)
        final_s1 = torch.where(need_swap, pos_d0, pos_d1)
        sparse_mask_g = torch.stack([final_s0, final_s1], dim=1).to(torch.int8)

        fv0 = torch.where(need_swap, val_d1, val_d0)
        fv1 = torch.where(need_swap, val_d0, val_d1)
        e0 = torch.floor(torch.log2(torch.abs(fv0).clamp(min=1e-38))).int()
        e1 = torch.floor(torch.log2(torch.abs(fv1).clamp(min=1e-38))).int()
        exp_d0_t = torch.tensor(exp_d0, dtype=torch.int32, device=device)
        exp_d1_t = torch.tensor(exp_d1, dtype=torch.int32, device=device)
        ovf0 = ~(e0.unsqueeze(-1) == exp_d0_t).any(dim=-1) & valid_g
        ovf1 = ~(e1.unsqueeze(-1) == exp_d1_t).any(dim=-1) & valid_g
        overflow_mask = torch.stack([ovf0, ovf1], dim=1)
        flag_bits = torch.zeros(R, 2, dtype=torch.int8, device=device)

        return sparse_mask_g, valid_g, need_swap.sum().item(), overflow_mask, flag_bits

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def process_weight_matrix(self, W, Hinv, blocksize, percdamp):
        device = W.device
        rows, cols = W.shape

        stats = {
            'overflow': 0, 'total': 0, 'swap_count': 0, 'total_pairs': 0,
            'overflow_up': 0, 'overflow_down': 0, 'flag_high_count': 0,
        }

        overflow_true_exp_counter = Counter()
        overflow_direction_counter = {'above': 0, 'below': 0}

        if cols % 4 != 0:
            print("    Warning: cols % 4 != 0, standard path")
            Q, stats = self._process_standard(W, Hinv, blocksize, stats)
            return Q, stats, None

        num_groups = cols // 4
        rbs = self.row_block_size
        num_rb = (rows + rbs - 1) // rbs if rbs > 0 else 1

        W_g = W.reshape(rows, num_groups, 4)
        nz_g = (W_g != 0)
        nz_cnt = nz_g.sum(dim=2)
        valid_row_mask = (nz_cnt == 2)

        pos_t4 = torch.arange(4, device=device).view(1, 1, 4).expand(rows, num_groups, 4)
        mp = torch.where(nz_g, pos_t4,
                         torch.tensor(99, dtype=torch.long, device=device))
        sp, _ = torch.sort(mp, dim=2)
        sparse_mask = sp[:, :, :2].clamp(0, 3).to(torch.int8)

        nonzero_col_mask = (W != 0)
        nonzero_col_float = nonzero_col_mask.float()
        stats['total_pairs'] = valid_row_mask.sum().item()

        cache_exp_d0 = {}
        cache_exp_d1 = {}
        cache_swap = {}

        all_exp_d0 = torch.zeros(rows, num_groups, 4, dtype=torch.float32, device=device)
        all_exp_d1 = torch.zeros(rows, num_groups, 4, dtype=torch.float32, device=device)
        all_sm = torch.zeros(rows, num_groups, 2, dtype=torch.int8, device=device)
        all_vg = torch.zeros(rows, num_groups, dtype=torch.bool, device=device)

        err_diag = {'per_group_col_err': []}

        Q = torch.zeros_like(W)

        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            block_nz_float = nonzero_col_float[:, i1:i2]
            block_nz_bool = nonzero_col_mask[:, i1:i2]

            Hinv1 = Hinv[i1:i2, i1:i2] if Hinv is not None else None

            current_group = -1

            for i in range(count):
                col = i1 + i
                g = col // 4
                lc = col % 4

                if g != current_group:
                    current_group = g
                    g_local_start = g * 4 - i1
                    g_local_end = g_local_start + 4

                    if g_local_start >= 0 and g_local_end <= count:
                        W_group_4 = W1[:, g_local_start:g_local_end]
                    else:
                        gs = g * 4
                        W_group_4 = W[:, gs:gs + 4]

                    if rbs > 0:
                        for rb in range(num_rb):
                            key = (g, rb)
                            if key not in cache_exp_d0:
                                rs = rb * rbs
                                re = min(rs + rbs, rows)
                                blk = W_group_4[rs:re, :]
                                vm = valid_row_mask[rs:re, g]

                                ed0, ed1 = self._select_shared_exp_from_block(
                                    blk, vm, device)
                                cache_exp_d0[key] = ed0
                                cache_exp_d1[key] = ed1

                                sm, vg, sc, _, _ = self._recompute_swap_for_group(
                                    blk, ed0, ed1, device)
                                cache_swap[key] = (sm, vg)
                                stats['swap_count'] += sc

                                ed0_t = torch.tensor(ed0, dtype=torch.float32, device=device)
                                ed1_t = torch.tensor(ed1, dtype=torch.float32, device=device)
                                all_exp_d0[rs:re, g, :] = ed0_t
                                all_exp_d1[rs:re, g, :] = ed1_t
                                all_sm[rs:re, g, :] = sm
                                all_vg[rs:re, g] = vg
                    else:
                        if g not in cache_exp_d0:
                            vm = valid_row_mask[:, g]
                            ed0, ed1 = self._select_shared_exp_from_block(
                                W_group_4, vm, device)
                            cache_exp_d0[g] = ed0
                            cache_exp_d1[g] = ed1

                            sm, vg, sc, _, _ = self._recompute_swap_for_group(
                                W_group_4, ed0, ed1, device)
                            cache_swap[g] = (sm, vg)
                            stats['swap_count'] += sc

                            ed0_t = torch.tensor(ed0, dtype=torch.float32, device=device)
                            ed1_t = torch.tensor(ed1, dtype=torch.float32, device=device)
                            all_exp_d0[:, g, :] = ed0_t
                            all_exp_d1[:, g, :] = ed1_t
                            all_sm[:, g, :] = sm
                            all_vg[:, g] = vg

                w = W1[:, i]
                d = Hinv1[i, i].item() if Hinv1 is not None else 1.0
                if d == 0 or math.isnan(d):
                    d = 1.0

                vg = all_vg[:, g]
                sm0 = all_sm[:, g, 0]
                sm1 = all_sm[:, g, 1]

                is_s0 = (sm0 == lc)
                is_s1 = (sm1 == lc)

                use_d0_mask = vg & is_s0
                use_d1_mask = vg & is_s1

                per_row_exp = all_exp_d0[:, g, :].clone()
                per_row_exp[use_d1_mask] = all_exp_d1[use_d1_mask, g, :]

                nz_col = block_nz_bool[:, i]
                q, overflow_col, flag, ovf_cnt, total_cnt, ovf_exp_info = \
                    self._quantize_column_unified(w, per_row_exp, nz_col, device)

                stats['overflow'] += ovf_cnt
                stats['total'] += total_cnt
                stats['flag_high_count'] += (flag == 1).sum().item()

                if self.print_err and ovf_exp_info is not None:
                    for te in ovf_exp_info['true_exps']:
                        overflow_true_exp_counter[te] += 1
                    cb = ovf_exp_info['codebook_exps']
                    if cb:
                        cb_min, cb_max = min(cb), max(cb)
                        for te in ovf_exp_info['true_exps']:
                            if te < cb_min:
                                overflow_direction_counter['below'] += 1
                            elif te > cb_max:
                                overflow_direction_counter['above'] += 1

                if self.print_err and total_cnt > 0:
                    active = nz_col & (w != 0)
                    abs_err = torch.abs(w[active] - q[active])
                    rel_err = abs_err / torch.abs(w[active]).clamp(min=1e-38)
                    
                    d0_cnt = use_d0_mask.sum().item()
                    d1_cnt = use_d1_mask.sum().item()
                    slot_type = "D0" if d0_cnt > d1_cnt else ("D1" if d1_cnt > d0_cnt else "mix")
                    
                    err_diag['per_group_col_err'].append({
                        'group': g, 'local_col': lc, 'slot': slot_type,
                        'mean_abs_err': abs_err.mean().item(),
                        'max_abs_err': abs_err.max().item(),
                        'mean_rel_err': rel_err.mean().item(),
                        'max_rel_err': rel_err.max().item(),
                        'overflow_cnt': ovf_cnt,
                        'total': total_cnt,
                        'd0_count': d0_cnt, 'd1_count': d1_cnt,
                    })

                Q1[:, i] = q

                err1 = (w - q) / d
                Err1[:, i] = err1

                if Hinv1 is not None and i < count - 1:
                    ow = torch.where(overflow_col,
                                     torch.tensor(self.overflow_weight, device=device),
                                     torch.tensor(1.0, device=device))
                    err1_w = err1 * ow
                    hinv_row = Hinv1[i, (i + 1):]
                    compensation = err1_w.unsqueeze(1) * hinv_row.unsqueeze(0)
                    W1[:, (i + 1):] -= compensation * block_nz_float[:, (i + 1):]

            Q[:, i1:i2] = Q1

            if Hinv is not None and i2 < cols:
                fm = nonzero_col_float[:, i2:]
                full_comp = Err1 @ Hinv[i1:i2, i2:]
                W[:, i2:] -= full_comp * fm

        # ---- 打印 ----
        mode_str = f"4x{rbs}块" if rbs > 0 else "整列"
        fr = stats['flag_high_count'] / max(stats['total'], 1)
        print(f"    [{mode_str}共享] swap={stats['swap_count']}/{stats['total_pairs']} "
              f"({stats['swap_count'] / max(stats['total_pairs'], 1):.2%}), "
              f"溢出: {stats['overflow']}/{stats['total']} "
              f"({stats['overflow'] / max(stats['total'], 1):.2%}), "
              f"flag=1: {fr:.2%}")

        if self.print_err and err_diag['per_group_col_err']:
            records = err_diag['per_group_col_err']
            
            d0_records = [r for r in records if r['slot'] == 'D0']
            d1_records = [r for r in records if r['slot'] == 'D1']
            
            if d0_records:
                d0_mean = sum(r['mean_abs_err'] for r in d0_records) / len(d0_records)
                d0_max = max(r['max_abs_err'] for r in d0_records)
                d0_ovf = sum(r['overflow_cnt'] for r in d0_records)
                d0_tot = sum(r['total'] for r in d0_records)
                d0_rel = sum(r['mean_rel_err'] for r in d0_records) / len(d0_records)
                print(f"    [ERR D0(large)] mean_abs={d0_mean:.6f}, max_abs={d0_max:.6f}, "
                      f"mean_rel={d0_rel:.4%}, overflow={d0_ovf}/{d0_tot}")
            
            if d1_records:
                d1_mean = sum(r['mean_abs_err'] for r in d1_records) / len(d1_records)
                d1_max = max(r['max_abs_err'] for r in d1_records)
                d1_ovf = sum(r['overflow_cnt'] for r in d1_records)
                d1_tot = sum(r['total'] for r in d1_records)
                d1_rel = sum(r['mean_rel_err'] for r in d1_records) / len(d1_records)
                print(f"    [ERR D1(small)] mean_abs={d1_mean:.6f}, max_abs={d1_max:.6f}, "
                      f"mean_rel={d1_rel:.4%}, overflow={d1_ovf}/{d1_tot}")
            
            if overflow_true_exp_counter:
                total_ovf = sum(overflow_true_exp_counter.values())
                sorted_ovf = sorted(overflow_true_exp_counter.items())
                ovf_str = ", ".join(f"e={e}:{c}({c/total_ovf:.1%})" for e, c in sorted_ovf)
                print(f"    [OVERFLOW EXP DIST] {ovf_str}")
                
                above = overflow_direction_counter['above']
                below = overflow_direction_counter['below']
                print(f"    [OVERFLOW DIR] above={above}({above/max(total_ovf,1):.1%}), "
                      f"below={below}({below/max(total_ovf,1):.1%}), "
                      f"other={total_ovf-above-below}")
            
            worst = max(records, key=lambda r: r['max_rel_err'])
            print(f"    [WORST] group={worst['group']}, col={worst['local_col']}, "
                  f"slot={worst['slot']}, max_rel={worst['max_rel_err']:.2%}, "
                  f"overflow={worst['overflow_cnt']}/{worst['total']}")
            
            if len(records) >= 8:
                first_groups = sorted(set(r['group'] for r in records))[:2]
                for gid in first_groups:
                    g_records = sorted([r for r in records if r['group'] == gid],
                                       key=lambda r: r['local_col'])
                    trend = " → ".join(
                        f"c{r['local_col']}({r['slot']},d0={r['d0_count']},d1={r['d1_count']}): "
                        f"rel={r['mean_rel_err']:.3%},ovf={r['overflow_cnt']}"
                        for r in g_records
                    )
                    print(f"    [TREND g={gid}] {trend}")

        return Q, stats, sparse_mask

    # ------------------------------------------------------------------
    # 非2:4标准处理
    # ------------------------------------------------------------------

    def _process_standard(self, W, Hinv, blocksize, stats):
        device = W.device
        rows, cols = W.shape
        num_rb = (rows + self.block_size - 1) // self.block_size
        Q = torch.zeros_like(W)
        nz_mask = (W != 0)
        nz_float = nz_mask.float()

        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            bnz = nz_float[:, i1:i2]
            Hinv1 = Hinv[i1:i2, i1:i2] if Hinv is not None else None

            for i in range(i2 - i1):
                w = W1[:, i]
                d = Hinv1[i, i].item() if Hinv1 is not None else 1.0
                if d == 0:
                    d = 1.0

                q = torch.zeros_like(w)
                for rb in range(num_rb):
                    rs = rb * self.block_size
                    re = min(rs + self.block_size, rows)
                    wb = w[rs:re]
                    if (wb != 0).any():
                        nz = wb[wb != 0]
                        exps = torch.floor(torch.log2(torch.abs(nz).clamp(min=1e-38))).int()
                        el = exps.cpu().tolist()
                        ec = Counter(el)
                        se = [e for e, _ in ec.most_common()][:4]
                        while len(se) < 4:
                            se.append(se[-1] + 1)
                        se = sorted(se)
                        exp_t = torch.tensor(se, dtype=torch.float32, device=device)
                        prb = exp_t.unsqueeze(0).expand(re - rs, 4)
                        nz_block = wb != 0
                        qb, _, _, _, _, _ = self._quantize_column_unified(
                            wb, prb, nz_block, device)
                        q[rs:re] = qb
                        stats['total'] += nz_block.sum().item()

                Q1[:, i] = q
                err1 = (w - q) / d
                Err1[:, i] = err1
                if Hinv1 is not None and i < i2 - i1 - 1:
                    hinv_row = Hinv1[i, (i + 1):]
                    comp = err1.unsqueeze(1) * hinv_row.unsqueeze(0)
                    W1[:, (i + 1):] -= comp * bnz[:, (i + 1):]

            Q[:, i1:i2] = Q1
            if Hinv is not None and i2 < cols:
                fm = nz_float[:, i2:]
                fc = Err1 @ Hinv[i1:i2, i2:]
                W[:, i2:] -= fc * fm

        return Q, stats


class SparseGPTWithSwapFast:

    def __init__(self, layer, device=None, mantissa_bits=4,
                 row_block_size=-1, skip_mantissa_quant=False,
                 debug_layer_name=None, print_err=False):
        self.layer = layer
        self.dev = device if device is not None else layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = SharedExpQuantizerFast(
            mantissa_bits=mantissa_bits,
            row_block_size=row_block_size,
            skip_mantissa_quant=skip_mantissa_quant,
            debug_layer_name=debug_layer_name,
            print_err=print_err,
        )

    def set_layer_name(self, name):
        self.quantizer.set_current_layer(name)

    def add_batch(self, inp, out, blocksize=1024):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if torch.isnan(inp).any():
            inp = torch.nan_to_num(inp, nan=0.0)

        inp_max = inp.abs().max().item()
        if inp_max > 1e4:
            inp = inp / (inp_max / 100.0)
        elif inp_max < 1e-8:
            return

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        scale = math.sqrt(2 / self.nsamples)
        inp = scale * inp.float()

        try:
            with torch.cuda.amp.autocast(enabled=False):
                update = inp.matmul(inp.t())
                if torch.isnan(update).any():
                    return
                self.H += update
                if torch.isnan(self.H).any():
                    self.H = torch.eye(self.columns, device=self.dev) * 0.01
        except Exception as e:
            print(f"Warning: Error updating H matrix: {e}")
            if not hasattr(self, 'H') or self.H is None:
                self.H = torch.eye(self.columns, device=self.dev) * 0.01

    def fasterprune_with_shared_exp(self, blocksize=128, percdamp=0.01):
        W = self.layer.weight.data.clone().to(self.dev)
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        try:
            H = torch.linalg.cholesky(H)
        except torch._C._LinAlgError:
            H[diag, diag] += damp * 10
            try:
                H = torch.linalg.cholesky(H)
            except:
                print("Error: Cholesky decomposition failed")
                raise

        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Q, stats, sparse_mask = self.quantizer.process_weight_matrix(
            W, Hinv, blocksize, percdamp)

        self.sparse_mask = sparse_mask

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(
            self.layer.weight.shape).to(self.layer.weight.data.dtype)

        stats['time'] = time.time() - tick
        return stats

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


@torch.no_grad()
def opt_shared_exp_with_swap(model, dataloader, dev, args):
    print('\nStarting Column-wise Shared Exp Quant (Bruteforce Optimal)...')
    print(f'  Mode: {"4x" + str(args.row_block_size) + " block" if args.row_block_size > 0 else "full column"}')
    print(f'  Mantissa bits: {args.mantissa_bits}')
    print(f'  Skip mantissa quant: {getattr(args, "skip_mantissa_quant", False)}')
    print(f'  Print err: {getattr(args, "print_err", False)}')

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
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
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

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    per_tensor_stats = []

    print('Ready.')
    total_stats = {'overflow': 0, 'total': 0, 'swap_count': 0, 'total_pairs': 0}

    for layer_idx in range(len(layers)):
        print(f'\n{"=" * 60}')
        print(f'Layer {layer_idx}')
        print(f'{"=" * 60}')

        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPTWithSwapFast(
                subset[name],
                mantissa_bits=args.mantissa_bits,
                row_block_size=args.row_block_size,
                skip_mantissa_quant=getattr(args, 'skip_mantissa_quant', False),
                debug_layer_name=getattr(args, 'debug_layer', None),
                print_err=getattr(args, 'print_err', False),
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(f'\n  {name}')
            gpts[name].set_layer_name(f"layer{layer_idx}.{name}")
            stats = gpts[name].fasterprune_with_shared_exp(
                blocksize=args.blocksize,
                percdamp=args.percdamp,
            )
            per_tensor_stats.append({
                'layer': layer_idx,
                'name': name,
                'overflow': stats['overflow'],
                'total': stats['total'],
            })
            print(f"    Time: {stats['time']:.1f}s")

            total_stats['overflow'] += stats['overflow']
            total_stats['total'] += stats['total']
            total_stats['swap_count'] += stats.get('swap_count', 0)
            total_stats['total_pairs'] += stats.get('total_pairs', 0)

            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[layer_idx] = layer.cpu()
        del layer, gpts
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return total_stats, per_tensor_stats


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
        print(i)
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

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
    parser.add_argument('model', type=str, help='Path to pruned model')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--blocksize', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=0.01)
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--row_block_size', type=int, default=-1,
                        help='Row block size. -1 = full column')
    parser.add_argument('--debug_layer', type=str, default=None)
    parser.add_argument('--mantissa_bits', type=int, default=4,
                        help='Mantissa quantization bits')
    parser.add_argument('--skip_mantissa_quant', action='store_true',
                        help='Skip mantissa quantization')
    parser.add_argument('--print_err', action='store_true',
                        help='Print error diagnostics')
    args = parser.parse_args()

    DEV = torch.device('cuda:0')

    print('=' * 70)
    print('2:4 Sparse + Column-wise Shared Exp Quant (Bruteforce Optimal)')
    print(f'Row block size: {args.row_block_size} (-1 = full column)')
    print(f'Mantissa bits: {args.mantissa_bits}')
    print(f'Skip mantissa quant: {args.skip_mantissa_quant}')
    print('=' * 70)
    print(f'Model: {args.model}')
    print(f'Tokenizer: {args.base_model}')

    print('\nLoading model...')
    model = get_opt(args.model)
    model.eval()

    print('\nLoading data...')
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=args.base_model)

    print('\n' + '=' * 70)
    print('Quantization')
    print('=' * 70)
    tick = time.time()
    total_stats, per_tensor_stats = opt_shared_exp_with_swap(
        model, dataloader, DEV, args)
    elapsed = time.time() - tick

    print('\n' + '=' * 70)
    print('Final')
    print('=' * 70)
    final_ppl = opt_eval(model, testloader, DEV)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_path = os.path.join(args.save, f'mantissa_{args.mantissa_bits}bit')
        os.makedirs(save_path, exist_ok=True)
        print('\n' + '=' * 70)
        print('Per-Tensor Overflow Statistics')
        print('=' * 70)
        print(f'{"Layer":<10} {"Name":<30} {"Overflow":>10} {"Total":>10} {"Rate":>10}')
        print('-' * 70)
        for item in per_tensor_stats:
            rate = item['overflow'] / max(item['total'], 1)
            print(f'{item["layer"]:<10} {item["name"]:<30} '
                  f'{item["overflow"]:>10} {item["total"]:>10} {rate:>9.2%}')

        model.save_pretrained(save_path)
        print(f'\nSaved to {save_path}')

    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    print(f'Final PPL:    {final_ppl:.3f}')
    print(f'Overflow:     {total_stats["overflow"]}/{total_stats["total"]} '
          f'({total_stats["overflow"] / max(total_stats["total"], 1):.2%})')
    print(f'Swap:         {total_stats["swap_count"]}/{total_stats["total_pairs"]} '
          f'({total_stats["swap_count"] / max(total_stats["total_pairs"], 1):.2%})')
    print(f'Time:         {elapsed:.1f}s')