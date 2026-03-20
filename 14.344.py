"""
2:4剪枝后的列交换优化 + 共享指数量化
CUDA_VISIBLE_DEVICES=3 python opt_shared_exp_swap.py /home/LHZ/opt/code/sparsegpt/compressed_models/stage1_pruned wikitext2 --base_model /home/LHZ/opt/model/opt-6.7b --save /home/LHZ/opt/code/sparsegpt/compressed_models/stage2_sharedxp_0131/16 --mantissa_bits 4 --skip_mantissa_quant --row_block_size 16
=========================================================
"""

import time
import math
from collections import Counter
import torch
import torch.nn as nn
import transformers

from datautils import get_loaders
from modelutils import find_layers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_opt(model_path):
    def skip(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


class SharedExpQuantizerWithSwapV3:
    def __init__(self, block_size=128, num_shared_exp=4, debug_layer_name=None, 
                 row_block_size=-1, mantissa_bits=4, overlap_bits=2,
                 skip_mantissa_quant=False):  # 新增
        self.block_size = block_size
        self.num_shared_exp = num_shared_exp
        self.debug_layer_name = debug_layer_name
        self.current_layer_name = None
        self.row_block_size = row_block_size
 
        # BBFP参数
        self.mantissa_bits = mantissa_bits
        self.overlap_bits = overlap_bits
        self.offset = mantissa_bits - overlap_bits
        
        # 不量化尾数选项
        self.skip_mantissa_quant = skip_mantissa_quant
        
        # OBS补偿权重
        self.overflow_weight = 1.5
        
        # 调试计数
        self.debug_count = 0
    
    def set_current_layer(self, layer_name):
        self.current_layer_name = layer_name

    def select_shared_exp_for_block(self, val_d0, val_d1, valid_mask, device):
        """
        选择共享指数 - 改为返回4个最优指数的列表（可非连续）
        
        Returns:
            exp_list_d0: D0的4个共享指数列表
            exp_list_d1: D1的4个共享指数列表
        """
        val_d0_valid = val_d0[valid_mask]
        val_d1_valid = val_d1[valid_mask]
        
        if len(val_d0_valid) == 0:
            return [-8, -7, -6, -5], [-9, -8, -7, -6]
        
        # 计算指数
        exp_d0 = torch.floor(torch.log2(torch.abs(val_d0_valid).clamp(min=1e-38))).int()
        exp_d1 = torch.floor(torch.log2(torch.abs(val_d1_valid).clamp(min=1e-38))).int()
        
        # 统计指数频率
        exp_d0_list = exp_d0.cpu().tolist()
        exp_d1_list = exp_d1.cpu().tolist()
        
        from collections import Counter
        exp_d0_counts = Counter(exp_d0_list)
        exp_d1_counts = Counter(exp_d1_list)
        
        # 选择频率最高的4个指数
        def select_top4(exp_counts):
            if not exp_counts:
                return [-8, -7, -6, -5]
            sorted_exps = [e for e, _ in exp_counts.most_common()]
            result = sorted_exps[:4]
            while len(result) < 4:
                result.append(result[-1] + 1)
            return sorted(result)
        
        exp_list_d0 = select_top4(exp_d0_counts)
        exp_list_d1 = select_top4(exp_d1_counts)
        
        return exp_list_d0, exp_list_d1

# ============================================================
# 1. quantize_with_clamp() 修改
# ============================================================

    def quantize_with_clamp(self, w_vec, shared_exp_list, device):
        nonzero_mask = w_vec != 0
        total_count = nonzero_mask.sum().item()
        
        if total_count == 0:
            return (w_vec.clone(), 
                    torch.zeros_like(w_vec, dtype=torch.bool),
                    torch.zeros_like(w_vec, dtype=torch.int8),
                    0)
        
        q_vec = torch.zeros_like(w_vec)
        overflow_mask = torch.zeros_like(w_vec, dtype=torch.bool)
        flag_bits = torch.zeros_like(w_vec, dtype=torch.int8)
        
        nonzero_w = w_vec[nonzero_mask]
        signs = torch.sign(nonzero_w)
        abs_w = torch.abs(nonzero_w)
        
        shared_exp = torch.tensor(shared_exp_list, dtype=torch.float32, device=device)
        scales = (2.0 ** shared_exp).unsqueeze(1)
        mantissas = abs_w.unsqueeze(0) / scales
        
        # ========== 修改点 1: 扩展有效范围 ==========
        # 原来: valid_mask_per_exp = (mantissas >= 1.0) & (mantissas < 2.0)
        max_mantissa = 2.0 ** self.overlap_bits   # overlap_bits=2 → 4.0
        valid_mask_per_exp = (mantissas >= 1.0) & (mantissas < max_mantissa)
        
        # ========== 修改点 2: 扩展 clamp 范围 ==========
        # 原来: clamped_mantissas = torch.clamp(mantissas, 1.0, 1.9999999)
        clamped_mantissas = torch.clamp(mantissas, 1.0, max_mantissa - 1e-7)
        
        # ========== 修改点 3: 用 offset 替代 mantissa_bits ==========
        if self.skip_mantissa_quant:
            quantized_mantissas = clamped_mantissas
        else:
            # 原来: quant_step = 2.0 ** (-self.mantissa_bits)
            quant_step = 2.0 ** (-self.offset)  # offset = mantissa_bits - overlap_bits
            quantized_mantissas = torch.round(clamped_mantissas / quant_step) * quant_step
            # 防止 round 后超出范围
            quantized_mantissas = torch.clamp(quantized_mantissas, 1.0, max_mantissa - quant_step)
        
        reconstructed = quantized_mantissas * scales
        errors = torch.abs(abs_w.unsqueeze(0) - reconstructed)
        
        best_k = torch.argmin(errors, dim=0)
        idx = torch.arange(len(best_k), device=device)
        
        best_valid = valid_mask_per_exp[best_k, idx]
        overflow_nonzero = ~best_valid
        
        best_mantissa = quantized_mantissas[best_k, idx]
        best_scale = scales[best_k, 0]
        best_q = signs * best_mantissa * best_scale
        
        q_vec[nonzero_mask] = best_q
        overflow_mask[nonzero_mask] = overflow_nonzero
        flag_bits[nonzero_mask] = best_k.to(torch.int8)
        
        return q_vec, overflow_mask, flag_bits, total_count


# ============================================================
# 2. _compute_quant_error() 修改 —— 同样的三处
# ============================================================

    def _compute_quant_error(self, val, shared_exp_list, device):
        abs_val = torch.abs(val).clamp(min=1e-38)
        
        shared_exp = torch.tensor(shared_exp_list, dtype=torch.float32, device=device)
        scales = (2.0 ** shared_exp).view(-1, 1)
        mantissas = abs_val.unsqueeze(0) / scales
        
        # ========== 同样的三处修改 ==========
        max_mantissa = 2.0 ** self.overlap_bits
        clamped_mantissas = torch.clamp(mantissas, 1.0, max_mantissa - 1e-7)
        
        if self.skip_mantissa_quant:
            quantized_mantissas = clamped_mantissas
        else:
            quant_step = 2.0 ** (-self.offset)
            quantized_mantissas = torch.round(clamped_mantissas / quant_step) * quant_step
            quantized_mantissas = torch.clamp(quantized_mantissas, 1.0, max_mantissa - quant_step)
        
        reconstructed = quantized_mantissas * scales
        errors = torch.abs(abs_val.unsqueeze(0) - reconstructed)
        
        min_errors, _ = errors.min(dim=0)
        return min_errors

    def _recompute_swap_for_group(self, W_group, exp_list_d0, exp_list_d1, device):
        """基于量化误差计算交换决策（先按值大小排序）"""
        rows = W_group.shape[0]
        n_cols = W_group.shape[1]
        
        nonzero_mask = W_group != 0
        nonzero_count = nonzero_mask.sum(dim=1)
        valid_g = (nonzero_count == 2)
        
        # 找非零位置
        pos_template = torch.arange(n_cols, device=device).view(1, -1).expand(rows, n_cols)
        masked_pos = torch.where(nonzero_mask, pos_template, 
                                torch.tensor(99, dtype=torch.long, device=device))
        sorted_pos, _ = torch.sort(masked_pos, dim=1)
        
        pos0 = sorted_pos[:, 0].clamp(0, n_cols-1)
        pos1 = sorted_pos[:, 1].clamp(0, n_cols-1)
        
        row_idx = torch.arange(rows, device=device)
        val0 = W_group[row_idx, pos0]
        val1 = W_group[row_idx, pos1]
        
        # 按绝对值大小排序
        abs_val0 = torch.abs(val0)
        abs_val1 = torch.abs(val1)
        need_sort_swap = abs_val0 < abs_val1
        
        val_d0 = torch.where(need_sort_swap, val1, val0)  # 大值
        val_d1 = torch.where(need_sort_swap, val0, val1)  # 小值
        pos_d0 = torch.where(need_sort_swap, pos1, pos0)
        pos_d1 = torch.where(need_sort_swap, pos0, pos1)
        
        # 计算不交换时的误差
        err_d0_default = self._compute_quant_error(val_d0, exp_list_d0, device)
        err_d1_default = self._compute_quant_error(val_d1, exp_list_d1, device)
        err_no_swap = err_d0_default + err_d1_default
        
        # 计算交换时的误差
        err_d0_swap = self._compute_quant_error(val_d0, exp_list_d1, device)
        err_d1_swap = self._compute_quant_error(val_d1, exp_list_d0, device)
        err_swap = err_d0_swap + err_d1_swap
        
        # 选择误差更小的配置
        need_exp_swap = (err_swap < err_no_swap) & valid_g
        
        # 生成 sparse_mask
        final_pos_slot0 = torch.where(need_exp_swap, pos_d1, pos_d0)
        final_pos_slot1 = torch.where(need_exp_swap, pos_d0, pos_d1)
        
        sparse_mask_g = torch.stack([final_pos_slot0, final_pos_slot1], dim=1).to(torch.int8)
        
        # 溢出检测
        final_val_slot0 = torch.where(need_exp_swap, val_d1, val_d0)
        final_val_slot1 = torch.where(need_exp_swap, val_d0, val_d1)
        
        exp_list_d0_t = torch.tensor(exp_list_d0, dtype=torch.int32, device=device)
        exp_list_d1_t = torch.tensor(exp_list_d1, dtype=torch.int32, device=device)
        
        exp_v0 = torch.floor(torch.log2(torch.abs(final_val_slot0).clamp(min=1e-38))).int()
        exp_v1 = torch.floor(torch.log2(torch.abs(final_val_slot1).clamp(min=1e-38))).int()
        
        overflow_v0 = ~(exp_v0.unsqueeze(-1) == exp_list_d0_t).any(dim=-1)
        overflow_v1 = ~(exp_v1.unsqueeze(-1) == exp_list_d1_t).any(dim=-1)
        
        overflow_mask = torch.stack([overflow_v0 & valid_g, overflow_v1 & valid_g], dim=1)
        
        # flag位（选中的指数索引）
        flag_bits = torch.zeros(rows, 2, dtype=torch.int8, device=device)
        
        return sparse_mask_g, valid_g, need_exp_swap.sum().item(), overflow_mask, flag_bits

    def _select_shared_exp_from_block(self, W_block, valid_mask, sparse_mask_g, device):
        """基于当前块的权重选择共享指数（按绝对值大小排序后）"""
        rows = W_block.shape[0]
        n_cols = W_block.shape[1]
        
        nonzero_mask = W_block != 0
        pos_template = torch.arange(n_cols, device=device).view(1, -1).expand(rows, n_cols)
        masked_pos = torch.where(nonzero_mask, pos_template, 
                                torch.tensor(99, dtype=torch.long, device=device))
        sorted_pos, _ = torch.sort(masked_pos, dim=1)
        
        pos0 = sorted_pos[:, 0].clamp(0, n_cols-1)
        pos1 = sorted_pos[:, 1].clamp(0, n_cols-1)
        
        row_idx = torch.arange(rows, device=device)
        val0 = W_block[row_idx, pos0]
        val1 = W_block[row_idx, pos1]
        
        # 按绝对值大小排序
        abs_val0 = torch.abs(val0)
        abs_val1 = torch.abs(val1)
        need_sort_swap = abs_val0 < abs_val1
        
        val_d0 = torch.where(need_sort_swap, val1, val0)  # 大值
        val_d1 = torch.where(need_sort_swap, val0, val1)  # 小值
        
        # 返回4个指数的列表
        exp_list_d0, exp_list_d1 = self.select_shared_exp_for_block(val_d0, val_d1, valid_mask, device)
        
        # 调试信息
        if self.debug_count < 3 and self.current_layer_name:
            self.debug_count += 1
            val_d0_valid = val_d0[valid_mask]
            val_d1_valid = val_d1[valid_mask]
            if len(val_d0_valid) > 0:
                e0 = torch.floor(torch.log2(torch.abs(val_d0_valid).clamp(min=1e-38))).int()
                e1 = torch.floor(torch.log2(torch.abs(val_d1_valid).clamp(min=1e-38))).int()
                print(f"      [DEBUG] valid={len(val_d0_valid)}, "
                      f"D0: range=[{e0.min().item()}, {e0.max().item()}], exp_list={exp_list_d0}, "
                      f"D1: range=[{e1.min().item()}, {e1.max().item()}], exp_list={exp_list_d1}")
        
        return exp_list_d0, exp_list_d1

    def process_weight_matrix(self, W, Hinv, blocksize, percdamp):
        """处理权重矩阵 - BBFP量化 + 加权OBS补偿"""
        device = W.device
        rows, cols = W.shape
        
        stats = {
            'overflow': 0, 'total': 0, 'swap_count': 0, 'total_pairs': 0,
            'overflow_up': 0, 'overflow_down': 0, 'flag_high_count': 0
        }
        
        if cols % 4 != 0:
            print("    Warning: cols % 4 != 0, using standard processing")
            Q, stats = self._process_standard_local(W, Hinv, blocksize, stats)
            return Q, stats, None
        
        num_groups = cols // 4
        num_row_blocks = (rows + self.row_block_size - 1) // self.row_block_size if self.row_block_size > 0 else 1
        
        # Step 1: 提取sparse_mask
        W_grouped = W.reshape(rows, num_groups, 4)
        nonzero_mask_grouped = W_grouped != 0
        nonzero_count = nonzero_mask_grouped.sum(dim=2)
        valid_row_mask = nonzero_count == 2
        
        pos_template = torch.arange(4, device=device).view(1, 1, 4).expand(rows, num_groups, 4)
        masked_pos = torch.where(nonzero_mask_grouped, pos_template, 
                                torch.tensor(99, dtype=torch.long, device=device))
        sorted_pos, _ = torch.sort(masked_pos, dim=2)
        sparse_mask = sorted_pos[:, :, :2].clamp(0, 3).to(torch.int8)
        
        nonzero_col_mask = (W != 0)
        stats['total_pairs'] = valid_row_mask.sum().item()
        
        # 存储共享指数
        shared_exp_d0 = {}
        shared_exp_d1 = {}
        
        # 存储交换决策和溢出信息
        swap_decisions = {}
        overflow_info = {}
        
        Q = torch.zeros_like(W)
        
        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            count = i2 - i1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Overflow1 = torch.zeros(rows, count, dtype=torch.bool, device=device)
            
            block_nonzero_mask = nonzero_col_mask[:, i1:i2]
            
            if Hinv is not None:
                Hinv1 = Hinv[i1:i2, i1:i2]
            
            current_group = -1
            
            for i in range(count):
                col = i1 + i
                g = col // 4
                local_col = col % 4
                
                # 进入新group时，选择共享指数并计算交换
                if g != current_group:
                    current_group = g
                    g_local_start = g * 4 - i1
                    g_local_end = g_local_start + 4
                    
                    if g_local_start >= 0 and g_local_end <= count:
                        W1_group = W1[:, g_local_start:g_local_end]
                        
                        if self.row_block_size > 0:
                            for rb in range(num_row_blocks):
                                rs = rb * self.row_block_size
                                re = min(rs + self.row_block_size, rows)
                                
                                key = (g, rb)
                                if key not in shared_exp_d0:
                                    W1_block = W1_group[rs:re, :]
                                    valid_block = valid_row_mask[rs:re, g]
                                    sparse_mask_block = sparse_mask[rs:re, g, :]
                                    
                                    exp_d0, exp_d1 = self._select_shared_exp_from_block(
                                        W1_block, valid_block, sparse_mask_block, device
                                    )
                                    shared_exp_d0[key] = exp_d0
                                    shared_exp_d1[key] = exp_d1
                                    
                                    sm, vg, sc, ovf, flags = self._recompute_swap_for_group(
                                        W1_block, exp_d0, exp_d1, device
                                    )
                                    swap_decisions[key] = (sm, vg)
                                    overflow_info[key] = ovf
                                    stats['swap_count'] += sc
                        else:
                            if g not in shared_exp_d0:
                                valid_g = valid_row_mask[:, g]
                                sparse_mask_g = sparse_mask[:, g, :]
                                
                                exp_d0, exp_d1 = self._select_shared_exp_from_block(
                                    W1_group, valid_g, sparse_mask_g, device
                                )
                                shared_exp_d0[g] = exp_d0
                                shared_exp_d1[g] = exp_d1
                                
                                sm, vg, sc, ovf, flags = self._recompute_swap_for_group(
                                    W1_group, exp_d0, exp_d1, device
                                )
                                swap_decisions[g] = (sm, vg)
                                overflow_info[g] = ovf
                                stats['swap_count'] += sc
                    else:
                        g_start_global = g * 4
                        W_group_global = W[:, g_start_global:g_start_global+4]
                        
                        if self.row_block_size > 0:
                            for rb in range(num_row_blocks):
                                rs = rb * self.row_block_size
                                re = min(rs + self.row_block_size, rows)
                                key = (g, rb)
                                if key not in shared_exp_d0:
                                    W_block = W_group_global[rs:re, :]
                                    valid_block = valid_row_mask[rs:re, g]
                                    sparse_mask_block = sparse_mask[rs:re, g, :]
                                    
                                    exp_d0, exp_d1 = self._select_shared_exp_from_block(
                                        W_block, valid_block, sparse_mask_block, device
                                    )
                                    shared_exp_d0[key] = exp_d0
                                    shared_exp_d1[key] = exp_d1
                                    
                                    sm, vg, sc, ovf, flags = self._recompute_swap_for_group(
                                        W_block, exp_d0, exp_d1, device
                                    )
                                    swap_decisions[key] = (sm, vg)
                                    overflow_info[key] = ovf
                                    stats['swap_count'] += sc
                        else:
                            if g not in shared_exp_d0:
                                valid_g = valid_row_mask[:, g]
                                sparse_mask_g = sparse_mask[:, g, :]
                                
                                exp_d0, exp_d1 = self._select_shared_exp_from_block(
                                    W_group_global, valid_g, sparse_mask_g, device
                                )
                                shared_exp_d0[g] = exp_d0
                                shared_exp_d1[g] = exp_d1
                                
                                sm, vg, sc, ovf, flags = self._recompute_swap_for_group(
                                    W_group_global, exp_d0, exp_d1, device
                                )
                                swap_decisions[g] = (sm, vg)
                                overflow_info[g] = ovf
                                stats['swap_count'] += sc
                
                w = W1[:, i]
                
                if Hinv is not None:
                    d = Hinv1[i, i].item()
                    if d == 0 or math.isnan(d):
                        d = 1.0
                else:
                    d = 1.0
                
                q = torch.zeros_like(w)
                overflow_col = torch.zeros(rows, dtype=torch.bool, device=device)
                
                # 按行块量化
# 按行块量化
                if self.row_block_size > 0:
                    # 构建每行对应的共享指数
                    all_exp_d0 = []
                    all_exp_d1 = []
                    all_sparse_mask_list = []
                    all_valid_list = []
                    
                    for rb in range(num_row_blocks):
                        rs = rb * self.row_block_size
                        re = min(rs + self.row_block_size, rows)
                        block_rows = re - rs
                        
                        key = (g, rb)
                        exp_d0_t = torch.tensor(shared_exp_d0[key], dtype=torch.float32, device=device)
                        exp_d1_t = torch.tensor(shared_exp_d1[key], dtype=torch.float32, device=device)
                        
                        all_exp_d0.append(exp_d0_t.unsqueeze(0).expand(block_rows, 4))
                        all_exp_d1.append(exp_d1_t.unsqueeze(0).expand(block_rows, 4))
                        
                        sm, vg = swap_decisions[key]
                        all_sparse_mask_list.append(sm)
                        all_valid_list.append(vg)
                    
                    all_exp_d0 = torch.cat(all_exp_d0, dim=0)
                    all_exp_d1 = torch.cat(all_exp_d1, dim=0)
                    all_sparse_mask = torch.cat(all_sparse_mask_list, dim=0)
                    all_valid = torch.cat(all_valid_list, dim=0)
                    
                    is_slot0 = all_sparse_mask[:, 0] == local_col
                    is_slot1 = all_sparse_mask[:, 1] == local_col
                    use_d0 = all_valid & is_slot0
                    use_d1 = all_valid & is_slot1
                    
                    if use_d0.any():
                        q_d0, ovf_d0, flag_d0, t0 = self._batch_quantize_varied_exp(
                            w[use_d0], all_exp_d0[use_d0], device
                        )
                        q[use_d0] = q_d0
                        overflow_col[use_d0] = ovf_d0
                        stats['overflow'] += ovf_d0.sum().item()
                        stats['total'] += t0
                        stats['flag_high_count'] += (flag_d0 == 1).sum().item()
                    
                    if use_d1.any():
                        q_d1, ovf_d1, flag_d1, t1 = self._batch_quantize_varied_exp(
                            w[use_d1], all_exp_d1[use_d1], device
                        )
                        q[use_d1] = q_d1
                        overflow_col[use_d1] = ovf_d1
                        stats['overflow'] += ovf_d1.sum().item()
                        stats['total'] += t1
                        stats['flag_high_count'] += (flag_d1 == 1).sum().item()
                    
                    invalid_nz = (~all_valid) & (w != 0)
                    if invalid_nz.any():
                        q_inv, ovf_inv, flag_inv, t_inv = self._batch_quantize_varied_exp(
                            w[invalid_nz], all_exp_d0[invalid_nz], device
                        )
                        q[invalid_nz] = q_inv
                        overflow_col[invalid_nz] = ovf_inv
                        stats['overflow'] += ovf_inv.sum().item()
                        stats['total'] += t_inv
                else:
                    current_exp_d0 = shared_exp_d0[g]
                    current_exp_d1 = shared_exp_d1[g]
                    current_sparse_mask_g, current_valid_g = swap_decisions[g]
                    
                    is_slot0 = current_sparse_mask_g[:, 0] == local_col
                    is_slot1 = current_sparse_mask_g[:, 1] == local_col
                    
                    use_d0 = current_valid_g & is_slot0
                    use_d1 = current_valid_g & is_slot1
                    
                    if use_d0.any():
                        q_d0, ovf_d0, flag_d0, t0 = self.quantize_with_clamp(
                            w[use_d0], current_exp_d0, device
                        )
                        q[use_d0] = q_d0
                        
                        # ovf_d0 现在是布尔张量，可以直接使用
                        use_d0_indices = use_d0.nonzero(as_tuple=True)[0]
                        nz_in_d0 = w[use_d0] != 0
                        overflow_col[use_d0_indices[nz_in_d0]] = ovf_d0[nz_in_d0]
                        
                        # 统计
                        stats['overflow'] += ovf_d0.sum().item()
                        stats['total'] += t0
                        stats['flag_high_count'] += (flag_d0 == 1).sum().item()
                    if use_d1.any():
                        q_d1, ovf_d1, flag_d1, t1 = self.quantize_with_clamp(
                            w[use_d1], current_exp_d1, device
                        )
                        q[use_d1] = q_d1
                        use_d1_indices = use_d1.nonzero(as_tuple=True)[0]
                        nz_in_d1 = w[use_d1] != 0
                        overflow_col[use_d1_indices[nz_in_d1]] = ovf_d1[nz_in_d1]
                        stats['overflow'] += ovf_d1.sum().item()
                        stats['total'] += t1
                        stats['flag_high_count'] += (flag_d1 == 1).sum().item()
                    
                    invalid_nz = (~current_valid_g) & (w != 0)
                    if invalid_nz.any():
                        q_inv, ovf_inv, flag_inv, t_inv = self.quantize_with_clamp(
                            w[invalid_nz], current_exp_d0, device
                        )
                        q[invalid_nz] = q_inv
                        inv_indices = invalid_nz.nonzero(as_tuple=True)[0]
                        nz_in_inv = w[invalid_nz] != 0
                        overflow_col[inv_indices[nz_in_inv]] = ovf_inv[nz_in_inv]
                        stats['overflow'] += ovf_inv.sum().item()
                        stats['total'] += t_inv
                
                Q1[:, i] = q
                Overflow1[:, i] = overflow_col
                
                # 加权OBS补偿
                err1 = (w - q) / d
                Err1[:, i] = err1
                
                if Hinv is not None and i < count - 1:
                    future_mask = block_nonzero_mask[:, (i+1):]
                    
                    # 加权：溢出位置给予更大权重
                    overflow_weight = torch.where(
                        overflow_col, 
                        torch.tensor(self.overflow_weight, device=device),
                        torch.tensor(1.0, device=device)
                    )
                    err1_weighted = err1 * overflow_weight
                    
                    # 计算补偿量
                    compensation = err1_weighted.unsqueeze(1) @ Hinv1[i, (i+1):].unsqueeze(0)
                    
                    # 关键：只补偿到非零位置，保持稀疏性
                    W1[:, (i+1):] -= compensation * future_mask.float()
            
            Q[:, i1:i2] = Q1
            
            # block间补偿
            if Hinv is not None and i2 < cols:
                future_mask = nonzero_col_mask[:, i2:]
                full_compensation = Err1 @ Hinv[i1:i2, i2:]
                W[:, i2:] -= full_compensation * future_mask.float()
        
        mode_str = f"4x{self.row_block_size}块" if self.row_block_size > 0 else "整列"
        flag_ratio = stats['flag_high_count'] / max(stats['total'], 1)
        print(f"    [{mode_str}共享] swap={stats['swap_count']}/{stats['total_pairs']} "
              f"({stats['swap_count']/max(stats['total_pairs'],1):.2%}), "
              f"溢出: {stats['overflow']}/{stats['total']} "
              f"({stats['overflow']/max(stats['total'],1):.2%}), "
              f"flag=1: {flag_ratio:.2%}")
        
        return Q, stats, sparse_mask


    def _batch_quantize_varied_exp(self, w_vec, exp_matrix, device):
        nonzero_mask = w_vec != 0
        total_count = nonzero_mask.sum().item()
        
        if total_count == 0:
            return (w_vec.clone(),
                    torch.zeros_like(w_vec, dtype=torch.bool),
                    torch.zeros_like(w_vec, dtype=torch.int8),
                    0)
        
        q_vec = torch.zeros_like(w_vec)
        overflow_mask = torch.zeros_like(w_vec, dtype=torch.bool)
        flag_bits = torch.zeros_like(w_vec, dtype=torch.int8)
        
        nz_w = w_vec[nonzero_mask]
        nz_exp = exp_matrix[nonzero_mask]
        
        signs = torch.sign(nz_w)
        abs_w = torch.abs(nz_w)
        
        scales = 2.0 ** nz_exp
        mantissas = abs_w.unsqueeze(1) / scales
        
        # ========== 同样的三处修改 ==========
        max_mantissa = 2.0 ** self.overlap_bits
        valid_mask_per_exp = (mantissas >= 1.0) & (mantissas < max_mantissa)
        clamped_mantissas = torch.clamp(mantissas, 1.0, max_mantissa - 1e-7)
        
        if self.skip_mantissa_quant:
            quantized_mantissas = clamped_mantissas
        else:
            quant_step = 2.0 ** (-self.offset)
            quantized_mantissas = torch.round(clamped_mantissas / quant_step) * quant_step
            quantized_mantissas = torch.clamp(quantized_mantissas, 1.0, max_mantissa - quant_step)
        
        reconstructed = quantized_mantissas * scales
        errors = torch.abs(abs_w.unsqueeze(1) - reconstructed)
        
        best_k = torch.argmin(errors, dim=1)
        idx = torch.arange(len(best_k), device=device)
        
        best_valid = valid_mask_per_exp[idx, best_k]
        overflow_nz = ~best_valid
        
        best_mantissa = quantized_mantissas[idx, best_k]
        best_scale = scales[idx, best_k]
        best_q = signs * best_mantissa * best_scale
        
        q_vec[nonzero_mask] = best_q
        overflow_mask[nonzero_mask] = overflow_nz
        flag_bits[nonzero_mask] = best_k.to(torch.int8)
        
        return q_vec, overflow_mask, flag_bits, total_count

    def _process_standard_local(self, W, Hinv, blocksize, stats):
        """非2:4情况的标准处理"""
        device = W.device
        rows, cols = W.shape
        num_row_blocks = (rows + self.block_size - 1) // self.block_size
        Q = torch.zeros_like(W)
        nonzero_mask = (W != 0)
        
        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            block_nz_mask = nonzero_mask[:, i1:i2]
            
            if Hinv is not None:
                Hinv1 = Hinv[i1:i2, i1:i2]
            
            for i in range(i2 - i1):
                w = W1[:, i]
                d = Hinv1[i, i].item() if Hinv is not None else 1.0
                if d == 0:
                    d = 1.0
                
                q = torch.zeros_like(w)
                for rb in range(num_row_blocks):
                    rs = rb * self.block_size
                    re = min(rs + self.block_size, rows)
                    w_block = w[rs:re]
                    
                    if (w_block != 0).any():
                        nz = w_block[w_block != 0]
                        abs_nz = torch.abs(nz).clamp(min=1e-38)
                        exps = torch.floor(torch.log2(abs_nz)).int()
                        
                        # ✅ 修正：选择频率最高的4个指数
                        exp_list = exps.cpu().tolist()
                        exp_counts = Counter(exp_list)
                        sorted_exps = [e for e, _ in exp_counts.most_common()]
                        shared_exp_list = sorted_exps[:4]
                        while len(shared_exp_list) < 4:
                            shared_exp_list.append(shared_exp_list[-1] + 1)
                        shared_exp_list = sorted(shared_exp_list)
                        
                        q_block, _, _, _ = self.quantize_with_clamp(w_block, shared_exp_list, device)
                        q[rs:re] = q_block
                        stats['total'] += (w_block != 0).sum().item()
                
                Q1[:, i] = q
                err1 = (w - q) / d
                Err1[:, i] = err1
                
                if Hinv is not None and i < i2 - i1 - 1:
                    future_mask = block_nz_mask[:, (i+1):]
                    compensation = err1.unsqueeze(1) @ Hinv1[i, (i+1):].unsqueeze(0)
                    W1[:, (i+1):] -= compensation * future_mask.float()
            
            Q[:, i1:i2] = Q1
            if Hinv is not None and i2 < cols:
                future_mask = nonzero_mask[:, i2:]
                full_comp = Err1 @ Hinv[i1:i2, i2:]
                W[:, i2:] -= full_comp * future_mask.float()
        
        return Q, stats


class SparseGPTWithSwapV3:
    """SparseGPT变体，支持BBFP风格共享指数量化 + 列交换"""
    
    def __init__(self, layer, device=None, debug_layer_name=None, row_block_size=-1,
                 mantissa_bits=4, overlap_bits=2, skip_mantissa_quant=False):
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
        self.quantizer = SharedExpQuantizerWithSwapV3(
            debug_layer_name=debug_layer_name, 
            row_block_size=row_block_size,
            mantissa_bits=mantissa_bits,
            overlap_bits=overlap_bits,
            skip_mantissa_quant=skip_mantissa_quant
        )
        self.quantizer.mantissa_bits = mantissa_bits
    
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
        
        Q, stats, sparse_mask = self.quantizer.process_weight_matrix(W, Hinv, blocksize, percdamp)
        
        self.sparse_mask = sparse_mask
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        
        stats['time'] = time.time() - tick
        
        return stats
    
    def free(self):
        self.H = None
        torch.cuda.empty_cache()


@torch.no_grad()
def opt_shared_exp_with_swap(model, dataloader, dev, args):
    print('\nStarting BBFP-style Shared Exponent Quantization V4...')
    print(f'  Mode: {"4x" + str(args.row_block_size) + " block" if args.row_block_size > 0 else "full column"}')
    print(f'  Mantissa bits: {args.mantissa_bits}, Overlap bits: {args.overlap_bits}')
    print(f'  Skip mantissa quant: {getattr(args, "skip_mantissa_quant", False)}')  # 新增
    
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
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    per_tensor_stats = []
    
    print('Ready.')
    
    total_stats = {'overflow': 0, 'total': 0, 'swap_count': 0, 'total_pairs': 0}
    
   # ===== 删除后 =====
    for layer_idx in range(len(layers)):
        print(f'\n{"="*60}')
        print(f'Layer {layer_idx}')
        print(f'{"="*60}')
        
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPTWithSwapV3(
                subset[name], 
                debug_layer_name=getattr(args, 'debug_layer', None),
                row_block_size=args.row_block_size,
                mantissa_bits=args.mantissa_bits,
                overlap_bits=args.overlap_bits,
                skip_mantissa_quant=getattr(args, 'skip_mantissa_quant', False)  # 新增
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
                percdamp=args.percdamp
            )
            per_tensor_stats.append({
                'layer': layer_idx,
                'name': name,
                'overflow': stats['overflow'],
                'total': stats['total']
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
                        help='Row block size. -1 for full column, 128 for 4x128 block')
    parser.add_argument('--overlap_bits', type=int, default=2,
                        help='Overlap bit width for BBFP quantization')
    parser.add_argument('--debug_layer', type=str, default=None)
    parser.add_argument('--mantissa_bits', type=int, default=4,
                        help='Mantissa quantization bits: 3, 4, or 5')
    parser.add_argument('--skip_mantissa_quant', action='store_true',
                    help='Skip mantissa quantization (keep full precision mantissa)')
    args = parser.parse_args()
    
    DEV = torch.device('cuda:0')
    
    print('='*70)
    print('2:4 Sparse + BBFP-style Shared Exponent Quantization V4')
    print(f'Row block size: {args.row_block_size} (-1 = full column)')
    print(f'Mantissa bits: {args.mantissa_bits}')
    print(f'BBFP config: mantissa={args.mantissa_bits}, overlap={args.overlap_bits}')
    print('='*70)
    print(f'Model: {args.model}')
    print(f'Tokenizer: {args.base_model}')
    
    print('\nLoading model...')
    model = get_opt(args.model)
    model.eval()
    
    print('\nLoading data...')
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=args.base_model
    )
    
    print('\n' + '='*70)
    print('Quantization')
    print('='*70)
    tick = time.time()
    total_stats, per_tensor_stats = opt_shared_exp_with_swap(model, dataloader, DEV, args)
    
    elapsed = time.time() - tick
    
    print('\n' + '='*70)
    print('Final')
    print('='*70)
    final_ppl = opt_eval(model, testloader, DEV)
    
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_path = os.path.join(args.save, f'mantissa_{args.mantissa_bits}bit')
        os.makedirs(save_path, exist_ok=True)
        print('\n' + '='*70)
        print('Per-Tensor Overflow Statistics')
        print('='*70)
        print(f'{"Layer":<10} {"Name":<30} {"Overflow":>10} {"Total":>10} {"Rate":>10}')
        print('-'*70)
        for item in per_tensor_stats:
            rate = item['overflow'] / max(item['total'], 1)
            print(f'{item["layer"]:<10} {item["name"]:<30} {item["overflow"]:>10} {item["total"]:>10} {rate:>9.2%}')
        
        model.save_pretrained(save_path)
        print(f'\nSaved to {args.save}')
    
    print('\n' + '='*70)
    print('Summary')
    print('='*70)
    print(f'Final PPL:    {final_ppl:.3f}')
    print(f'Overflow:     {total_stats["overflow"]}/{total_stats["total"]} ({total_stats["overflow"]/max(total_stats["total"],1):.2%})')
    print(f'Swap:         {total_stats["swap_count"]}/{total_stats["total_pairs"]} ({total_stats["swap_count"]/max(total_stats["total_pairs"],1):.2%})')
    print(f'Time:         {elapsed:.1f}s')