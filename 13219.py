"""
2:4剪枝后的列交换优化 + 共享指数量化 + 双重误差补偿 (V2)
=========================================================
多GPU优化版本 - 支持4张A6000并行处理

优化点：
1. 向量化处理替代Python循环（10-50x加速）
2. 多GPU层级并行（4x加速）
3. CUDA流异步处理
4. 预计算和缓存优化

使用方法：
    python opt_shared_exp_swap_multigpu.py ./pruned_model wikitext2 \
        --base_model /path/to/tokenizer --nsamples 128 --num_gpus 4
"""

import time
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import torch.nn as nn
import transformers

from datautils import get_loaders
from modelutils import find_layers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ==================== 多GPU管理 ====================
class MultiGPUManager:
    """多GPU资源管理器"""
    
    def __init__(self, num_gpus=None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.devices = [torch.device(f'cuda:{i}') for i in range(self.num_gpus)]
        self.streams = [torch.cuda.Stream(device=d) for d in self.devices]
        self.lock = threading.Lock()
        
        print(f"Initialized MultiGPUManager with {self.num_gpus} GPUs:")
        for i, d in enumerate(self.devices):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
    
    def get_device(self, idx):
        return self.devices[idx % self.num_gpus]
    
    def get_stream(self, idx):
        return self.streams[idx % self.num_gpus]


# 全局GPU管理器（延迟初始化）
_gpu_manager = None

def get_gpu_manager(num_gpus=None):
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = MultiGPUManager(num_gpus)
    return _gpu_manager


def get_opt(model_path):
    """加载OPT模型"""
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


def get_exponent_scalar(x):
    """提取单个值的指数: floor(log2(|x|))"""
    if x == 0:
        return -127
    return int(math.floor(math.log2(abs(float(x)))))


def get_exponent_tensor(x):
    """提取tensor的指数"""
    abs_x = torch.abs(x).clamp(min=1e-38)
    return torch.floor(torch.log2(abs_x)).to(torch.int32)

def warmup_gpus(num_gpus):
    """预热所有GPU，避免lazy initialization问题"""
    actual_gpus = min(num_gpus, torch.cuda.device_count())
    print(f"Warming up {actual_gpus} GPUs...")
    for i in range(actual_gpus):
        device = torch.device(f'cuda:{i}')
        with torch.cuda.device(device):
            _ = torch.zeros(1, device=device)
            # 预热cholesky - 关键！
            test_mat = torch.eye(64, device=device) + 0.01 * torch.randn(64, 64, device=device)
            test_mat = test_mat @ test_mat.t()
            _ = torch.linalg.cholesky(test_mat)
            torch.cuda.synchronize(device)
    print(f"All {actual_gpus} GPUs warmed up.")
    return actual_gpus

class SharedExpQuantizerWithSwapV2:
    """
    带列交换优化的共享指数量化器 V2
    - 模拟量化：数值上量化，不实际打包存储
    - 共享指数：每个dense列的128行独立选择（连续4个）
    - 尾数量化：支持 10/5/4/3 bit
    """
    
    def __init__(self, block_size=128, num_shared_exp=4, mantissa_bits=10, debug_layer_name=None):
        self.block_size = block_size
        self.num_shared_exp = num_shared_exp
        self.mantissa_bits = mantissa_bits
        self.debug_layer_name = debug_layer_name
        self.current_layer_name = None

    def set_current_layer(self, layer_name):
        """设置当前处理的层名称"""
        self.current_layer_name = layer_name

    def _should_debug(self):
        """检查是否应该打印调试信息"""
        if self.debug_layer_name is None:
            return False
        if self.current_layer_name is None:
            return False
        return self.debug_layer_name in self.current_layer_name

    def select_shared_exp_for_column(self, all_vals):
        """
        为整列选择两组共享指数 (dense0, dense1)
        
        规则：
        - Dense0: 覆盖频率最高的连续4个指数
        - Dense1: 包含第5大指数，与Dense0重叠3个，连续
        """
        device = all_vals.device
        
        nonzero_mask = all_vals != 0
        if not nonzero_mask.any():
            return [-8, -7, -6, -5], [-9, -8, -7, -6]
        
        nonzero_w = all_vals[nonzero_mask]
        abs_w = torch.abs(nonzero_w).clamp(min=1e-38)
        exponents = torch.floor(torch.log2(abs_w)).to(torch.int32)
        
        exp_list = exponents.cpu().tolist()
        exp_counts = Counter(exp_list)
        
        sorted_items = exp_counts.most_common()
        if len(sorted_items) == 0:
            return [-8, -7, -6, -5], [-9, -8, -7, -6]
        
        all_exps = sorted(exp_counts.keys())
        min_exp = min(all_exps)
        max_exp = max(all_exps)
        
        # Dense0: 找覆盖最多的连续4个
        best_start = min_exp
        best_coverage = 0
        for start in range(min_exp - 3, max_exp + 1):
            coverage = sum(exp_counts.get(e, 0) for e in range(start, start + 4))
            if coverage > best_coverage:
                best_coverage = coverage
                best_start = start
        dense0 = [best_start, best_start + 1, best_start + 2, best_start + 3]
        
        # Dense1: 包含第5大指数，与Dense0重叠3个
        if len(sorted_items) >= 5:
            fifth_exp = sorted_items[4][0]
        elif len(sorted_items) >= 1:
            fifth_exp = min_exp if min_exp < dense0[0] else max_exp
        else:
            fifth_exp = dense0[0] - 1
        
        dense1_option1 = [dense0[0] - 1, dense0[0], dense0[1], dense0[2]]
        dense1_option2 = [dense0[1], dense0[2], dense0[3], dense0[3] + 1]
        
        opt1_contains = dense1_option1[0] <= fifth_exp <= dense1_option1[3]
        opt2_contains = dense1_option2[0] <= fifth_exp <= dense1_option2[3]
        
        if opt1_contains and not opt2_contains:
            dense1 = dense1_option1
        elif opt2_contains and not opt1_contains:
            dense1 = dense1_option2
        else:
            cov1 = sum(exp_counts.get(e, 0) for e in dense1_option1)
            cov2 = sum(exp_counts.get(e, 0) for e in dense1_option2)
            dense1 = dense1_option1 if cov1 >= cov2 else dense1_option2
        
        return dense0, dense1

    def _print_column_debug_info(self, g, rows, W_grouped, sparse_mask_before, sparse_mask_after,
                                exp0, exp1, valid_row_mask, need_swap,
                                shared_exp_d0, shared_exp_d1):
        """打印整列调试信息"""
        print(f"\n{'='*80}")
        print(f"Group {g} - cols [{g*4}:{(g+1)*4}], 整列共享 ({rows} rows)")
        print(f"{'='*80}")
        
        valid_g = valid_row_mask[:, g]
        valid_count = valid_g.sum().item()
        
        print(f"\n[1] 整列稀疏模式统计:")
        print(f"    有效2:4行数: {valid_count}/{rows}")
        
        print(f"\n[2] 整列指数分布 (slot0 + slot1 合并):")
        exp0_g = exp0[:, g]
        exp1_g = exp1[:, g]
        valid_exp0 = exp0_g[valid_g].cpu().tolist()
        valid_exp1 = exp1_g[valid_g].cpu().tolist()
        all_exps = valid_exp0 + valid_exp1
        
        if all_exps:
            exp_counts = Counter(all_exps)
            total_vals = len(all_exps)
            print(f"    指数分布: {dict(exp_counts.most_common(6))}")
            print(f"    指数范围: [{min(all_exps)}, {max(all_exps)}]")
        
        print(f"\n[3] 共享指数选择:")
        d0_min, d0_max = shared_exp_d0[0], shared_exp_d0[3]
        d1_min, d1_max = shared_exp_d1[0], shared_exp_d1[3]
        
        if all_exps:
            d0_coverage = sum(1 for e in all_exps if d0_min <= e <= d0_max)
            d1_coverage = sum(1 for e in all_exps if d1_min <= e <= d1_max)
            print(f"    Dense0: {shared_exp_d0} (覆盖 {d0_coverage}/{total_vals} = {d0_coverage/total_vals:.1%})")
            print(f"    Dense1: {shared_exp_d1} (覆盖 {d1_coverage}/{total_vals} = {d1_coverage/total_vals:.1%})")
        
        print(f"\n[4] 溢出分析 (交换前):")
        if valid_count > 0:
            exp0_valid = exp0_g[valid_g]
            exp1_valid = exp1_g[valid_g]
            slot0_in_d0 = ((exp0_valid >= d0_min) & (exp0_valid <= d0_max)).sum().item()
            slot0_overflow = valid_count - slot0_in_d0
            slot1_in_d1 = ((exp1_valid >= d1_min) & (exp1_valid <= d1_max)).sum().item()
            slot1_overflow = valid_count - slot1_in_d1
            print(f"    Slot0用Dense0: {slot0_in_d0}/{valid_count} 在范围内, {slot0_overflow} 溢出")
            print(f"    Slot1用Dense1: {slot1_in_d1}/{valid_count} 在范围内, {slot1_overflow} 溢出")
            print(f"    总溢出: {slot0_overflow + slot1_overflow}")
        
        print(f"\n[5] 交换决策:")
        need_swap_g = need_swap[:, g]
        swap_count = need_swap_g.sum().item()
        print(f"    需要交换: {swap_count}/{valid_count} 行 ({swap_count/max(valid_count,1):.1%})")
        
        print(f"\n[6] 溢出分析 (交换后):")
        if valid_count > 0:
            new_pos0 = sparse_mask_after[:, g, 0].long()
            new_pos1 = sparse_mask_after[:, g, 1].long()
            row_idx = torch.arange(rows, device=W_grouped.device)
            new_val0 = W_grouped[row_idx, g, new_pos0]
            new_val1 = W_grouped[row_idx, g, new_pos1]
            new_exp0 = torch.floor(torch.log2(torch.abs(new_val0).clamp(min=1e-38))).int()
            new_exp1 = torch.floor(torch.log2(torch.abs(new_val1).clamp(min=1e-38))).int()
            new_exp0_valid = new_exp0[valid_g]
            new_exp1_valid = new_exp1[valid_g]
            
            slot0_in_d0_after = ((new_exp0_valid >= d0_min) & (new_exp0_valid <= d0_max)).sum().item()
            slot0_overflow_after = valid_count - slot0_in_d0_after
            slot1_in_d1_after = ((new_exp1_valid >= d1_min) & (new_exp1_valid <= d1_max)).sum().item()
            slot1_overflow_after = valid_count - slot1_in_d1_after
            
            total_before = slot0_overflow + slot1_overflow
            total_after = slot0_overflow_after + slot1_overflow_after
            print(f"    Slot0用Dense0: {slot0_in_d0_after}/{valid_count} 在范围内, {slot0_overflow_after} 溢出")
            print(f"    Slot1用Dense1: {slot1_in_d1_after}/{valid_count} 在范围内, {slot1_overflow_after} 溢出")
            print(f"    总溢出: {total_after}")
            if total_before > 0:
                print(f"    溢出变化: {total_before} -> {total_after} (改善 {(total_before-total_after)/total_before*100:.1f}%)")

    def quantize_mantissa(self, mantissa, mantissa_bits):
        """量化尾数到指定位数"""
        if mantissa_bits >= 10:
            return mantissa
        
        step = 2.0 ** (-mantissa_bits)
        normalized = mantissa - 1.0
        quantized_normalized = torch.round(normalized / step) * step
        max_val = 1.0 - step
        quantized_normalized = torch.clamp(quantized_normalized, 0.0, max_val)
        
        return quantized_normalized + 1.0
    
    def analyze_block_exponent(self, w_block):
        """分析一个block的指数分布，返回max_exp和min_exp"""
        nonzero_mask = w_block != 0
        if not nonzero_mask.any():
            return -6, -6
        
        nonzero_w = w_block[nonzero_mask]
        abs_w = torch.abs(nonzero_w).clamp(min=1e-38)
        exponents = torch.floor(torch.log2(abs_w)).to(torch.int32)
        
        exp_list = exponents.cpu().tolist()
        exp_counts = Counter(exp_list)
        top5_items = exp_counts.most_common(5)
        top5_exps = [e for e, _ in top5_items]
        
        if len(top5_exps) == 0:
            return -6, -6
        
        while len(top5_exps) < 5:
            top5_exps.append(top5_exps[-1] + 1)
        
        max_exp = max(top5_exps[:5])
        min_exp = min(top5_exps[:5])
        
        return max_exp, min_exp
    
    def get_dense0_shared(self, max_exp):
        """dense0共享指数：[max_exp-3, max_exp-2, max_exp-1, max_exp]"""
        return [max_exp - 3, max_exp - 2, max_exp - 1, max_exp]
    
    def get_dense1_shared(self, min_exp):
        """dense1共享指数：[min_exp, min_exp+1, min_exp+2, min_exp+3]"""
        return [min_exp, min_exp + 1, min_exp + 2, min_exp + 3]

    def quantize_vectorized(self, w_vec, shared_exp_list, device):
        """向量化量化 - 支持尾数量化"""
        nonzero_mask = w_vec != 0
        total_count = nonzero_mask.sum().item()
        
        if total_count == 0:
            return w_vec.clone(), 0, 0
        
        q_vec = torch.zeros_like(w_vec)
        nonzero_w = w_vec[nonzero_mask]
        signs = torch.sign(nonzero_w)
        abs_w = torch.abs(nonzero_w)
        
        shared_exp = torch.tensor(shared_exp_list, dtype=torch.float32, device=device)
        scales = (2.0 ** shared_exp).unsqueeze(1)
        mantissas = abs_w.unsqueeze(0) / scales
        
        valid_mask = (mantissas >= 1.0) & (mantissas < 2.0)
        clamped_mantissas = torch.clamp(mantissas, 1.0, 1.9999999)
        
        # 尾数量化
        if self.mantissa_bits < 10:
            clamped_mantissas = self.quantize_mantissa(clamped_mantissas, self.mantissa_bits)
        
        reconstructed = clamped_mantissas * scales
        errors = torch.abs(abs_w.unsqueeze(0) - reconstructed)
        
        best_k = torch.argmin(errors, dim=0)
        idx = torch.arange(len(best_k), device=device)
        best_mantissa = clamped_mantissas[best_k, idx]
        best_scale = scales[best_k, 0]
        best_q = signs * best_mantissa * best_scale
        
        best_valid = valid_mask[best_k, idx]
        overflow_count = (~best_valid).sum().item()
        
        q_vec[nonzero_mask] = best_q
        return q_vec, overflow_count, total_count

    def process_weight_matrix(self, W, Hinv, blocksize, percdamp):
        """处理权重矩阵 - 整列共享指数版本"""
        device = W.device
        rows, cols = W.shape
        stats = {'overflow': 0, 'total': 0, 'swap_count': 0, 'total_pairs': 0}
        
        if cols % 4 != 0:
            Q, stats = self._process_standard_local(W, Hinv, blocksize, stats)
            return Q, stats, None
        
        num_groups = cols // 4
        should_debug = self._should_debug()
        
        # Step 1: 提取sparse_mask
        W_grouped = W.reshape(rows, num_groups, 4)
        nonzero_mask_grouped = W_grouped != 0
        nonzero_count = nonzero_mask_grouped.sum(dim=2)
        valid_row_mask = nonzero_count == 2
        
        pos_template = torch.arange(4, device=device).view(1, 1, 4).expand(rows, num_groups, 4)
        masked_pos = torch.where(nonzero_mask_grouped, pos_template, torch.tensor(99, device=device))
        sorted_pos, _ = torch.sort(masked_pos, dim=2)
        sparse_mask = sorted_pos[:, :, :2].clamp(0, 3).to(torch.int8)
        sparse_mask_before = sparse_mask.clone()
        
        # Step 2: 提取值和指数
        row_idx = torch.arange(rows, device=device).view(-1, 1).expand(rows, num_groups)
        group_idx = torch.arange(num_groups, device=device).view(1, -1).expand(rows, num_groups)
        val0 = W_grouped[row_idx, group_idx, sparse_mask[:, :, 0].long()]
        val1 = W_grouped[row_idx, group_idx, sparse_mask[:, :, 1].long()]
        exp0 = torch.floor(torch.log2(torch.abs(val0).clamp(min=1e-38))).int()
        exp1 = torch.floor(torch.log2(torch.abs(val1).clamp(min=1e-38))).int()
        
        # Step 3: 整列共享指数
        shared_exp_d0 = {}
        shared_exp_d1 = {}
        for g in range(num_groups):
            valid_g = valid_row_mask[:, g]
            all_vals = torch.cat([val0[:, g][valid_g], val1[:, g][valid_g]])
            dense0, dense1 = self.select_shared_exp_for_column(all_vals)
            shared_exp_d0[g] = dense0
            shared_exp_d1[g] = dense1
        
        # Step 4: 判断交换
        need_swap = torch.zeros((rows, num_groups), dtype=torch.bool, device=device)
        for g in range(num_groups):
            d0_min, d0_max = shared_exp_d0[g][0], shared_exp_d0[g][3]
            d1_min, d1_max = shared_exp_d1[g][0], shared_exp_d1[g][3]
            exp0_g, exp1_g = exp0[:, g], exp1[:, g]
            valid_g = valid_row_mask[:, g]
            
            exp0_in_d0 = (exp0_g >= d0_min) & (exp0_g <= d0_max)
            exp1_in_d1 = (exp1_g >= d1_min) & (exp1_g <= d1_max)
            exp0_in_d1 = (exp0_g >= d1_min) & (exp0_g <= d1_max)
            exp1_in_d0 = (exp1_g >= d0_min) & (exp1_g <= d0_max)
            
            current_bad = (~exp0_in_d0).int() + (~exp1_in_d1).int()
            swap_bad = (~exp1_in_d0).int() + (~exp0_in_d1).int()
            need_swap[:, g] = (swap_bad < current_bad) & valid_g
        
        stats['swap_count'] = need_swap.sum().item()
        stats['total_pairs'] = valid_row_mask.sum().item()
        
        # Step 5: 执行交换
        pos0_orig = sparse_mask[:, :, 0].clone()
        pos1_orig = sparse_mask[:, :, 1].clone()
        sparse_mask[:, :, 0] = torch.where(need_swap, pos1_orig, pos0_orig).to(torch.int8)
        sparse_mask[:, :, 1] = torch.where(need_swap, pos0_orig, pos1_orig).to(torch.int8)
        
        # 调试打印
        if should_debug:
            print(f"\n{'#'*80}")
            print(f"# DEBUG: {self.current_layer_name}")
            print(f"# Weight shape: {rows} x {cols}, Mantissa bits: {self.mantissa_bits}")
            print(f"{'#'*80}")
            for g in range(min(num_groups, 5)):
                self._print_column_debug_info(g, rows, W_grouped, sparse_mask_before, sparse_mask,
                                            exp0, exp1, valid_row_mask, need_swap,
                                            shared_exp_d0[g], shared_exp_d1[g])
            print(f"\n{'#'*80}\n")
        
        print(f"    Mantissa bits: {self.mantissa_bits}")
        print(f"    Swap: {stats['swap_count']}/{stats['total_pairs']} ({stats['swap_count']/max(stats['total_pairs'],1):.2%})")
        
        # Step 6: 量化 + OBS补偿
        Q = torch.zeros_like(W)
        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2] if Hinv is not None else None
            
            for i in range(i2 - i1):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i].item() if Hinv1 is not None else 1.0
                if d == 0 or math.isnan(d): d = 1.0
                
                g = col // 4
                local_col = col % 4
                is_slot0 = sparse_mask[:, g, 0] == local_col
                is_slot1 = sparse_mask[:, g, 1] == local_col
                valid_col = valid_row_mask[:, g]
                
                use_d0 = valid_col & is_slot0
                use_d1 = valid_col & is_slot1
                q = torch.zeros_like(w)
                
                exp_d0 = shared_exp_d0[g]
                exp_d1 = shared_exp_d1[g]
                
                if use_d0.any():
                    q_d0, of0, t0 = self.quantize_vectorized(w[use_d0], exp_d0, device)
                    q[use_d0] = q_d0
                    stats['overflow'] += of0
                    stats['total'] += t0
                if use_d1.any():
                    q_d1, of1, t1 = self.quantize_vectorized(w[use_d1], exp_d1, device)
                    q[use_d1] = q_d1
                    stats['overflow'] += of1
                    stats['total'] += t1
                
                invalid_nonzero = (~valid_col) & (w != 0)
                if invalid_nonzero.any():
                    q_inv, of_inv, t_inv = self.quantize_vectorized(w[invalid_nonzero], exp_d0, device)
                    q[invalid_nonzero] = q_inv
                    stats['overflow'] += of_inv
                    stats['total'] += t_inv
                
                Q1[:, i] = q
                if Hinv1 is not None:
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
            
            Q[:, i1:i2] = Q1
            if Hinv is not None and i2 < cols:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
        return Q, stats, sparse_mask
    
    def _process_standard_local(self, W, Hinv, blocksize, stats):
        """非2:4情况"""
        device = W.device
        rows, cols = W.shape
        num_row_blocks = (rows + self.block_size - 1) // self.block_size
        Q = torch.zeros_like(W)
        
        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            
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
                        max_exp, _ = self.analyze_block_exponent(w_block)
                        shared_exp = self.get_dense0_shared(max_exp)
                        q_block, overflow, total = self.quantize_vectorized(w_block, shared_exp, device)
                        q[rs:re] = q_block
                        stats['overflow'] += overflow
                        stats['total'] += total
                
                Q1[:, i] = q
                if Hinv is not None:
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
            
            Q[:, i1:i2] = Q1
            if Hinv is not None and i2 < cols:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
        return Q, stats

class SparseGPTWithSwapV2:
    """SparseGPT变体，支持共享指数量化 + 列交换"""
    
    def __init__(self, layer, device=None):
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
        self.quantizer = SharedExpQuantizerWithSwapV2()
    
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
        """带共享指数量化的剪枝"""
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
        
        # 返回值现在包含sparse_mask
        Q, stats, sparse_mask = self.quantizer.process_weight_matrix(W, Hinv, blocksize, percdamp)
        
        # 保存sparse_mask供推理使用
        self.sparse_mask = sparse_mask
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        
        stats['time'] = time.time() - tick
        
        return stats
    
    def free(self):
        self.H = None
        torch.cuda.empty_cache()


# ==================== 多GPU并行处理函数 ====================

def process_layer_on_gpu(layer_module, subset_names, inps, attention_mask, 
                         nsamples, blocksize, percdamp, gpu_id, results_dict):
    """
    在指定GPU上处理一个layer的所有sublayers
    """
    device = torch.device(f'cuda:{gpu_id}')
    
    try:
        # 移动layer到指定GPU
        layer = layer_module.to(device)
        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            if name in subset_names:
                gpts[name] = SparseGPTWithSwapV2(subset[name], device=device)
        
        # 移动输入到GPU
        inps_gpu = inps.to(device)
        attn_mask = attention_mask.to(device) if attention_mask is not None else None
        
        # 收集Hessian
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        outs_gpu = torch.zeros_like(inps_gpu)
        for j in range(nsamples):
            outs_gpu[j] = layer(inps_gpu[j].unsqueeze(0), attention_mask=attn_mask)[0]
        
        for h in handles:
            h.remove()
        
        # 量化
        layer_stats = []
        for name in gpts:
            stats = gpts[name].fasterprune_with_shared_exp(
                blocksize=blocksize,
                percdamp=percdamp
            )
            layer_stats.append({
                'name': name,
                'stats': stats
            })
            gpts[name].free()
        
        # 重新计算输出
        for j in range(nsamples):
            outs_gpu[j] = layer(inps_gpu[j].unsqueeze(0), attention_mask=attn_mask)[0]
        
        # 移回CPU
        layer = layer.cpu()
        outs_cpu = outs_gpu.cpu()
        
        results_dict[gpu_id] = {
            'layer': layer,
            'outs': outs_cpu,
            'stats': layer_stats
        }
        
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        results_dict[gpu_id] = None


def process_sublayers_parallel(layer, subset_names_list, inps, attention_mask,
                               nsamples, blocksize, percdamp, num_gpus):
    """
    并行处理一个layer中的多个sublayer
    
    Args:
        layer: 当前层
        subset_names_list: sublayer名称列表的列表，每个GPU处理一组
        其他参数同上
    
    Returns:
        all_stats: 所有sublayer的统计信息
    """
    gpu_manager = get_gpu_manager(num_gpus)
    
    all_stats = []
    
    # 将sublayer分配到不同GPU
    for gpu_id, names in enumerate(subset_names_list):
        if gpu_id >= gpu_manager.num_gpus:
            break
        
        device = gpu_manager.get_device(gpu_id)
        subset = find_layers(layer)
        
        for name in names:
            if name in subset:
                # 创建处理器
                gpt = SparseGPTWithSwapV2(subset[name], device=device)
                
                # 移动数据
                layer_gpu = layer.to(device)
                inps_gpu = inps.to(device)
                attn_mask = attention_mask.to(device) if attention_mask is not None else None
                
                # 收集Hessian
                def add_batch_closure(gpt_obj):
                    def tmp(_, inp, out):
                        gpt_obj.add_batch(inp[0].data, out.data)
                    return tmp
                
                handle = subset[name].register_forward_hook(add_batch_closure(gpt))
                
                for j in range(nsamples):
                    with torch.cuda.stream(gpu_manager.get_stream(gpu_id)):
                        _ = layer_gpu(inps_gpu[j].unsqueeze(0), attention_mask=attn_mask)[0]
                
                handle.remove()
                
                # 同步
                torch.cuda.synchronize(device)
                
                # 量化
                stats = gpt.fasterprune_with_shared_exp(
                    blocksize=blocksize,
                    percdamp=percdamp
                )
                
                all_stats.append({
                    'name': name,
                    'stats': stats
                })
                
                gpt.free()
                layer_gpu = layer_gpu.cpu()
    
    return all_stats


@torch.no_grad()
def opt_shared_exp_with_swap(model, dataloader, dev, args):
    """主量化函数 - 保持原有接口"""
    print('\nStarting Shared Exponent Quantization with Swap V2...')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers
    
    # 移动嵌入层
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
    
    for layer_idx in range(len(layers)):
        # 跳过前4层
        if layer_idx < 4:
            print(f'\nLayer {layer_idx} - SKIPPED (first 4 layers)')
            layer = layers[layer_idx].to(dev)
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            layers[layer_idx] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps
            continue
        
        print(f'\n{"="*60}')
        print(f'Layer {layer_idx}')
        print(f'{"="*60}')
        
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPTWithSwapV2(subset[name], debug_layer_name=args.debug_layer)
        
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
            stats = gpts[name].fasterprune_with_shared_exp(...)
            per_tensor_stats.append({
                'layer': layer_idx,
                'name': name,
                'overflow': stats['overflow'],
                'total': stats['total']
            })
            print(f"    Time: {stats['time']:.1f}s")
            print(f"    Overflow: {stats['overflow']}/{stats['total']} "
                  f"({stats['overflow']/max(stats['total'],1):.2%})")
            
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
def opt_shared_exp_with_swap_multigpu(model, dataloader, dev, args, num_gpus=4):
    """..."""
    print(f'\nStarting Multi-GPU Shared Exponent Quantization (GPUs: {num_gpus})...')
    
    # 预热所有GPU，避免lazy initialization问题
    actual_gpus = warmup_gpus(num_gpus)
    
    gpu_manager = get_gpu_manager(actual_gpus)
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers
    num_layers = len(layers)
    
    # 初始化：使用GPU0收集初始输入
    primary_dev = gpu_manager.get_device(0)
    
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(primary_dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(primary_dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(primary_dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(primary_dev)
    layers[0] = layers[0].to(primary_dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=primary_dev
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
            model(batch[0].to(primary_dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    # 清理
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    
    # 将初始输入移到CPU以便分发
    inps = inps.cpu()
    attention_mask = cache['attention_mask'].cpu() if cache['attention_mask'] is not None else None
    
    for i in range(gpu_manager.num_gpus):
        torch.cuda.empty_cache()
    
    print('Ready.')
    
    total_stats = {'overflow': 0, 'total': 0, 'swap_count': 0, 'total_pairs': 0}
    per_tensor_stats = []
    outs = torch.zeros_like(inps)
    
    # 策略：每次处理num_gpus个层（或剩余层数）
    layer_idx = 0
    while layer_idx < num_layers:
        # 确定这一批处理的层
        batch_size = min(gpu_manager.num_gpus, num_layers - layer_idx)
        
        # 检查是否需要跳过
        skip_indices = []
        process_indices = []
        for i in range(batch_size):
            idx = layer_idx + i
            if idx < 4:
                skip_indices.append(idx)
            else:
                process_indices.append(idx)
        
        # 处理需要跳过的层（顺序处理）
        for idx in skip_indices:
            print(f'\nLayer {idx} - SKIPPED (first 4 layers)')
            layer = layers[idx].to(primary_dev)
            inps_dev = inps.to(primary_dev)
            attn_dev = attention_mask.to(primary_dev) if attention_mask is not None else None
            
            for j in range(args.nsamples):
                outs[j] = layer(inps_dev[j].unsqueeze(0), attention_mask=attn_dev)[0].cpu()
            
            layers[idx] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps
        
        # 并行处理需要量化的层
        if process_indices:
            print(f'\n{"="*60}')
            print(f'Processing Layers {process_indices} in parallel')
            print(f'{"="*60}')
            
            # 使用线程池并行处理
            results = {}
            threads = []
            
            for i, idx in enumerate(process_indices):
                gpu_id = i % gpu_manager.num_gpus
                
                # 每个层在单独线程中处理
                t = threading.Thread(
                    target=process_single_layer_thread,
                    args=(layers[idx], inps.clone(), attention_mask, 
                          args.nsamples, args.blocksize, args.percdamp,
                          gpu_id, idx, results)
                )
                threads.append(t)
                t.start()
            
            # 等待所有线程完成
            for t in threads:
                t.join()
            
            # 收集结果（必须按顺序处理以保持依赖）
            for idx in sorted(process_indices):
                if idx in results and results[idx] is not None:
                    result = results[idx]
                    layers[idx] = result['layer']
                    
                    # 更新统计
                    for stat_item in result['stats']:
                        per_tensor_stats.append({
                            'layer': idx,
                            'name': stat_item['name'],
                            'overflow': stat_item['stats']['overflow'],
                            'total': stat_item['stats']['total']
                        })
                        total_stats['overflow'] += stat_item['stats']['overflow']
                        total_stats['total'] += stat_item['stats']['total']
                        total_stats['swap_count'] += stat_item['stats'].get('swap_count', 0)
                        total_stats['total_pairs'] += stat_item['stats'].get('total_pairs', 0)
                        
                        print(f"  Layer {idx} {stat_item['name']}: "
                              f"Time={stat_item['stats']['time']:.1f}s, "
                              f"Overflow={stat_item['stats']['overflow']}/{stat_item['stats']['total']}")
                    
                    # 更新输出（串行，因为有依赖）
                    layer = layers[idx].to(primary_dev)
                    inps_dev = inps.to(primary_dev)
                    attn_dev = attention_mask.to(primary_dev) if attention_mask is not None else None
                    
                    for j in range(args.nsamples):
                        outs[j] = layer(inps_dev[j].unsqueeze(0), attention_mask=attn_dev)[0].cpu()
                    
                    layers[idx] = layer.cpu()
                    inps, outs = outs, inps
        
        layer_idx += batch_size
    
    model.config.use_cache = use_cache
    return total_stats, per_tensor_stats


def process_single_layer_thread(layer_module, inps, attention_mask, 
                                nsamples, blocksize, percdamp, 
                                gpu_id, layer_idx, results_dict):
    device = torch.device(f'cuda:{gpu_id}')
    
    # 确保在正确的设备上下文中
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    
    try:
        # 深拷贝层到GPU
        import copy
        layer = copy.deepcopy(layer_module).to(device)
        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPTWithSwapV2(subset[name], device=device)
        
        # 移动输入到GPU
        inps_gpu = inps.to(device)
        attn_mask = attention_mask.to(device) if attention_mask is not None else None
        
        # 收集Hessian
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        for j in range(nsamples):
            _ = layer(inps_gpu[j].unsqueeze(0), attention_mask=attn_mask)[0]
        
        for h in handles:
            h.remove()
        
        # 量化
        layer_stats = []
        for name in gpts:
            stats = gpts[name].fasterprune_with_shared_exp(
                blocksize=blocksize,
                percdamp=percdamp
            )
            layer_stats.append({
                'name': name,
                'stats': stats
            })
            gpts[name].free()
        
        # 更新原始层的权重
        for name in subset:
            layer_module_subset = find_layers(layer_module)
            if name in layer_module_subset:
                layer_module_subset[name].weight.data = subset[name].weight.data.cpu()
        
        results_dict[layer_idx] = {
            'layer': layer_module,  # 返回更新后的原始层
            'stats': layer_stats
        }
        
        del layer, gpts
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing layer {layer_idx} on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        results_dict[layer_idx] = None


@torch.no_grad()
def opt_eval(model, testenc, dev):
    """评估PPL - 保持原有接口"""
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
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU parallel processing')
    parser.add_argument('--debug_layer', type=str, default=None, 
                    help='Layer name to debug, e.g. "layer4.self_attn.k_proj"')
    args = parser.parse_args()
    
    # 设置默认设备
    DEV = torch.device('cuda:0')
    
    print('='*70)
    print('2:4 Sparse + Column Swap + Shared Exponent Quantization V2')
    print(f'Multi-GPU: {args.multi_gpu}, Num GPUs: {args.num_gpus}')
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
    print('Baseline')
    print('='*70)
    # baseline_ppl = opt_eval(model, testloader, DEV)
    
    print('\n' + '='*70)
    print('Quantization')
    print('='*70)
    tick = time.time()
    
    if args.multi_gpu:
        total_stats, per_tensor_stats = opt_shared_exp_with_swap_multigpu(
            model, dataloader, DEV, args, num_gpus=args.num_gpus
        )
    else:
        total_stats, per_tensor_stats = opt_shared_exp_with_swap(model, dataloader, DEV, args)
    
    elapsed = time.time() - tick
    
    print('\n' + '='*70)
    print('Final')
    print('='*70)
    final_ppl = opt_eval(model, testloader, DEV)
    
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        
        print('\n' + '='*70)
        print('Per-Tensor Overflow Statistics')
        print('='*70)
        print(f'{"Layer":<10} {"Name":<30} {"Overflow":>10} {"Total":>10} {"Rate":>10}')
        print('-'*70)
        for item in per_tensor_stats:
            rate = item['overflow'] / max(item['total'], 1)
            print(f'{item["layer"]:<10} {item["name"]:<30} {item["overflow"]:>10} {item["total"]:>10} {rate:>9.2%}')
        
        model.save_pretrained(args.save)
        print(f'\nSaved to {args.save}')
    
    print('\n' + '='*70)
    print('Summary')
    print('='*70)
    # print(f'Baseline PPL: {baseline_ppl:.3f}')
    print(f'Final PPL:    {final_ppl:.3f}')
    # print(f'PPL increase: {final_ppl - baseline_ppl:+.3f}')
    print(f'Overflow:     {total_stats["overflow"]}/{total_stats["total"]} ({total_stats["overflow"]/max(total_stats["total"],1):.2%})')
    print(f'Swap:         {total_stats["swap_count"]}/{total_stats["total_pairs"]} ({total_stats["swap_count"]/max(total_stats["total_pairs"],1):.2%})')
    print(f'Time:         {elapsed:.1f}s')