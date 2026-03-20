"""
2:4剪枝后的列交换优化 + 共享指数量化（预计算加速版）
=========================================================
与 opt_shared_exp_swap_swap.py 逻辑完全一致，但跳过两大耗时步骤：
  1. 暴力枚举最优4指数 → 从 codebook_tables.json 查表
  2. 交换决策计算       → 从 sparse_masks.pt 直接读取位置

前置步骤:
  python test.py /path/to/pruned_model --mantissa_bits 4 \
      --row_block_size 256 --save_codebooks_per_group --save_masks \
      --output_dir /path/to/precomputed

使用方法:
  python opt_shared_exp_swap_quant.py /home/hej/model/float/stage1_pruned wikitext2 \
      --base_model /home/hej/model/float/opt-6.7b --mantissa_bits 4 \
      --skip_mantissa_quant --row_block_size 256 \
      --precomputed_dir /home/hej/model/float/sparsegpt/compressed_256_4bits/mantissa_4bit
"""

import time
import math
import json
from collections import Counter
import torch
import torch.nn as nn
import transformers

from datautils import get_loaders
from modelutils import find_layers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ======================================================================
# 预计算数据加载器
# ======================================================================

class PrecomputedLoader:
    """
    加载 test.py 产出的 codebook_tables.json 和 sparse_masks.pt
    
    codebook_tables.json 格式:
      { "layer0.self_attn.q_proj": { "g0_rb0": {"d0": [-7,-6,-5,-4], "d1": [-9,-8,-7,-6]}, ... }, ... }
    
    sparse_masks.pt 格式:
      { "layer0.self_attn.q_proj": tensor[rows, num_groups, 2] int8 }
      其中 dim2 = [pos_d0, pos_d1]，即大值和小值在4元素组内的位置
    """

    def __init__(self, precomputed_dir):
        self.codebooks = {}   # { tensor_name: { (g,rb): {"d0": [...], "d1": [...]} } }
        self.masks = {}       # { tensor_name: tensor[rows, groups, 2] }

        # ---------- 加载码表 ----------
        cb_path = f"{precomputed_dir}/codebook_tables.json"
        print(f"  Loading codebook: {cb_path}")
        with open(cb_path, 'r') as f:
            raw_cb = json.load(f)

        for tensor_name, entries in raw_cb.items():
            parsed = {}
            for key, val in entries.items():
                # key = "g123_rb4" → (123, 4)
                parts = key.split('_')
                g = int(parts[0][1:])
                rb = int(parts[1][2:])
                parsed[(g, rb)] = val
            self.codebooks[tensor_name] = parsed

        print(f"    Loaded codebooks for {len(self.codebooks)} tensors")

        # ---------- 加载 mask ----------
        mask_path = f"{precomputed_dir}/sparse_masks.pt"
        print(f"  Loading masks: {mask_path}")
        self.masks = torch.load(mask_path, map_location='cpu')
        print(f"    Loaded masks for {len(self.masks)} tensors")

    def get_codebook(self, tensor_name, g, rb):
        """返回 (exp_d0_list, exp_d1_list)，每个是4个int的list"""
        entry = self.codebooks[tensor_name][(g, rb)]
        return entry['d0'], entry['d1']

    def get_mask(self, tensor_name):
        """返回 tensor[rows, num_groups, 2] int8，dim2=[pos_d0, pos_d1]"""
        return self.masks[tensor_name]

    def has_tensor(self, tensor_name):
        return tensor_name in self.codebooks and tensor_name in self.masks


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


# ======================================================================
# 量化器（预计算加速版）
# ======================================================================

class SharedExpQuantizerPrecomputed:
    """
    与 SharedExpQuantizerFast 完全等价，但码表和mask从预计算文件加载。
    
    跳过:
      - _select_top4_bruteforce (暴力枚举 C(n,4))
      - _recompute_swap_for_group (交换决策)
    保留:
      - _quantize_column_unified (逐列量化，不可预计算)
      - GPTQ误差补偿循环 (依赖实时补偿后的权重)
    """

    def __init__(self, mantissa_bits=4, row_block_size=-1,
                 skip_mantissa_quant=False, print_err=False):
        self.mantissa_bits = mantissa_bits
        self.row_block_size = row_block_size
        self.skip_mantissa_quant = skip_mantissa_quant
        self.overflow_weight = 1.5
        self.print_err = print_err
        self.current_layer_name = None

        # 预计算数据（由外部注入）
        self.precomputed_loader = None

    def set_current_layer(self, layer_name):
        self.current_layer_name = layer_name

    def set_precomputed(self, loader):
        self.precomputed_loader = loader

    # ------------------------------------------------------------------
    # 量化（与原版完全相同）
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
        return q, overflow, flag, ovf_cnt, total_count, None

    # ------------------------------------------------------------------
    # 主入口（预计算加速版）
    # ------------------------------------------------------------------

    def process_weight_matrix(self, W, Hinv, blocksize, percdamp):
        device = W.device
        rows, cols = W.shape
        loader = self.precomputed_loader
        layer_name = self.current_layer_name
        rbs = self.row_block_size

        stats = {
            'overflow': 0, 'total': 0, 'swap_count': 0, 'total_pairs': 0,
            'flag_high_count': 0,
        }

        if cols % 4 != 0:
            print("    Warning: cols % 4 != 0")
            return torch.zeros_like(W), stats, None

        num_groups = cols // 4
        num_rb = (rows + rbs - 1) // rbs if rbs > 0 else 1

        # ========================================================
        # 关键改动1: 从预计算mask加载稀疏位置 → 替代 _recompute_swap_for_group
        # ========================================================
        # mask_tensor: [rows, num_groups, 2] int8, dim2 = [pos_d0, pos_d1]
        # 这就是原版中 all_sm 的含义：每行每组的两个非零值位置
        # pos_d0 = 大值（分配给D0码表），pos_d1 = 小值（分配给D1码表）
        mask_tensor = loader.get_mask(layer_name).to(device)
        all_sm = mask_tensor  # [rows, num_groups, 2]

        # 有效性掩码：检查每行每组恰好2个非零
        W_g = W.reshape(rows, num_groups, 4)
        nz_cnt = (W_g != 0).sum(dim=2)
        all_vg = (nz_cnt == 2)
        stats['total_pairs'] = all_vg.sum().item()

        # ========================================================
        # 关键改动2: 从预计算码表加载指数 → 替代 _select_shared_exp_from_block
        # ========================================================
        # all_exp_d0[r, g, :] = 该 (g, rb_of_r) 的D0码表（4个指数）
        # all_exp_d1[r, g, :] = 该 (g, rb_of_r) 的D1码表（4个指数）
        all_exp_d0 = torch.zeros(rows, num_groups, 4, dtype=torch.float32, device=device)
        all_exp_d1 = torch.zeros(rows, num_groups, 4, dtype=torch.float32, device=device)

        for g in range(num_groups):
            if rbs > 0:
                for rb in range(num_rb):
                    rs = rb * rbs
                    re = min(rs + rbs, rows)
                    ed0, ed1 = loader.get_codebook(layer_name, g, rb)
                    all_exp_d0[rs:re, g, :] = torch.tensor(ed0, dtype=torch.float32, device=device)
                    all_exp_d1[rs:re, g, :] = torch.tensor(ed1, dtype=torch.float32, device=device)
            else:
                ed0, ed1 = loader.get_codebook(layer_name, g, 0)
                all_exp_d0[:, g, :] = torch.tensor(ed0, dtype=torch.float32, device=device)
                all_exp_d1[:, g, :] = torch.tensor(ed1, dtype=torch.float32, device=device)

        # ========================================================
        # 统计交换次数（mask中 pos_d0 != 原始排序位置 的即为交换）
        # 这里简化处理：与原版保持统计兼容
        # ========================================================
        # （可选：如果需要精确swap统计，对比mask和原始位置排序即可）

        # ========================================================
        # 以下为标准GPTQ循环，与原版完全一致
        # ========================================================

        nonzero_col_mask = (W != 0)
        nonzero_col_float = nonzero_col_mask.float()

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

            for i in range(count):
                col = i1 + i
                g = col // 4
                lc = col % 4

                w = W1[:, i]
                d = Hinv1[i, i].item() if Hinv1 is not None else 1.0
                if d == 0 or math.isnan(d):
                    d = 1.0

                # ---- 确定每行用 D0 还是 D1 码表 ----
                # all_sm[:, g, 0] = pos_d0 (大值位置)
                # all_sm[:, g, 1] = pos_d1 (小值位置)
                # 当前列 lc 如果是该行的大值位置 → 用 D0 码表
                # 当前列 lc 如果是该行的小值位置 → 用 D1 码表
                vg = all_vg[:, g]
                sm0 = all_sm[:, g, 0]  # pos_d0
                sm1 = all_sm[:, g, 1]  # pos_d1

                is_s0 = (sm0 == lc)
                is_s1 = (sm1 == lc)

                use_d0_mask = vg & is_s0
                use_d1_mask = vg & is_s1

                # 默认用 D0 码表，D1 位置的行覆盖为 D1 码表
                per_row_exp = all_exp_d0[:, g, :].clone()
                per_row_exp[use_d1_mask] = all_exp_d1[use_d1_mask, g, :]

                nz_col = block_nz_bool[:, i]
                q, overflow_col, flag, ovf_cnt, total_cnt, _ = \
                    self._quantize_column_unified(w, per_row_exp, nz_col, device)

                stats['overflow'] += ovf_cnt
                stats['total'] += total_cnt
                stats['flag_high_count'] += (flag == 1).sum().item()

                Q1[:, i] = q

                err1 = (w - q) / d
                Err1[:, i] = err1

                # GPTQ误差补偿
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

        mode_str = f"4x{rbs}块" if rbs > 0 else "整列"
        fr = stats['flag_high_count'] / max(stats['total'], 1)
        print(f"    [{mode_str}预计算] "
              f"溢出: {stats['overflow']}/{stats['total']} "
              f"({stats['overflow'] / max(stats['total'], 1):.2%}), "
              f"flag=1: {fr:.2%}")

        sparse_mask = all_sm
        return Q, stats, sparse_mask


# ======================================================================
# SparseGPT 包装器
# ======================================================================

class SparseGPTWithPrecomputed:

    def __init__(self, layer, device=None, mantissa_bits=4,
                 row_block_size=-1, skip_mantissa_quant=False,
                 print_err=False):
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
        self.quantizer = SharedExpQuantizerPrecomputed(
            mantissa_bits=mantissa_bits,
            row_block_size=row_block_size,
            skip_mantissa_quant=skip_mantissa_quant,
            print_err=print_err,
        )

    def set_layer_name(self, name):
        self.quantizer.set_current_layer(name)

    def set_precomputed(self, loader):
        self.quantizer.set_precomputed(loader)

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


# ======================================================================
# 主量化流程
# ======================================================================

@torch.no_grad()
def opt_shared_exp_precomputed(model, dataloader, dev, args, precomputed_loader):
    print('\nStarting Column-wise Shared Exp Quant (Precomputed Accelerated)...')
    print(f'  Precomputed dir: {args.precomputed_dir}')
    print(f'  Row block size: {args.row_block_size}')
    print(f'  Mantissa bits: {args.mantissa_bits}')

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
            gpts[name] = SparseGPTWithPrecomputed(
                subset[name],
                device=dev,
                mantissa_bits=args.mantissa_bits,
                row_block_size=args.row_block_size,
                skip_mantissa_quant=getattr(args, 'skip_mantissa_quant', False),
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
            full_name = f"layer{layer_idx}.{name}"
            print(f'\n  {full_name}')

            # ---- 注入预计算数据 ----
            gpts[name].set_layer_name(full_name)
            gpts[name].set_precomputed(precomputed_loader)

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


# ======================================================================
# 评估（与原版一致）
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


# ======================================================================
# 主函数
# ======================================================================

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to pruned model')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--precomputed_dir', type=str, required=True,
                        help='Path to test.py output (codebook_tables.json + sparse_masks.pt)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--blocksize', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=0.01)
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--row_block_size', type=int, default=-1)
    parser.add_argument('--mantissa_bits', type=int, default=4)
    parser.add_argument('--skip_mantissa_quant', action='store_true')
    parser.add_argument('--print_err', action='store_true')
    args = parser.parse_args()

    DEV = torch.device('cuda:0')

    print('=' * 70)
    print('2:4 Sparse + Shared Exp Quant (PRECOMPUTED ACCELERATED)')
    print(f'Precomputed dir: {args.precomputed_dir}')
    print(f'Row block size: {args.row_block_size}')
    print(f'Mantissa bits: {args.mantissa_bits}')
    print('=' * 70)

    # ---- 加载预计算数据 ----
    print('\nLoading precomputed codebooks and masks...')
    precomputed_loader = PrecomputedLoader(args.precomputed_dir)

    print('\nLoading model...')
    model = get_opt(args.model)
    model.eval()

    print('\nLoading data...')
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=args.base_model)

    print('\n' + '=' * 70)
    print('Quantization (Precomputed)')
    print('=' * 70)
    tick = time.time()
    total_stats, per_tensor_stats = opt_shared_exp_precomputed(
        model, dataloader, DEV, args, precomputed_loader)
    elapsed = time.time() - tick

    print('\n' + '=' * 70)
    print('Evaluation')
    print('=' * 70)
    final_ppl = opt_eval(model, testloader, DEV)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_path = os.path.join(args.save, f'mantissa_{args.mantissa_bits}bit')
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        print(f'\nSaved to {save_path}')

    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    print(f'Final PPL:    {final_ppl:.3f}')
    print(f'Overflow:     {total_stats["overflow"]}/{total_stats["total"]} '
          f'({total_stats["overflow"] / max(total_stats["total"], 1):.2%})')
    print(f'Time:         {elapsed:.1f}s')