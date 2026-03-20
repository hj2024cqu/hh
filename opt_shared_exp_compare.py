"""
共享指数量化对比实验 - 8种配置
1. row_128_noncont: 每列按128行分块，非连续指数
2. row_128_cont: 每列按128行分块，连续指数
3. col_128_noncont: 每128列共享指数选择，非连续指数
4. col_128_cont: 每128列共享指数选择，连续指数
5. col_4x128_noncont: 4列×128行共享指数，非连续指数
6. col_4x128_cont: 4列×128行共享指数，连续指数
7. fullcol_noncont: 整列共享，非连续指数
8. fullcol_cont: 整列共享，连续指数

python opt_shared_exp_compare.py ./compressed_models/stage1_pruned wikitext2 --base_model /path/to/opt-6.7b --nsamples 128
"""

import time, math, torch, torch.nn as nn, transformers
from datautils import get_loaders
from modelutils import find_layers, DEV

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


def select_exponents(weights, continuous=False):
    """选择4个共享指数（向量化）"""
    dev = weights.device
    mask = weights != 0
    if not mask.any():
        return torch.tensor([-8, -7, -6, -5], dtype=torch.int32, device=dev)
    
    nz = weights[mask]
    abs_w = torch.abs(nz).clamp(min=1e-38)
    exps = torch.floor(torch.log2(abs_w)).to(torch.int32)
    
    e_min, e_max = exps.min().item(), exps.max().item()
    span = e_max - e_min
    
    if span <= 3:
        r = list(range(int(e_min), int(e_max) + 1))
        while len(r) < 4: r.append(r[-1] + 1)
        return torch.tensor(r[:4], dtype=torch.int32, device=dev)
    
    bins = int(span + 1)
    hist = torch.histc(exps.float(), bins=bins, min=float(e_min), max=float(e_max))
    
    if continuous:
        if bins >= 4:
            conv = torch.nn.functional.conv1d(
                hist.view(1, 1, -1), torch.ones(1, 1, 4, device=dev), padding=0
            ).squeeze()
            best_idx = conv.argmax().item()
        else:
            best_idx = 0
        return torch.tensor([e_min + best_idx + j for j in range(4)], dtype=torch.int32, device=dev)
    else:
        k = min(4, bins)
        _, idx = torch.topk(hist, k)
        r = sorted([int(e_min + i.item()) for i in idx])
        while len(r) < 4: r.append(r[-1] + 1)
        return torch.tensor(r[:4], dtype=torch.int32, device=dev)


def select_exponents_batch(W_block, continuous=False):
    """批量为每行选择共享指数 W_block: [rows, cols]"""
    dev = W_block.device
    rows, cols = W_block.shape
    result = torch.zeros((rows, 4), dtype=torch.int32, device=dev)
    
    if continuous:
        mask = W_block != 0
        abs_w = torch.abs(W_block).clamp(min=1e-38)
        exps = torch.floor(torch.log2(abs_w))
        
        INF = 1e9
        exps_for_min = torch.where(mask, exps, torch.tensor(INF, device=dev))
        exps_for_max = torch.where(mask, exps, torch.tensor(-INF, device=dev))
        
        e_min = exps_for_min.min(dim=1).values
        e_max = exps_for_max.max(dim=1).values
        
        all_zero = ~mask.any(dim=1)
        e_min = torch.where(all_zero, torch.tensor(-8.0, device=dev), e_min)
        e_max = torch.where(all_zero, torch.tensor(-5.0, device=dev), e_max)
        
        e_center = ((e_min + e_max) / 2).floor()
        for i in range(4):
            result[:, i] = (e_center - 1 + i).to(torch.int32)
    else:
        for r in range(rows):
            result[r] = select_exponents(W_block[r, :], False)
    
    return result

def select_exponents_all_blocks(W, num_groups, row_block_size, continuous, device):
    """向量化为所有(group, row_block)块计算共享指数"""
    rows, cols = W.shape
    num_row_blocks = (rows + row_block_size - 1) // row_block_size
    
    # padding到整块
    padded_rows = num_row_blocks * row_block_size
    W_padded = torch.zeros((padded_rows, cols), device=device, dtype=W.dtype)
    W_padded[:rows, :] = W
    
    # [num_row_blocks, row_block_size, num_groups, 4]
    W_reshaped = W_padded.reshape(num_row_blocks, row_block_size, num_groups, 4)
    # [num_row_blocks, num_groups, row_block_size * 4]
    W_blocks = W_reshaped.permute(0, 2, 1, 3).reshape(num_row_blocks, num_groups, -1)
    
    mask = W_blocks != 0
    abs_w = torch.abs(W_blocks).clamp(min=1e-38)
    exps = torch.floor(torch.log2(abs_w))
    
    INF = 1e9
    exps_min = torch.where(mask, exps, torch.tensor(INF, device=device)).min(dim=2).values
    exps_max = torch.where(mask, exps, torch.tensor(-INF, device=device)).max(dim=2).values
    
    all_zero = ~mask.any(dim=2)
    exps_min = torch.where(all_zero, torch.tensor(-8.0, device=device), exps_min)
    exps_max = torch.where(all_zero, torch.tensor(-5.0, device=device), exps_max)
    
    # [num_row_blocks, num_groups, 4]
    result = torch.zeros((num_row_blocks, num_groups, 4), dtype=torch.int32, device=device)
    
    if continuous:
        e_center = ((exps_min + exps_max) / 2).floor()
        for i in range(4):
            result[:, :, i] = (e_center - 1 + i).to(torch.int32)
    else:
        # 非连续：用min/max附近的4个（近似）
        span = exps_max - exps_min
        small_span = span <= 3
        
        # 小范围直接用[min, min+1, min+2, min+3]
        for i in range(4):
            result[:, :, i] = (exps_min + i).to(torch.int32)
        
        # 大范围用中心扩展（近似top-k）
        e_center = ((exps_min + exps_max) / 2).floor()
        for i in range(4):
            result[:, :, i] = torch.where(small_span, result[:, :, i], (e_center - 1 + i).to(torch.int32))
    
    return result

def quantize_with_exp(w, exp):
    """用给定共享指数量化（向量化）"""
    dev = w.device
    q = w.clone()
    mask = w != 0
    if not mask.any():
        return q, 0, 0
    
    nz_w = w[mask]
    signs = torch.sign(nz_w)
    abs_w = torch.abs(nz_w)
    
    scales = (2.0 ** exp.float()).unsqueeze(1)
    mantissas = abs_w.unsqueeze(0) / scales
    valid = (mantissas >= 1.0) & (mantissas < 2.0)
    clamped = torch.clamp(mantissas, 1.0, 1.9999999)
    recon = clamped * scales
    errors = torch.abs(abs_w.unsqueeze(0) - recon)
    
    best_k = torch.argmin(errors, dim=0)
    idx = torch.arange(len(best_k), device=dev)
    best_m = clamped[best_k, idx]
    best_s = scales[best_k, 0]
    best_q = signs * best_m * best_s
    
    overflow = (~valid[best_k, idx]).sum().item()
    q[mask] = best_q
    return q, overflow, mask.sum().item()

def quantize_column_with_block_exps(w_col, shared_exps_col, row_block_size, device):
    """向量化量化整列，每个row_block用不同的共享指数
    w_col: [rows]
    shared_exps_col: [num_row_blocks, 4]
    """
    rows = len(w_col)
    num_row_blocks = shared_exps_col.shape[0]
    q = w_col.clone()
    
    mask = w_col != 0
    if not mask.any():
        return q, 0, 0
    
    nz_idx = mask.nonzero(as_tuple=True)[0]
    nz_w = w_col[nz_idx]
    nz_rb = nz_idx // row_block_size  # 每个非零元素属于哪个row_block
    nz_rb = nz_rb.clamp(max=num_row_blocks-1)
    
    signs = torch.sign(nz_w)
    abs_w = torch.abs(nz_w)
    
    # 获取每个非零元素对应的4个共享指数 [num_nz, 4]
    nz_exps = shared_exps_col[nz_rb]
    scales = 2.0 ** nz_exps.float()  # [num_nz, 4]
    
    mantissas = abs_w.unsqueeze(1) / scales  # [num_nz, 4]
    valid = (mantissas >= 1.0) & (mantissas < 2.0)
    clamped = torch.clamp(mantissas, 1.0, 1.9999999)
    recon = clamped * scales
    errors = torch.abs(abs_w.unsqueeze(1) - recon)
    
    best_k = torch.argmin(errors, dim=1)
    idx = torch.arange(len(best_k), device=device)
    best_m = clamped[idx, best_k]
    best_s = scales[idx, best_k]
    best_q = signs * best_m * best_s
    
    overflow = (~valid[idx, best_k]).sum().item()
    q[nz_idx] = best_q
    
    return q, overflow, mask.sum().item()

def quantize_column_row_block(col, block_size, continuous):
    """按行分块量化单列"""
    dev = col.device
    rows = len(col)
    Q = torch.zeros_like(col)
    total_ov, total_nz = 0, 0
    
    for r in range(0, rows, block_size):
        re = min(r + block_size, rows)
        block = col[r:re]
        exp = select_exponents(block, continuous)
        qb, ov, nz = quantize_with_exp(block, exp)
        Q[r:re] = qb
        total_ov += ov
        total_nz += nz
    
    return Q, total_ov, total_nz


def quantize_column_full(col, continuous):
    """整列共享指数量化"""
    exp = select_exponents(col, continuous)
    return quantize_with_exp(col, exp)


def quantize_col_with_row_exps(w_col, row_exps):
    """向量化：用每行对应的共享指数量化整列"""
    dev = w_col.device
    q = w_col.clone()
    
    mask = w_col != 0
    if not mask.any():
        return q, 0, 0
    
    nz_idx = mask.nonzero(as_tuple=True)[0]
    nz_w = w_col[nz_idx]
    nz_exps = row_exps[nz_idx]
    
    signs = torch.sign(nz_w)
    abs_w = torch.abs(nz_w)
    
    scales = 2.0 ** nz_exps.float()
    mantissas = abs_w.unsqueeze(1) / scales
    valid = (mantissas >= 1.0) & (mantissas < 2.0)
    clamped = torch.clamp(mantissas, 1.0, 1.9999999)
    recon = clamped * scales
    errors = torch.abs(abs_w.unsqueeze(1) - recon)
    
    best_k = torch.argmin(errors, dim=1)
    idx = torch.arange(len(best_k), device=dev)
    best_m = clamped[idx, best_k]
    best_s = scales[idx, best_k]
    best_q = signs * best_m * best_s
    
    overflow = (~valid[idx, best_k]).sum().item()
    q[nz_idx] = best_q
    
    return q, overflow, mask.sum().item()


def process_layer_row_mode(layer_idx, layer, inps, attn_mask, args, block_size, continuous, dev, skip=False):
    """处理单层 - 按行分块模式"""
    layer = layer.to(dev)
    subset = find_layers(layer)
    nsamples = inps.shape[0]
    outs = torch.zeros_like(inps)
    
    if skip:
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
        layer = layer.cpu()
        torch.cuda.empty_cache()
        return layer, outs, {'overflow': 0, 'total': 0}
    
    H_dict, ns_dict = {}, {}
    for name in subset:
        W = subset[name].weight.data
        if isinstance(subset[name], transformers.Conv1D): W = W.t()
        cols = W.shape[1]
        H_dict[name] = torch.zeros((cols, cols), device=dev)
        ns_dict[name] = 0

    def hook(name):
        def fn(_, inp, out):
            x = inp[0]
            if len(x.shape) == 2: x = x.unsqueeze(0)
            if len(x.shape) == 3: x = x.reshape(-1, x.shape[-1])
            x = x.t().float()
            n = x.shape[1]
            H_dict[name] *= ns_dict[name] / (ns_dict[name] + n)
            ns_dict[name] += n
            x = math.sqrt(2 / ns_dict[name]) * x
            H_dict[name] += x @ x.t()
        return fn
    
    handles = [subset[n].register_forward_hook(hook(n)) for n in subset]
    for j in range(nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
    for h in handles: h.remove()

    stats = {'overflow': 0, 'total': 0}
    
    for name in subset:
        tick = time.time()
        W = subset[name].weight.data.clone()
        orig_shape, orig_dtype = W.shape, W.dtype
        if isinstance(subset[name], transformers.Conv1D): W = W.t()
        W = W.float().to(dev)
        rows, cols = W.shape
        
        # 非零mask
        nonzero_mask = (W != 0)
        
        H = H_dict[name]
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        damp = args.percdamp * torch.mean(torch.diag(H))
        H[torch.arange(cols, device=dev), torch.arange(cols, device=dev)] += damp
        
        try:
            H = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(H)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
        except:
            Hinv = torch.eye(cols, device=dev)
        
        Q = torch.zeros_like(W)
        Losses = torch.zeros(rows, device=dev)
        sub_ov, sub_nz = 0, 0
        
        # SparseGPT风格按块处理
        obs_bs = 128
        for c1 in range(0, cols, obs_bs):
            c2 = min(c1 + obs_bs, cols)
            W1 = W[:, c1:c2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[c1:c2, c1:c2]
            block_nz_mask = nonzero_mask[:, c1:c2]
            
            for i in range(c2 - c1):
                w_col = W1[:, i]
                d = Hinv1[i, i]
                
                if block_size <= 0:
                    q_col, ov, nz = quantize_column_full(w_col, continuous)
                else:
                    q_col, ov, nz = quantize_column_row_block(w_col, block_size, continuous)
                
                Q1[:, i] = q_col
                sub_ov += ov
                sub_nz += nz
                
                err = (w_col - q_col) / d
                Losses += (w_col - q_col) ** 2 / d ** 2
                Err1[:, i] = err
                
                # 块内补偿（只补偿非零位置）
                if i < c2 - c1 - 1:
                    compensation = err.unsqueeze(1) @ Hinv1[i, i+1:].unsqueeze(0)
                    future_mask = block_nz_mask[:, i+1:]
                    W1[:, i+1:] -= compensation * future_mask.float()
            
            Q[:, c1:c2] = Q1
            # 块间补偿（只补偿非零位置）
            if c2 < cols:
                full_comp = Err1 @ Hinv[c1:c2, c2:]
                future_mask = nonzero_mask[:, c2:]
                W[:, c2:] -= full_comp * future_mask.float()
        
        if isinstance(subset[name], transformers.Conv1D): Q = Q.t()
        subset[name].weight.data = Q.reshape(orig_shape).to(orig_dtype)
        
        stats['overflow'] += sub_ov
        stats['total'] += sub_nz
        
        elapsed = time.time() - tick
        loss = torch.sum(Losses).item() / 2
        print(f'  L{layer_idx} {name} t={elapsed:.1f}s err={loss:.1f} ov={sub_ov}/{sub_nz}')
        del H_dict[name]
    
    for j in range(nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
    
    layer = layer.cpu()
    torch.cuda.empty_cache()
    return layer, outs, stats


def process_layer_col_mode(layer_idx, layer, inps, attn_mask, args, col_block_size, continuous, dev, skip=False):
    """处理单层 - 按列分块模式（每col_block_size列共享指数选择）"""
    layer = layer.to(dev)
    subset = find_layers(layer)
    nsamples = inps.shape[0]
    outs = torch.zeros_like(inps)
    
    if skip:
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
        layer = layer.cpu()
        torch.cuda.empty_cache()
        return layer, outs, {'overflow': 0, 'total': 0}
    
    H_dict, ns_dict = {}, {}
    for name in subset:
        W = subset[name].weight.data
        if isinstance(subset[name], transformers.Conv1D): W = W.t()
        cols = W.shape[1]
        H_dict[name] = torch.zeros((cols, cols), device=dev)
        ns_dict[name] = 0

    def hook(name):
        def fn(_, inp, out):
            x = inp[0]
            if len(x.shape) == 2: x = x.unsqueeze(0)
            if len(x.shape) == 3: x = x.reshape(-1, x.shape[-1])
            x = x.t().float()
            n = x.shape[1]
            H_dict[name] *= ns_dict[name] / (ns_dict[name] + n)
            ns_dict[name] += n
            x = math.sqrt(2 / ns_dict[name]) * x
            H_dict[name] += x @ x.t()
        return fn
    
    handles = [subset[n].register_forward_hook(hook(n)) for n in subset]
    for j in range(nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
    for h in handles: h.remove()

    stats = {'overflow': 0, 'total': 0}
    
    for name in subset:
        tick = time.time()
        W = subset[name].weight.data.clone()
        orig_shape, orig_dtype = W.shape, W.dtype
        if isinstance(subset[name], transformers.Conv1D): W = W.t()
        W = W.float().to(dev)
        rows, cols = W.shape
        
        # 非零mask
        nonzero_mask = (W != 0)
        
        H = H_dict[name]
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        damp = args.percdamp * torch.mean(torch.diag(H))
        H[torch.arange(cols, device=dev), torch.arange(cols, device=dev)] += damp
        
        try:
            H = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(H)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
        except:
            Hinv = torch.eye(cols, device=dev)
        
        Q = torch.zeros_like(W)
        Losses = torch.zeros(rows, device=dev)
        sub_ov, sub_nz = 0, 0
        
        # 按列块处理
        for c1 in range(0, cols, col_block_size):
            c2 = min(c1 + col_block_size, cols)
            
            W_block = W[:, c1:c2]
            row_exps = select_exponents_batch(W_block, continuous)
            
            W1 = W[:, c1:c2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[c1:c2, c1:c2]
            block_nz_mask = nonzero_mask[:, c1:c2]
            
            for i in range(c2 - c1):
                w_col = W1[:, i]
                d = Hinv1[i, i]
                
                q_col, ov, nz = quantize_col_with_row_exps(w_col, row_exps)
                
                Q1[:, i] = q_col
                sub_ov += ov
                sub_nz += nz
                
                err = (w_col - q_col) / d
                Losses += (w_col - q_col) ** 2 / d ** 2
                Err1[:, i] = err
                
                # 块内补偿（只补偿非零位置）
                if i < c2 - c1 - 1:
                    compensation = err.unsqueeze(1) @ Hinv1[i, i+1:].unsqueeze(0)
                    future_mask = block_nz_mask[:, i+1:]
                    W1[:, i+1:] -= compensation * future_mask.float()
            
            Q[:, c1:c2] = Q1
            # 块间补偿（只补偿非零位置）
            if c2 < cols:
                full_comp = Err1 @ Hinv[c1:c2, c2:]
                future_mask = nonzero_mask[:, c2:]
                W[:, c2:] -= full_comp * future_mask.float()
        
        if isinstance(subset[name], transformers.Conv1D): Q = Q.t()
        subset[name].weight.data = Q.reshape(orig_shape).to(orig_dtype)
        
        stats['overflow'] += sub_ov
        stats['total'] += sub_nz
        
        elapsed = time.time() - tick
        loss = torch.sum(Losses).item() / 2
        print(f'  L{layer_idx} {name} t={elapsed:.1f}s err={loss:.1f} ov={sub_ov}/{sub_nz}')
        del H_dict[name]
    
    for j in range(nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
    
    layer = layer.cpu()
    torch.cuda.empty_cache()
    return layer, outs, stats


def process_layer_col_block_mode(layer_idx, layer, inps, attn_mask, args, row_block_size, continuous, dev, skip=False):
    """处理单层 - 4列×row_block_size行共享指数模式"""
    layer = layer.to(dev)
    subset = find_layers(layer)
    nsamples = inps.shape[0]
    outs = torch.zeros_like(inps)
    
    if skip:
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
        layer = layer.cpu()
        torch.cuda.empty_cache()
        return layer, outs, {'overflow': 0, 'total': 0}
    
    H_dict, ns_dict = {}, {}
    for name in subset:
        W = subset[name].weight.data
        if isinstance(subset[name], transformers.Conv1D): W = W.t()
        cols = W.shape[1]
        H_dict[name] = torch.zeros((cols, cols), device=dev)
        ns_dict[name] = 0

    def hook(name):
        def fn(_, inp, out):
            x = inp[0]
            if len(x.shape) == 2: x = x.unsqueeze(0)
            if len(x.shape) == 3: x = x.reshape(-1, x.shape[-1])
            x = x.t().float()
            n = x.shape[1]
            H_dict[name] *= ns_dict[name] / (ns_dict[name] + n)
            ns_dict[name] += n
            x = math.sqrt(2 / ns_dict[name]) * x
            H_dict[name] += x @ x.t()
        return fn
    
    handles = [subset[n].register_forward_hook(hook(n)) for n in subset]
    for j in range(nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
    for h in handles: h.remove()

    stats = {'overflow': 0, 'total': 0}
    
    for name in subset:
        tick = time.time()
        W = subset[name].weight.data.clone()
        orig_shape, orig_dtype = W.shape, W.dtype
        if isinstance(subset[name], transformers.Conv1D): W = W.t()
        W = W.float().to(dev)
        rows, cols = W.shape
        
        # 非零mask
        nonzero_mask = (W != 0)
        
        H = H_dict[name]
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        damp = args.percdamp * torch.mean(torch.diag(H))
        H[torch.arange(cols, device=dev), torch.arange(cols, device=dev)] += damp
        
        try:
            H = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(H)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
        except:
            Hinv = torch.eye(cols, device=dev)
        
        Q = torch.zeros_like(W)
        Losses = torch.zeros(rows, device=dev)
        sub_ov, sub_nz = 0, 0
        
        num_groups = cols // 4 if cols % 4 == 0 else 0
        num_row_blocks = (rows + row_block_size - 1) // row_block_size
        
        # 预计算每个(group, row_block)的共享指数: 4列×row_block_size行
        shared_exps = {}
        # 预计算所有(group, row_block)的共享指数 - 向量化
        if num_groups > 0:
            # [num_row_blocks, num_groups, 4]
            all_shared_exps = select_exponents_all_blocks(W, num_groups, row_block_size, continuous, dev)
        
        # 预计算行到row_block的映射
        row_to_rb = torch.arange(rows, device=dev) // row_block_size
        row_to_rb = row_to_rb.clamp(max=num_row_blocks-1)
        
        # OBS按块处理
        obs_bs = 128
        for c1 in range(0, cols, obs_bs):
            c2 = min(c1 + obs_bs, cols)
            W1 = W[:, c1:c2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[c1:c2, c1:c2]
            block_nz_mask = nonzero_mask[:, c1:c2]
            
            for i in range(c2 - c1):
                col = c1 + i
                g = col // 4 if num_groups > 0 else -1
                w_col = W1[:, i]
                d = Hinv1[i, i]
                
                # 按row_block量化
# 向量化量化整列
                if num_groups > 0 and g >= 0:
                    # all_shared_exps: [num_row_blocks, num_groups, 4]
                    shared_exps_col = all_shared_exps[:, g, :]  # [num_row_blocks, 4]
                    q_col, ov, nz = quantize_column_with_block_exps(w_col, shared_exps_col, row_block_size, dev)
                else:
                    q_col, ov, nz = quantize_column_full(w_col, continuous)
                sub_ov += ov
                sub_nz += nz
                
                Q1[:, i] = q_col
                
                err = (w_col - q_col) / d
                Losses += (w_col - q_col) ** 2 / d ** 2
                Err1[:, i] = err
                
                # 块内补偿（只补偿非零位置）
                if i < c2 - c1 - 1:
                    compensation = err.unsqueeze(1) @ Hinv1[i, i+1:].unsqueeze(0)
                    future_mask = block_nz_mask[:, i+1:]
                    W1[:, i+1:] -= compensation * future_mask.float()
            
            Q[:, c1:c2] = Q1
            # 块间补偿（只补偿非零位置）
            if c2 < cols:
                full_comp = Err1 @ Hinv[c1:c2, c2:]
                future_mask = nonzero_mask[:, c2:]
                W[:, c2:] -= full_comp * future_mask.float()
        
        if isinstance(subset[name], transformers.Conv1D): Q = Q.t()
        subset[name].weight.data = Q.reshape(orig_shape).to(orig_dtype)
        
        stats['overflow'] += sub_ov
        stats['total'] += sub_nz
        
        elapsed = time.time() - tick
        loss = torch.sum(Losses).item() / 2
        print(f'  L{layer_idx} {name} t={elapsed:.1f}s err={loss:.1f} ov={sub_ov}/{sub_nz}')
        del H_dict[name]
    
    for j in range(nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
    
    layer = layer.cpu()
    torch.cuda.empty_cache()
    return layer, outs, stats


@torch.no_grad()
def run_quantization(model, dataloader, dev, args, mode, block_size, continuous):
    """运行量化"""
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
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, inp, **kw):
            inps[cache['i']] = inp; cache['i'] += 1
            cache['attention_mask'] = kw['attention_mask']
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try: model(batch[0].to(dev))
        except ValueError: pass
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
    attn_mask = cache['attention_mask']
    
    total_stats = {'overflow': 0, 'total': 0}
    
    for i in range(len(layers)):
        skip = (i < 4)
        
        if mode == 'col':
            layer, outs, st = process_layer_col_mode(
                i, layers[i], inps, attn_mask, args, block_size, continuous, dev, skip)
        elif mode == 'col_block':
            layer, outs, st = process_layer_col_block_mode(
                i, layers[i], inps, attn_mask, args, block_size, continuous, dev, skip)
        else:  # row or full_col
            layer, outs, st = process_layer_row_mode(
                i, layers[i], inps, attn_mask, args, block_size, continuous, dev, skip)
        
        layers[i] = layer
        total_stats['overflow'] += st['overflow']
        total_stats['total'] += st['total']
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return total_stats


@torch.no_grad()
def eval_ppl(model, testenc, dev):
    """评估PPL"""
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
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, inp, **kw):
            inps[cache['i']] = inp; cache['i'] += 1
            cache['attention_mask'] = kw['attention_mask']
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try: model(batch)
        except ValueError: pass
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
    attn_mask = cache['attention_mask']

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
        layers[i] = layer.cpu()
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
        h = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            h = model.model.decoder.final_layer_norm(h)
        if model.model.decoder.project_out is not None:
            h = model.model.decoder.project_out(h)
        logits = model.lm_head(h)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * model.seqlen)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    return ppl.item()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--base_model', type=str, default='/home/LHZ/opt/model/opt-6.7b')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=0.01)
    args = parser.parse_args()

    # 8种配置: (name, mode, block_size, continuous)
    configs = [
        ('row_128_noncont', 'row', 128, False),
        ('row_128_cont', 'row', 128, True),
        ('col_128_noncont', 'col', 128, False),
        ('col_128_cont', 'col', 128, True),
        ('col_4x128_noncont', 'col_block', 128, False),
        ('col_4x128_cont', 'col_block', 128, True),
        ('fullcol_noncont', 'row', -1, False),
        ('fullcol_cont', 'row', -1, True),
    ]

    print('='*70)
    print('Shared Exponent Quantization - 8 Configurations Comparison')
    print('='*70)

    print('\nLoading model and data...')
    base_model = get_opt(args.model)
    base_model.eval()
    
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=base_model.seqlen, model=args.base_model
    )

    print('\n--- Baseline ---')
    baseline_ppl = eval_ppl(base_model, testloader, DEV)
    print(f'Baseline PPL: {baseline_ppl:.3f}')

    results = []
    
    for name, mode, bs, cont in configs:
        print(f'\n{"="*70}')
        print(f'Config: {name} (mode={mode}, bs={bs}, cont={cont})')
        print('='*70)
        
        model = get_opt(args.model)
        model.eval()
        
        tick = time.time()
        stats = run_quantization(model, dataloader, DEV, args, mode, bs, cont)
        elapsed = time.time() - tick
        
        ppl = eval_ppl(model, testloader, DEV)
        
        ov_rate = stats['overflow'] / max(stats['total'], 1) * 100
        results.append({
            'name': name, 'ppl': ppl, 'overflow': stats['overflow'],
            'total': stats['total'], 'ov_rate': ov_rate, 'time': elapsed
        })
        
        print(f'\nResult: PPL={ppl:.3f}, Overflow={ov_rate:.2f}%, Time={elapsed:.1f}s')
        
        del model
        torch.cuda.empty_cache()

    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'{"Config":<20} {"PPL":>8} {"PPL Δ":>8} {"Overflow%":>10} {"Time(s)":>8}')
    print('-'*70)
    print(f'{"Baseline":<20} {baseline_ppl:>8.3f} {0:>8.3f} {"N/A":>10} {"N/A":>8}')
    for r in results:
        delta = r['ppl'] - baseline_ppl
        print(f'{r["name"]:<20} {r["ppl"]:>8.3f} {delta:>+8.3f} {r["ov_rate"]:>9.2f}% {r["time"]:>8.1f}')
    print('='*70)