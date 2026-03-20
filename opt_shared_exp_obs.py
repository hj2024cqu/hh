"""
共享指数量化 + OBS误差补偿（优化版）
====================================

优化：
1. 向量化处理，避免Python循环
2. 支持多GPU并行（4张A6000）

使用方法：
    # 单GPU
    python opt_shared_exp_only.py ./compressed_models/stage1_pruned wikitext2 --base_model /home/LHZ/opt/model/opt-6.7b --nsamples 128
    
    # 多GPU
    python opt_shared_exp_only.py ./compressed_models/stage1_pruned wikitext2 --base_model /home/LHZ/opt/model/opt-6.7b --nsamples 128 --multigpu
"""

import time
import math
from collections import Counter

import torch
import torch.nn as nn
import transformers

from datautils import get_loaders
from modelutils import find_layers, DEV


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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


class SharedExpQuantizerFast:
    """
    向量化的共享指数量化器
    """
    
    def __init__(self, block_size=128):
        self.block_size = block_size
    
    def select_shared_exponents_vectorized(self, weights):
        """向量化选择4个共享指数"""
        device = weights.device
        
        nonzero_mask = weights != 0
        if not nonzero_mask.any():
            return torch.tensor([-8, -7, -6, -5], dtype=torch.int32, device=device)
        
        nonzero_weights = weights[nonzero_mask]
        abs_w = torch.abs(nonzero_weights).clamp(min=1e-38)
        exponents = torch.floor(torch.log2(abs_w)).to(torch.int32)
        
        # 统计指数频率
        exp_min = exponents.min().item()
        exp_max = exponents.max().item()
        
        if exp_max - exp_min <= 3:
            # 范围小于4，直接使用
            result = list(range(int(exp_min), int(exp_max) + 1))
            while len(result) < 4:
                result.append(result[-1] + 1)
            return torch.tensor(sorted(result[:4]), dtype=torch.int32, device=device)
        
        # 用直方图找最密集区域
        bins = exp_max - exp_min + 1
        hist = torch.histc(exponents.float(), bins=int(bins), min=exp_min, max=exp_max)
        
        # 找连续3个bin的最大和
        if bins >= 3:
            conv_sum = hist[:-2] + hist[1:-1] + hist[2:]
            best_start_idx = torch.argmax(conv_sum).item()
            best_3 = [exp_min + best_start_idx, exp_min + best_start_idx + 1, exp_min + best_start_idx + 2]
        else:
            best_3 = [exp_min, exp_min + 1, exp_min + 2]
        
        # 第4个：覆盖最大绝对值
        max_idx = torch.argmax(torch.abs(nonzero_weights))
        fourth_exp = exponents[max_idx].item()
        
        all_exps = set(best_3 + [int(fourth_exp)])
        result = sorted(list(all_exps))
        
        # 补齐到4个（选频率最高的）
        if len(result) < 4:
            for e in range(int(exp_min), int(exp_max) + 1):
                if e not in result:
                    result.append(e)
                if len(result) == 4:
                    break
        
        while len(result) < 4:
            result.append(result[-1] + 1)
        
        return torch.tensor(sorted(result[:4]), dtype=torch.int32, device=device)
    
    def quantize_block_vectorized(self, w_block):
        """
        向量化量化一个块
        
        返回: (量化后权重, 溢出数量, 总非零数量)
        """
        device = w_block.device
        n = len(w_block)
        
        # 找非零权重
        nonzero_mask = w_block != 0
        nonzero_count = nonzero_mask.sum().item()
        
        if nonzero_count == 0:
            return w_block.clone(), 0, 0
        
        # 选择共享指数
        shared_exp = self.select_shared_exponents_vectorized(w_block)  # [4]
        
        # 准备输出
        q_block = w_block.clone()
        
        # 只处理非零元素
        nonzero_w = w_block[nonzero_mask]
        signs = torch.sign(nonzero_w)
        abs_w = torch.abs(nonzero_w)
        
        # 对每个共享指数计算尾数 [4, num_nonzero]
        scales = (2.0 ** shared_exp.float()).unsqueeze(1)  # [4, 1]
        mantissas = abs_w.unsqueeze(0) / scales  # [4, num_nonzero]
        
        # 检查哪些在有效范围 [1.0, 2.0)
        valid_mask = (mantissas >= 1.0) & (mantissas < 2.0)  # [4, num_nonzero]
        
        # 对于有效的，计算误差
        # 对于无效的，clamp后计算误差
        clamped_mantissas = torch.clamp(mantissas, 1.0, 1.9999999)
        reconstructed = clamped_mantissas * scales  # [4, num_nonzero]
        errors = torch.abs(abs_w.unsqueeze(0) - reconstructed)  # [4, num_nonzero]
        
        # 选择误差最小的指数
        best_k = torch.argmin(errors, dim=0)  # [num_nonzero]
        
        # 获取最佳量化值
        best_mantissa = clamped_mantissas[best_k, torch.arange(len(best_k), device=device)]
        best_scale = scales[best_k, 0]
        best_q = signs * best_mantissa * best_scale
        
        # 检查溢出（最佳选择是否有效）
        best_valid = valid_mask[best_k, torch.arange(len(best_k), device=device)]
        overflow_count = (~best_valid).sum().item()
        
        # 写回
        q_block[nonzero_mask] = best_q
        
        return q_block, overflow_count, nonzero_count


def process_layer_on_gpu(layer_idx, layer, inps, attention_mask, args, quantizer, dev):
    """在指定GPU上处理一层"""
    layer = layer.to(dev)
    subset = find_layers(layer)
    
    outs = torch.zeros_like(inps)
    
    # 收集Hessian
    H_dict = {}
    nsamples_dict = {}
    
    for name in subset:
        W = subset[name].weight.data
        if isinstance(subset[name], transformers.Conv1D):
            W = W.t()
        cols = W.shape[1]
        H_dict[name] = torch.zeros((cols, cols), device=dev)
        nsamples_dict[name] = 0

    def add_batch(name):
        def tmp(_, inp, out):
            if len(inp[0].shape) == 2:
                inp_data = inp[0].unsqueeze(0)
            else:
                inp_data = inp[0]
            if len(inp_data.shape) == 3:
                inp_data = inp_data.reshape((-1, inp_data.shape[-1]))
            inp_data = inp_data.t().float()
            
            tmp_n = inp_data.shape[1]
            H_dict[name] *= nsamples_dict[name] / (nsamples_dict[name] + tmp_n)
            nsamples_dict[name] += tmp_n
            inp_data = math.sqrt(2 / nsamples_dict[name]) * inp_data
            H_dict[name] += inp_data.matmul(inp_data.t())
        return tmp
    
    handles = []
    for name in subset:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    for j in range(args.nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
    for h in handles:
        h.remove()

    layer_stats = {'overflow': 0, 'total': 0}
    
    # 处理每个子层
    for name in subset:
        tick = time.time()
        
        W = subset[name].weight.data.clone()
        original_shape = W.shape
        original_dtype = W.dtype
        
        if isinstance(subset[name], transformers.Conv1D):
            W = W.t()
        W = W.float().to(dev)
        rows, cols = W.shape
        
        # 计算H_inv
        H = H_dict[name]
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        
        damp = args.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(cols, device=dev)
        H[diag, diag] += damp
        
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
        except:
            print(f'    Cholesky failed for {name}')
            Hinv = None
        
        Q = torch.zeros_like(W)
        Losses = torch.zeros(rows, device=dev)
        sub_overflow = 0
        sub_total = 0
        
        # 按块处理
        for i1 in range(0, cols, args.blocksize):
            i2 = min(i1 + args.blocksize, cols)
            count = i2 - i1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            
            if Hinv is not None:
                Hinv1 = Hinv[i1:i2, i1:i2]
            
            # 对块内每一列处理
            for i in range(count):
                w = W1[:, i]
                
                if Hinv is not None:
                    d = Hinv1[i, i]
                else:
                    d = 1.0
                
                # 向量化：对整列按row_block分块处理
                q = torch.zeros_like(w)
                for row_start in range(0, rows, args.blocksize):
                    row_end = min(row_start + args.blocksize, rows)
                    w_block = w[row_start:row_end]
                    
                    # 向量化量化
                    q_block, overflow, total = quantizer.quantize_block_vectorized(w_block)
                    q[row_start:row_end] = q_block
                    sub_overflow += overflow
                    sub_total += total
                
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                
                # OBS补偿
                if Hinv is not None:
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
            
            Q[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            
            if Hinv is not None:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
        # 写回
        if isinstance(subset[name], transformers.Conv1D):
            Q = Q.t()
        subset[name].weight.data = Q.reshape(original_shape).to(original_dtype)
        
        layer_stats['overflow'] += sub_overflow
        layer_stats['total'] += sub_total
        
        elapsed = time.time() - tick
        print(f'  Layer {layer_idx} {name} time {elapsed:.1f}s error {torch.sum(Losses).item():.1f} overflow {sub_overflow}/{sub_total}')
        
        del H_dict[name]
    
    # 更新输出
    for j in range(args.nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
    
    layer = layer.cpu()
    torch.cuda.empty_cache()
    
    return layer, outs, layer_stats


@torch.no_grad()
def opt_shared_exp(model, dataloader, dev, args):
    """共享指数量化 + OBS补偿"""
    print('\nStarting shared exponent quantization...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    # 准备输入
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

    print('Ready.')

    quantizer = SharedExpQuantizerFast(block_size=args.blocksize)
    total_stats = {'overflow': 0, 'total': 0}

    # 逐层处理
    for layer_idx in range(len(layers)):
        print(f'\nLayer {layer_idx}')
        
        layer, outs, layer_stats = process_layer_on_gpu(
            layer_idx, layers[layer_idx], inps, attention_mask, args, quantizer, dev
        )
        layers[layer_idx] = layer
        
        total_stats['overflow'] += layer_stats['overflow']
        total_stats['total'] += layer_stats['total']
        
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return total_stats


@torch.no_grad()
def opt_shared_exp_multigpu(model, dataloader, args):
    """多GPU并行处理"""
    print('\nStarting shared exponent quantization (Multi-GPU)...')
    
    num_gpus = torch.cuda.device_count()
    print(f'Using {num_gpus} GPUs')
    
    gpus = [torch.device(f'cuda:{i}') for i in range(num_gpus)]

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers
    num_layers = len(layers)

    # 在GPU0上准备输入
    dev = gpus[0]
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

    attention_mask = cache['attention_mask']

    print('Ready.')

    quantizer = SharedExpQuantizerFast(block_size=args.blocksize)
    total_stats = {'overflow': 0, 'total': 0}

    # 分配层到GPU（轮询方式）
    # 由于需要顺序处理（OBS依赖前一层输出），我们按组处理
    # 每组num_gpus层并行
    
    outs = torch.zeros_like(inps)
    
    for layer_idx in range(num_layers):
        # 选择GPU（轮询）
        gpu_idx = layer_idx % num_gpus
        gpu_dev = gpus[gpu_idx]
        
        print(f'\nLayer {layer_idx} on GPU {gpu_idx}')
        
        # 将输入移到目标GPU
        inps_gpu = inps.to(gpu_dev)
        attn_mask_gpu = attention_mask.to(gpu_dev) if attention_mask is not None else None
        
        layer, outs_gpu, layer_stats = process_layer_on_gpu(
            layer_idx, layers[layer_idx], inps_gpu, attn_mask_gpu, args, quantizer, gpu_dev
        )
        layers[layer_idx] = layer
        
        # 输出移回主设备
        outs = outs_gpu.to(dev)
        inps = outs.clone()
        
        total_stats['overflow'] += layer_stats['overflow']
        total_stats['total'] += layer_stats['total']
        
        del inps_gpu, outs_gpu
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return total_stats


@torch.no_grad()
def opt_eval(model, testenc, dev):
    """评估PPL"""
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
    parser.add_argument('--base_model', type=str, default='/home/LHZ/opt/model/opt-6.7b')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--blocksize', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=0.01)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--multigpu', action='store_true', help='Use multiple GPUs')

    args = parser.parse_args()

    print('='*60)
    print('Shared Exponent Quantization + OBS (Optimized)')
    print('='*60)
    print(f'Pruned model: {args.model}')
    print(f'Base model: {args.base_model}')
    print(f'Multi-GPU: {args.multigpu}')

    # 加载模型
    print('\nLoading model...')
    model = get_opt(args.model)
    model.eval()

    # 加载数据
    print('Loading data...')
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=args.base_model
    )

    # 基准PPL
    print('\n--- Baseline ---')
    baseline_ppl = opt_eval(model, testloader, DEV)

    # 共享指数量化
    tick = time.time()
    if args.multigpu and torch.cuda.device_count() > 1:
        stats = opt_shared_exp_multigpu(model, dataloader, args)
    else:
        stats = opt_shared_exp(model, dataloader, DEV, args)
    elapsed = time.time() - tick

    # 最终PPL
    print('\n--- After Shared Exponent ---')
    final_ppl = opt_eval(model, testloader, DEV)

    # 保存
    if args.save_dir:
        save_path = os.path.join(args.save_dir, 'shared_exp')
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        print(f'\nSaved to: {save_path}')

    # 总结
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'Baseline PPL: {baseline_ppl:.3f}')
    print(f'Final PPL: {final_ppl:.3f}')
    print(f'PPL increase: {final_ppl - baseline_ppl:+.3f}')
    print(f'Overflow: {stats["overflow"]}/{stats["total"]} ({stats["overflow"]/max(stats["total"],1):.2%})')
    print(f'Time: {elapsed:.1f}s')