"""
SparseGPT + 高通AIMET量化工具集成
支持INT8量化和FP8模拟
适配RTX 4090
"""

import time
import torch
import torch.nn as nn
import numpy as np
import json
import os

from quant import *
from sparsegpt import *
from modelutils import *

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

# ==================== 高通AIMET量化接口 ====================
class QualcommQuantizer:
    """
    高通量化器 - 支持AIMET INT8和FP8模拟
    """
    
    def __init__(self, quantization_type='int8'):
        self.quant_type = quantization_type
        self.aimet_available = False
        
        # 尝试导入AIMET
        try:
            import aimet_torch
            import aimet_common.defs as aimet_defs
            from aimet_torch.quantsim import QuantizationSimModel
            self.aimet_torch = aimet_torch
            self.aimet_defs = aimet_defs
            self.QuantizationSimModel = QuantizationSimModel
            self.aimet_available = True
            print("AIMET已加载，使用高通官方量化工具")
        except ImportError:
            print("AIMET未安装，使用模拟量化。安装方法：")
            print("pip install aimet-torch")
            print("或访问: https://github.com/quic/aimet")
    
    def quantize_with_aimet(self, model, dummy_input, num_calibration_batches=10):
        """使用AIMET进行INT8量化"""
        if not self.aimet_available:
            print("AIMET不可用，回退到模拟量化")
            return self.simulate_quantization(model)
        
        print("使用AIMET进行INT8量化...")
        
        # 创建量化模拟模型
        quant_sim = self.QuantizationSimModel(
            model=model,
            dummy_input=dummy_input,
            quant_scheme=self.aimet_defs.QuantScheme.post_training_tf_enhanced,
            default_output_bw=8,
            default_param_bw=8
        )
        
        # 设置量化参数
        quant_sim.config_file = None  # 使用默认配置
        
        # 计算编码（校准）
        def calibrate(model, calibration_loader):
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if i >= num_calibration_batches:
                        break
                    model(batch[0])
        
        print("运行校准...")
        quant_sim.compute_encodings(calibrate, forward_pass_callback_args=model)
        
        # 导出量化模型
        quant_sim.export(path="./aimet_quantized_model", filename_prefix="opt_int8")
        
        return quant_sim.model
    
    def simulate_quantization(self, model):
        """模拟INT8/FP8量化（当AIMET不可用时）"""
        if self.quant_type == 'fp8':
            return self._simulate_fp8(model)
        else:
            return self._simulate_int8(model)
    
    def _simulate_int8(self, model):
        """INT8量化模拟"""
        print("模拟INT8量化...")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 计算INT8量化参数
                weight = module.weight.data
                weight_scale = weight.abs().max() / 127.0
                
                # 量化和反量化
                weight_int8 = torch.round(weight / weight_scale).clamp(-128, 127)
                module.weight.data = weight_int8 * weight_scale
                
                if module.bias is not None:
                    bias_scale = module.bias.data.abs().max() / 127.0
                    bias_int8 = torch.round(module.bias.data / bias_scale).clamp(-128, 127)
                    module.bias.data = bias_int8 * bias_scale
        
        return model
    
    def _simulate_fp8(self, model):
        """FP8量化模拟（E4M3格式）"""
        print("模拟FP8量化 (E4M3格式)...")
        
        # FP8 E4M3参数
        max_val = 448.0
        min_val = -448.0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 量化权重
                weight = module.weight.data
                
                # Clamp到FP8范围
                weight_clamped = torch.clamp(weight, min_val, max_val)
                
                # 简化的FP8模拟：降低精度
                # 使用round来模拟精度损失
                precision_scale = 256.0  # 模拟3位尾数的精度
                weight_quantized = torch.round(weight_clamped * precision_scale) / precision_scale
                
                module.weight.data = weight_quantized
                
                # 量化偏置
                if module.bias is not None:
                    bias_clamped = torch.clamp(module.bias.data, min_val, max_val)
                    bias_quantized = torch.round(bias_clamped * precision_scale) / precision_scale
                    module.bias.data = bias_quantized
        
        return model


class FastFP8Quantizer:
    """快速FP8量化器，优化了性能"""
    
    @staticmethod
    def quantize_model(model, format='E4M3'):
        print(f"\n应用快速FP8量化 (格式: {format})...")
        
        # 设置FP8参数
        if format == 'E4M3':
            max_val = 448.0
            precision_bits = 3
        else:  # E5M2
            max_val = 57344.0
            precision_bits = 2
        
        precision_scale = 2 ** precision_bits
        
        # 批量处理所有Linear层
        linear_layers = [(name, module) for name, module in model.named_modules() 
                        if isinstance(module, nn.Linear)]
        
        total_params = 0
        total_error = 0
        
        for name, module in linear_layers:
            with torch.no_grad():
                # 权重量化
                w = module.weight.data
                w_abs_max = w.abs().max()
                
                if w_abs_max > 0:
                    # 动态缩放
                    scale = min(1.0, max_val / w_abs_max)
                    w_scaled = w * scale
                    
                    # 量化
                    w_quantized = torch.round(w_scaled * precision_scale) / precision_scale
                    w_dequantized = w_quantized / scale
                    
                    # 计算误差
                    error = (w - w_dequantized).abs().mean().item()
                    total_error += error * w.numel()
                    total_params += w.numel()
                    
                    module.weight.data = w_dequantized
                    
                    print(f"  {name}: 缩放={scale:.4f}, 误差={error:.6f}")
                
                # 偏置量化
                if module.bias is not None:
                    b = module.bias.data
                    b_quantized = torch.clamp(b, -max_val, max_val)
                    b_quantized = torch.round(b_quantized * precision_scale) / precision_scale
                    module.bias.data = b_quantized
        
        avg_error = total_error / total_params if total_params > 0 else 0
        print(f"\n平均量化误差: {avg_error:.6f}")
        
        return model

# ==================== 在原始代码中添加量化功能 ====================

# ... (保留原始的get_opt, opt_sequential, opt_eval函数不变) ...

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    # 原始参数（保持不变）
    parser.add_argument('model', type=str, help='OPT model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=.01)
    parser.add_argument('--sparsity', type=float, default=0)
    parser.add_argument('--prunen', type=int, default=0)
    parser.add_argument('--prunem', type=int, default=0)
    parser.add_argument('--blocksize', type=int, default=128)
    parser.add_argument('--gmp', action='store_true')
    parser.add_argument('--wbits', type=int, default=16)
    parser.add_argument('--minlayer', type=int, default=-1)
    parser.add_argument('--maxlayer', type=int, default=1000)
    parser.add_argument('--prune_only', type=str, default='')
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--log_wandb', action='store_true')
    
    # 新增量化参数
    parser.add_argument('--quantize', action='store_true', help='Apply quantization after pruning')
    parser.add_argument('--quant_type', type=str, default='fp8', 
                       choices=['int8', 'fp8', 'aimet'], help='Quantization type')
    parser.add_argument('--fp8_format', type=str, default='E4M3',
                       choices=['E4M3', 'E5M2'], help='FP8 format')
    parser.add_argument('--use_aimet', action='store_true', help='Try to use Qualcomm AIMET')

    args = parser.parse_args()

    # W&B日志
    if args.log_wandb:
        assert has_wandb, "wandb not installed"
        wandb.init(config=args)

    # 加载模型
    print("="*60)
    print(f"加载模型: {args.model}")
    print("="*60)
    
    model = get_opt(args.model)
    model.eval()

    # 数据加载
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, 
        model=args.model, seqlen=model.seqlen
    )

    # 步骤1: SparseGPT剪枝
    if (args.sparsity or args.prunen) and not args.gmp:
        print("\n" + "="*60)
        print("步骤1: SparseGPT剪枝")
        print("="*60)
        
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        
        # 统计稀疏性
        print("\n剪枝结果统计:")
        for n, p in model.named_parameters():
            if 'weight' in n:
                sparsity = (p == 0).float().mean().item()
                if sparsity > 0:
                    print(f"  {n}: {sparsity:.1%} 稀疏")
            if 'fc2' in n:
                break
        
        print(f"\n剪枝耗时: {time.time() - tick:.2f}秒")

    # 步骤2: 评估剪枝模型
    print("\n" + "="*60)
    print("步骤2: 评估剪枝后的模型")
    print("="*60)
    
    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(f"\n{dataset}:")
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    # 步骤3: 量化
    if args.quantize:
        print("\n" + "="*60)
        print(f"步骤3: {args.quant_type.upper()}量化")
        print("="*60)
        
        if args.use_aimet or args.quant_type == 'aimet':
            # 使用高通AIMET
            quantizer = QualcommQuantizer(args.quant_type)
            dummy_input = torch.randint(0, model.config.vocab_size, (1, model.seqlen)).to(DEV)
            
            if quantizer.aimet_available:
                model = quantizer.quantize_with_aimet(model, dummy_input, dataloader)
            else:
                model = quantizer.simulate_quantization(model)
        
        elif args.quant_type == 'fp8':
            # 使用快速FP8量化
            model = FastFP8Quantizer.quantize_model(model, format=args.fp8_format)
        
        else:  # int8
            quantizer = QualcommQuantizer('int8')
            model = quantizer.simulate_quantization(model)
        
        # 评估量化后的模型
        print("\n" + "="*60)
        print("步骤4: 评估量化后的模型")
        print("="*60)
        
        dataloader, testloader = get_loaders(
            'wikitext2', seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        opt_eval(model, testloader, DEV, 'wikitext2', args.log_wandb)

    # 保存模型
    if args.save:
        save_path = args.save
        if args.quantize:
            save_path += f"_{args.quant_type}"
            if args.quant_type == 'fp8':
                save_path += f"_{args.fp8_format.lower()}"
        
        print(f"\n保存模型到: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        
        # 保存配置信息
        config_info = {
            'model': args.model,
            'pruning': {
                'method': 'sparsegpt',
                'sparsity': args.sparsity,
                'n_m': f"{args.prunen}:{args.prunem}" if args.prunen else None
            },
            'quantization': {
                'enabled': args.quantize,
                'type': args.quant_type if args.quantize else None,
                'fp8_format': args.fp8_format if args.quant_type == 'fp8' else None,
                'tool': 'aimet' if args.use_aimet else 'simulated'
            }
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_info, f, indent=2)
        
        print("完成！")