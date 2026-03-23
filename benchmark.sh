#!/bin/bash
# ======================================================================
# 一键 benchmark: FP16 baseline / torchao Int4+Sparse / Custom SharedExp
# 适配 4090 24GB
# ======================================================================

# -------------------- 路径 --------------------
DENSE_MODEL="/home/hej/model/float/opt-6.7b"
PRUNED_MODEL="/home/hej/model/float/stage1_pruned"
BASE_MODEL="/home/hej/model/float/opt-6.7b"

# packed_model.pt 的路径 (由 pack.sh 产出)
# 可以测多个配置, 改这里即可
PACKED_PATH="/home/hej/model/float/sparsegpt/compressed_256_3bits/mantissa_3bit/packed/packed_model.pt"

# -------------------- 参数 --------------------
PROMPT_LEN=128
GEN_LEN=128
WARMUP=3
REPEATS=10
PPL_SAMPLES=40

OUTPUT_DIR="/home/hej/model/float/sparsegpt/benchmark_results"
mkdir -p ${OUTPUT_DIR}

echo "======================================================================"
echo "Step 0: 环境检查"
echo "======================================================================"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
try:
    import torchao
    print(f'torchao: {torchao.__version__}')
except:
    print('torchao: NOT INSTALLED')
    print('  Run: pip install torchao')
"

echo ""
echo "======================================================================"
echo "Step 1: FP16 Dense + torchao Int4+Sparse + Custom"
echo "======================================================================"

python benchmark.py \
    --model ${DENSE_MODEL} \
    --pruned_model ${PRUNED_MODEL} \
    --packed_path ${PACKED_PATH} \
    --base_model ${BASE_MODEL} \
    --prompt_len ${PROMPT_LEN} \
    --gen_len ${GEN_LEN} \
    --warmup ${WARMUP} \
    --repeats ${REPEATS} \
    --ppl_samples ${PPL_SAMPLES} \
    --output ${OUTPUT_DIR}/benchmark_all.json

echo ""
echo "======================================================================"
echo "Done! Results in ${OUTPUT_DIR}/benchmark_all.json"
echo "======================================================================"