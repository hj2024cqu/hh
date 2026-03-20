#!/bin/bash
# ============================================================
# 实验流程: 基线 vs Hessian加权 → 分析不连续指数模式
# ============================================================
# 用法:
#   chmod +x run_experiment.sh
#   ./run_experiment.sh
# ============================================================

set -e  # 出错即停

# ======================== 配置区 ========================
MODEL="/home/hej/model/float/stage1_pruned"
BASE_MODEL="/home/hej/model/float/opt-6.7b"
DATASET="wikitext2"

# 输出目录
BASELINE_DIR="/home/hej/model/float/sparsegpt/compressed_baseline"
HESSIAN_DIR="/home/hej/model/float/sparsegpt/compressed_hessian"

# 量化参数(按需修改)
NSAMPLES=128
BLOCKSIZE=128
MANTISSA_BITS=4
ROW_BLOCK_SIZE=-1   # -1=整列, 128=4x128块
# ========================================================

# echo "============================================================"
# echo "  Step 1/3: 基线量化 (纯频率top4选指数)"
# echo "============================================================"
# python opt_shared_exp_swap_v5.py \
#     ${MODEL} ${DATASET} \
#     --base_model ${BASE_MODEL} \
#     --nsamples ${NSAMPLES} \
#     --blocksize ${BLOCKSIZE} \
#     --mantissa_bits ${MANTISSA_BITS} \
#     --row_block_size ${ROW_BLOCK_SIZE} \
#     --save ${BASELINE_DIR}

# echo ""
# echo "============================================================"
echo "  Step 2/3: Hessian加权量化"
echo "============================================================"
python opt_shared_exp_swap.py \
    ${MODEL} ${DATASET} \
    --base_model ${BASE_MODEL} \
    --nsamples ${NSAMPLES} \
    --blocksize ${BLOCKSIZE} \
    --mantissa_bits ${MANTISSA_BITS} \
    --row_block_size ${ROW_BLOCK_SIZE} \
    --save ${HESSIAN_DIR} \
    --hessian_weighted

echo ""
echo "============================================================"
echo "  Step 3/3: 对比分析不连续指数模式"
echo "============================================================"
python analyze_patterns.py \
    # ${BASELINE_DIR}/non_continuous_exponents.json \
    ${HESSIAN_DIR}/non_continuous_exponents.json

echo ""
echo "============================================================"
echo "  完成! JSON文件位置:"
# echo "    基线:    ${BASELINE_DIR}/non_continuous_exponents.json"
echo "    Hessian: ${HESSIAN_DIR}/non_continuous_exponents.json"
echo "============================================================"