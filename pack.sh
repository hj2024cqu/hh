#!/bin/bash
set -e

# ============================================================
# 公共参数
# ============================================================
MODEL=/home/hej/model/float/stage1_pruned
BASE_MODEL=/home/hej/model/float/opt-6.7b
ROOT=/home/hej/model/float/sparsegpt
MANTISSA=3
DATASET=wikitext2

# ============================================================
# 实验配置:  block_size  precomputed_dir  output_dir
# ============================================================
#  full (-1) 的预计算目录比较特殊，单独列出
#  其余三组路径规律一致

declare -a CONFIGS=(
    # row_block_size | precomputed_dir | packed_output_dir
    # "-1|${ROOT}/compressed_full_10bits/mantissa_10|${ROOT}/compressed_full_3bits/mantissa_3/packed"
    "1024|${ROOT}/compressed_1024_3bits/mantissa_3bit|${ROOT}/compressed_1024_3bits/mantissa_3/packed"
    "512|${ROOT}/compressed_512_3bits/mantissa_3bit|${ROOT}/compressed_512_3bits/mantissa_3/packed"
    "256|${ROOT}/compressed_256_3bits/mantissa_3bit|${ROOT}/compressed_256_3bits/mantissa_3/packed"
)

# ============================================================
# 逐组跑 pack → unpack_eval
# ============================================================
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r BLOCK_SIZE PRECOMPUTED OUTPUT_DIR <<< "$cfg"

    echo ""
    echo "######################################################################"
    echo "# row_block_size=${BLOCK_SIZE}"
    echo "######################################################################"

    # ---- Pack ----
    echo "[Pack] block_size=${BLOCK_SIZE}"
    python pack.py \
        --model ${MODEL} \
        --precomputed_dir ${PRECOMPUTED} \
        --row_block_size ${BLOCK_SIZE} \
        --mantissa_bits ${MANTISSA} \
        --output_dir ${OUTPUT_DIR}

    # ---- Unpack + Eval ----
    echo "[Unpack+Eval] block_size=${BLOCK_SIZE}"
    python unpack_eval.py \
        --packed_path ${OUTPUT_DIR}/packed_model.pt \
        --model_structure ${MODEL} \
        --base_model ${BASE_MODEL} \
        --dataset ${DATASET}

done

echo ""
echo "======================================================================"
echo "All done!"
echo "======================================================================"