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
# 实验配置:  block_size | precomputed_dir | packed_output_dir
#
# precomputed_dir 必须与 test.sh 产出路径一致:
#   ${ROOT}/compressed_${tag}_${MANTISSA}bits/mantissa_${MANTISSA}bit
# 其中 tag: -1→full, 其余→数字本身
# ============================================================

declare -a CONFIGS=(
    "-1|${ROOT}/compressed_full_${MANTISSA}bits/mantissa_${MANTISSA}bit|${ROOT}/compressed_full_${MANTISSA}bits/mantissa_${MANTISSA}bit/packed"
    # "1024|${ROOT}/compressed_1024_${MANTISSA}bits/mantissa_${MANTISSA}bit|${ROOT}/compressed_1024_${MANTISSA}bits/mantissa_${MANTISSA}bit/packed"
    # "512|${ROOT}/compressed_512_${MANTISSA}bits/mantissa_${MANTISSA}bit|${ROOT}/compressed_512_${MANTISSA}bits/mantissa_${MANTISSA}bit/packed"
    # "256|${ROOT}/compressed_256_${MANTISSA}bits/mantissa_${MANTISSA}bit|${ROOT}/compressed_256_${MANTISSA}bits/mantissa_${MANTISSA}bit/packed"
)

# ============================================================
# 逐组跑 pack → unpack_eval
# ============================================================
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r BLOCK_SIZE PRECOMPUTED OUTPUT_DIR <<< "$cfg"

    # 跳过预计算目录不存在的配置
    if [ ! -d "${PRECOMPUTED}" ]; then
        echo "[SKIP] ${PRECOMPUTED} 不存在, 跳过 block_size=${BLOCK_SIZE}"
        continue
    fi

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