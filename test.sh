#!/bin/bash
# ======================================================================
# 一键运行: test.py 预计算 + 量化
# 全部输出为 .pt 格式 (int8), 兼容 pack.py v2 和 opt_shared_exp_quant.py
# ======================================================================

# -------------------- 路径参数 --------------------
PRUNED_MODEL="/home/hej/model/float/stage1_pruned"
BASE_MODEL="/home/hej/model/float/opt-6.7b"
OUTPUT_ROOT="/home/hej/model/float/sparsegpt"

# -------------------- 量化参数 --------------------
MANTISSA_BITS=3
DATASET="wikitext2"
SKIP_MANTISSA="--skip_mantissa_quant"   # 留空则不跳过: SKIP_MANTISSA=""

# -------------------- row_block_size 列表 --------------------
RBS_LIST=(-1 )
# RBS_LIST=(128 64 32 16)

# -------------------- checkpoint --------------------
CHECKPOINT_EVERY=4    # 每 N 层保存一次 checkpoint
RESUME=true           # true 则自动从断点恢复

# -------------------- 是否运行量化 --------------------
RUN_QUANTIZE=true     # false 则只跑 test.py 预计算

# -------------------- 保存量化后模型 --------------------
SAVE_QUANTIZED=true   # false 则不保存模型

# ======================================================================
# 以下无需修改
# ======================================================================

rbs_to_name() {
    local rbs=$1
    if [ "$rbs" -eq -1 ]; then
        echo "full"
    else
        echo "$rbs"
    fi
}

for rbs in "${RBS_LIST[@]}"; do
    tag=$(rbs_to_name "$rbs")
    outdir="${OUTPUT_ROOT}/compressed_${tag}_${MANTISSA_BITS}bits/mantissa_${MANTISSA_BITS}bit"

    echo ""
    echo "======================================================================"
    echo "  row_block_size=${rbs}  (${tag})"
    echo "  output: ${outdir}"
    echo "======================================================================"

    # ---- Step 1: test.py 预计算 ----
    echo "[Step 1] test.py 预计算..."

    RESUME_ARG=""
    if [ "${RESUME}" = true ]; then
        RESUME_ARG="--resume"
    fi

    python test.py "${PRUNED_MODEL}" \
        --mantissa_bits "${MANTISSA_BITS}" \
        --row_block_size "${rbs}" \
        --checkpoint_every "${CHECKPOINT_EVERY}" \
        ${RESUME_ARG} \
        --output_dir "${outdir}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] test.py failed for rbs=${rbs}, skipping..."
        continue
    fi

    # ---- Step 2: 量化 ----
    if [ "${RUN_QUANTIZE}" = true ]; then
        echo "[Step 2] 量化..."

        SAVE_ARG=""
        if [ "${SAVE_QUANTIZED}" = true ]; then
            SAVE_ARG="--save ${outdir}/quantized_model"
        fi

        python opt_shared_exp_quant.py "${PRUNED_MODEL}" "${DATASET}" \
            --base_model "${BASE_MODEL}" \
            --mantissa_bits "${MANTISSA_BITS}" \
            ${SKIP_MANTISSA} \
            --row_block_size "${rbs}" \
            --precomputed_dir "${outdir}" \
            ${SAVE_ARG}

        if [ $? -ne 0 ]; then
            echo "[ERROR] quantize failed for rbs=${rbs}"
        fi
    fi

    echo "[DONE] rbs=${rbs}"
done

echo ""
echo "======================================================================"
echo "  全部完成"
echo "======================================================================"