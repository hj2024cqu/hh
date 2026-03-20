#!/bin/bash
#2:4剪枝
python opt.py facebook/opt-6.7b wikitext2 \
    --prunen 2 --prunem 4 \
    --save /home/hej/model/float/stage1_pruned

# ======================================================================
# 一键运行: test.py 预计算 + 量化
# 修改下面的参数即可，路径和配置会自动传播
# ======================================================================

# -------------------- 路径参数 --------------------
PRUNED_MODEL="/home/hej/model/float/stage1_pruned"       # 2:4剪枝后的模型路径
BASE_MODEL="/home/hej/model/float/opt-6.7b"              # 用于加载tokenizer的基础模型路径（通常与pruned_model相同）
OUTPUT_ROOT="/home/hej/model/float/sparsegpt"            # 量化结果的根目录，最终会在这里生成 compressed_{rbs}_4bits/mantissa_{MANTISSA_BITS}bit 这样的子目录

# -------------------- 量化参数 --------------------
MANTISSA_BITS=4
DATASET="wikitext2"
SKIP_MANTISSA="--skip_mantissa_quant"   # 留空则不跳过: SKIP_MANTISSA=""

# -------------------- row_block_size 列表 --------------------
# -1 表示整列共享，会命名为 compressed_full_4bits
# 正数会命名为 compressed_{rbs}_4bits
# RBS_LIST=(-1 1024 512 256 128 64 32 16)
# RBS_LIST=(-1)
RBS_LIST=(128 64 32 16)

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
    python test.py "${PRUNED_MODEL}" \
        --mantissa_bits "${MANTISSA_BITS}" \
        --row_block_size "${rbs}" \
        --save_codebooks_per_group --save_masks \
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