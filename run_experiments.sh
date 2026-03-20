#!/bin/bash

# smooth 模型路径（search_alpha 的输出）
SMOOTH_MODEL="/home/LHZ/opt/code/sparsegpt/compressed/0316_smooth_alpha05"
BASE_MODEL="/home/LHZ/opt/model/opt-6.7b"
LOGDIR="/home/LHZ/opt/code/sparsegpt/compressed/0316_smooth_quant_logs"
mkdir -p ${LOGDIR}

# 检查 smooth 模型是否已生成
if [ ! -f "${SMOOTH_MODEL}/config.json" ]; then
    echo "ERROR: Smooth model not found at ${SMOOTH_MODEL}"
    echo "Wait for smooth.py --search_alpha to finish first."
    exit 1
fi

echo "Smooth model found: ${SMOOTH_MODEL}"
echo "Logs will be in: ${LOGDIR}"
echo ""

run_exp() {
    local gpu=$1
    local bs=$2
    local tag=$3
    echo "=============================================="
    echo "Running smooth + rb${bs} on GPU ${gpu} -> ${tag}"
    echo "=============================================="
    CUDA_VISIBLE_DEVICES=${gpu} python opt_shared_exp_swap.py \
        ${SMOOTH_MODEL} \
        wikitext2 \
        --base_model ${BASE_MODEL} \
        --save /home/LHZ/opt/code/sparsegpt/compressed/0316_smooth_${tag} \
        --mantissa_bits 4 \
        --skip_mantissa_quant \
        --row_block_size ${bs} \
        > ${LOGDIR}/${tag}.log 2>&1
}

# === Wave 1: GPU 2,3 各跑1个 ===
echo "========== Wave 1 =========="
run_exp 2 256  "rb256"  &
run_exp 3 512  "rb512"  &
wait
echo "========== Wave 1 done =========="

# === Wave 2 ===
echo "========== Wave 2 =========="
run_exp 2 1024 "rb1024" &
run_exp 3 -1   "full"  &
wait
echo "========== Wave 2 done =========="


echo ""
echo "========== PPL Summary (smooth model) =========="
for f in ${LOGDIR}/*.log; do
    tag=$(basename $f .log)
    ppl=$(grep "Final PPL:" $f 2>/dev/null | tail -1 | awk '{print $NF}')
    overflow=$(grep "^Overflow:" $f 2>/dev/null | tail -1)
    echo "  ${tag}: PPL=${ppl} ${overflow}"
done

echo ""
echo "========== Comparison: smooth vs non-smooth =========="
echo "  (non-smooth results from 0315 experiments)"
echo ""
printf "  %-12s %-15s %-15s\n" "Config" "Non-smooth" "Smooth"
printf "  %-12s %-15s %-15s\n" "------" "----------" "------"

for tag in full rb1024 rb512 rb256 rb128 rb64; do
    old_log="/home/LHZ/opt/code/sparsegpt/compressed/0315_logs/${tag}.log"
    new_log="${LOGDIR}/${tag}.log"
    old_ppl=$(grep "Final PPL:" ${old_log} 2>/dev/null | tail -1 | awk '{print $NF}')
    new_ppl=$(grep "Final PPL:" ${new_log} 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "  %-12s %-15s %-15s\n" "${tag}" "${old_ppl:-N/A}" "${new_ppl:-N/A}"
done

echo ""
echo "All done."