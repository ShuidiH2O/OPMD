#!/bin/bash
# ============================================================
# NTP Qwen3-1.7B-Base MegaMath Evaluation — Minerva & MATH-500
# ============================================================
set -euo pipefail

# Conda
source /mnt/cpfs/epic-user/zhengruize-20260612/miniconda3/etc/profile.d/conda.sh
set +u; conda activate opmd_eval_cu12; set -u

# 环境变量
export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export TMPDIR=/dev/shm/zhengruize-20260612/tmp
export VLLM_IPC_PATH=/dev/shm/zhengruize-20260612/vllm_ipc
export VLLM_USE_FLASHINFER_SAMPLER=0
export TRITON_CACHE_DIR=/dev/shm/zhengruize-20260612/triton_eval
mkdir -p "$TMPDIR" "$VLLM_IPC_PATH"
rm -rf "$TRITON_CACHE_DIR"

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
EVAL_DIR="${PROJECT_ROOT}/OPMD/scripts/eval"

# NTP 重跑 checkpoint（20260905_160044），遍历多个 step
CKPT_ROOT="${PROJECT_ROOT}/NTP/checkpoints/ntp_qwen3_1.7b_megamath_8192ctx_lr1e-5_bs64_steps-1_20260905_160044"
OUTPUT_ROOT="${PROJECT_ROOT}/NTP/eval_results/ntp_qwen3_1.7b_megamath_rerun"
STEPS=${STEPS:-"400 800 1200 1600 1902"}

GPU_IDS=${GPU_IDS:-"auto"}
mkdir -p "$OUTPUT_ROOT"

# GPU UUID → 索引转换（兼容 vLLM 0.23.0）
if echo "${CUDA_VISIBLE_DEVICES:-}" | grep -q "GPU-"; then
    echo "Converting GPU UUIDs to numeric indices..."
    GPU_INDICES=""
    IFS=',' read -ra UUID_LIST <<< "$CUDA_VISIBLE_DEVICES"
    for uuid in "${UUID_LIST[@]}"; do
        uuid=$(echo "$uuid" | xargs)
        idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | grep "$uuid" | cut -d',' -f1 | xargs)
        if [ -n "$idx" ]; then
            GPU_INDICES="${GPU_INDICES}${GPU_INDICES:+,}$idx"
        fi
    done
    if [ -n "$GPU_INDICES" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
fi

echo "=========================================="
echo "NTP Qwen3-1.7B-Base Evaluation (rerun)"
echo "Checkpoint root: ${CKPT_ROOT}"
echo "Steps:   ${STEPS}"
echo "Tasks:   Minerva, MATH-500"
echo "GPUs:    ${GPU_IDS}"
echo "Results: ${OUTPUT_ROOT}"
echo "Start:   $(date)"
echo "=========================================="

for step in ${STEPS}; do
    MODEL_PATH="${CKPT_ROOT}/checkpoint-${step}"
    OUTPUT_DIR="${OUTPUT_ROOT}/step_${step}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "=================================================="
    echo "[Step ${step}] Model:  ${MODEL_PATH}"
    echo "[Step ${step}] Output: ${OUTPUT_DIR}"
    echo "[Step ${step}] Start:  $(date)"
    echo "=================================================="

    # Step 1: vLLM 生成（Minerva + MATH-500 一起跑）
    echo "[Step ${step} 1/2] Generating responses with vLLM..."
    python "${EVAL_DIR}/gen_vllm.py" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --tasks "Minerva,MATH-500" \
        --gpu_ids "$GPU_IDS" \
        --max_model_len 8192 \
        --replace

    # Step 2: 评分
    echo "[Step ${step} 2/2] Grading responses..."
    python "${EVAL_DIR}/grade.py" \
        --input_dir "$OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR"

    echo "[Step ${step}] Done: $(date)"
done

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results:  ${OUTPUT_ROOT}"
echo "End:      $(date)"
echo "=========================================="

# 打印评分摘要
for step in ${STEPS}; do
    SUMMARY_JSON="${OUTPUT_ROOT}/step_${step}/grading_summary.json"
    if [ -f "$SUMMARY_JSON" ]; then
        echo ""
        echo "--- Grading Summary (step ${step}) ---"
        python -c "
import json
with open('${SUMMARY_JSON}') as f:
    results = json.load(f)
for r in results:
    print(f\"  pass@1={r['pass@1']:.4f} ({r['pass@1']*100:.1f}%)\")
    print(f\"  avg@k={r['avg@k']:.4f} ({r['avg@k']*100:.1f}%)\")
    print(f\"  questions={r['num_questions']}\")
    print(f\"  solve_none={r['solve_none']}\")
    print(f\"  solve_all={r['solve_all']}\")
"
    fi
done