#!/bin/bash
# ============================================================
# Coding extra benchmark eval: BigCodeBench + SantaCoderFIM
# 生成用 opmd_eval_cu12（vLLM 0.19.1，兼容当前 CUDA driver），
# 评分用 opmd（含 evalplus / bigcodebench / pyarrow / datasets）。
# ============================================================
set -euo pipefail

source /mnt/cpfs/epic-user/zhengruize-20260612/miniconda3/etc/profile.d/conda.sh

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
CKPT_ROOT="${PROJECT_ROOT}/NTP/checkpoints/ntp_qwen3_1.7b_StarCoderData_8192ctx_lr1e-5_bs64_steps-1_20260908_171721"
BASE_MODEL="${PROJECT_ROOT}/OPMD_models/Qwen3-1.7B-Base"
OUT_ROOT="${PROJECT_ROOT}/NTP/eval_results"
TASKS="BigCodeBench,SantaCoderFIM"
GPU_IDS=${GPU_IDS:-"auto"}

echo "=============================================="
echo "Coding extra eval: ${TASKS}"
echo "GPUs: ${GPU_IDS}"
echo "Start: $(date)"
echo "=============================================="

# name|model_path|out_dir
MODELS=(
  "coding_base|${BASE_MODEL}|${OUT_ROOT}/coding_base"
  "coding_step_400|${CKPT_ROOT}/checkpoint-400|${OUT_ROOT}/coding_step_400"
  "coding_step_800|${CKPT_ROOT}/checkpoint-800|${OUT_ROOT}/coding_step_800"
  "coding_step_1200|${CKPT_ROOT}/checkpoint-1200|${OUT_ROOT}/coding_step_1200"
  "coding_step_1600|${CKPT_ROOT}/checkpoint-1600|${OUT_ROOT}/coding_step_1600"
  "coding_step_1902|${CKPT_ROOT}/checkpoint-1902|${OUT_ROOT}/coding_step_1902"
)

# ---- Phase 1: generation (vLLM) ----
set +u; conda activate opmd_eval_cu12; set -u
echo "[env] generation: opmd_eval_cu12 (vLLM $(python -c 'import vllm; print(vllm.__version__)'))"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name model out <<< "$entry"
    mkdir -p "$out"
    echo ""
    echo "############## [${name}] generate ##############"
    python "${EVAL_DIR}/gen_vllm.py" \
        --model_path "$model" \
        --output_dir "$out" \
        --tasks "$TASKS" \
        --gpu_ids "$GPU_IDS" \
        --max_model_len 8192 \
        --replace
    echo "[${name}] generate done: $(date)"
done

# ---- Phase 2: grading ----
set +u; conda activate opmd; set -u
echo "[env] grading: opmd"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name model out <<< "$entry"
    echo ""
    echo "############## [${name}] grade ##############"
    python "${EVAL_DIR}/grade_code.py" \
        --input_dir "$out" \
        --output_dir "$out" \
        --only "bigcodebench,santacoderfim" \
        --summary_name grading_summary_extra.json
    echo "[${name}] grade done: $(date)"
done

echo ""
echo "=============================================="
echo "All done: $(date)"
echo "=============================================="