#!/bin/bash
# ============================================================
# NTP AIME25 Quick Eval — 仅评测 AIME25 快速验证训练效果
# ============================================================
set -euo pipefail

export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true

# 修复 vLLM IPC socket 路径过长问题 (Unix socket 限制 107 字符)
export TMPDIR=/tmp
export VLLM_IPC_PATH=/tmp

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
EVAL_DIR="${PROJECT_ROOT}/OPMD/scripts/eval"
NTP_MODEL_PATH="${PROJECT_ROOT}/NTP/checkpoints/ntp_qwen3_4b_megamath_8192ctx_lr2e-5_bs96_steps-1_20260622_124935"
OUTPUT_DIR="${PROJECT_ROOT}/NTP/eval_results"
GRAPH_DIR="${PROJECT_ROOT}/NTP/graph"

GPU_IDS=${GPU_IDS:-"auto"}

mkdir -p "$OUTPUT_DIR" "$GRAPH_DIR"

echo "=========================================="
echo "NTP AIME25 Quick Eval"
echo "Model: $NTP_MODEL_PATH"
echo "GPUs:  $GPU_IDS (auto=platform allocation)"
echo "Start: $(date)"
echo "=========================================="

# Step 1: Generate
echo "[Step 1/3] Generating AIME25 responses..."
python "${EVAL_DIR}/gen_vllm.py" \
    --model_path "$NTP_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "AIME25" \
    --gpu_ids "$GPU_IDS" \
    --max_model_len 8192

# Step 2: Grade
echo "[Step 2/3] Grading..."
python "${EVAL_DIR}/grade.py" \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

# Step 3: Plot
echo "[Step 3/3] Plotting..."
python "${EVAL_DIR}/plot_results.py" \
    --summary_path "${OUTPUT_DIR}/grading_summary.json" \
    --output_path "${GRAPH_DIR}/ntp_aime25_eval.png" \
    --title "NTP Phase 1 — AIME25" \
    --model_name "qwen3-4b-base + NTP on MegaMath+FineWeb (1.25B tokens)"

echo "Done: $(date)"
echo "Results: $OUTPUT_DIR"
echo "Figure:  ${GRAPH_DIR}/ntp_aime25_eval.png"