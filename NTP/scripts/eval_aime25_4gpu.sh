#!/bin/bash
# ============================================================
# NTP 模型 AIME2025 评测 — 4 GPU 版本
# ============================================================
# 用法:
#   bash NTP/scripts/eval_aime25_4gpu.sh
#
# 注意:
#   - 平台自动注入 GPU，直接运行即可，无需 sbatch
#   - 默认使用 --no_chat_template（NTP 模型在纯文本上训练）
#   - 如需对比 chat template 评测，设置 CHAT_MODE=1
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
NTP_MODEL_PATH="${PROJECT_ROOT}/NTP/checkpoints/ntp_qwen3_4b_finemath_8192ctx_lr1e-5_bs128_steps-1_20260701_141347"
OUTPUT_DIR="${PROJECT_ROOT}/NTP/eval_results"

# 评测模式: 默认 chat_template（与 base model 评测对齐）
# 设置 CHAT_MODE=0 使用 no_chat_template（NTP 模型在纯文本上训练）
CHAT_MODE=${CHAT_MODE:-1}

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "NTP AIME2025 Evaluation — Auto GPU Detection"
echo "=========================================="
echo "Model:      $NTP_MODEL_PATH"
echo "Chat Mode:  $CHAT_MODE (0=no_chat_template, 1=chat_template+thinking)"
echo "Output:     $OUTPUT_DIR"
echo "Start:      $(date)"
echo "=========================================="

# Step 1: Generate
echo ""
echo "[Step 1/2] Generating AIME25 responses..."

if [ "$CHAT_MODE" = "1" ]; then
    python "${EVAL_DIR}/gen_vllm.py" \
        --model_path "$NTP_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --tasks "AIME25" \
        --gpu_ids auto \
        --max_tokens 31744 \
        --enable_thinking \
        --replace
else
    python "${EVAL_DIR}/gen_vllm.py" \
        --model_path "$NTP_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --tasks "AIME25" \
        --gpu_ids auto \
        --max_tokens 31744 \
        --no_chat_template \
        --replace
fi

# Step 2: Grade
echo ""
echo "[Step 2/2] Grading responses..."
python "${EVAL_DIR}/grade.py" \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="