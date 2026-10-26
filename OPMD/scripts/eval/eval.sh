#!/bin/bash
# ============================================================
# OPMD Baseline Evaluation — Base Model (qwen3-4b-base)
# ============================================================
# 评测未训练的 base model 在 AIME25 / DAPO / MATH-500 上的正确率。
# 作为 Phase 1 的对比基线。
#
# 用法:
#   bash OPMD/scripts/eval/eval.sh
#
# 流程:
#   1. vLLM 生成推理结果
#   2. 规则评分
#   3. 可视化
# ============================================================

set -euo pipefail

# ============================================================
# 环境
# ============================================================
export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true

# 修复 vLLM IPC socket 路径过长问题 (Unix socket 限制 107 字符)
export TMPDIR=/tmp
export VLLM_IPC_PATH=/tmp

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
EVAL_DIR="${OPMD_DIR}/scripts/eval"

# ============================================================
# 配置
# ============================================================
# Base model (未训练)
BASE_MODEL_PATH="${PROJECT_ROOT}/OPMD_models/qwen3-4b-base"

# 输出目录
OUTPUT_DIR="${OPMD_DIR}/eval_results"
GRAPH_DIR="${PROJECT_ROOT}/OPMD_graph"

# GPU 配置 (评测用 8 卡并行)
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}

# 任务列表
TASKS="AIME25"

mkdir -p "$OUTPUT_DIR" "$GRAPH_DIR"

# ============================================================
# 打印配置
# ============================================================
echo "=========================================="
echo "OPMD Baseline Evaluation — Base Model"
echo "=========================================="
echo "Model:      $BASE_MODEL_PATH"
echo "Tasks:      $TASKS"
echo "GPUs:       $GPU_IDS"
echo "Output:     $OUTPUT_DIR"
echo "Graph:      $GRAPH_DIR"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================
# Step 1: Generate responses
# ============================================================
echo ""
echo "[Step 1/3] Generating responses with vLLM..."
python "${EVAL_DIR}/gen_vllm.py" \
    --model_path "$BASE_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "$TASKS" \
    --gpu_ids auto \
    --max_tokens 31744 \
    --enable_thinking \
    --replace

# ============================================================
# Step 2: Grade
# ============================================================
echo ""
echo "[Step 2/3] Grading responses..."
python "${EVAL_DIR}/grade.py" \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

# ============================================================
# Step 3: Plot
# ============================================================
echo ""
echo "[Step 3/3] Generating visualization..."
python "${EVAL_DIR}/plot_results.py" \
    --summary_path "${OUTPUT_DIR}/grading_summary.json" \
    --output_path "${GRAPH_DIR}/opmd_eval_results.png" \
    --title "OPMD Baseline — Base Model (Pre-Training)" \
    --model_name "qwen3-4b-base (no mid-training)"

# ============================================================
# Done
# ============================================================
echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results:  $OUTPUT_DIR"
echo "Figure:   ${GRAPH_DIR}/opmd_eval_results.png"
echo "End time: $(date)"
echo "=========================================="