#!/bin/bash
# ============================================================
# NTP Mid-Training — Evaluation
# ============================================================
# 评测 NTP 训练后的模型在 AIME25 / DAPO / MATH-500 上的正确率。
#
# 用法:
#   bash NTP/scripts/eval.sh
#
# 流程:
#   1. vLLM 生成推理结果 (gen_vllm.py)
#   2. 规则评分 (grade.py)
#   3. 可视化 (plot_results.py)
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
NTP_DIR="${PROJECT_ROOT}/NTP"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
EVAL_DIR="${OPMD_DIR}/scripts/eval"

# ============================================================
# 配置
# ============================================================
# NTP 训练好的模型路径
NTP_MODEL_PATH="${NTP_DIR}/checkpoints/ntp_qwen3_4b_megamath_8192ctx_lr2e-5_bs96_steps-1_20260622_124935"

# 输出目录
OUTPUT_DIR="${NTP_DIR}/eval_results"
GRAPH_DIR="${NTP_DIR}/graph"

# GPU 配置 (设为 auto 自动检测平台分配的 GPU)
GPU_IDS=${GPU_IDS:-"auto"}

# 任务列表
TASKS="AIME25,DAPO,MATH-500"

mkdir -p "$OUTPUT_DIR" "$GRAPH_DIR"

# ============================================================
# 打印配置
# ============================================================
echo "=========================================="
echo "NTP Mid-Training — Evaluation"
echo "=========================================="
echo "Model:      $NTP_MODEL_PATH"
echo "Tasks:      $TASKS"
echo "GPUs:       $GPU_IDS (auto=platform allocation)"
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
    --model_path "$NTP_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "$TASKS" \
    --gpu_ids "$GPU_IDS" \
    --max_model_len 8192

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
    --output_path "${GRAPH_DIR}/ntp_eval_results.png" \
    --title "NTP Mid-Training — Phase 1 Baseline" \
    --model_name "qwen3-4b-base + NTP on MegaMath+FineWeb (1.25B tokens)"

# ============================================================
# Done
# ============================================================
echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results:  $OUTPUT_DIR"
echo "Figure:   ${GRAPH_DIR}/ntp_eval_results.png"
echo "End time: $(date)"
echo "=========================================="