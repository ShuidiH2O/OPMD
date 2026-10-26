#!/bin/bash
# ============================================================
# Llama-3.2-3B MATH-500 Evaluation
# ============================================================
# 支持评测 baseline 和 NTP/OPD 训练后的 checkpoint。
#
# 用法:
#   # Baseline (默认)
#   bash OPMD/scripts/eval/eval_llama_baseline.sh
#
#   # NTP 训练后 checkpoint
#   MODEL_PATH=/path/to/checkpoint-1902 \
#   OUTPUT_DIR=OPMD/eval_results/ntp_llama_finemath \
#   MODEL_NAME="Llama-3.2-3B (NTP FineMath)" \
#   bash OPMD/scripts/eval/eval_llama_baseline.sh
#
# 环境变量:
#   MODEL_PATH    模型路径 (默认: baseline Llama-3.2-3B)
#   OUTPUT_DIR    输出目录 (默认: OPMD/eval_results)
#   MODEL_NAME    图表中显示的模型名 (默认: 自动从路径提取)
#   TASKS         评测任务 (默认: MATH-500)
#   SKIP_PLOT     设为 1 跳过绘图步骤
# ============================================================

set -eo pipefail

# ============================================================
# Conda 环境
# (conda deactivate 脚本会引用未绑定变量，与 set -u 冲突，此处关闭)
# ============================================================
source /mnt/cpfs/epic-user/zhengruize-20260612/miniconda3/etc/profile.d/conda.sh
set +u
conda activate opmd
set -u

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

# 禁用 flashinfer 采样，避免 gcc 版本不兼容导致的 JIT 编译失败
# (CUDA 12 不支持 gcc > 14，flashinfer JIT 编译会报 unsupported GNU version)
export VLLM_USE_FLASHINFER_SAMPLER=0

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
EVAL_DIR="${OPMD_DIR}/scripts/eval"

# ============================================================
# 配置 — 通过环境变量可覆盖
# ============================================================
# 模型路径
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/OPMD_models/Llama-3.2-3B}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-${OPMD_DIR}/eval_results}"
GRAPH_DIR="${PROJECT_ROOT}/OPMD_graph"

# 任务列表
TASKS="${TASKS:-MATH-500}"

# 图表显示名
if [ -z "${MODEL_NAME:-}" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH")
fi

# 是否跳过绘图
SKIP_PLOT="${SKIP_PLOT:-0}"

# 判断是否为 base model（无 chat template）
# 通过环境变量 USE_CHAT_TEMPLATE 控制:
#   USE_CHAT_TEMPLATE=1  使用 chat template (thinking 模式)
#   USE_CHAT_TEMPLATE=0  使用 --no_chat_template (base model, 默认)
CHAT_ARGS=""
if [ "${USE_CHAT_TEMPLATE:-0}" = "1" ]; then
    echo "[INFO] Using chat template (thinking mode)"
else
    echo "[INFO] Base model — using --no_chat_template"
    CHAT_ARGS="--no_chat_template"
fi

mkdir -p "$OUTPUT_DIR" "$GRAPH_DIR"

# ============================================================
# GPU 检测与 UUID→索引 转换
# ============================================================
# 平台通过 CUDA_VISIBLE_DEVICES (UUID 格式) 管理 GPU 分配
# vLLM 需要数字索引，这里做转换
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DETECTED_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    echo "Detected $DETECTED_GPUS GPUs from CUDA_VISIBLE_DEVICES"

    # 如果包含 UUID 格式 (GPU-xxx)，转为数字索引
    if echo "$CUDA_VISIBLE_DEVICES" | grep -q "GPU-"; then
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
        else
            echo "WARNING: Could not resolve GPU UUIDs to indices, keeping original"
        fi
    fi
else
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty; refusing to fall back to all physical GPUs." >&2
    exit 1
fi

# ============================================================
# 打印配置
# ============================================================
echo "=========================================="
echo "Llama-3.2-3B MATH-500 Evaluation"
echo "=========================================="
echo "Model:      $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "Tasks:      $TASKS"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Output:     $OUTPUT_DIR"
echo "Graph:      $GRAPH_DIR"
echo "Chat args:  ${CHAT_ARGS:---enable_thinking (default)}"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================
# Step 1: Generate responses
# ============================================================
echo ""
echo "[Step 1/3] Generating responses with vLLM..."
python "${EVAL_DIR}/gen_vllm.py" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "$TASKS" \
    --gpu_ids auto \
    --max_tokens 31744 \
    $CHAT_ARGS

# ============================================================
# Step 2: Grade
# ============================================================
echo ""
echo "[Step 2/3] Grading responses..."
python "${EVAL_DIR}/grade.py" \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

# ============================================================
# Step 3: Plot (optional)
# ============================================================
if [ "$SKIP_PLOT" != "1" ]; then
    echo ""
    echo "[Step 3/3] Generating visualization..."
    PLOT_OUTPUT="${GRAPH_DIR}/opmd_eval_${MODEL_NAME// /_}.png"
    python "${EVAL_DIR}/plot_results.py" \
        --summary_path "${OUTPUT_DIR}/grading_summary.json" \
        --output_path "$PLOT_OUTPUT" \
        --title "OPMD Eval — ${MODEL_NAME}" \
        --model_name "$MODEL_NAME"
    echo "Figure: $PLOT_OUTPUT"
fi

# ============================================================
# Done
# ============================================================
echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results:  $OUTPUT_DIR"
echo "End time: $(date)"
echo "=========================================="