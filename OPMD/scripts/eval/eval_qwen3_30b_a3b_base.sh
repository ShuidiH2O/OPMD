#!/bin/bash
# ============================================================
# Qwen3-30B-A3B-Base 基线模型评测 — MATH-500 + Minerva
# ============================================================
# 直接加载 HuggingFace 格式的 qwen3-30b-a3b-base（无需合并 checkpoint）。
# 模型为 MoE：128 experts / 8 experts per token / 激活约 3B。
# 用于得到 8B-OPMD 教师模型自身在这两个测试集上的成绩。
#
# 用法:
#   bash OPMD/scripts/eval/eval_qwen3_30b_a3b_base.sh
#
# 环境变量:
#   MODEL_PATH=<path>    默认为 OPMD_models/Qwen3-30B-A3B-Base
#   OUTPUT_DIR=<path>    默认为 OPMD/eval_results/qwen3_30b_a3b_base
#   N_RESPONSES=<n>      每题采样次数，默认 4（与 8b 评测一致）
#   GPU_IDS=0,1,2,3      指定 GPU 索引（有 32 heads，TP_SIZE 必须整除 32）
# ============================================================

set -euo pipefail

# ============================================================
# Conda 环境（最新评测环境）
# ============================================================
source /mnt/cpfs/epic-user/zhengruize-20260612/miniconda3/etc/profile.d/conda.sh
set +u
conda activate opmd_eval_cu12
set -u

# ============================================================
# 环境变量
# ============================================================
export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true

# /tmp 空间不足，临时目录改用 /dev/shm（1T tmpfs）
export TMPDIR=/dev/shm/zhengruize-20260612/tmp
export VLLM_IPC_PATH=/dev/shm/zhengruize-20260612/vllm_ipc
export VLLM_USE_FLASHINFER_SAMPLER=0
export TRITON_CACHE_DIR=/dev/shm/zhengruize-20260612/triton_eval
mkdir -p "$TMPDIR" "$VLLM_IPC_PATH"
rm -rf "$TRITON_CACHE_DIR"

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
EVAL_DIR="${OPMD_DIR}/scripts/eval"

# ============================================================
# 配置
# ============================================================
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/OPMD_models/Qwen3-30B-A3B-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-${OPMD_DIR}/eval_results/qwen3_30b_a3b_base}"
N_RESPONSES=${N_RESPONSES:-4}
GPU_IDS=${GPU_IDS:-"auto"}

mkdir -p "$OUTPUT_DIR"

# GPU UUID → 索引转换 (vLLM 0.23.0 兼容)
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
echo "Qwen3-30B-A3B-Base — MATH-500 + Minerva"
echo "=========================================="
echo "Model:   ${MODEL_PATH}"
echo "Tasks:   MATH-500 (4096) + Minerva (8192)"
echo "Rollout: ${N_RESPONSES} per question"
echo "GPUs:    ${GPU_IDS}"
echo "Output:  ${OUTPUT_DIR}"
echo "Start:   $(date)"
echo "=========================================="

# ============================================================
# Step 1: vLLM 生成
# ============================================================
echo ""
echo "[Step 1/2] Generating responses with vLLM..."
python "${EVAL_DIR}/gen_vllm.py" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "MATH-500,Minerva" \
    --gpu_ids "$GPU_IDS" \
    --n_responses "$N_RESPONSES" \
    --replace

# ============================================================
# Step 2: 评分
# ============================================================
echo ""
echo "[Step 2/2] Grading responses..."
python "${EVAL_DIR}/grade.py" \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

# 打印评分摘要
SUMMARY_JSON="${OUTPUT_DIR}/grading_summary.json"
if [ -f "$SUMMARY_JSON" ]; then
    echo ""
    echo "--- Grading Summary ---"
    python -c "
import json
with open('${SUMMARY_JSON}') as f:
    results = json.load(f)
for r in results:
    task = r['file'].split('/')[-1].split('_')[0].upper()
    print(f\"  {task:<10} pass@1={r['pass@1']:.4f} ({r['pass@1']*100:.1f}%)  avg@k={r['avg@k']:.4f} ({r['avg@k']*100:.1f}%)  q={r['num_questions']}\")
"
fi

echo ""
echo "Evaluation complete! Results: ${OUTPUT_DIR}  $(date)"