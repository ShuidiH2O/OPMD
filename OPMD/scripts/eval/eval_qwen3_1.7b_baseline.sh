#!/bin/bash
# ============================================================
# Qwen3-1.7B-Base AIME25 Baseline Evaluation
# ============================================================
# 评测未训练的 Qwen3-1.7B-Base 在 AIME25 上的正确率。
#
# 用法:
#   bash OPMD/scripts/eval/eval_qwen3_1.7b_baseline.sh
# ============================================================

set -euo pipefail

# ============================================================
# Conda 环境
# ============================================================
source /mnt/cpfs/epic-user/zhengruize-20260612/miniconda3/etc/profile.d/conda.sh
set +u
conda activate opmd
set -u

# ============================================================
# 环境变量
# ============================================================
export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true

# 修复 vLLM IPC socket 路径过长问题 (Unix socket 限制 107 字符)
export TMPDIR=/tmp
export VLLM_IPC_PATH=/tmp

# 禁用 flashinfer 采样，避免 gcc 版本不兼容导致的 JIT 编译失败
export VLLM_USE_FLASHINFER_SAMPLER=0

# Triton cache 重定向到 /tmp，避免 stale cache 版本冲突
export TRITON_CACHE_DIR=/tmp/triton_cache_baseline
rm -rf "$TRITON_CACHE_DIR"

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
EVAL_DIR="${OPMD_DIR}/scripts/eval"

# ============================================================
# 配置
# ============================================================
MODEL_PATH="${PROJECT_ROOT}/OPMD_models/Qwen3-1.7B-Base"
OUTPUT_DIR="${OPMD_DIR}/eval_results/qwen3_1.7b_baseline"

mkdir -p "$OUTPUT_DIR"

# GPU UUID → 索引转换
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

# ============================================================
# 打印配置
# ============================================================
echo "=========================================="
echo "Qwen3-1.7B-Base AIME25 Baseline Evaluation"
echo "=========================================="
echo "Model:      ${MODEL_PATH}"
echo "Tasks:      AIME25"
echo "Output:     ${OUTPUT_DIR}"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================
# Step 1: vLLM 生成
# ============================================================
echo ""
echo "[Step 1/2] Generating responses with vLLM..."

# Qwen3-1.7B-Base 使用 chat template（不加 enable_thinking）
python "${EVAL_DIR}/gen_vllm.py" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "Minerva" \
    --gpu_ids auto \
    --replace

# ============================================================
# Step 2: 评分
# ============================================================
echo ""
echo "[Step 2/2] Grading responses..."

python "${EVAL_DIR}/grade.py" \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

# ============================================================
# 完成
# ============================================================
echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results:  ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=========================================="

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
    print(f\"  pass@1={r['pass@1']:.4f} ({r['pass@1']*100:.1f}%)\")
    print(f\"  avg@k={r['avg@k']:.4f} ({r['avg@k']*100:.1f}%)\")
    print(f\"  questions={r['num_questions']}\")
    print(f\"  solve_none={r['solve_none']}\")
    print(f\"  solve_all={r['solve_all']}\")
"
fi