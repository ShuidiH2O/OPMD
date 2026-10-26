#!/bin/bash
# ============================================================
# OPMD Llama-3.2-3B MegaMath-Web-Pro Checkpoint 500 — MATH-500 Evaluation
# ============================================================
# 评测 OPD 训练 step 500 的 checkpoint 在 MATH-500 上的表现。
#
# 流程:
#   1. 合并 FSDP 分片 → HuggingFace 格式
#   2. vLLM 生成推理结果 (16 rollouts, --no_chat_template)
#   3. 规则评分
#
# 用法:
#   bash OPMD/scripts/eval/eval_llama_opmd.sh
#
# 环境变量:
#   GPU_IDS=auto              指定 GPU 索引，默认 auto（从 CUDA_VISIBLE_DEVICES 自动检测）
#   SKIP_MERGE=1              跳过合并（已合并过时使用）
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

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
EVAL_DIR="${OPMD_DIR}/scripts/eval"

# ============================================================
# 配置
# ============================================================
# Checkpoint 路径
CKPT_RUN_DIR="${OPMD_DIR}/checkpoints/opmd_MegaMath-Web-Pro-OPMD_Llama-3.2-3B_Qwen2.5-Math-7B-Instruct_512-T_1.0-n_1-mbs_64-topk_16-only_stu-rw_teacher_p-2026-07-27_08-27-43"
ACTOR_DIR="${CKPT_RUN_DIR}/global_step_500/actor"

# 输出目录
OUTPUT_DIR="${OPMD_DIR}/eval_results/opmd_llama_megamath"
MERGE_DIR="${OUTPUT_DIR}/merged_model"

# GPU 配置
GPU_IDS=${GPU_IDS:-"auto"}

# 是否跳过已合并
SKIP_MERGE=${SKIP_MERGE:-0}

mkdir -p "$OUTPUT_DIR" "$MERGE_DIR"

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

# ============================================================
# 打印配置
# ============================================================
echo "=========================================="
echo "OPMD Llama-3.2-3B MegaMath — MATH-500 Evaluation"
echo "=========================================="
echo "Checkpoint:  ${ACTOR_DIR}"
echo "Tasks:       MATH-500"
echo "GPUs:        ${GPU_IDS}"
echo "Output:      ${OUTPUT_DIR}"
echo "Skip merge:  ${SKIP_MERGE}"
echo "Start time:  $(date)"
echo "=========================================="

# ============================================================
# Step 1: 合并 FSDP checkpoint → HuggingFace
# ============================================================
echo ""
echo "[Step 1/3] Merging FSDP checkpoint → HuggingFace..."

if [ "$SKIP_MERGE" = "1" ] && [ -f "${MERGE_DIR}/config.json" ]; then
    echo "Skipping merge (already merged at ${MERGE_DIR})"
else
    python "${OPMD_DIR}/scripts/merge_checkpoint.py" \
        --checkpoint_path "$ACTOR_DIR" \
        --output_path "$MERGE_DIR"
    echo "Merge complete: ${MERGE_DIR}"
fi

# ============================================================
# Step 2: vLLM 生成推理结果
# ============================================================
echo ""
echo "[Step 2/3] Generating responses with vLLM..."

# Llama-3.2-3B 是 base model，使用 --no_chat_template
python "${EVAL_DIR}/gen_vllm.py" \
    --model_path "$MERGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "MATH-500" \
    --gpu_ids "$GPU_IDS" \
    --max_tokens 4096 \
    --no_chat_template \
    --replace

# ============================================================
# Step 3: 评分
# ============================================================
echo ""
echo "[Step 3/3] Grading responses..."

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