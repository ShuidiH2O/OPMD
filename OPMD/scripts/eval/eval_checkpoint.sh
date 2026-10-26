#!/bin/bash
# ============================================================
# OPMD Checkpoint Evaluation — AIME 2025
# ============================================================
# 评测 OPMD Phase 2 训练的 checkpoint（step 100/200/300/400）在 AIME25 上的表现。
#
# 流程:
#   1. 合并 FSDP 分片 → HuggingFace 格式
#   2. vLLM 生成推理结果 (16 rollouts, thinking mode)
#   3. 规则评分
#   4. 汇总各 step 对比
#
# 用法:
#   bash OPMD/scripts/eval/eval_checkpoint.sh
#
# 环境变量覆盖:
#   GPU_IDS=0,1,2,3,4,5,6,7   指定 GPU（默认自动检测）
#   STEPS="100 200 300 400"    指定要评测的 step（默认所有）
#   SKIP_MERGE=1               跳过合并步骤（已合并过时使用）
# ============================================================

set -eo pipefail

# ============================================================
# Conda 环境
# ============================================================
source /mnt/cpfs/epic-user/zhengruize-20260612/miniconda3/etc/profile.d/conda.sh
conda activate opmd

# ============================================================
# 环境变量
# ============================================================
export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export TMPDIR=/tmp
export VLLM_IPC_PATH=/tmp
export HYDRA_FULL_ERROR=1
export VLLM_USE_FLASHINFER_SAMPLER=0

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
EVAL_DIR="${OPMD_DIR}/scripts/eval"

# ============================================================
# 配置
# ============================================================
# Checkpoint 路径
CKPT_RUN_DIR="${OPMD_DIR}/checkpoints/opmd_FineMATH-4plus-OPMD_qwen3-4b-base_self-teacher_512-T_1.0-n_1-mbs_64-topk_16-only_stu-rw_teacher_p-2026-07-06_05-48-09"

# 要评测的 step
STEPS=${STEPS:-"100 200 300 400"}

# 评测任务
TASKS="AIME25"

# vLLM 生成参数
MAX_TOKENS=31744

# GPU 配置
GPU_IDS=${GPU_IDS:-"auto"}

# 输出根目录
OUTPUT_ROOT="${OPMD_DIR}/eval_results/ckpt_eval"
MERGE_ROOT="${OUTPUT_ROOT}/merged_models"

# 跳过合并
SKIP_MERGE=${SKIP_MERGE:-0}

mkdir -p "$OUTPUT_ROOT" "$MERGE_ROOT"

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
echo "OPMD Checkpoint Evaluation — AIME 2025"
echo "=========================================="
echo "Checkpoint:  ${CKPT_RUN_DIR}"
echo "Steps:       ${STEPS}"
echo "Tasks:       ${TASKS}"
echo "GPUs:        ${GPU_IDS}"
echo "Max tokens:  ${MAX_TOKENS}"
echo "Output:      ${OUTPUT_ROOT}"
echo "Start time:  $(date)"
echo "=========================================="

# ============================================================
# 汇总文件
# ============================================================
SUMMARY_FILE="${OUTPUT_ROOT}/all_steps_summary.txt"
echo "OPMD Checkpoint Evaluation Summary" > "$SUMMARY_FILE"
echo "Run: $(basename "$CKPT_RUN_DIR")" >> "$SUMMARY_FILE"
echo "Time: $(date)" >> "$SUMMARY_FILE"
echo "Baseline (untrained qwen3-4b-base): pass@1=30.0%, avg@k=7.1%" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# ============================================================
# 主循环：对每个 step 进行评测
# ============================================================
for STEP in $STEPS; do
    echo ""
    echo "############################################################"
    echo "# Step ${STEP}"
    echo "############################################################"

    CKPT_DIR="${CKPT_RUN_DIR}/global_step_${STEP}"
    ACTOR_DIR="${CKPT_DIR}/actor"

    if [ ! -d "$ACTOR_DIR" ]; then
        echo "WARNING: Checkpoint not found at ${ACTOR_DIR}, skipping."
        echo "  Step ${STEP}: CHECKPOINT NOT FOUND" >> "$SUMMARY_FILE"
        continue
    fi

    # --- Step 1: 合并 FSDP checkpoint ---
    MERGED_MODEL_PATH="${MERGE_ROOT}/step_${STEP}"

    if [ "$SKIP_MERGE" = "1" ] && [ -f "${MERGED_MODEL_PATH}/config.json" ]; then
        echo "[Step ${STEP}] Skipping merge (already merged at ${MERGED_MODEL_PATH})"
    else
        echo "[Step ${STEP}] Merging FSDP checkpoint → HuggingFace..."
        python "${OPMD_DIR}/scripts/merge_checkpoint.py" \
            --checkpoint_path "$ACTOR_DIR" \
            --output_path "$MERGED_MODEL_PATH"
        echo "[Step ${STEP}] Merge complete: ${MERGED_MODEL_PATH}"
    fi

    # --- Step 2: vLLM 生成 ---
    STEP_OUTPUT_DIR="${OUTPUT_ROOT}/step_${STEP}"
    mkdir -p "$STEP_OUTPUT_DIR"

    echo "[Step ${STEP}] Generating responses with vLLM..."
    python "${EVAL_DIR}/gen_vllm.py" \
        --model_path "$MERGED_MODEL_PATH" \
        --output_dir "$STEP_OUTPUT_DIR" \
        --tasks "$TASKS" \
        --gpu_ids "$GPU_IDS" \
        --max_tokens "$MAX_TOKENS" \
        --enable_thinking \
        --replace

    # --- Step 3: 评分 ---
    echo "[Step ${STEP}] Grading responses..."
    python "${EVAL_DIR}/grade.py" \
        --input_dir "$STEP_OUTPUT_DIR" \
        --output_dir "$STEP_OUTPUT_DIR"

    # --- 提取结果 ---
    SUMMARY_JSON="${STEP_OUTPUT_DIR}/grading_summary.json"
    if [ -f "$SUMMARY_JSON" ]; then
        echo "" >> "$SUMMARY_FILE"
        echo "--- Step ${STEP} ---" >> "$SUMMARY_FILE"
        python -c "
import json
with open('${SUMMARY_JSON}') as f:
    results = json.load(f)
for r in results:
    task = r['file'].split('/')[-1].split('_')[0].upper()
    print(f\"  {task}: pass@1={r['pass@1']:.4f} ({r['pass@1']*100:.1f}%), avg@k={r['avg@k']:.4f} ({r['avg@k']*100:.1f}%), questions={r['num_questions']}\")
" | tee -a "$SUMMARY_FILE"
    else
        echo "  Step ${STEP}: GRADING FAILED (no grading_summary.json)" >> "$SUMMARY_FILE"
    fi

    echo "[Step ${STEP}] Done."
done

# ============================================================
# 最终汇总
# ============================================================
echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Results: ${OUTPUT_ROOT}"
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "--- Final Summary ---"
cat "$SUMMARY_FILE"
echo ""
echo "End time: $(date)"
echo "=========================================="