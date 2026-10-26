#!/bin/bash
# ============================================================
# Coding OPMD checkpoint 评测 — HumanEval+ / MBPP+ / LiveCodeBench
# ============================================================
# 评测 coding 语料 OPMD 训练的 Qwen3-1.7B-Base checkpoint 在代码基准上的表现。
#
# 流程（MODE=checkpoint，默认）:
#   1. 合并 FSDP 分片 (global_step_${EVAL_STEP}/actor) → HuggingFace
#   2. vLLM 生成（coding 任务为 raw 补全 + greedy + n=1）
#   3. 代码执行评分（grade_code.py）
#
# 对比 base 模型（MODE=base）:
#   直接加载 Qwen3-1.7B-Base，跳过合并。
#
# 用法:
#   # 评测 100 step checkpoint
#   bash OPMD/scripts/eval/eval_qwen3_1.7b_coding_opmd.sh
#
#   # 评测 base 模型（用于对比）
#   MODE=base bash OPMD/scripts/eval/eval_qwen3_1.7b_coding_opmd.sh
#
# 环境变量:
#   CKPT_RUN_DIR=<path>   checkpoint run 根目录（MODE=checkpoint 时使用）
#   EVAL_STEP=100         要评测的 global step
#   MODE=checkpoint|base  评测模式
#   OUTPUT_ROOT=<path>    输出根目录
#   TASKS=...             默认 HumanEvalPlus,MBPPPlus,LiveCodeBench
#   GPU_IDS=0,1,2,3       生成阶段 GPU（Qwen3-1.7B 有 16 heads）
# ============================================================

set -euo pipefail

source /mnt/cpfs/epic-user/zhengruize-20260612/miniconda3/etc/profile.d/conda.sh

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
EVAL_DIR="${OPMD_DIR}/scripts/eval"

CKPT_RUN_DIR="${CKPT_RUN_DIR:-${OPMD_DIR}/checkpoints/opmd_StarCoderData-OPMD-rand-0.05_Qwen3-1.7B-Base_qwen3-14b-base_512-T_1.0-n_1-mbs_64-topk_16-only_stu-rw_teacher_p-2026-09-12_12-15-28}"
EVAL_STEP="${EVAL_STEP:-100}"
MODE="${MODE:-checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OPMD_DIR}/eval_results/coding_opmd_qwen3_1.7b}"
TASKS="${TASKS:-HumanEvalPlus,MBPPPlus,LiveCodeBench}"
GPU_IDS="${GPU_IDS:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# ============================================================
# 环境变量
# ============================================================
export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export TMPDIR=/dev/shm/zhengruize-20260612/tmp
export VLLM_IPC_PATH=/dev/shm/zhengruize-20260612/vllm_ipc
export VLLM_USE_FLASHINFER_SAMPLER=0
export TRITON_CACHE_DIR=/dev/shm/zhengruize-20260612/triton_eval
mkdir -p "$TMPDIR" "$VLLM_IPC_PATH"
rm -rf "$TRITON_CACHE_DIR"

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
# 确定 model 路径与输出目录
# ============================================================
if [ "$MODE" = "base" ]; then
    MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/OPMD_models/Qwen3-1.7B-Base}"
    OUTPUT_DIR="${OUTPUT_ROOT}/coding_base"
elif [ "$MODE" = "checkpoint" ]; then
    ACTOR_DIR="${CKPT_RUN_DIR}/global_step_${EVAL_STEP}/actor"
    MERGE_DIR="${OUTPUT_ROOT}/merged_model_step${EVAL_STEP}"
    OUTPUT_DIR="${OUTPUT_ROOT}/step_${EVAL_STEP}_coding"
    if [ ! -d "$ACTOR_DIR" ]; then
        echo "ERROR: checkpoint not found at ${ACTOR_DIR}" >&2
        exit 1
    fi
    MODEL_PATH="$MERGE_DIR"
else
    echo "ERROR: unknown MODE=${MODE} (expected checkpoint|base)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_DIR"

echo "=========================================="
echo "Coding OPMD evaluation"
echo "Mode:     ${MODE}"
if [ "$MODE" = "checkpoint" ]; then
    echo "Checkpoint step: ${EVAL_STEP}"
fi
echo "Model:    ${MODEL_PATH}"
echo "Tasks:    ${TASKS}"
echo "GPUs:     ${GPU_IDS}"
echo "Output:   ${OUTPUT_DIR}"
echo "Start:    $(date)"
echo "=========================================="

# ============================================================
# 1. 合并（仅 checkpoint 模式）
# ============================================================
MERGE_DONE=0
if [ "$MODE" = "checkpoint" ]; then
    echo ""
    echo "[Step 1/3] Merging FSDP checkpoint -> HuggingFace ..."
    if [ -f "${MERGE_DIR}/config.json" ]; then
        echo "Skipping merge (already at ${MERGE_DIR})"
    else
        set +u
        conda activate opmd_eval_cu12
        set -u
        python "${OPMD_DIR}/scripts/merge_checkpoint.py" \
            --checkpoint_path "$ACTOR_DIR" \
            --output_path "$MERGE_DIR"
    fi
    MERGE_DONE=1
fi

# ============================================================
# 2. 生成（coding: raw + greedy + n=1）
# ============================================================
echo ""
echo "[Step 2/3] Generating with vLLM ..."
set +u
conda activate opmd_eval_cu12
set -u
python "${EVAL_DIR}/gen_vllm.py" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "$TASKS" \
    --gpu_ids "$GPU_IDS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --replace

# ============================================================
# 3. 评分（需要 opmd 环境：evalplus / bigcodebench 依赖）
# ============================================================
echo ""
echo "[Step 3/3] Grading ..."
set +u
conda activate opmd
set -u
python "${EVAL_DIR}/grade_code.py" \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

# ============================================================
# 打印摘要
# ============================================================
SUMMARY_JSON="${OUTPUT_DIR}/grading_summary.json"
if [ -f "$SUMMARY_JSON" ]; then
    echo ""
    echo "--- Grading Summary ---"
    python -c "
import json
with open('${SUMMARY_JSON}') as f:
    results = json.load(f)
for r in results:
    name = r['file'].split('/')[-1].split('_')[0]
    print(f\"  {name:<16} pass@1={r['pass@1']:.4f} ({r['pass@1']*100:.1f}%)  q={r['num_questions']}\")
"
fi

echo ""
echo "Done: ${OUTPUT_DIR}  $(date)"