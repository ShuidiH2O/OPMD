#!/bin/bash
# ============================================================
# OPMD Qwen3-1.7B-Base MegaMath-Web-Pro Checkpoint Evaluation
# — 4b-teacher arm × multi-step sweep (MATH-500 + Minerva)
# ============================================================
# 评测 OPD 训练 Qwen3-1.7B-Base（teacher=qwen3-4b-base）多个 checkpoint
# 在 MATH-500 与 Minerva 上的表现。
#
# 流程（每个 (arm, step) 组合）:
#   1. 合并 FSDP 分片 → HuggingFace 格式
#   2. vLLM 生成推理结果 (16 rollouts, chat template)
#   3. 规则评分
#
# 用法:
#   # 默认：teacher arm，step 100/200/300/400/500
#   bash OPMD/scripts/eval/eval_qwen3_1.7b_4b_opmd.sh
#
#   # 只跑部分 step
#   EVAL_STEPS=100,200 bash OPMD/scripts/eval/eval_qwen3_1.7b_4b_opmd.sh
#
#   # 已合并过时跳过合并
#   SKIP_MERGE=1 bash OPMD/scripts/eval/eval_qwen3_1.7b_4b_opmd.sh
#
# 环境变量:
#   EVAL_STEPS=100,200,300,400,500  要评测的 global step 列表（逗号分隔）
#   ARMS_ENV="name|path,..."        覆盖 arm 配置
#   SKIP_MERGE=1                    跳过合并步骤
#   GPU_IDS=0,1,2,3                 指定 GPU 索引（Qwen3-1.7B 有 16 heads，TP_SIZE 必须整除 16，默认 auto）
# ============================================================

set -euo pipefail

# ============================================================
# Conda 环境
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

# /tmp 空间不足（挂在 /dev/vda 上、仅 30G 且已满），临时目录改用 /dev/shm（1T tmpfs）
export TMPDIR=/dev/shm/zhengruize-20260612/tmp
export VLLM_IPC_PATH=/dev/shm/zhengruize-20260612/vllm_ipc

# 禁用 flashinfer 采样，避免 gcc 版本不兼容导致的 JIT 编译失败
export VLLM_USE_FLASHINFER_SAMPLER=0

# Triton cache 重定向到 /dev/shm（大 tmpfs），避免 /tmp 写满
export TRITON_CACHE_DIR=/dev/shm/zhengruize-20260612/triton_eval
mkdir -p "$TMPDIR" "$VLLM_IPC_PATH"
rm -rf "$TRITON_CACHE_DIR"

PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
EVAL_DIR="${OPMD_DIR}/scripts/eval"

# ============================================================
# 配置
# ============================================================
# 要评测的 step 列表
EVAL_STEPS="${EVAL_STEPS:-100,200,300,400,500}"
IFS=',' read -r -a STEP_LIST <<< "$EVAL_STEPS"

# 输出根目录前缀（针对 4b-teacher 场景，避免和其它 teacher 混在一起）
OUTPUT_PREFIX="opmd_qwen3_1.7b_megamath_4b"

# 臂配置：每个元素为 "输出名|checkpoint 运行目录"
ARMS=(
  "teacher|${OPMD_DIR}/checkpoints/opmd_MegaMath-Web-Pro-OPMD-0.05_Qwen3-1.7B-Base_qwen3-4b-base_512-T_1.0-n_1-mbs_64-topk_16-only_stu-rw_teacher_p-2026-09-21_19-52-56"
)

# 允许通过环境变量 ARMS_ENV 覆盖（格式 "name|path,name|path,..."）
if [ -n "${ARMS_ENV:-}" ]; then
    IFS=',' read -r -a ARMS <<< "$ARMS_ENV"
fi

# GPU 配置
GPU_IDS=${GPU_IDS:-"auto"}
# 是否跳过已合并
SKIP_MERGE=${SKIP_MERGE:-0}

# GPU UUID → 索引转换 (vLLM 0.23.0 兼容)，只做一次
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
# 单次评测函数 (arm_name, ckpt_run_dir, step)
# ============================================================
eval_one() {
    local arm_name="$1"
    local ckpt_run_dir="$2"
    local step="$3"

    local actor_dir="${ckpt_run_dir}/global_step_${step}/actor"
    local output_dir="${OPMD_DIR}/eval_results/${OUTPUT_PREFIX}_${arm_name}/step_${step}_math500_minerva"
    local merge_dir="${output_dir}/merged_model_step${step}"
    local summary_json="${output_dir}/grading_summary.json"

    echo ""
    echo "=========================================="
    echo "Arm:   ${arm_name}"
    echo "Step:  ${step}"
    echo "Actor: ${actor_dir}"
    echo "Output: ${output_dir}"
    echo "Start: $(date)"
    echo "=========================================="

    mkdir -p "$output_dir"

    # ---- Step 1: merge ----
    echo ""
    echo "[Step 1/3] Merging FSDP checkpoint → HuggingFace..."
    if [ "$SKIP_MERGE" = "1" ] && [ -f "${merge_dir}/config.json" ]; then
        echo "Skipping merge (already merged at ${merge_dir})"
    else
        python "${OPMD_DIR}/scripts/merge_checkpoint.py" \
            --checkpoint_path "$actor_dir" \
            --output_path "$merge_dir"
        echo "Merge complete: ${merge_dir}"
    fi

    # ---- Step 2: generate ----
    echo ""
    echo "[Step 2/3] Generating responses with vLLM (MATH-500 + Minerva)..."
    python "${EVAL_DIR}/gen_vllm.py" \
        --model_path "$merge_dir" \
        --output_dir "$output_dir" \
        --tasks "MATH-500,Minerva" \
        --gpu_ids "$GPU_IDS" \
        --replace

    # ---- Step 3: grade ----
    echo ""
    echo "[Step 3/3] Grading responses..."
    python "${EVAL_DIR}/grade.py" \
        --input_dir "$output_dir" \
        --output_dir "$output_dir"

    # ---- 打印摘要 ----
    if [ -f "$summary_json" ]; then
        echo ""
        echo "--- Grading Summary (${arm_name} / step ${step}) ---"
        python -c "
import json
with open('${summary_json}') as f:
    results = json.load(f)
for r in results:
    task = r['file'].split('/')[-1].split('_')[0].upper()
    print(f\"  {task}: pass@1={r['pass@1']*100:.1f}%  avg@k={r['avg@k']*100:.1f}%  q={r['num_questions']}\")
"
    fi
    echo "Done: ${output_dir}  $(date)"
}

# ============================================================
# 主循环：臂 × step
# ============================================================
ARM_NAMES=$(printf '%s ' "${ARMS[@]}" | sed 's/|[^ ]*//g')
echo "=========================================="
echo "OPMD Qwen3-1.7B-Base MegaMath — 4b-teacher sweep evaluation"
echo "Arms:   ${ARM_NAMES}"
echo "Steps:  ${EVAL_STEPS}"
echo "Tasks:  MATH-500 (max_tokens=4096) + Minerva (max_tokens=8192)"
echo "GPUs:   ${GPU_IDS}"
echo "Skip merge: ${SKIP_MERGE}"
echo "Start time: $(date)"
echo "=========================================="

for arm_spec in "${ARMS[@]}"; do
    arm_name="${arm_spec%%|*}"
    ckpt_run_dir="${arm_spec#*|}"
    for step in "${STEP_LIST[@]}"; do
        eval_one "$arm_name" "$ckpt_run_dir" "$step"
    done
done

# ============================================================
# 最终汇总（跨 step）
# ============================================================
echo ""
echo "=========================================="
echo "All evaluations complete. Final summary:"
echo "=========================================="
for arm_spec in "${ARMS[@]}"; do
    arm_name="${arm_spec%%|*}"
    for step in "${STEP_LIST[@]}"; do
        sjson="${OPMD_DIR}/eval_results/${OUTPUT_PREFIX}_${arm_name}/step_${step}_math500_minerva/grading_summary.json"
        if [ -f "$sjson" ]; then
            echo ""
            echo "--- ${arm_name} / step ${step} ---"
            python - "$sjson" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])):
    task = r["file"].split("/")[-1].split("_")[0].upper()
    print(f"  {task}: pass@1={r['pass@1']*100:.1f}%  avg@k={r['avg@k']*100:.1f}%  q={r['num_questions']}")
PY
        fi
    done
done
echo ""
echo "End time: $(date)"
echo "=========================================="