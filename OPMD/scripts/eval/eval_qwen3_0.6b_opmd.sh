#!/bin/bash
# ============================================================
# OPMD Qwen3-0.6B-Base MegaMath-Web-Pro (teacher=qwen3-8b-base, arm=0.05) Checkpoint Evaluation
# — MATH-500 + Minerva, all saved global_steps sweep
# ============================================================
# 评测 OPMD 0.6B 实验（0.05 arm，8B 教师蒸馏 Qwen3-0.6B-Base）
# 的多个 checkpoint 在 MATH-500 与 Minerva 上的表现。
#
# 流程（每个 step）:
#   1. 合并 FSDP 分片 (global_step_${step}/actor) → HuggingFace 格式
#   2. vLLM 生成推理结果 (默认 4 rollout，可用 N_RESPONSES 覆盖；chat template，base 模型不开启 thinking)
#   3. 规则评分 (rule-based matching)
#
# 用法:
#   # 默认：自动评测该 run 下所有已保存的 global_step
#   bash OPMD/scripts/eval/eval_qwen3_0.6b_opmd.sh
#
#   # 只跑部分 step
#   EVAL_STEPS=100 bash OPMD/scripts/eval/eval_qwen3_0.6b_opmd.sh
#
#   # 已合并过时跳过合并
#   SKIP_MERGE=1 bash OPMD/scripts/eval/eval_qwen3_0.6b_opmd.sh
#
#   # 断点续传（已生成部分 rollout 时不重新生成）
#   RESUME=1 bash OPMD/scripts/eval/eval_qwen3_0.6b_opmd.sh
#
# 环境变量:
#   CKPT_RUN_DIR=<path>         checkpoint run 根目录（含 global_step_* 子目录）
#   EVAL_STEPS=100,500,1000     要评测的 global step 列表（逗号分隔）；留空=自动发现全部
#   OUTPUT_ROOT=<path>          输出根目录
#   GPU_IDS=0,1,2,3             指定 GPU 索引（Qwen3-0.6B 有 16 heads，TP_SIZE 必须整除 16）
#   SKIP_MERGE=1                跳过合并步骤（已合并过时使用）
#   RESUME=1                    保留已有生成结果，断点续传（默认每次重新生成）
#   N_RESPONSES=16              每题采样次数（默认 16，即标准 16-rollout 评测）
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

# /tmp 空间不足（挂在 /dev/vda 上），临时目录改用 /dev/shm（1T tmpfs）
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
CKPT_RUN_DIR="${CKPT_RUN_DIR:-${OPMD_DIR}/checkpoints/opmd_MegaMath-Web-Pro-OPMD-0.05_Qwen3-0.6B-Base_qwen3-8b-base_512-T_1.0-n_1-mbs_64-topk_16-only_stu-rw_teacher_p-2026-09-21_11-06-34}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OPMD_DIR}/eval_results/opmd_qwen3_0.6b_megamath_8b_teacher}"
GPU_IDS=${GPU_IDS:-"auto"}
SKIP_MERGE=${SKIP_MERGE:-0}
RESUME=${RESUME:-0}
N_RESPONSES=${N_RESPONSES:-16}

# step 列表：优先使用显式 EVAL_STEPS，否则自动发现所有 global_step_*
if [ -n "${EVAL_STEPS:-}" ]; then
    IFS=',' read -r -a STEP_LIST <<< "$EVAL_STEPS"
else
    mapfile -t STEP_LIST < <(
        find "$CKPT_RUN_DIR" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' \
            | sed 's/global_step_//' | sort -n
    ) || true
fi

if [ "${#STEP_LIST[@]}" -eq 0 ]; then
    echo "ERROR: no global_step_* checkpoints found under ${CKPT_RUN_DIR}" >&2
    exit 1
fi

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

mkdir -p "$OUTPUT_ROOT"

# ============================================================
# 单次评测函数 (step)
# ============================================================
eval_one() {
    local step="$1"

    local actor_dir="${CKPT_RUN_DIR}/global_step_${step}/actor"
    local output_dir="${OUTPUT_ROOT}/step_${step}_math500_minerva"
    local merge_dir="${output_dir}/merged_model_step${step}"
    local summary_json="${output_dir}/grading_summary.json"

    echo ""
    echo "=========================================="
    echo "Step:  ${step}"
    echo "Actor: ${actor_dir}"
    echo "Output: ${output_dir}"
    echo "Start: $(date)"
    echo "=========================================="

    if [ ! -d "$actor_dir" ]; then
        echo "WARNING: checkpoint not found at ${actor_dir}, skipping."
        return 0
    fi

    mkdir -p "$output_dir" "$merge_dir"

    # ---- Step 1: merge ----
    echo ""
    echo "[Step 1/3] Merging FSDP checkpoint -> HuggingFace..."
    if [ "$SKIP_MERGE" = "1" ] && [ -f "${merge_dir}/config.json" ]; then
        echo "Skipping merge (already merged at ${merge_dir})"
    else
        python "${OPMD_DIR}/scripts/merge_checkpoint.py" \
            --checkpoint_path "$actor_dir" \
            --output_path "$merge_dir"
        echo "Merge complete: ${merge_dir}"
    fi

    # ---- Step 2: generate ----
    gen_args=(--replace)
    if [ "$RESUME" = "1" ]; then
        gen_args=()
    fi

    echo ""
    echo "[Step 2/3] Generating responses with vLLM (MATH-500 + Minerva)..."
    python "${EVAL_DIR}/gen_vllm.py" \
        --model_path "$merge_dir" \
        --output_dir "$output_dir" \
        --tasks "MATH-500,Minerva" \
        --gpu_ids "$GPU_IDS" \
        --n_responses "$N_RESPONSES" \
        "${gen_args[@]}"

    # ---- Step 3: grade ----
    echo ""
    echo "[Step 3/3] Grading responses..."
    python "${EVAL_DIR}/grade.py" \
        --input_dir "$output_dir" \
        --output_dir "$output_dir"

    # ---- 打印摘要 ----
    if [ -f "$summary_json" ]; then
        echo ""
        echo "--- Grading Summary (step ${step}) ---"
        python -c "
import json
with open('${summary_json}') as f:
    results = json.load(f)
for r in results:
    task = r['file'].split('/')[-1].split('_')[0].upper()
    print(f\"  {task:<10} pass@1={r['pass@1']:.4f} ({r['pass@1']*100:.1f}%)  avg@k={r['avg@k']:.4f} ({r['avg@k']*100:.1f}%)  q={r['num_questions']}\")
"
    fi
    echo "Done: ${output_dir}  $(date)"
}

# ============================================================
# 主循环：step
# ============================================================
echo "=========================================="
echo "OPMD Qwen3-0.6B-Base (teacher=qwen3-8b-base, arm=0.05) evaluation"
echo "Checkpoint run: ${CKPT_RUN_DIR}"
echo "Steps:  $(printf '%s ' "${STEP_LIST[@]}")"
echo "Tasks:  MATH-500 (max_tokens=4096) + Minerva (max_tokens=8192)"
echo "Rollout: ${N_RESPONSES} per question"
echo "GPUs:   ${GPU_IDS}"
echo "Skip merge: ${SKIP_MERGE}"
echo "Resume:     ${RESUME}"
echo "Output:     ${OUTPUT_ROOT}"
echo "Start time: $(date)"
echo "=========================================="

for step in "${STEP_LIST[@]}"; do
    eval_one "$step"
done

# ============================================================
# 汇总所有 step 的结果
# ============================================================
echo ""
echo "=========================================="
echo "Writing all-steps summary ..."
echo "=========================================="
python - "$OUTPUT_ROOT" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
rows = []
for d in sorted(glob.glob(os.path.join(root, "step_*"))):
    base = os.path.basename(d)  # e.g. step_100_math500_minerva
    parts = base.split("_")
    if len(parts) < 3 or not parts[1].isdigit():
        continue
    step = int(parts[1])
    js = os.path.join(d, "grading_summary.json")
    if not os.path.isfile(js):
        continue
    with open(js, encoding="utf-8") as f:
        results = json.load(f)
    for r in results:
        task = os.path.basename(r["file"]).split("_")[0]
        rows.append((step, task, r["num_questions"], r["pass@1"], r["avg@k"]))

rows.sort(key=lambda x: (x[0], x[1]))

csv_path = os.path.join(root, "all_steps_summary.csv")
txt_path = os.path.join(root, "all_steps_summary.txt")

with open(csv_path, "w", encoding="utf-8") as f:
    f.write("step,task,num_questions,pass@1,avg@k\n")
    for step, task, q, p, a in rows:
        f.write(f"{step},{task},{q},{p:.6f},{a:.6f}\n")

lines = ["OPMD Qwen3-0.6B-Base (teacher=qwen3-8b-base, arm=0.05) — MATH-500 + Minerva"]
for step, task, q, p, a in rows:
    lines.append(f"step {step:>4}  {task:<10}  pass@1={p*100:5.1f}%  avg@k={a*100:5.1f}%  q={q}")

summary_text = "\n".join(lines) + "\n"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(summary_text, end="")
print(f"Saved: {csv_path}")
print(f"Saved: {txt_path}")
PY

echo ""
echo "=========================================="
echo "All evaluations complete."
echo "End time: $(date)"
echo "=========================================="