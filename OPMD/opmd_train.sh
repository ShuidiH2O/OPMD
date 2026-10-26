#!/bin/bash
# OPMD (On-Policy Mid-training Distillation) Training Script
# ===========================================================
# 基于 OPD/on_policy_distillation.sh，适配 mid-training 场景。
#
# 核心思路：
#   1. 代理模型预先选取 anchor tokens（高熵 token）
#   2. Student 从包含 anchor 的上下文生成 continuation
#   3. Teacher 评估 continuation 质量 → Reverse KL distillation
#   4. token_reward_direct + PPO loss 实现 D_KL(P_student || P_teacher) 最小化
#   5. KL loss penalty 防止策略偏离参考模型过远
#
# 用法:
#   bash OPMD/opmd_train.sh                                    # 从头训练 (默认 teacher=qwen3-14b-base, student=qwen3-1.7b-base)
#   TEACHER_MODEL=self bash OPMD/opmd_train.sh                  # self-distillation
#   ARM=low bash OPMD/opmd_train.sh                             # 低熵 anchor 臂
#   ARM=rand bash OPMD/opmd_train.sh                            # 随机 anchor 臂
#   STUDENT_MODEL=qwen3-8b-base TEACHER_MODEL=qwen3-30b-a3b-base ARM=rand \
#       bash OPMD/opmd_train.sh                                 # scaling-law: 30B-A3B 教师 → 8B 学生
#   RESUME_CKPT=<checkpoint_dir> bash OPMD/opmd_train.sh        # 断点续传
#
# 环境变量覆盖:
#   STUDENT_MODEL=  qwen3-1.7b-base | qwen3-8b-base
#   TEACHER_MODEL=  qwen3-14b-base | qwen3-8b-base | qwen3-30b-a3b-base | self | qwen3-4b-base | qwen3-8b | DeepSeek-R1-Distill-Qwen-1.5B
#   ARM=             high | low | rand    (anchor 选取方式，默认 high)
#   MAX_PROMPT_LENGTH=1024
#   MAX_RESP_LENGTH=512
#   N_RESPONSES=1
#   LOG_PROB_TOP_K=16
#   TOP_K_STRATEGY=only_stu
#   REWARD_WEIGHT_MODE=teacher_p
#   MINI_BATCH_SIZE=64
#   TEMPERATURE=1.0
#   LEARNING_RATE=1e-6
#   USE_KL=True            (KL loss penalty, 推荐开启)
#   MODEL_DTYPE=bf16
#   LOSS_AGG_MODE=token-mean
#   REWARD_MBS=16          (reward model micro batch size，30B 教师可调小防 OOM)
#   DEBUG=true             (单步 debug 模式)
#   RESUME_CKPT=<path>     (断点续传，指定已有 checkpoint 目录路径)
# ===========================================================

set -x

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT="/mnt/cpfs/epic-user/zhengruize-20260612"
OPMD_DIR="${PROJECT_ROOT}/OPMD"
OPD_DIR="${PROJECT_ROOT}/OPD"

# 添加 Python 路径
export PYTHONPATH="${OPMD_DIR}:${OPD_DIR}/verl:${PYTHONPATH}"

# ============================================================
# 日志配置
# ============================================================
if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR="${OPMD_DIR}/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/opmd_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "OPMD Training Log: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

# ============================================================
# Ray 环境
# ============================================================
ray stop --force 2>/dev/null || true
# 清理 Ray 临时目录
if [ -n "$RAY_TMPDIR" ] && [ -d "$RAY_TMPDIR" ]; then
    rm -rf "$RAY_TMPDIR" 2>/dev/null || true
fi

# 避免 socket 路径过长（AF_UNIX 限制 107 字节）
# ray 的 unix socket 需要短路径；/tmp 是共享盘且已满，改用同样短且有空间的 /var/tmp
OPMD_TMP_DIR="/var/tmp"
export TMPDIR="$OPMD_TMP_DIR"
export TEMP="$OPMD_TMP_DIR"
export TMP="$OPMD_TMP_DIR"
# RAY_TMPDIR 用短名（仅 pid），避免长用户名把路径撑过 107 字节
export RAY_TMPDIR="${OPMD_TMP_DIR}/ray_$$"
mkdir -p "$RAY_TMPDIR"

# 平台通过 CUDA_VISIBLE_DEVICES (UUID 格式) 管理 GPU 分配
# PyTorch 原生支持 UUID 格式，但 vLLM 0.23.0 需要数字索引，需要转换
# 从 CUDA_VISIBLE_DEVICES 自动检测 GPU 数量
DETECTED_GPUS=0
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    DETECTED_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    echo "Detected $DETECTED_GPUS GPUs from CUDA_VISIBLE_DEVICES"
else
    # 无 CUDA_VISIBLE_DEVICES 时，拒绝回退到整机卡，避免绕过平台授权
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty; refusing to fall back to all physical GPUs." >&2
    exit 1
fi
if [ -z "${N_GPUS:-}" ]; then
    N_GPUS=$DETECTED_GPUS
fi

# vLLM 0.23.0 不支持 UUID 格式 CUDA_VISIBLE_DEVICES，需转换为数字索引
if echo "$CUDA_VISIBLE_DEVICES" | grep -q "GPU-"; then
    echo "Converting GPU UUIDs to numeric indices for vLLM compatibility..."
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

export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export PROJECT_NAME='OPMD'
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # 禁用 flashinfer 采样，避免 CUDA 13.0 兼容性问题
# 平台把 TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR 默认指到 CPFS 共享盘，
# vLLM 的 torch.compile 在 CPFS 写缓存后 import 会报 Errno 116 Stale file handle，
# 因此统一改到本机 /var/tmp，并用 $$ 隔离并发任务。
export TRITON_CACHE_DIR="$OPMD_TMP_DIR/triton_cache_$$"
export TORCHINDUCTOR_CACHE_DIR="$OPMD_TMP_DIR/torchinductor_$$"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# ============================================================
# OPD 核心参数（保持不变）
# ============================================================
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export GRPO_OUTCOME_WEIGHT=1.0

# ============================================================
# OPMD 训练参数
# ============================================================
# --- 序列长度 ---
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}      # Anchor 上下文长度
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-512}            # CoT 生成长度
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH + 256 ))

# --- 采样参数 ---
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export N_RESPONSES=${N_RESPONSES:-1}                      # 每个 anchor 生成 1 条 CoT

# --- OPD 蒸馏参数 ---
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"teacher_p"}
export USE_KL=${USE_KL:-True}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"}

# --- NTP loss 权重 ---
export LAMBDA_NTP=${LAMBDA_NTP:-1.0}

# --- reward model 微批大小（30B-A3B 教师可按需调小防 OOM）---
export REWARD_MBS=${REWARD_MBS:-16}

# --- 模型 ---
export MODEL_DTYPE=${MODEL_DTYPE:-bf16}

# --- 学习率 ---
export LEARNING_RATE=${LEARNING_RATE:-1e-6}

# ============================================================
# 模型配置
# ============================================================
# Student Model（可通过 STUDENT_MODEL 环境变量覆盖）
# 选项: qwen3-0.6b-base | qwen3-1.7b-base | qwen3-8b-base | qwen3.5-0.8b-base
STUDENT_MODEL=${STUDENT_MODEL:-"qwen3-1.7b-base"}
case "$STUDENT_MODEL" in
    qwen3-0.6b-base)
        export ACTOR_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3-0.6B-Base"
        export ACTOR_MODEL_NAME="Qwen3-0.6B-Base"
        ;;
    qwen3-1.7b-base)
        export ACTOR_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3-1.7B-Base"
        export ACTOR_MODEL_NAME="Qwen3-1.7B-Base"
        ;;
    qwen3-8b-base)
        export ACTOR_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3-8B-Base"
        export ACTOR_MODEL_NAME="Qwen3-8B-Base"
        ;;
    qwen3.5-0.8b-base)
        export ACTOR_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3.5-0.8B-Base"
        export ACTOR_MODEL_NAME="Qwen3.5-0.8B-Base"
        ;;
    *)
        echo "ERROR: Unknown STUDENT_MODEL='${STUDENT_MODEL}' (expected qwen3-0.6b-base | qwen3-1.7b-base | qwen3-8b-base | qwen3.5-0.8b-base)"
        exit 1
        ;;
esac

# Teacher Model（可配置多种教师）
# 推荐: qwen3-14b-base (同系列更强 base model, 分布兼容)
# 选项: qwen3-14b-base | qwen3-8b-base | qwen3-30b-a3b-base | self | qwen3-4b-base | qwen3-8b | DeepSeek-R1-Distill-Qwen-1.5B
# 说明: qwen3-30b-base 是 qwen3-30b-a3b-base (MoE, 3B 激活) 的别名，指向同一目录
TEACHER_MODEL=${TEACHER_MODEL:-"qwen3-14b-base"}
case "$TEACHER_MODEL" in
    self)
        export REWARD_MODEL_PATH="${ACTOR_MODEL_PATH}"
        export REWARD_MODEL_NAME="self-teacher"
        ;;
    qwen3-4b-base)
        export REWARD_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3-4B-Base"
        export REWARD_MODEL_NAME="qwen3-4b-base"
        ;;
    qwen3-14b-base)
        export REWARD_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3-14B-Base"
        export REWARD_MODEL_NAME="qwen3-14b-base"
        ;;
    qwen3-30b-a3b-base|qwen3-30b-base)
        export REWARD_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3-30B-A3B-Base"
        export REWARD_MODEL_NAME="qwen3-30b-a3b-base"
        ;;
    qwen3-8b-base)
        export REWARD_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3-8B-Base"
        export REWARD_MODEL_NAME="qwen3-8b-base"
        ;;
    qwen3.5-9b-base)
        export REWARD_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3.5-9B-Base"
        export REWARD_MODEL_NAME="qwen3.5-9b-base"
        ;;
    qwen3-8b)
        export REWARD_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen3-8B"
        export REWARD_MODEL_NAME="qwen3-8b"
        ;;
    DeepSeek-R1-Distill-Qwen-1.5B)
        export REWARD_MODEL_PATH="${OPMD_DIR}/../OPMD_models/DeepSeek-R1-Distill-Qwen-1.5B"
        export REWARD_MODEL_NAME="DS-R1-Distill-Qwen-1.5B"
        ;;
    Llama-3.1-8B-Instruct)
        export REWARD_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Llama-3.1-8B-Instruct"
        export REWARD_MODEL_NAME="Llama-3.1-8B-Instruct"
        ;;
    Qwen2.5-Math-7B-Instruct)
        export REWARD_MODEL_PATH="${OPMD_DIR}/../OPMD_models/Qwen2.5-Math-7B-Instruct"
        export REWARD_MODEL_NAME="Qwen2.5-Math-7B-Instruct"
        ;;
    *)
        echo "Unknown TEACHER_MODEL: $TEACHER_MODEL"
        echo "Using actor model as teacher (self-teacher)"
        export REWARD_MODEL_PATH="${ACTOR_MODEL_PATH}"
        export REWARD_MODEL_NAME="self-teacher"
        ;;
esac

# ============================================================
# Debug 模式检测
# ============================================================
if [ "${DEBUG:-false}" = "true" ]; then
    echo "=== DEBUG MODE ==="
    DEBUG_ARGS="trainer.n_gpus_per_node=${N_GPUS} \
    trainer.total_epochs=1 \
    data.train_batch_size=6 \
    data.filter_overlong_prompts_workers=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.model.enable_activation_offload=True \
    reward_model.micro_batch_size_per_gpu=2"
else
    N_GPUS=${N_GPUS:-$DETECTED_GPUS}
    DEBUG_ARGS=""
fi

echo "=========================================="
echo "OPMD Training Configuration"
echo "=========================================="
echo "  GPUs:     ${N_GPUS}"
echo "  Student:  ${ACTOR_MODEL_NAME} (${ACTOR_MODEL_PATH})"
echo "  Teacher:  ${REWARD_MODEL_NAME} (${REWARD_MODEL_PATH})"
echo "  Prompt:   ${MAX_PROMPT_LENGTH} tokens"
echo "  Response: ${MAX_RESP_LENGTH} tokens"
echo "  N:        ${N_RESPONSES}"
echo "  Top-K:    ${LOG_PROB_TOP_K} (${TOP_K_STRATEGY})"
echo "  Weight:   ${REWARD_WEIGHT_MODE}"
echo "  LR:       ${LEARNING_RATE}"
echo "  KL:       ${USE_KL}"
echo "  λ_NTP:    ${LAMBDA_NTP}"
echo "  MBS:      ${MINI_BATCH_SIZE}"
echo "  Batch:    $((MINI_BATCH_SIZE * N_GPUS))"
echo "=========================================="

# ============================================================
# 数据集
# ============================================================
# 实验臂选择（anchor 选取方式），默认 high（与原脚本行为一致）。
#   high = 高熵 anchor（主实验，已跑）
#   low  = 低熵 anchor（本次要跑的副实验）
#   rand = 随机 anchor（已跑）
# 显式设置 TRAIN_DATASET / TRAIN_DATASET_NAME 可覆盖 ARM 自动推导的值。
export ARM=${ARM:-high}
case "$ARM" in
    high)
        ARM_DATASET="${OPMD_DIR}/datasets/megamath_web_pro_opmd_0.05_train.parquet"
        ARM_DATASET_NAME="MegaMath-Web-Pro-OPMD-0.05"
        ;;
    low)
        ARM_DATASET="${OPMD_DIR}/datasets/megamath_web_pro_opmd_low_0.05_train.parquet"
        ARM_DATASET_NAME="MegaMath-Web-Pro-OPMD-low-0.05"
        ;;
    rand)
        ARM_DATASET="${OPMD_DIR}/datasets/megamath_web_pro_opmd_rand_0.05_train.parquet"
        ARM_DATASET_NAME="MegaMath-Web-Pro-OPMD-rand-0.05"
        ;;
    *)
        echo "ERROR: Unknown ARM='${ARM}' (expected high|low|rand)"
        exit 1
        ;;
esac

export TRAIN_DATASET=${TRAIN_DATASET:-"$ARM_DATASET"}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-"$ARM_DATASET_NAME"}

# 检查数据文件是否存在
if [ ! -f "$TRAIN_DATASET" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATASET"
    echo "Please run first:"
    echo "  python OPMD/scripts/preprocess/anchor_selection.py"
    echo "  python OPMD/scripts/preprocess/prepare_opmd_data.py"
    exit 1
fi

# ============================================================
# Checkpoint 路径
# ============================================================
export PROJECT_PATH="${OPMD_DIR}/checkpoints"

if [ -n "${RESUME_CKPT:-}" ]; then
    # 断点续传：使用已有 checkpoint 目录
    export CKPT_PATH="${RESUME_CKPT}"
    echo "Resuming from checkpoint: ${CKPT_PATH}"
else
    # 从头训练：生成新的 checkpoint 目录
    export CKPT_PATH="${PROJECT_PATH}/opmd_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)"
fi
export EXPERIMENT_NAME=$(basename "$CKPT_PATH")

mkdir -p "$PROJECT_PATH"

# ============================================================
# KL 参数
# ============================================================
KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

# ============================================================
# GPU 内存计算
# ============================================================
PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"

# ============================================================
# 训练 (Ray 由 verl 内部 ray.init() 自动启动，避免 Raylet daemon 导致进程树断裂)
# ============================================================
cd "$OPMD_DIR"

python3 ${OPMD_DIR}/custom/train_entry.py \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.custom_cls.path="${OPMD_DIR}/custom/dataset.py" \
    data.custom_cls.name=OPMDDataset \
    data.shuffle=True \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TRAIN_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${N_GPUS})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.8} \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=${REWARD_MBS} \
    custom_reward_function.path="${OPMD_DIR}/custom/reward.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_PATH" \
    $DEBUG_ARGS "$@"

# ============================================================
# 清理
# ============================================================
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi

ray stop --force 2>/dev/null || true
# 清理临时目录（Ray / Triton / torch inductor 缓存）
for _d in "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"; do
    if [ -n "$_d" ] && [ -d "$_d" ]; then
        rm -rf "$_d" 2>/dev/null || true
    fi
done