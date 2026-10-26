#!/bin/bash
# ============================================================
# NTP Mid-Training — Phase 1 Baseline
# Next Token Prediction on MegaMath + FineWeb Mixture (1.25B tokens)
# Model: qwen3-4b-base
#
# 用法:
#   # 单卡调试 (自动跳过 DeepSpeed)
#   bash submit_ntp.sh
#
#   # 6 卡生产训练 (默认)
#   NUM_GPUS=6 bash submit_ntp.sh
#
#   # 使用预处理数据 (跳过在线 tokenize)
#   PRETRAINED_DATA_PATH=processed_data bash submit_ntp.sh
#
# 生产 6 卡参数 (推荐):
#   NUM_GPUS=6 PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8 \
#   MAX_SEQ_LENGTH=8192 MAX_STEPS=-1 bash submit_ntp.sh
# ============================================================
#SBATCH --job-name=ntp_train
# These are overridden by the platform at submit time; kept as defaults
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --gres=gpu:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -x

# ============================================================
# Logging setup (works both inside and outside SBATCH)
# ============================================================
if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

# ============================================================
# 环境变量
# ============================================================
export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1

# ============================================================
# 项目路径
# ============================================================
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ============================================================
# 模型 & 数据路径
# ============================================================
MODEL_PATH="/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/qwen3-4b-base"
DATA_PATH="/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_fineweb_mixture_1.25B.jsonl"

# 预处理数据路径 (如果设置了，则跳过在线 tokenize)
PRETRAINED_DATA_PATH="${PRETRAINED_DATA_PATH:-}"

# ============================================================
# 超参数 (可通过环境变量覆盖)
# ============================================================
# 生产 6 卡推荐值: MAX_SEQ_LENGTH=8192, PER_DEVICE_BATCH_SIZE=2, GRADIENT_ACCUMULATION_STEPS=8
# 此时 total_batch_size = 2*8*6 = 96, effective tokens/batch = 96*8192 ≈ 786K
# 1.25B tokens 需要约 1592 步跑完 1 epoch
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
MAX_STEPS=${MAX_STEPS:--1}          # -1 = 跑完整个 epoch
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_EPOCHS=${NUM_EPOCHS:-1}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
SAVE_STEPS=${SAVE_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-5}

# ============================================================
# GPU 配置 — 自动检测可用 GPU 数量和序号，不硬编码
# ============================================================
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES is empty; refusing to fall back to all physical GPUs." >&2
    exit 1
fi
# DeepSpeed 需要整数 slot，将平台注入的 UUID 转换为物理数字索引
if echo "$CUDA_VISIBLE_DEVICES" | grep -q "GPU-"; then
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
    else
        echo "[ERROR] could not resolve GPU UUIDs to indices" >&2
        exit 1
    fi
fi
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
echo "[INFO] Using $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"
TOTAL_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

# DeepSpeed 配置 (可通过环境变量强制覆盖)
if [ -n "${USE_DEEPSPEED+x}" ]; then
    if [ "$USE_DEEPSPEED" = "true" ]; then
        DEEPSPEED_CONFIG="config/ds_z2_config.json"
        echo "[INFO] DeepSpeed manually enabled"
    else
        echo "[INFO] DeepSpeed manually disabled"
    fi
elif [ "$NUM_GPUS" -eq 1 ]; then
    USE_DEEPSPEED=false
    echo "[INFO] Single GPU detected — DeepSpeed disabled (set USE_DEEPSPEED=true to override)"
else
    USE_DEEPSPEED=true
    DEEPSPEED_CONFIG="config/ds_z2_config.json"
fi

# ============================================================
# 输出路径 — 支持通过环境变量 OUTPUT_DIR 指定已有目录来续训
# ============================================================
if [ -n "$OUTPUT_DIR" ]; then
    RUN_NAME=$(basename "$OUTPUT_DIR")
    echo "[INFO] Resuming from existing output dir: $OUTPUT_DIR"
else
    RUN_NAME="ntp_qwen3_4b_megamath_${MAX_SEQ_LENGTH}ctx_lr${LEARNING_RATE}_bs${TOTAL_BATCH_SIZE}_steps${MAX_STEPS}_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="checkpoints/${RUN_NAME}"
fi

mkdir -p "$OUTPUT_DIR" logs

# ============================================================
# 打印配置
# ============================================================
echo "=========================================="
echo "NTP Mid-Training — Phase 1 Baseline"
echo "=========================================="
echo "Model:            $MODEL_PATH"
echo "Data:             $DATA_PATH"
if [ -n "$PRETRAINED_DATA_PATH" ]; then
    echo "Preprocessed:     $PRETRAINED_DATA_PATH"
fi
echo "Max Seq Length:   $MAX_SEQ_LENGTH"
echo "Per Device BS:    $PER_DEVICE_BATCH_SIZE"
echo "Grad Accum:       $GRADIENT_ACCUMULATION_STEPS"
echo "Total Batch Size: $TOTAL_BATCH_SIZE"
echo "Max Steps:        $MAX_STEPS"
echo "Learning Rate:    $LEARNING_RATE"
echo "Epochs:           $NUM_EPOCHS"
echo "Warmup Ratio:     $WARMUP_RATIO"
echo "LR Scheduler:     $LR_SCHEDULER"
echo "Weight Decay:     $WEIGHT_DECAY"
echo "Num GPUs:         $NUM_GPUS"
echo "DeepSpeed:        $USE_DEEPSPEED"
if [ "$USE_DEEPSPEED" = true ]; then
    echo "DS Config:        $DEEPSPEED_CONFIG"
fi
echo "Output Dir:       $OUTPUT_DIR"
echo "Run Name:         $RUN_NAME"
echo "Start time:       $(date)"
echo "=========================================="

# ============================================================
# 构建训练命令
# ============================================================
# 公共参数
COMMON_ARGS=(
    --do_train
    --model_name_or_path "$MODEL_PATH"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --preprocessing_num_workers 4
    --output_dir "$OUTPUT_DIR"
    --run_name "$RUN_NAME"
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --num_train_epochs "$NUM_EPOCHS"
    --max_steps "$MAX_STEPS"
    --lr_scheduler_type "$LR_SCHEDULER"
    --warmup_ratio "$WARMUP_RATIO"
    --weight_decay "$WEIGHT_DECAY"
    --save_steps "$SAVE_STEPS"
    --logging_steps "$LOGGING_STEPS"
    --save_total_limit 2
    --bf16 true
    --tf32 true
    --gradient_checkpointing true
    --dataloader_num_workers 1
    --remove_unused_columns false
    --report_to "none"
    --use_flash_attn false
    --seed 42
)

# 数据路径
if [ -n "$PRETRAINED_DATA_PATH" ]; then
    COMMON_ARGS+=(--preprocessed_data_path "$PRETRAINED_DATA_PATH")
else
    COMMON_ARGS+=(--data_path "$DATA_PATH")
fi

# 启动训练
# 注意: 不传 --num_gpus, 让 DeepSpeed 从 CUDA_VISIBLE_DEVICES 自动检测
if [ "$USE_DEEPSPEED" = true ]; then
    # 多卡 DeepSpeed 模式
    COMMON_ARGS+=(--deepspeed "$DEEPSPEED_CONFIG")
    deepspeed train_ntp.py "${COMMON_ARGS[@]}"
else
    # 单卡模式 (无 DeepSpeed)
    python -u train_ntp.py "${COMMON_ARGS[@]}"
fi

EXIT_CODE=$?

# ============================================================
# 完成
# ============================================================
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

exit $EXIT_CODE