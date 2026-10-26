#!/bin/bash
# ============================================================
# NTP Mid-Training — Phase 1 Baseline (MegaMath-Web-Pro)
# Next Token Prediction on MegaMath-Web-Pro (1B tokens)
# Model: Llama-3.2-3B
#
# 用法:
#   # 直接运行 (已在 GPU 节点上)
#   bash submit_ntp_llama_megamath.sh
#
#   # 续训
#   OUTPUT_DIR=checkpoints/<existing_dir> bash submit_ntp_llama_megamath.sh
# ============================================================
#SBATCH --job-name=ntp_llama_mm
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --gres=gpu:8
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
export TMPDIR=/tmp
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
MODEL_PATH="/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/Llama-3.2-3B"
DATA_PATH="/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_web_pro_1B.jsonl"

# 预处理数据路径 (Llama tokenizer 预处理)
PRETRAINED_DATA_PATH="${PRETRAINED_DATA_PATH:-processed_data/megamath_web_pro_1B_llama}"

# ============================================================
# 超参数 (与 FineMath 实验保持一致)
# ============================================================
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
MAX_STEPS=${MAX_STEPS:--1}          # -1 = 跑完整个 epoch
LEARNING_RATE=${LEARNING_RATE:-1e-5}
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
    RUN_NAME="ntp_llama3.2_3b_megamath_${MAX_SEQ_LENGTH}ctx_lr${LEARNING_RATE}_bs${TOTAL_BATCH_SIZE}_steps${MAX_STEPS}_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="checkpoints/${RUN_NAME}"
fi

mkdir -p "$OUTPUT_DIR" logs

# ============================================================
# 打印配置
# ============================================================
echo "=========================================="
echo "NTP Mid-Training — Phase 1 Baseline (MegaMath-Web-Pro)"
echo "Model: Llama-3.2-3B"
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
    COMMON_ARGS+=(--deepspeed "$DEEPSPEED_CONFIG")
    deepspeed train_ntp.py "${COMMON_ARGS[@]}"
else
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