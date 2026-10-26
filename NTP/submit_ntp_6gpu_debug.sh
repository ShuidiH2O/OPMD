#!/bin/bash
# ============================================================
# NTP Mid-Training — 6 卡小样本调试脚本
# ============================================================
# 目的：在正式训练前验证 6 卡分布式训练正常
#   - 验证 DeepSpeed ZeRO-2 多卡通信
#   - 验证 NCCL 无误
#   - 验证 loss 下降趋势与单卡一致
#
# 用法:
#   bash submit_ntp_6gpu_debug.sh
#
# 与单卡调试的区别:
#   - 6 卡并行，测试 DeepSpeed 分布式
#   - 稍多数据（5000 文档，50 步），验证梯度同步
# ============================================================

set -euo pipefail

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
# 路径配置
# ============================================================
MODEL_PATH="/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/qwen3-4b-base"
DATA_PATH="/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_fineweb_mixture_1.25B.jsonl"
PROCESSED_DATA_DIR="processed_data/debug_5k_docs_6gpu"

# ============================================================
# 调试超参数
# ============================================================
# 自动检测可用 GPU 数量和序号
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
MAX_SEQ_LENGTH=8192                # 与正式训练保持一致
PER_DEVICE_BATCH_SIZE=1            # 调试时每卡较小 batch
GRADIENT_ACCUMULATION_STEPS=2      # 调试时较小累积步数
MAX_STEPS=50                       # 50 步，快速验证多卡收敛
LEARNING_RATE=2e-5
NUM_EPOCHS=1
WARMUP_RATIO=0.03
LR_SCHEDULER=cosine
WEIGHT_DECAY=0.01
SAVE_STEPS=100
LOGGING_STEPS=1
MAX_DOCS=5000                      # 5000 条文档做小样本
PREPROCESS_NUM_WORKERS=4

TOTAL_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

# ============================================================
# 输出路径
# ============================================================
RUN_NAME="ntp_6gpu_debug_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="checkpoints/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR" logs

# ============================================================
# 打印配置
# ============================================================
echo "=========================================="
echo "NTP Mid-Training — 6 GPU DEBUG (Distributed Validation)"
echo "=========================================="
echo "Model:             $MODEL_PATH"
echo "Data:              $DATA_PATH"
echo "Max Docs:          $MAX_DOCS (subset)"
echo "Processed Data:    $PROCESSED_DATA_DIR"
echo "Max Seq Length:    $MAX_SEQ_LENGTH"
echo "Per Device BS:     $PER_DEVICE_BATCH_SIZE"
echo "Grad Accum:        $GRADIENT_ACCUMULATION_STEPS"
echo "Total Batch Size:  $TOTAL_BATCH_SIZE"
echo "Max Steps:         $MAX_STEPS"
echo "Learning Rate:     $LEARNING_RATE"
echo "Num GPUs:          $NUM_GPUS"
echo "DeepSpeed:         true (ZeRO-2)"
echo "Output Dir:        $OUTPUT_DIR"
echo "Start time:        $(date)"
echo "=========================================="

# ============================================================
# Step 1: 数据预处理 (仅首次运行或 forced)
# ============================================================
if [ ! -f "${PROCESSED_DATA_DIR}/dataset_info.json" ]; then
    echo ""
    echo "[Step 1/2] Preprocessing data (${MAX_DOCS} documents)..."
    python -u preprocess_data.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$PROCESSED_DATA_DIR" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --preprocessing_num_workers "$PREPROCESS_NUM_WORKERS" \
        --max_samples "$MAX_DOCS"
    echo "[Step 1/2] Done."
else
    echo ""
    echo "[Step 1/2] Skipped — preprocessed data already exists at $PROCESSED_DATA_DIR"
    echo "          Remove it to force reprocessing: rm -rf $PROCESSED_DATA_DIR"
fi

# ============================================================
# Step 2: 训练 (6 卡 DeepSpeed ZeRO-2)
# ============================================================
echo ""
echo "[Step 2/2] Starting training (${NUM_GPUS} GPUs, DeepSpeed ZeRO-2)..."

deepspeed train_ntp.py \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --preprocessed_data_path "$PROCESSED_DATA_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --preprocessing_num_workers "$PREPROCESS_NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --lr_scheduler_type "$LR_SCHEDULER" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --save_total_limit 2 \
    --bf16 true \
    --tf32 true \
    --gradient_checkpointing true \
    --dataloader_num_workers 1 \
    --remove_unused_columns false \
    --report_to "none" \
    --use_flash_attn false \
    --deepspeed config/ds_z2_config.json \
    --seed 42

EXIT_CODE=$?

# ============================================================
# 完成
# ============================================================
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 6 GPU debug passed! Ready for full training."
    echo "   Run: bash submit_ntp.sh"
else
    echo ""
    echo "❌ 6 GPU debug FAILED (exit code: $EXIT_CODE)."
    echo "   Check logs in $OUTPUT_DIR"
fi

exit $EXIT_CODE