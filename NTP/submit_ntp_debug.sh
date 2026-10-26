#!/bin/bash
# ============================================================
# NTP Mid-Training — 快速调试脚本 (单卡、无 DeepSpeed、小数据)
# ============================================================
# 三步流程:
#   1. 预处理小数据集子集 (1000 条文档)
#   2. 单卡训练 (无 DeepSpeed)
#   3. 验证 loss 正常下降
#
# 用法:
#   bash submit_ntp_debug.sh
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
PROCESSED_DATA_DIR="processed_data/debug_1k_docs"

# ============================================================
# 调试超参数
# ============================================================
MAX_SEQ_LENGTH=8192
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
MAX_STEPS=20                  # 只跑 20 步，快速验证
LEARNING_RATE=2e-5
NUM_EPOCHS=1
WARMUP_RATIO=0.03
LR_SCHEDULER=cosine
WEIGHT_DECAY=0.01
SAVE_STEPS=50
LOGGING_STEPS=1
MAX_DOCS=1000                 # 只用 1000 条文档
PREPROCESS_NUM_WORKERS=4

TOTAL_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))

# ============================================================
# 输出路径
# ============================================================
RUN_NAME="ntp_debug_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="checkpoints/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR" logs

# ============================================================
# 打印配置
# ============================================================
echo "=========================================="
echo "NTP Mid-Training — DEBUG (Fast Iteration)"
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
echo "DeepSpeed:         false (single GPU)"
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
# Step 2: 训练 (单卡，无 DeepSpeed)
# ============================================================
echo ""
echo "[Step 2/2] Starting training (single GPU, no DeepSpeed)..."

python -u train_ntp.py \
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
    --use_flash_attn true \
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

exit $EXIT_CODE