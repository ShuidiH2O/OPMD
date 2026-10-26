#!/bin/bash
# ============================================================
# 环境验证脚本 — 在提交训练任务前运行
# 用法: bash scripts/check_env.sh
# ============================================================

set -e

echo "=========================================="
echo "NTP 训练环境检查"
echo "=========================================="

# 1. Python
echo -n "[1] Python: "
python --version

# 2. CUDA
echo -n "[2] CUDA available: "
python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "FAILED"

# 3. GPU count
echo -n "[3] GPU count: "
python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "FAILED"

# 4. GPU info
echo "[4] GPU info:"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory/1024**3:.0f} GB)')
" 2>/dev/null || echo "  FAILED"

# 5. Required packages
echo "[5] Required packages:"
for pkg in torch transformers datasets deepspeed accelerate swanlab; do
    echo -n "  $pkg: "
    python -c "import $pkg; print('OK')" 2>/dev/null || echo "MISSING"
done

# 6. Model
echo -n "[6] Model path: "
MODEL_PATH="/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/qwen3-4b-base"
if [ -d "$MODEL_PATH" ]; then
    echo "OK ($MODEL_PATH)"
else
    echo "MISSING ($MODEL_PATH)"
fi

# 7. Data
echo -n "[7] Data path: "
DATA_PATH="/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_fineweb_mixture_1.25B.jsonl"
if [ -f "$DATA_PATH" ]; then
    SIZE=$(du -h "$DATA_PATH" | cut -f1)
    LINES=$(wc -l < "$DATA_PATH")
    echo "OK ($SIZE, $LINES lines)"
else
    echo "MISSING ($DATA_PATH)"
fi

# 8. Flash Attention
echo -n "[8] Flash Attention 2: "
python -c "from flash_attn import flash_attn_func; print('OK')" 2>/dev/null || echo "MISSING (will fallback to sdpa)"

echo "=========================================="
echo "环境检查完成"
echo "=========================================="