# NTP Mid-Training — Phase 1 Baseline

Next Token Prediction (NTP) 训练，在 FineMath-4+ (1B tokens) 上进行 mid-training。

## 项目结构

```
NTP/
├── train_ntp.py                     # 主训练代码 (HuggingFace Trainer + DeepSpeed)
├── preprocess_data.py               # 数据预处理脚本 (tokenize + chunk)
├── submit_ntp_finemath.sh           # Qwen3-4B FineMath 训练脚本 (8卡)
├── submit_ntp_llama_finemath.sh     # Llama-3.2-3B FineMath 训练脚本 (8卡)
├── config/
│   └── ds_z2_config.json            # DeepSpeed ZeRO-2 配置
├── scripts/
│   └── check_env.sh                 # 环境检查脚本
├── checkpoints/                     # 模型 checkpoint 输出
├── processed_data/                  # 预处理后的 Arrow 数据
│   ├── finemath_4plus_1B/           #   Qwen3 tokenizer 预处理
│   └── finemath_4plus_1B_llama/     #   Llama tokenizer 预处理 (需生成)
└── logs/                            # 训练日志
```

## 环境要求

- Python 3.11+
- PyTorch 2.5+ (CUDA 12.1+)
- 1× A100/A800 80GB GPU (调试) / 8× A100/A800 80GB GPU (生产)

### 依赖安装

```bash
conda activate opmd
pip install deepspeed accelerate swanlab flash-attn --no-build-isolation
```

## 使用方法

### 1. 环境检查

```bash
bash scripts/check_env.sh
```

### 2. 提交训练任务（调试模式 — 单卡）

```bash
# 通过 Slurm 提交
sbatch submit_ntp.sh

# 或直接运行（已在 GPU 节点上）
bash submit_ntp.sh
```

### 3. 自定义训练参数

```bash
# 修改超参数通过环境变量
MAX_STEPS=100 MAX_SEQ_LENGTH=2048 bash submit_ntp.sh
```

## 当前调试配置 (单卡)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_SEQ_LENGTH` | 1024 | 最大序列长度 |
| `PER_DEVICE_BATCH_SIZE` | 1 | 每 GPU batch size |
| `GRADIENT_ACCUMULATION_STEPS` | 2 | 梯度累积步数 |
| `MAX_STEPS` | 50 | 最大训练步数 |
| `LEARNING_RATE` | 2e-5 | 学习率 |
| `NUM_EPOCHS` | 1 | 训练轮数 |
| `WARMUP_RATIO` | 0.03 | 预热比例 |
| `LOGGING_STEPS` | 1 | 日志记录间隔 |

## 生产 8 卡配置 (调试通过后切换)

```bash
# 修改 submit_ntp.sh 中的参数：
#   #SBATCH --gres=gpu:8
#   MAX_SEQ_LENGTH=8192
#   PER_DEVICE_BATCH_SIZE=2
#   GRADIENT_ACCUMULATION_STEPS=8
#   NUM_GPUS=8
#   MAX_STEPS=""  (删除，使用 --num_train_epochs 1)
#   preprocessing_num_workers=64
#   dataloader_num_workers=4
```

## 数据

- 来源: FineMath-4+
- 大小: 1B tokens, 700,702 文档
- 格式: JSONL, 每行 `{"text": "..."}`
- 路径: `../OPMD/datasets/finemath_4plus_1B.jsonl`

## 模型

| 模型 | 参数量 | 路径 |
|------|--------|------|
| qwen3-4b-base | ~4B | `../OPMD_models/qwen3-4b-base` |
| Llama-3.2-3B | ~3B | `../OPMD_models/Llama-3.2-3B` |

## Llama-3.2-3B 实验流程

### Step 1: 数据预处理（必须先做）

Llama 的 tokenizer 与 Qwen 不同，必须重新预处理数据：

```bash
cd /mnt/cpfs/epic-user/zhengruize-20260612/NTP
conda activate opmd
python preprocess_data.py \
    --model_name_or_path /mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/Llama-3.2-3B \
    --data_path /mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/finemath_4plus_1B.jsonl \
    --output_dir processed_data/finemath_4plus_1B_llama \
    --max_seq_length 8192 \
    --preprocessing_num_workers 8
```

### Step 2: 训练

```bash
bash submit_ntp_llama_finemath.sh
```

### 训练参数 (与 Qwen 实验一致)

| 参数 | 值 |
|------|-----|
| max_seq_length | 8192 |
| per_device_batch_size | 2 |
| gradient_accumulation_steps | 8 |
| total_batch_size | 128 |
| learning_rate | 1e-5 |
| lr_scheduler | cosine |
| warmup_ratio | 0.03 |
| weight_decay | 0.01 |
| epochs | 1 |
| deepspeed | ZeRO-2 |
| precision | bf16 |

### Qwen 训练参考数据

| 指标 | 值 |
|------|-----|
| 预处理 blocks | 128,607 |
| 总 tokens | ~1.05B |
| 训练步数 | 1,005 |
| 训练时间 | ~10h 22min (8×A800) |
| 每步耗时 | ~35.9s |
| 最终 loss | 1.053 |