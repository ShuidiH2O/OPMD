"""
Next Token Prediction (NTP) Mid-Training Script
================================================
基于 HuggingFace Trainer 的因果语言模型训练，用于 Phase 1 Baseline。

用法:
    deepspeed train_ntp.py --deepspeed config/ds_z2_config.json
    python train_ntp.py  # 单卡训练

数据格式: JSONL, 每行 {"text": "..."}
"""

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


# ============================================================
# 自定义参数
# ============================================================
@dataclass
class ModelArguments:
    """模型相关参数"""

    model_name_or_path: str = field(
        metadata={"help": "预训练模型路径 (本地或 HuggingFace ID)"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任 remote code (Qwen 模型需要)"},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "是否使用 Flash Attention 2"},
    )


@dataclass
class DataArguments:
    """数据相关参数"""

    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "JSONL 数据文件路径 (与 preprocessed_data_path 二选一)"}
    )
    preprocessed_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "预处理好的 Arrow 数据集路径 (跳过在线 tokenize)"}
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "最大序列长度 (token 数)"},
    )
    preprocessing_num_workers: int = field(
        default=16,
        metadata={"help": "数据预处理并行数 (仅在线 tokenize 时使用)"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "是否覆盖预处理缓存 (仅在线 tokenize 时使用)"},
    )


# ============================================================
# 数据预处理
# ============================================================
def tokenize_function(examples, tokenizer, max_seq_length):
    """将 text 字段 tokenize 并拼接、分块。"""
    # 合并所有 text 为连续 token 流
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )

    # 拼接所有 token 序列
    all_input_ids = []
    for ids in tokenized["input_ids"]:
        all_input_ids.extend(ids)
        # 可选：文档间添加 eos_token 分隔
        all_input_ids.append(tokenizer.eos_token_id)

    # 按 max_seq_length 分块
    total_len = len(all_input_ids)
    block_size = max_seq_length
    # 丢弃最后不足一个 block 的部分
    total_len = (total_len // block_size) * block_size

    result = {
        "input_ids": [
            all_input_ids[i : i + block_size]
            for i in range(0, total_len, block_size)
        ]
    }
    # labels 在 DataCollator 中自动处理 (shifted)
    result["labels"] = [ids[:] for ids in result["input_ids"]]
    return result


def load_and_prepare_dataset(data_args, tokenizer):
    """加载并预处理数据集。支持在线 tokenize 或加载预处理的 Arrow 数据。"""
    # 如果提供了预处理数据路径，直接加载
    if data_args.preprocessed_data_path is not None:
        if not os.path.isdir(data_args.preprocessed_data_path):
            raise FileNotFoundError(
                f"Preprocessed data path not found: {data_args.preprocessed_data_path}"
            )
        logger.info(f"Loading preprocessed dataset from: {data_args.preprocessed_data_path}")
        tokenized_dataset = load_from_disk(data_args.preprocessed_data_path)
        logger.info(f"Preprocessed dataset loaded: {len(tokenized_dataset)} blocks")
        return tokenized_dataset

    # 在线 tokenize 模式
    if data_args.data_path is None:
        raise ValueError("Either --data_path or --preprocessed_data_path must be provided")

    logger.info(f"Loading dataset from: {data_args.data_path}")

    # 加载 JSONL 文件
    dataset = load_dataset(
        "json",
        data_files=data_args.data_path,
        split="train",
    )

    logger.info(f"Dataset loaded: {len(dataset)} documents")

    # Tokenize 并分块
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": data_args.max_seq_length,
        },
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Tokenizing and chunking",
    )

    logger.info(
        f"Tokenized dataset: {len(tokenized_dataset)} blocks "
        f"(max_seq_length={data_args.max_seq_length})"
    )

    return tokenized_dataset


# ============================================================
# 主函数
# ============================================================
def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 从 JSON 文件加载参数
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 设置 seed
    set_seed(training_args.seed)

    logger.info("=" * 60)
    logger.info("NTP Mid-Training — Phase 1 Baseline")
    logger.info(f"Model: {model_args.model_name_or_path}")
    if data_args.preprocessed_data_path:
        logger.info(f"Data:  {data_args.preprocessed_data_path} (preprocessed)")
    else:
        logger.info(f"Data:  {data_args.data_path}")
    logger.info(f"Max Seq Length: {data_args.max_seq_length}")
    logger.info(f"Output Dir: {training_args.output_dir}")
    logger.info(f"DeepSpeed: {training_args.deepspeed is not None}")
    logger.info("=" * 60)

    # ============================================================
    # 加载 Tokenizer & Model
    # ============================================================
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    # Qwen tokenizer 没有 pad_token，设置 pad_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    logger.info("Loading model...")
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    model_kwargs = {}
    if model_args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        **model_kwargs,
    )

    # 启用 gradient checkpointing 节省显存
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # gradient checkpointing 需要关闭 cache

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params/1e9:.2f}B")

    # ============================================================
    # 加载数据
    # ============================================================
    train_dataset = load_and_prepare_dataset(data_args, tokenizer)

    # ============================================================
    # Data Collator
    # ============================================================
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型，非 MLM
    )

    # ============================================================
    # Trainer
    # ============================================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ============================================================
    # 训练
    # ============================================================
    # 检查是否有 checkpoint 可恢复
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # 保存最终模型
    trainer.save_model()
    trainer.save_state()

    # 输出训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training completed!")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()