"""
数据预处理脚本：将 JSONL 文本数据 tokenize 并分块，保存为 Arrow 格式。
============================================================
在训练前运行此脚本，避免训练时 GPU 空闲等待数据预处理。

用法:
    python preprocess_data.py \
        --model_name_or_path /path/to/model \
        --data_path /path/to/data.jsonl \
        --output_dir /path/to/processed \
        --max_seq_length 1024
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import transformers
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL text data for causal LM training"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Tokenizer 路径 (本地或 HuggingFace ID)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="JSONL 数据文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录 (保存 Arrow 格式的预处理数据)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help="最大序列长度 (token 数)",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="数据预处理并行数",
    )
    parser.add_argument(
        "--map_batch_size",
        type=int,
        default=1000,
        help="map 的 batch size (文档数)",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=True,
        help="是否信任 remote code",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="覆盖已有预处理数据",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大文档数 (调试用，None=全部)",
    )
    return parser.parse_args()


def tokenize_and_chunk(examples, tokenizer, max_seq_length):
    """
    将一批文档 tokenize 并拼接成连续 token 流，再按 max_seq_length 分块。
    与 train_ntp.py 中的 tokenize_function 逻辑保持一致。
    """
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )

    # 拼接所有文档的 token 序列，文档间用 eos_token 分隔
    all_input_ids = []
    for ids in tokenized["input_ids"]:
        all_input_ids.extend(ids)
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
    # labels 与 input_ids 相同，DataCollator 会做 shift
    result["labels"] = [ids[:] for ids in result["input_ids"]]
    return result


def main():
    args = parse_args()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_info()

    logger.info("=" * 60)
    logger.info("Data Preprocessing for NTP Mid-Training")
    logger.info(f"Model:  {args.model_name_or_path}")
    logger.info(f"Data:   {args.data_path}")
    logger.info(f"Max Seq Length: {args.max_seq_length}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查是否已有预处理数据
    processed_path = os.path.join(args.output_dir, "dataset.arrow")
    if os.path.exists(processed_path) and not args.overwrite_cache:
        logger.info(f"Preprocessed data already exists at {processed_path}")
        logger.info("Use --overwrite_cache to force reprocessing.")
        return

    # 加载 tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # 加载数据
    logger.info(f"Loading dataset from: {args.data_path}")
    dataset = load_dataset(
        "json",
        data_files=args.data_path,
        split="train",
    )
    logger.info(f"Dataset loaded: {len(dataset)} documents")

    # 取子集 (调试用)
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        logger.info(f"Using subset: {len(dataset)} documents")

    # Tokenize 并分块
    logger.info("Tokenizing and chunking...")
    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": args.max_seq_length,
        },
        load_from_cache_file=False,
        desc="Tokenizing and chunking",
    )

    logger.info(
        f"Tokenized dataset: {len(tokenized_dataset)} blocks "
        f"(max_seq_length={args.max_seq_length})"
    )

    # 保存为 Arrow 格式
    save_path = os.path.join(args.output_dir, "dataset.arrow")
    logger.info(f"Saving preprocessed dataset to {save_path}...")
    tokenized_dataset.save_to_disk(args.output_dir)

    # 也保存一些元数据
    meta_path = os.path.join(args.output_dir, "preprocess_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"model_name_or_path: {args.model_name_or_path}\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"max_seq_length: {args.max_seq_length}\n")
        f.write(f"num_blocks: {len(tokenized_dataset)}\n")
        f.write(f"num_documents: {len(dataset)}\n")
        f.write(f"vocab_size: {tokenizer.vocab_size}\n")

    logger.info("Preprocessing completed!")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Blocks: {len(tokenized_dataset)}")
    logger.info(f"Estimated tokens: {len(tokenized_dataset) * args.max_seq_length}")


if __name__ == "__main__":
    main()