"""
从 FineMath-4+ 中流式抽样约 1B tokens，保存为 JSONL 格式。

用法:
    python scripts/sample_finemath.py \
        --target_tokens 1000000000 \
        --output_path ../OPMD/datasets/finemath_4plus_1B.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time

from datasets import load_dataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample ~1B tokens from FineMath-4+ dataset"
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=1_000_000_000,
        help="目标 token 数 (默认 1B)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="finemath_4plus_1B.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="最大文档数 (调试用，None=不限制)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (用于 shuffle)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("=" * 60)
    logger.info("FineMath-4+ Sampling")
    logger.info(f"Target tokens: {args.target_tokens / 1e9:.2f}B")
    logger.info(f"Output: {args.output_path}")
    logger.info("=" * 60)

    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 流式加载 FineMath-4+
    logger.info("Loading FineMath-4+ (streaming mode)...")
    ds = load_dataset(
        "HuggingFaceTB/finemath",
        "finemath-4plus",
        split="train",
        streaming=True,
    )

    # Shuffle 打乱顺序，避免数据分布偏差
    logger.info(f"Shuffling with seed={args.seed}...")
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    cumulative_tokens = 0
    doc_count = 0
    skipped = 0
    start_time = time.time()

    with open(args.output_path, "w", encoding="utf-8") as f:
        for sample in ds:
            text = sample.get("text", "")
            token_count = sample.get("token_count", 0)

            # 跳过空文档或 token_count 异常大的文档 (可能是噪声)
            if not text or token_count <= 0:
                skipped += 1
                continue

            # 写入 JSONL
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            cumulative_tokens += token_count
            doc_count += 1

            # 进度日志
            if doc_count % 10000 == 0:
                elapsed = time.time() - start_time
                rate = doc_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {doc_count:,} docs, "
                    f"{cumulative_tokens / 1e9:.3f}B tokens, "
                    f"{rate:.0f} docs/s"
                )

            # 达到目标 token 数
            if cumulative_tokens >= args.target_tokens:
                break

            # 调试模式：限制文档数
            if args.max_docs and doc_count >= args.max_docs:
                break

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Sampling completed!")
    logger.info(f"Documents:     {doc_count:,}")
    logger.info(f"Skipped:       {skipped:,}")
    logger.info(f"Total tokens:  {cumulative_tokens / 1e9:.3f}B")
    logger.info(f"Time elapsed:  {elapsed / 60:.1f} min")
    logger.info(f"Output file:   {args.output_path}")
    logger.info(f"File size:     {os.path.getsize(args.output_path) / 1e6:.1f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()