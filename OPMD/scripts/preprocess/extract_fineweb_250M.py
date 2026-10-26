"""
从 HuggingFaceFW/fineweb 流式提取 250M token，保存为 JSONL 格式。
与 MegaMath 数据格式一致，用于 OPMD mid-training mixture。

用法:
    conda activate opmd
    python OPMD/scripts/preprocess/extract_fineweb_250M.py
"""

import json
import os
import sys
import time
from pathlib import Path

# 使用 HF 镜像加速下载（国内网络）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from transformers import AutoTokenizer

# ============ 配置 ============
MODEL_PATH = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/qwen3-4b-base"
OUTPUT_DIR = Path("/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets")
OUTPUT_FILE = OUTPUT_DIR / "fineweb_250M.jsonl"
TARGET_TOKENS = 250_000_000  # 250M
BATCH_SIZE = 500  # 每批 tokenize 的行数
WRITE_BATCH = 1000  # 每批写入的行数
# =============================


def main():
    print("=" * 60)
    print("FineWeb → 250M Token 提取")
    print(f"目标: {TARGET_TOKENS:,} tokens")
    print(f"输出: {OUTPUT_FILE}")
    print("=" * 60)

    # 加载 tokenizer
    print("\n[1/3] 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  Vocab size: {tok.vocab_size}")

    # 流式加载数据集
    # FineWeb 的默认 split 是 "train"，sample 是 "sample-10BT" 子集
    print("\n[2/3] 流式加载数据集 (HuggingFaceFW/fineweb, sample-10BT) ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    print("  数据集加载完成，开始流式处理...")

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_rows = 0
    write_buffer = []
    start_time = time.time()
    last_report_time = start_time

    # 打开输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        text_batch = []

        for row in ds:
            text = row["text"]
            text_batch.append(text)
            total_rows += 1

            # 攒够一批就 tokenize
            if len(text_batch) >= BATCH_SIZE:
                enc = tok(text_batch, add_special_tokens=False)
                batch_tokens = sum(len(ids) for ids in enc["input_ids"])
                total_tokens += batch_tokens

                # 写入缓冲区
                for t in text_batch:
                    write_buffer.append(json.dumps({"text": t}, ensure_ascii=False))

                text_batch = []

                # 写入磁盘
                if len(write_buffer) >= WRITE_BATCH:
                    f.write("\n".join(write_buffer) + "\n")
                    f.flush()
                    write_buffer = []

                # 进度报告 (每 30 秒)
                now = time.time()
                if now - last_report_time >= 30:
                    elapsed = now - start_time
                    speed = total_tokens / elapsed if elapsed > 0 else 0
                    pct = total_tokens / TARGET_TOKENS * 100
                    eta = (TARGET_TOKENS - total_tokens) / speed if speed > 0 else 0
                    print(
                        f"  [进度] {total_tokens/1e6:.1f}M tokens "
                        f"({pct:.1f}%) | "
                        f"{total_rows} 行 | "
                        f"速度: {speed/1e6:.1f}M tok/s | "
                        f"预计剩余: {eta/60:.0f} min"
                    )
                    last_report_time = now

                # 达到目标
                if total_tokens >= TARGET_TOKENS:
                    break

        # 处理剩余的 text_batch
        if text_batch and total_tokens < TARGET_TOKENS:
            enc = tok(text_batch, add_special_tokens=False)
            batch_tokens = sum(len(ids) for ids in enc["input_ids"])
            total_tokens += batch_tokens
            for t in text_batch:
                write_buffer.append(json.dumps({"text": t}, ensure_ascii=False))

        # 写入剩余的 write_buffer
        if write_buffer:
            f.write("\n".join(write_buffer) + "\n")
            f.flush()

    # 最终统计
    elapsed = time.time() - start_time
    print("\n[3/3] 完成!")
    print(f"  总行数: {total_rows:,}")
    print(f"  总 tokens: {total_tokens:,} ({total_tokens/1e9:.3f}B)")
    print(f"  耗时: {elapsed/60:.1f} min")
    print(f"  平均速度: {total_tokens/elapsed/1e6:.1f}M tok/s")

    # 文件大小
    file_size = OUTPUT_FILE.stat().st_size
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  文件大小: {file_size/1024/1024:.1f} MB ({file_size/1024/1024/1024:.2f} GB)")

    # 验证
    print("\n验证前 3 行:")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            obj = json.loads(line)
            print(
                f"  [{i}] text 长度: {len(obj['text'])} chars, "
                f"预览: {obj['text'][:80]}..."
            )


if __name__ == "__main__":
    main()