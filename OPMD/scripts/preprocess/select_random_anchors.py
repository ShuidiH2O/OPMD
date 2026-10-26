"""
Random anchor selection (rand arm) — 跳过代理模型熵计算。

直接从 JSONL 语料中用 student tokenizer 随机抽取 anchor token 位置，
产出 anchors parquet（text / anchor_positions / entropy_values），
供 prepare_opmd_data.py 使用。entropy_values 填 0（rand 臂不依赖熵）。

用法:
    conda activate opmd
    python OPMD/scripts/preprocess/select_random_anchors.py \
        --dataset_path OPMD/datasets/coding_1b/train.jsonl \
        --output_path OPMD/datasets/coding_1b_anchors_rand_0.05.parquet \
        --tokenizer_path OPMD_models/Qwen3-1.7B-Base \
        --anchor_ratio 0.05 --seed 42
"""

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Random anchor selection (rand arm)")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--anchor_ratio", type=float, default=0.05,
                        help="每个文档抽取的 anchor 占比（token 数比例）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批 tokenize 的文档数")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    rng = np.random.default_rng(args.seed)

    # 流式读取所有文档 text
    texts = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text") or ""
            if t.strip():
                texts.append(t)
    print(f"Loaded docs: {len(texts):,}")

    rows = []
    total_anchors = 0
    for i in tqdm(range(0, len(texts), args.batch_size), desc="select random anchors"):
        batch = texts[i : i + args.batch_size]
        enc = tok(batch, add_special_tokens=False, truncation=False)
        for j, ids in enumerate(enc["input_ids"]):
            n = len(ids)
            if n == 0:
                continue
            k = max(1, int(n * args.anchor_ratio))
            positions = sorted(int(x) for x in rng.choice(n, size=min(k, n), replace=False))
            rows.append({
                "text": batch[j],
                "anchor_positions": positions,
                "entropy_values": [0.0] * len(positions),
            })
            total_anchors += len(positions)

    df = pd.DataFrame(rows, columns=["text", "anchor_positions", "entropy_values"])
    df.to_parquet(args.output_path, index=False)
    print(f"docs={len(df):,}, total_anchors={total_anchors:,}")
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()