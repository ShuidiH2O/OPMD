"""
Generate low-entropy and random anchor variants from cached full per-token entropies.

副实验数据生成：
  - low:  每文档选取熵值最低的 bottom-5% token 作为 anchor
  - rand: 每文档从所有 token 位置均匀随机抽取 5% 作为 anchor

直接从 anchor_selection.py 写出的 entropy_cache 重筛，无需重跑代理模型。

Input:  OPMD/datasets/entropy_cache/doc_*.pt   (每份含 text + 完整逐 token entropy)
Output:
  OPMD/datasets/megamath_web_pro_1B_anchors_low_0.05.parquet
  OPMD/datasets/megamath_web_pro_1B_anchors_rand_0.05.parquet

Usage:
    python OPMD/scripts/preprocess/select_anchor_variants.py
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
CACHE_DIR = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/entropy_cache"
OUT_LOW = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_web_pro_1B_anchors_low_0.05.parquet"
OUT_RAND = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_web_pro_1B_anchors_rand_0.05.parquet"

ANCHOR_RATIO = 0.05          # 每文档选取比例（与高熵 0.05 对齐，近似）
RANDOM_SEED = 42             # 随机抽取种子（可复现）


def load_cache(cache_path: str):
    """加载单个缓存文件，返回 (doc_idx, text, entropies)。损坏/空文件返回 None。"""
    if os.path.getsize(cache_path) == 0:
        return None
    try:
        data = torch.load(cache_path, weights_only=False)
    except Exception:
        return None
    text = data.get("text", "")
    entropies = data.get("entropies")
    doc_idx = data.get("doc_idx", -1)
    if not isinstance(entropies, list) or len(entropies) == 0:
        return None
    return doc_idx, text, entropies


def select_low(entropies: list[float], k: int) -> tuple[list[int], list[float]]:
    """选取熵值最低的 k 个 token 位置（按位置升序返回）。"""
    arr = np.asarray(entropies)
    idx = np.argsort(arr, kind="stable")[:k]  # 升序 → 最低熵
    positions = sorted(int(i) for i in idx)
    values = [float(entropies[i]) for i in positions]
    return positions, values


def select_random(n_total: int, k: int, seed: int) -> list[int]:
    """从 [0, n_total) 均匀随机抽取 k 个不同位置（按位置升序返回）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=k, replace=False)
    return sorted(int(i) for i in idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--out_low", type=str, default=OUT_LOW)
    parser.add_argument("--out_rand", type=str, default=OUT_RAND)
    parser.add_argument("--anchor_ratio", type=float, default=ANCHOR_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 个 cache 文件（用于测试）")
    args = parser.parse_args()

    cache_files = sorted(glob.glob(os.path.join(args.cache_dir, "doc_*.pt")))
    if args.limit is not None:
        cache_files = cache_files[: args.limit]
    print(f"Processing {len(cache_files):,} cache files under {args.cache_dir}")

    low_rows = []
    rand_rows = []
    total_low = 0
    total_rand = 0

    for path in tqdm(cache_files, desc="Re-screening anchors"):
        loaded = load_cache(path)
        if loaded is None:
            continue
        doc_idx, text, entropies = loaded
        n_total = len(entropies)
        k = max(1, int(n_total * args.anchor_ratio))
        if k > n_total:
            k = n_total

        # low: bottom-k entropy
        low_pos, low_val = select_low(entropies, k)
        low_rows.append({
            "text": text,
            "anchor_positions": low_pos,
            "entropy_values": low_val,
        })
        total_low += len(low_pos)

        # rand: uniform random k positions（per-doc 种子 = seed + doc_idx，可复现）
        rand_pos = select_random(n_total, k, args.seed + doc_idx)
        rand_val = [float(entropies[i]) for i in rand_pos]
        rand_rows.append({
            "text": text,
            "anchor_positions": rand_pos,
            "entropy_values": rand_val,
        })
        total_rand += len(rand_pos)

    low_df = pd.DataFrame(low_rows)
    rand_df = pd.DataFrame(rand_rows)

    os.makedirs(os.path.dirname(args.out_low), exist_ok=True)
    low_df.to_parquet(args.out_low, index=False)
    rand_df.to_parquet(args.out_rand, index=False)

    print(f"\nSaved LOW  → {args.out_low}")
    print(f"  docs={len(low_df):,}, anchors={total_low:,}, mean_anchors/doc={total_low/max(len(low_df),1):.2f}")
    print(f"Saved RAND → {args.out_rand}")
    print(f"  docs={len(rand_df):,}, anchors={total_rand:,}, mean_anchors/doc={total_rand/max(len(rand_df),1):.2f}")

    # 熵值 sanity check：低熵臂每文档平均熵应明显低于随机臂
    if len(low_df) > 0:
        low_mean_ent = np.mean([np.mean(v) for v in low_df["entropy_values"]])
        rand_mean_ent = np.mean([np.mean(v) for v in rand_df["entropy_values"]])
        print(f"\nSanity: mean per-doc mean-entropy  LOW={low_mean_ent:.3f}  RAND={rand_mean_ent:.3f}")
        if low_mean_ent >= rand_mean_ent:
            print("WARNING: low-entropy mean is NOT lower than random mean, please check.")


if __name__ == "__main__":
    main()