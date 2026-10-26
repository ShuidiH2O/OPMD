"""
Prepare OPMD Training Data for verl (OPD + NTP Hybrid)
=======================================================
将 anchor_selection.py 的输出（高熵 anchor 位置）格式化为 verl 兼容的 parquet。

核心改动（v2）：
- 读取 anchor_selection.py 输出的 anchor parquet（含 anchor_positions + entropy_values）
- 以每个 anchor 位置为中心截取上下文作为 prompt
- 支持 FineMATH-4+、MegaMath-Web-Pro 等 OPD 数据源
- 可选混合 FineWeb NTP 数据（task_type="ntp"）
- 每文档产生多个 OPD 样本（不再限制 1 个）

输入:
  - anchor_selection.py 的输出 parquet（列: text, anchor_positions, entropy_values）
  - 可选：FineWeb JSONL（NTP 数据）

输出:
  - verl 兼容的训练 parquet（列: prompt, reward_model, data_source, task_type, extra_info）

用法:
    python OPMD/scripts/preprocess/prepare_opmd_data.py
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ============================================================
# 配置
# ============================================================
# 输入路径
ANCHOR_PARQUET_PATH = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_web_pro_1B_anchors.parquet"

# 输出路径
OUTPUT_PATH = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_web_pro_opmd_train.parquet"

# Tokenizer 路径
TOKENIZER_PATH = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/Qwen3-1.7B-Base"

# OPD 参数
MAX_PROMPT_LENGTH = 1024  # Anchor 上下文窗口大小（tokens）
SAMPLES_PER_ANCHOR = 1    # 每个 anchor 产生的样本数
DEDUP_WINDOW = 512        # Anchor 去重窗口大小（tokens），相邻 anchor 间距 < 此值视为冗余

# NTP 参数
NTP_CHUNK_SIZE = 1024     # FineWeb 文本 chunk 大小（tokens）
NTP_SAMPLES_PER_DOC = 3   # 每个 FineWeb 文档生成的 NTP 样本数

# NTP 数据 JSONL 路径（可选，None = 不混合 NTP 数据）
NTP_JSONL_PATH: Optional[str] = None

# 最大样本数（None = 全部）
MAX_SAMPLES: Optional[int] = None

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def deduplicate_anchors(
    anchor_positions: list[int],
    entropy_values: list[float],
    dedup_window: int = 512,
) -> tuple[list[int], list[float]]:
    """
    对同文档内的 anchor 按窗口去重，消除冗余训练样本。

    原理：相邻 anchor 的上下文窗口（1024 tokens）高度重叠，产生几乎相同的训练信号。
    将文档划分为 dedup_window 大小的非重叠窗口，每个窗口只保留熵值最高的 anchor。

    Args:
        anchor_positions: anchor token 位置列表
        entropy_values: 对应的熵值列表
        dedup_window: 去重窗口大小（tokens），默认 512（= max_prompt_length // 2）

    Returns:
        deduped_positions: 去重后的 anchor 位置
        deduped_entropies: 对应的熵值
    """
    if len(anchor_positions) <= 1:
        return anchor_positions, entropy_values

    # 按位置排序后，按窗口分组
    pairs = sorted(zip(anchor_positions, entropy_values), key=lambda x: x[0])

    window_best: dict[int, tuple[int, float]] = {}  # window_id -> (pos, entropy)
    for pos, ent in pairs:
        w = pos // dedup_window
        if w not in window_best or ent > window_best[w][1]:
            window_best[w] = (pos, ent)

    # 按位置排序输出
    deduped = sorted(window_best.values(), key=lambda x: x[0])
    deduped_positions = [p for p, _ in deduped]
    deduped_entropies = [e for _, e in deduped]

    return deduped_positions, deduped_entropies


def create_opd_samples_from_anchors(
    text: str,
    anchor_positions: list[int],
    entropy_values: list[float],
    tokenizer,
    max_prompt_length: int,
    doc_idx: int,
    data_source: str = "finemath_4plus",
    samples_per_anchor: int = 1,
    dedup_window: int = 512,
) -> list[dict]:
    """
    以 entropy-based anchor 位置为中心创建 OPD 训练样本。

    与旧版的关键区别：
    - 不再使用简单 stride 选取，而是使用 anchor_selection.py 输出的高熵位置
    - 每个 anchor 位置产生一个上下文窗口
    - 通过 dedup_window 对同文档内冗余 anchor 去重

    Args:
        text: 原始文本
        anchor_positions: 高熵 token 的位置列表（token 索引）
        entropy_values: 对应的熵值列表
        tokenizer: tokenizer
        max_prompt_length: 上下文窗口大小
        doc_idx: 文档索引（用于 extra_info）
        data_source: 数据来源标识
        samples_per_anchor: 每个 anchor 产生的样本数（保留用于未来扩展）
        dedup_window: 去重窗口大小（0 = 不去重）

    Returns:
        samples: verl 兼容的样本列表
    """
    # ---- 去重 ----
    if dedup_window > 0 and len(anchor_positions) > 1:
        anchor_positions, entropy_values = deduplicate_anchors(
            anchor_positions, entropy_values, dedup_window
        )

    samples = []
    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding["input_ids"][0]
    total_len = len(input_ids)

    if total_len < max_prompt_length:
        return samples

    for i, anchor_pos in enumerate(anchor_positions):
        # 确保 anchor 位置在有效范围内
        if anchor_pos >= total_len or anchor_pos < 0:
            continue

        # 提示仅包含 anchor 之前的上下文 x_{<t}，不包含 x_t
        # Student 将从 x_{<t} 出发自主生成，prediction 应以 anchor token 开始
        start = max(0, anchor_pos - max_prompt_length)
        end = anchor_pos  # 不包含 anchor token 本身

        # 如果 x_{<t} 不足 max_prompt_length tokens，保持 start 不变（左填充由 tokenizer 处理）
        if end - start < 1:
            continue

        context_ids = input_ids[start:end]
        context_text = tokenizer.decode(context_ids, skip_special_tokens=True)

        if len(context_text.strip()) < 20:
            continue

        # anchor 在当前窗口之后（第一个生成 token）
        anchor_offset = anchor_pos - start

        # 提取 anchor token 文本作为 ground_truth（student 应预测的第一个 token）
        anchor_token_id = input_ids[anchor_pos].item()
        anchor_token_text = tokenizer.decode([anchor_token_id], skip_special_tokens=True)

        sample = {
            "prompt": [{"role": "user", "content": context_text}],
            "reward_model": {"ground_truth": anchor_token_text, "style": "rule"},
            "data_source": data_source,
            "task_type": "opd",
            "extra_info": {
                "context_length": int(len(context_ids)),
                "doc_index": int(doc_idx),
                "anchor_position": int(anchor_pos),
                "anchor_entropy": float(entropy_values[i]) if i < len(entropy_values) else 0.0,
            },
        }
        samples.append(sample)

    return samples


def create_ntp_samples(
    text: str,
    tokenizer,
    chunk_size: int,
    samples_per_doc: int,
    doc_idx: int,
    data_source: str = "fineweb",
) -> list[dict]:
    """
    为通用文本（如 FineWeb）创建 NTP 样本。
    将文本切分为固定大小的 chunk，作为标准的 causal LM 训练数据。
    """
    samples = []
    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding["input_ids"][0]
    total_len = len(input_ids)

    if total_len < chunk_size:
        return samples

    import random
    random.seed(42 + doc_idx)

    max_start = total_len - chunk_size
    if max_start <= 0:
        return samples

    for _ in range(min(samples_per_doc, max(1, max_start // chunk_size))):
        start = random.randint(0, max_start)
        chunk_ids = input_ids[start : start + chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        if len(chunk_text.strip()) < 50:
            continue

        sample = {
            "prompt": [{"role": "user", "content": chunk_text}],
            "reward_model": {"ground_truth": "", "style": "rule"},
            "data_source": data_source,
            "task_type": "ntp",
            "extra_info": {
                "chunk_size": int(chunk_size),
                "doc_index": int(doc_idx),
                "position": int(start),
            },
        }
        samples.append(sample)

    return samples


def prepare_hybrid_samples(
    anchor_parquet_path: str,
    output_path: str,
    tokenizer_path: str,
    max_prompt_length: int = 1024,
    ntp_jsonl_path: Optional[str] = None,
    ntp_chunk_size: int = 1024,
    ntp_samples_per_doc: int = 3,
    max_samples: Optional[int] = None,
    data_source_label: str = "finemath_4plus",
    dedup_window: int = 512,
):
    """
    从 anchor parquet 生成 OPD 训练样本，可选混合 NTP 数据。

    Args:
        anchor_parquet_path: anchor_selection.py 输出的 parquet 路径
        output_path: 输出 parquet 路径
        tokenizer_path: tokenizer 路径
        max_prompt_length: OPD 上下文窗口大小
        ntp_jsonl_path: NTP 数据 JSONL 路径（None = 不混合）
        ntp_chunk_size: NTP chunk 大小
        ntp_samples_per_doc: 每文档 NTP 样本数
        max_samples: 最大样本数
        data_source_label: OPD 数据的 data_source 标签
        dedup_window: anchor 去重窗口大小（0 = 不去重）
    """
    logger.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # ============================================================
    # 1. 读取 anchor parquet
    # ============================================================
    logger.info(f"Reading anchor data from {anchor_parquet_path}...")
    if not os.path.exists(anchor_parquet_path):
        logger.error(f"Anchor parquet not found: {anchor_parquet_path}")
        logger.error("Please run anchor_selection.py first.")
        sys.exit(1)

    anchor_df = pd.read_parquet(anchor_parquet_path)
    logger.info(f"Loaded {len(anchor_df)} documents with anchors.")

    # ============================================================
    # 2. 生成 OPD 样本
    # ============================================================
    samples = []
    opd_count = 0

    for doc_idx, row in tqdm(
        anchor_df.iterrows(),
        total=len(anchor_df),
        desc="Creating OPD samples",
    ):
        text = row["text"]
        anchor_positions = row["anchor_positions"]
        entropy_values = row.get("entropy_values", [0.0] * len(anchor_positions))

        opd_samples = create_opd_samples_from_anchors(
            text=text,
            anchor_positions=anchor_positions,
            entropy_values=entropy_values,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            doc_idx=doc_idx,
            data_source=data_source_label,
            dedup_window=dedup_window,
        )
        samples.extend(opd_samples)
        opd_count += len(opd_samples)

        if max_samples and len(samples) >= max_samples:
            break

    logger.info(f"OPD samples created: {opd_count}")

    # 统计去重效果
    if dedup_window > 0:
        total_anchors_before = sum(len(row["anchor_positions"]) for _, row in anchor_df.iterrows())
        reduction = (1 - opd_count / max(total_anchors_before, 1)) * 100
        logger.info(
            f"Dedup: {total_anchors_before:,} anchors → {opd_count:,} samples "
            f"({reduction:.1f}% reduction, dedup_window={dedup_window})"
        )

    # ============================================================
    # 3. 可选：混合 NTP 数据
    # ============================================================
    ntp_count = 0
    if ntp_jsonl_path and os.path.exists(ntp_jsonl_path):
        logger.info(f"Reading NTP data from {ntp_jsonl_path}...")

        with open(ntp_jsonl_path, "r", encoding="utf-8") as f:
            for doc_idx, line in enumerate(
                tqdm(f, desc="Creating NTP samples")
            ):
                data = json.loads(line.strip())
                text = data.get("text", "")
                source = data.get("source", "fineweb")

                ntp_samples = create_ntp_samples(
                    text=text,
                    tokenizer=tokenizer,
                    chunk_size=ntp_chunk_size,
                    samples_per_doc=ntp_samples_per_doc,
                    doc_idx=doc_idx,
                    data_source=source,
                )
                samples.extend(ntp_samples)
                ntp_count += len(ntp_samples)

                if max_samples and len(samples) >= max_samples:
                    break

        logger.info(f"NTP samples created: {ntp_count}")

    # ============================================================
    # 4. 保存
    # ============================================================
    logger.info(f"Total samples: {len(samples)} (OPD: {opd_count}, NTP: {ntp_count})")

    df = pd.DataFrame(samples)

    # 将嵌套列序列化为 JSON 字符串，避免 parquet 中产生 list<struct> 等嵌套类型
    # pyarrow 对嵌套数据的 chunked array 转换未实现，会导致 ArrowNotImplementedError
    nested_cols = ["prompt", "reward_model", "extra_info"]
    for col in nested_cols:
        if col in df.columns:
            df[col] = df[col].apply(json.dumps)

    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # 验证
    logger.info("Verification of first 3 samples:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        prompt_raw = row["prompt"]
        # 反序列化 JSON 以验证内容
        prompt = json.loads(prompt_raw) if isinstance(prompt_raw, str) else prompt_raw
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            content = prompt[0].get("content", "")
            preview = content[:80].replace("\n", "\\n")
        else:
            preview = str(prompt)[:80]
        logger.info(
            f"  [{i}] task_type={row.get('task_type', '?')}, "
            f"source={row.get('data_source', '?')}, "
            f"preview: {preview}..."
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare OPMD hybrid data from anchors")
    parser.add_argument("--anchor_parquet", type=str, default=ANCHOR_PARQUET_PATH,
                        help="anchor_selection.py 输出的 parquet 路径")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--ntp_jsonl", type=str, default=NTP_JSONL_PATH,
                        help="NTP 数据 JSONL 路径（可选）")
    parser.add_argument("--ntp_chunk_size", type=int, default=NTP_CHUNK_SIZE)
    parser.add_argument("--ntp_samples_per_doc", type=int, default=NTP_SAMPLES_PER_DOC)
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--data_source_label", type=str, default="megamath_web_pro",
                        help="OPD 数据的 data_source 标签")
    parser.add_argument("--dedup_window", type=int, default=DEDUP_WINDOW,
                        help="Anchor 去重窗口大小（tokens），0=不去重")
    args = parser.parse_args()

    prepare_hybrid_samples(
        anchor_parquet_path=args.anchor_parquet,
        output_path=args.output_path,
        tokenizer_path=args.tokenizer_path,
        max_prompt_length=args.max_prompt_length,
        ntp_jsonl_path=args.ntp_jsonl,
        ntp_chunk_size=args.ntp_chunk_size,
        ntp_samples_per_doc=args.ntp_samples_per_doc,
        max_samples=args.max_samples,
        data_source_label=args.data_source_label,
        dedup_window=args.dedup_window,
    )


if __name__ == "__main__":
    main()