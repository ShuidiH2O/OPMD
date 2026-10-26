"""
合并 MegaMath 和 FineWeb 数据集为 OPMD mid-training mixture。
比例: 1B MegaMath + 250M FineWeb = 1.25B tokens

用法:
    python OPMD/scripts/preprocess/merge_mixture.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============ 配置 ============
DATASET_DIR = Path("/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets")
MEGAMATH_FILE = DATASET_DIR / "megamath_web_pro_1B.jsonl"
FINEWEB_FILE = DATASET_DIR / "fineweb_250M.jsonl"
OUTPUT_FILE = DATASET_DIR / "megamath_fineweb_mixture_1.25B.jsonl"
# =============================


def merge_datasets(
    megamath_path: Path,
    fineweb_path: Path,
    output_path: Path,
    shuffle: bool = True,
):
    """
    合并两个 JSONL 数据集，可选 shuffle。
    """
    # 检查输入文件
    for path, name in [(megamath_path, "MegaMath"), (fineweb_path, "FineWeb")]:
        if not path.exists():
            logger.error(f"{name} 文件不存在: {path}")
            sys.exit(1)
        size_mb = path.stat().st_size / 1024 / 1024
        logger.info(f"{name}: {path} ({size_mb:.1f} MB)")

    # 读取并合并
    logger.info("读取数据...")
    all_lines = []

    for path, source in [
        (megamath_path, "megamath_web_pro"),
        (fineweb_path, "fineweb"),
    ]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                data["source"] = source  # 标记数据来源
                all_lines.append(json.dumps(data, ensure_ascii=False))
        logger.info(f"  {source}: {len(all_lines)} 行累计")

    logger.info(f"总行数: {len(all_lines)}")

    # 可选 shuffle（保留原始顺序以便复现，但 shuffle 更安全）
    if shuffle:
        import random
        random.seed(42)
        random.shuffle(all_lines)
        logger.info("已 shuffle (seed=42)")

    # 写入
    logger.info(f"写入 {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")

    output_size = output_path.stat().st_size
    logger.info(f"完成! 输出: {output_path} ({output_size/1024/1024:.1f} MB)")

    # 验证
    logger.info("验证前 3 行:")
    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            obj = json.loads(line)
            logger.info(
                f"  [{i}] source={obj.get('source', '?')}, "
                f"text 长度={len(obj['text'])} chars, "
                f"预览: {obj['text'][:80]}..."
            )


def main():
    parser = argparse.ArgumentParser(description="Merge MegaMath + FineWeb mixture")
    parser.add_argument(
        "--megamath",
        type=Path,
        default=MEGAMATH_FILE,
        help="MegaMath JSONL 路径",
    )
    parser.add_argument(
        "--fineweb",
        type=Path,
        default=FINEWEB_FILE,
        help="FineWeb JSONL 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="输出路径",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不 shuffle",
    )
    args = parser.parse_args()

    merge_datasets(
        megamath_path=args.megamath,
        fineweb_path=args.fineweb,
        output_path=args.output,
        shuffle=not args.no_shuffle,
    )


if __name__ == "__main__":
    main()