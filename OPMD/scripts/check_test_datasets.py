#!/usr/bin/env python3
"""
检查 coding benchmark 与现有 math benchmark 在 OPMD test-data 规范下是否一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

TEST_DATA_DIR = Path("/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/test_data")

# math 的 canonical 列（以 MATH-500 / Minerva / AIME25 的共同列为准）
CANONICAL_COLS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

MATH = ["MATH-500", "Minerva", "AIME25"]
CODING = ["HumanEvalPlus", "MBPPPlus", "LiveCodeBench", "BigCodeBench", "SantaCoderFIM"]


def parquet_columns(name: str) -> list[str]:
    p = TEST_DATA_DIR / name / "test.parquet"
    return pq.read_schema(p).names


def main():
    print("=" * 72)
    print("OPMD test-data 一致性检查 (math vs coding)")
    print("=" * 72)

    math_cols = None
    for name in MATH:
        cols = parquet_columns(name)
        if math_cols is None:
            math_cols = cols
        missing = [c for c in CANONICAL_COLS if c not in cols]
        print(f"Math   [{name:<10}] columns={cols}")
        print(f"         canonical_missing={missing if missing else 'NONE'}")

    print()
    for name in CODING:
        cols = parquet_columns(name)
        # coding 必须包含全部 canonical math 列
        missing = [c for c in CANONICAL_COLS if c not in cols]
        extra = [c for c in cols if c not in CANONICAL_COLS]
        compatible = not missing
        print(f"Coding [{name:<14}] format=parquet")
        print(f"         columns={cols}")
        print(f"         canonical_missing={missing if missing else 'NONE'}")
        print(f"         extra_columns={extra if extra else 'NONE'}")
        print(f"         compatible={'YES' if compatible else 'NO'}")

    # 读取各 coding benchmark 的 task count
    print("\n--- 任务数统计 ---")
    meta_path = TEST_DATA_DIR / "coding_benchmarks_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for k, v in meta.items():
            print(f"  {k:<18} tasks={v.get('num_tasks')}")
    for name in CODING:
        p = TEST_DATA_DIR / name / "test.parquet"
        print(f"  {name:<18} parquet_rows={pq.read_metadata(p).num_rows}")


if __name__ == "__main__":
    main()