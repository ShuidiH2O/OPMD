#!/usr/bin/env python3
"""
将两个新增 coding benchmark 转成 OPMD 现有 math test 数据格式：

1. BigCodeBench  (complete split, base-model friendly)  -> BigCodeBench/
2. SantaCoder-FIM (bigcode/santacoder-fim-task)         -> SantaCoderFIM/

canonical parquet 列（与 MATH-500 / Minerva / AIME25 及已有 coding 数据一致）：
    data_source   string
    prompt        list<struct<content, role>>
    ability       string
    reward_model  struct<ground_truth, style>
    extra_info    struct<index, split>
    coding_info   string   （JSON 字符串，保留 coding 特有 evaluation metadata）

用法:
    conda activate opmd
    cd /mnt/cpfs/epic-user/zhengruize-20260612/OPMD/scripts
    python build_extra_coding_benchmarks.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

ROOT = Path("/mnt/cpfs/epic-user/zhengruize-20260612")
TEST_DATA_DIR = ROOT / "OPMD/datasets/test_data"

SCHEMA = pa.schema([
    pa.field("data_source", pa.string()),
    pa.field("prompt", pa.list_(pa.struct([
        pa.field("content", pa.string()),
        pa.field("role", pa.string()),
    ]))),
    pa.field("ability", pa.string()),
    pa.field("reward_model", pa.struct([
        pa.field("ground_truth", pa.string()),
        pa.field("style", pa.string()),
    ])),
    pa.field("extra_info", pa.struct([
        pa.field("index", pa.string()),
        pa.field("split", pa.string()),
    ])),
    pa.field("coding_info", pa.string()),
])

# FIM placeholder tokens（StarCoder / bigcode-evaluation-harness `starcoder_fim` 约定）。
# 评估时根据具体模型可替换为其原生 FIM 特殊 token（Qwen3 base 无 FIM 训练，按原文拼接）。
FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bigcodebench():
    """BigCodeBench v0.1.4（1140 tasks，complete 风格）。"""
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    records = []
    for i in range(len(ds)):
        d = ds[i]
        records.append({
            "data_source": "BigCodeBench",
            "prompt": d["complete_prompt"],
            "ground_truth": d["canonical_solution"],
            "index": d["task_id"],
            "coding_info": {
                "task_id": d["task_id"],
                "complete_prompt": d["complete_prompt"],
                "instruct_prompt": d["instruct_prompt"],
                "canonical_solution": d["canonical_solution"],
                "code_prompt": d["code_prompt"],
                "test": d["test"],
                "entry_point": d["entry_point"],
                "doc_struct": d["doc_struct"],
                "libs": d["libs"],
            },
        })
    return records


def load_santacoder_fim():
    """SantaCoder-FIM（4792 单行 infilling tasks）。"""
    ds = load_dataset("bigcode/santacoder-fim-task", split="train")
    records = []
    for i in range(len(ds)):
        d = ds[i]
        # 模型实际见到的 FIM 输入：prefix + suffix + middle 占位。
        fim_prompt = f"{FIM_PREFIX}{d['prompt']}{FIM_SUFFIX}{d['suffix']}{FIM_MIDDLE}"
        records.append({
            "data_source": "SantaCoderFIM",
            "prompt": fim_prompt,
            "ground_truth": d["canonical_solution"],
            "index": d["name"],
            "coding_info": {
                "name": d["name"],
                "language": d["language"],
                "prefix": d["prompt"],
                "suffix": d["suffix"],
                "canonical_solution": d["canonical_solution"],
                "tests": d["tests"],
                "fim_prompt": fim_prompt,
            },
        })
    return records


def make_rows(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        rows.append({
            "data_source": rec["data_source"],
            "prompt": [{"content": rec["prompt"], "role": "user"}],
            "ability": "code",
            "reward_model": {"ground_truth": rec["ground_truth"], "style": "rule"},
            "extra_info": {"index": rec["index"], "split": "test"},
            "coding_info": json.dumps(rec["coding_info"], ensure_ascii=False, default=str),
        })
    return rows


def write_benchmark(name: str, records: list[dict]) -> dict:
    out_dir = TEST_DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = make_rows(records)

    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    parquet_path = out_dir / "test.parquet"
    pq.write_table(table, parquet_path, compression="snappy")

    first_row = table.to_pandas().iloc[0].to_dict()
    (out_dir / "test.parquet_first_row.json").write_text(
        json.dumps(first_row, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    (out_dir / "test.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "name": name,
        "num_tasks": len(records),
        "parquet": str(parquet_path),
        "sha256": sha256_file(parquet_path),
    }


def tongue_distribution(coding_infos: list[dict]):
    from collections import Counter
    c = Counter(ci.get("language") for ci in coding_infos)
    return {k: v for k, v in sorted(c.items(), key=lambda kv: -kv[1])}


def main():
    print("=" * 72)
    print("Build extra coding benchmarks (BigCodeBench + SantaCoder-FIM)")
    print("=" * 72)

    print("\n[1/2] BigCodeBench v0.1.4 (complete) ...")
    bcb = load_bigcodebench()
    bcb_meta = write_benchmark("BigCodeBench", bcb)
    print(f"  tasks={bcb_meta['num_tasks']}  sha256={bcb_meta['sha256'][:16]}...")

    print("\n[2/2] SantaCoder-FIM ...")
    fim = load_santacoder_fim()
    fim_meta = write_benchmark("SantaCoderFIM", fim)
    lang = tongue_distribution([r["coding_info"] for r in fim])
    print(f"  tasks={fim_meta['num_tasks']}  sha256={fim_meta['sha256'][:16]}...")
    print(f"  language={lang}")

    # ---- update metadata ----
    meta_path = TEST_DATA_DIR / "coding_benchmarks_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["bigcodebench"] = {
        "source": "bigcode-project/bigcodebench (GitHub) / bigcode/bigcodebench (Hugging Face)",
        "version": "v0.1.4 (data split); package bigcodebench 0.2.5",
        "num_tasks": bcb_meta["num_tasks"],
        "split": "complete",
        "languages": ["python"],
        "python_pct": 100.0,
        "sha256": bcb_meta["sha256"],
    }
    meta["santacoder_fim"] = {
        "source": "bigcode/santacoder-fim-task (Hugging Face, built on MultiPL-E HumanEval+MBPP)",
        "version": "1 file / train split",
        "num_tasks": fim_meta["num_tasks"],
        "languages": {"javascript": 1969, "java": 1780, "python": 1043},
        "python_pct": round(1043 / 4792 * 100, 2),
        "sha256": fim_meta["sha256"],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote: {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()