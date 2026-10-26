#!/usr/bin/env python3
"""
将 HumanEval+ / MBPP+ / LiveCodeBench 转成 OPMD 现有 math test 数据格式。

- math test 的 canonical parquet 列（以 MATH-500 / Minerva / AIME25 为准）：
    data_source: string
    prompt:      list<struct<content: string, role: string>>
    ability:     string
    reward_model: struct<ground_truth: string, style: string>
    extra_info:   struct<index: string, split: string>
- coding benchmark 在保持上述 5 列的基础上，额外增加一列 `coding_info`
  （JSON string），保留 coding 特有的 evaluation metadata
  （entry_point / tests / plus_input / atol / private_test_cases 等），不丢数据。

数据源：
- HumanEval+: evalplus 官方 release v0.1.10
- MBPP+     : evalplus 官方 release v0.2.0
- LiveCodeBench: ModelScope `AI-ModelScope/code_generation_lite`（6 个 test*.jsonl，
  即 release_latest 全量）

用法:
    conda activate opmd
    python build_coding_benchmarks.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
import urllib.request
import zlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from modelscope.hub.api import HubApi

ROOT = Path("/mnt/cpfs/epic-user/zhengruize-20260612")
TEST_DATA_DIR = ROOT / "OPMD/datasets/test_data"
MODELSCOPE_NS = "AI-ModelScope"
MODELSCOPE_DATASET = "code_generation_lite"
LCB_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_lcb_file(api: HubApi, fp: str) -> str:
    url = api.get_dataset_file_url(
        file_name=fp, dataset_name=MODELSCOPE_DATASET,
        namespace=MODELSCOPE_NS, revision="master")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def decode_lcb_private_test_cases(s: str):
    """LCB 的 private_test_cases 是 base64(zlib(pickle(json_str)))。"""
    if not s:
        return []
    dec = zlib.decompress(base64.b64decode(s))
    obj = pickle.loads(dec)  # 官方 LCB 数据，可信来源
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj


def get_humaneval_plus():
    from evalplus import data as D
    return D.get_human_eval_plus(), D.get_human_eval_plus_hash()


def get_mbpp_plus():
    from evalplus import data as D
    return D.get_mbpp_plus(), D.get_mbpp_plus_hash()


def get_livecodebench(api: HubApi):
    rows = []
    for fp in LCB_FILES:
        txt = fetch_lcb_file(api, fp)
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pub = d.get("public_test_cases")
            if isinstance(pub, str):
                pub = json.loads(pub)
            priv = decode_lcb_private_test_cases(d.get("private_test_cases") or "")
            rows.append({
                "question_title": d.get("question_title", ""),
                "question_content": d.get("question_content", ""),
                "platform": d.get("platform", ""),
                "question_id": d.get("question_id", ""),
                "contest_id": d.get("contest_id", ""),
                "contest_date": d.get("contest_date", ""),
                "starter_code": d.get("starter_code", ""),
                "difficulty": d.get("difficulty", ""),
                "public_test_cases": pub,
                "private_test_cases": priv,
                "metadata": d.get("metadata", {}),
            })
    return rows


def make_rows(records) -> list[dict]:
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

    # 首行 dump（与 math 目录里的 test.parquet_first_row.json 一致）
    first_row = table.to_pandas().iloc[0].to_dict()
    (out_dir / "test.parquet_first_row.json").write_text(
        json.dumps(first_row, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 源 JSON（原始 benchmark 数据，便于溯源）
    (out_dir / "test.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "name": name,
        "num_tasks": len(records),
        "parquet": str(parquet_path),
        "sha256": sha256_file(parquet_path),
    }


def main():
    api = HubApi()
    print("=" * 70)
    print("Build coding benchmark datasets (HumanEval+ / MBPP+ / LiveCodeBench)")
    print("=" * 70)

    # ---- HumanEval+ ----
    print("\n[1/3] HumanEval+ ...")
    he, he_hash = get_humaneval_plus()
    he_records = []
    for task_id in sorted(he.keys()):
        d = he[task_id]
        he_records.append({
            "data_source": "HumanEvalPlus",
            "prompt": d["prompt"],
            "ground_truth": d.get("canonical_solution", ""),
            "index": task_id,
            "coding_info": {
                k: d.get(k) for k in
                ("task_id", "entry_point", "canonical_solution", "base_input",
                 "plus_input", "atol", "test", "contract")
            },
        })
    he_meta = write_benchmark("HumanEvalPlus", he_records)
    print(f"  tasks={he_meta['num_tasks']}  hash={he_hash}  sha256={he_meta['sha256'][:16]}...")

    # ---- MBPP+ ----
    print("\n[2/3] MBPP+ ...")
    mb, mb_hash = get_mbpp_plus()
    mb_records = []
    for task_id in sorted(mb.keys()):
        d = mb[task_id]
        mb_records.append({
            "data_source": "MBPPPlus",
            "prompt": d["prompt"],
            "ground_truth": d.get("canonical_solution", ""),
            "index": task_id,
            "coding_info": {
                k: d.get(k) for k in
                ("task_id", "entry_point", "canonical_solution", "base_input",
                 "plus_input", "atol", "contract", "assertion")
            },
        })
    mb_meta = write_benchmark("MBPPPlus", mb_records)
    print(f"  tasks={mb_meta['num_tasks']}  hash={mb_hash}  sha256={mb_meta['sha256'][:16]}...")

    # ---- LiveCodeBench ----
    print("\n[3/3] LiveCodeBench ...")
    lcb_rows = get_livecodebench(api)
    lcb_records = []
    for d in lcb_rows:
        lcb_records.append({
            "data_source": "LiveCodeBench",
            "prompt": d["question_content"],
            "ground_truth": "",
            "index": d["question_id"],
            "coding_info": d,
        })
    lcb_meta = write_benchmark("LiveCodeBench", lcb_records)
    dates = sorted(x["contest_date"] for x in lcb_rows if x.get("contest_date"))
    print(f"  tasks={lcb_meta['num_tasks']}  sha256={lcb_meta['sha256'][:16]}...")
    print(f"  contest_date range: {dates[0] if dates else 'N/A'} .. {dates[-1] if dates else 'N/A'}")

    # ---- metadata ----
    metadata = {
        "humaneval_plus": {
            "source": "EvalPlus official release (github.com/evalplus/humanevalplus_release)",
            "version": "v0.1.10",
            "num_tasks": he_meta["num_tasks"],
            "sha256": he_meta["sha256"],
            "evalplus_data_hash": he_hash,
        },
        "mbpp_plus": {
            "source": "EvalPlus official release (github.com/evalplus/mbppplus_release)",
            "version": "v0.2.0",
            "num_tasks": mb_meta["num_tasks"],
            "sha256": mb_meta["sha256"],
            "evalplus_data_hash": mb_hash,
        },
        "livecodebench": {
            "source": "ModelScope AI-ModelScope/code_generation_lite (mirror of livecodebench/code_generation_lite)",
            "version": "release_latest (all 6 test*.jsonl)",
            "num_tasks": lcb_meta["num_tasks"],
            "sha256": lcb_meta["sha256"],
            "cutoff": (dates[-1] if dates else None),
            "date_range": [dates[0] if dates else None, dates[-1] if dates else None],
            "languages": sorted({x["platform"] for x in lcb_rows if x.get("platform")}),
        },
    }
    meta_path = TEST_DATA_DIR / "coding_benchmarks_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote: {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()