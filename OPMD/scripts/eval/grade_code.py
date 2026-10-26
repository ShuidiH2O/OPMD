#!/usr/bin/env python3
"""
Coding benchmark 代码执行评分（配合 gen_vllm.py 生成的 JSONL）。

- HumanEval+ / MBPP+：复用 EvalPlus 官方 evaluator（执行 base + plus tests）。
- LiveCodeBench：对 private/public test cases 逐个 stdin/stdout 执行比对。

输出 grading_summary.json，字段与 math 的 grade.py 保持一致
（pass@1 / avg@k / num_questions / solve_none / solve_all 等）。

用法:
    python grade_code.py --input_dir <dir> --output_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path("/mnt/cpfs/epic-user/zhengruize-20260612")
TEST_DATA_DIR = PROJECT_ROOT / "OPMD/datasets/test_data"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================ HumanEval+ / MBPP+ ============================
def grade_evalplus(dataset: str, records: list[dict]) -> dict:
    tmp = Path(tempfile.mkdtemp())
    samples_path = tmp / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "task_id": r["index"],
                "completion": r["response"],
            }, ensure_ascii=False) + "\n")

    from evalplus.evaluate import evaluate
    evaluate(dataset=dataset, samples=str(samples_path), base_only=False, parallel=None)

    result_path = Path(str(samples_path).replace(".jsonl", "_eval_results.json"))
    data = json.loads(result_path.read_text(encoding="utf-8"))
    ev = data["eval"]
    n = len(ev)
    base_ok = 0
    plus_ok = 0
    for tid, lst in ev.items():
        if not lst:
            continue
        res = lst[0]  # greedy：每个 task 一个 sample
        if res.get("base_status") == "pass":
            base_ok += 1
            if res.get("plus_status") == "pass":
                plus_ok += 1

    return {
        "num_questions": n,
        "num_responses_per_question": 1,
        "base_pass@1": base_ok / n if n else 0.0,
        "plus_pass@1": plus_ok / n if n else 0.0,
    }


# ============================ LiveCodeBench ============================
def extract_code(text: str) -> str:
    # 优先取 ```python ... ``` 代码块
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # 去掉常见的 markdown 包围
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def run_program(code: str, input_data: str, timeout: float = 10.0) -> tuple[str, str]:
    """执行代码，返回 (stdout, status)。"""
    tmp = Path(tempfile.mkdtemp())
    code_path = tmp / "main.py"
    code_path.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(code_path)],
            input=input_data.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
        return proc.stdout.decode("utf-8", errors="replace"), ("ok" if proc.returncode == 0 else f"rc{proc.returncode}")
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except Exception as e:
        return "", f"err:{type(e).__name__}"


def grade_livecodebench(records: list[dict]) -> dict:
    # 从 LCB parquet 读 coding_info，按 question_id 建索引
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    df = pd.read_parquet(TEST_DATA_DIR / "LiveCodeBench" / "test.parquet")
    qid_to_tests = {}
    for i in range(len(df)):
        qid = str(df.at[i, "extra_info"]["index"])
        ci = json.loads(df.at[i, "coding_info"])
        qid_to_tests[qid] = ci

    def check_cases(code: str, tests: list) -> bool:
        if not tests:
            return True
        if not code.strip():
            # 没提取到代码 → 该题判失败（不能把空代码当通过）
            return False
        fd, prog = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)
            for tc in tests:
                inp = tc.get("input", "") or ""
                expected = (tc.get("output", "") or "").rstrip("\n")
                try:
                    ex = subprocess.run(
                        [sys.executable, prog], input=inp.encode("utf-8"),
                        capture_output=True, timeout=10.0,
                    )
                except (subprocess.TimeoutExpired, OSError):
                    # 程序死循环 / 执行异常 → 该测试判为失败，不崩整个评分
                    return False
                if ex.returncode != 0:
                    return False
                out = ex.stdout.decode("utf-8", errors="replace").rstrip("\n").strip()
                if out != expected.strip():
                    return False
            return True
        finally:
            try:
                os.unlink(prog)
            except OSError:
                pass

    def process_task(rec: dict):
        qid = rec["index"]
        ci = qid_to_tests.get(qid)
        if ci is None:
            return None
        code = extract_code(rec["response"])
        pub = ci.get("public_test_cases") or []
        priv = ci.get("private_test_cases") or []
        return qid, check_cases(code, pub), check_cases(code, priv)

    n = 0
    ok_public = 0
    ok_private = 0
    workers = min(16, os.cpu_count() or 4)
    total = len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_task, r) for r in records]
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res is not None:
                qid, pub_ok, priv_ok = res
                n += 1
                ok_public += int(pub_ok)
                ok_private += int(priv_ok)
            if done % 200 == 0 or done == total:
                print(f"  LCB progress: {done}/{total}", flush=True)

    return {
        "num_questions": n,
        "num_responses_per_question": 1,
        "public_pass@1": ok_public / n if n else 0.0,
        "private_pass@1": ok_private / n if n else 0.0,
        "lcb_pass@1": ok_private / n if n else 0.0,
    }


# ==================== SantaCoder-FIM ====================
def grade_santacoder_fim(records: list[dict]) -> dict:
    """FIM 单行 exact match（strip 后相等），per-language 报告。"""
    from collections import defaultdict

    import pyarrow.parquet as pq

    t = pq.read_table(TEST_DATA_DIR / "SantaCoderFIM" / "test.parquet").to_pandas()
    name2lang = {}
    for i in range(len(t)):
        ci = json.loads(t.iloc[i]["coding_info"])
        name2lang[t.iloc[i]["extra_info"]["index"]] = ci["language"]

    acc, cnt = defaultdict(int), defaultdict(int)
    for r in records:
        lang = name2lang.get(r["index"])
        if lang is None:
            continue
        cnt[lang] += 1
        acc[lang] += int((r["response"] or "").strip() == (r["answer"] or "").strip())

    total_ok = sum(acc.values())
    total_n = sum(cnt.values())
    per_lang = {lang: (acc[lang] / cnt[lang] if cnt[lang] else 0.0)
                for lang in ["py", "js", "java"]}
    return {
        "num_questions": total_n,
        "num_responses_per_question": 1,
        "em_pass@1": total_ok / total_n if total_n else 0.0,
        "em_per_language": per_lang,
    }


# ==================== BigCodeBench ====================
_BIGCODEBENCH_PROBLEMS: dict = {}
_BIGCODEBENCH_CWD: str = ""

# unittest 运行器：跑 TestCases，全部通过则 exit 0，否则 exit 1。
_BIGCODEBENCH_RUNNER = (
    "import unittest\n"
    "_loader = unittest.TestLoader()\n"
    "_suite = _loader.loadTestsFromTestCase(TestCases)\n"
    "_result = unittest.TestResult()\n"
    "_suite.run(_result)\n"
    "import sys\n"
    "sys.exit(0 if not (_result.failures or _result.errors) else 1)\n"
)


def _bcb_check(r):
    from bigcodebench.sanitize import sanitize

    tid = r["index"]
    p = _BIGCODEBENCH_PROBLEMS.get(tid)
    if p is None:
        return None
    try:
        completion = r["response"]
        # 坍缩模型补全可能是超长退化文本（step_400 median ~6.8KB），会让 sanitize
        # 的 code_extract 退化成 O(n^2)/O(n^3) 而卡死。canonical_solution 最长 1811 字符，
        # 截到 2000 足够且不会截断任何正确解。
        if len(completion) > 2000:
            completion = completion[:2000]
        sol = sanitize(p["complete_prompt"] + completion, p["entry_point"])
    except Exception:
        sol = ""
    if not (sol or "").strip():
        return 0

    full = sol + "\n" + p["test"] + "\n" + _BIGCODEBENCH_RUNNER
    timeout = float(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", "10"))
    import signal
    try:
        # 用独立进程组 + DEVNULL，超时后整组 SIGKILL，避免子进程产生的
        # 后代进程占住管道导致 communicate 永久阻塞。
        proc = subprocess.Popen(
            [sys.executable, "-c", full],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=_BIGCODEBENCH_CWD or None,
        )
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()
            return 0
        return int(proc.returncode == 0)
    except Exception:
        return 0


def grade_bigcodebench(records: list[dict]) -> dict:
    """BigCodeBench-Complete：官方 sanitize + 子进程 unittest 执行（Original Pass@1）。"""
    import pyarrow.parquet as pq
    from bigcodebench.sanitize import sanitize  # 预热 import，避免重复导入
    from concurrent.futures import ProcessPoolExecutor

    global _BIGCODEBENCH_PROBLEMS, _BIGCODEBENCH_CWD

    t = pq.read_table(TEST_DATA_DIR / "BigCodeBench" / "test.parquet").to_pandas()
    problems = {}
    for i in range(len(t)):
        ci = json.loads(t.iloc[i]["coding_info"])
        problems[t.iloc[i]["extra_info"]["index"]] = ci
    _BIGCODEBENCH_PROBLEMS = problems

    # 生成代码会在当前目录写文件；切到临时目录，避免污染项目根目录。
    import shutil
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="bigcodebench_eval_")
    _BIGCODEBENCH_CWD = tmpdir
    try:
        workers = min(16, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = [x for x in ex.map(_bcb_check, records) if x is not None]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _BIGCODEBENCH_CWD = ""

    n = len(res)
    ok = sum(res)
    return {
        "num_questions": n,
        "num_responses_per_question": 1,
        "original_pass@1": ok / n if n else 0.0,
    }


def infer_task(filename: str) -> str:
    fn = filename.lower()
    if fn.startswith("humanevalplus"):
        return "humaneval"
    if fn.startswith("mbppplus"):
        return "mbpp"
    if fn.startswith("livecodebench"):
        return "livecodebench"
    if fn.startswith("bigcodebench"):
        return "bigcodebench"
    if fn.startswith("santacoderfim"):
        return "santacoderfim"
    return ""


def grade_file(path: Path) -> dict:
    records = load_jsonl(path)
    task = infer_task(path.name)
    if task in ("humaneval", "mbpp"):
        r = grade_evalplus(task, records)
        r["file"] = str(path)
        r["pass@1"] = r["plus_pass@1"]  # HumanEval+/MBPP+ 的正式指标
        r["avg@k"] = r["plus_pass@1"]
        r["solve_none"] = 0
        r["solve_all"] = 0
        return r
    elif task == "livecodebench":
        r = grade_livecodebench(records)
        r["file"] = str(path)
        r["pass@1"] = r["private_pass@1"]  # LCB 正式指标（private tests）
        r["avg@k"] = r["private_pass@1"]
        r["solve_none"] = 0
        r["solve_all"] = 0
        return r
    elif task == "bigcodebench":
        r = grade_bigcodebench(records)
        r["file"] = str(path)
        r["pass@1"] = r["original_pass@1"]  # BigCodeBench 正式指标（Original Pass@1）
        r["avg@k"] = r["original_pass@1"]
        r["solve_none"] = 0
        r["solve_all"] = 0
        return r
    elif task == "santacoderfim":
        r = grade_santacoder_fim(records)
        r["file"] = str(path)
        r["pass@1"] = r["em_pass@1"]  # FIM 正式指标（overall exact match）
        r["avg@k"] = r["em_pass@1"]
        r["solve_none"] = 0
        r["solve_all"] = 0
        return r
    else:
        return {"file": str(path), "num_questions": 0, "pass@1": 0.0, "avg@k": 0.0,
                "solve_none": 0, "solve_all": 0, "error": "unknown task"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--only", type=str, default=None,
                        help="逗号分隔的文件名前缀，仅评分这些 benchmark（默认评分全部）")
    parser.add_argument("--summary_name", type=str, default="grading_summary.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    only_prefixes = []
    if args.only:
        only_prefixes = [p.strip().lower() for p in args.only.split(",") if p.strip()]

    all_results = []
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        if only_prefixes and not any(jsonl_file.name.lower().startswith(p) for p in only_prefixes):
            continue
        print(f"Grading: {jsonl_file.name}")
        result = grade_file(jsonl_file)
        all_results.append(result)
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print()

    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()