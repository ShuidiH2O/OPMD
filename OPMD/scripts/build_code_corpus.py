#!/usr/bin/env python3
"""
从 ModelScope 「AI-ModelScope/starcoderdata」 构建 coding-domain JSONL 语料，
格式与 OPMD megamath_web_pro_1B.jsonl 对齐（每行一个 {"text": ...} 文档），
并额外保留 language 元数据。

设计要点：
- 仅使用纯 programming-language 子集（排除 issues / commits / notebooks 及
  markdown / json / yaml / html / ... 等文档与结构化数据语言，即 Code-focused 口径）。
- 流式读取（不整体下载），tokenize on-the-fly，按语言规模 proportional 分配 token 配额。
- 严格控制最终 token 数 == target_tokens（最后一条文档按需截断）。
- 支持 checkpoint / resume；语言文件列表缓存到 language_files.json，避免重复走 API。

用法:
    conda activate opmd
    python -u build_code_corpus.py --target_tokens 10000000     # 10M dry run
    python -u build_code_corpus.py --target_tokens 1000000000   # 正式 1B
    python -u build_code_corpus.py --resume                     # 断点续跑
    python -u build_code_corpus.py --refresh_cache              # 强制重新列文件

注：脚本落盘完成后调用 os._exit(0)，以规避 datasets/pyarrow 在解释器
退出时的 teardown 崩溃（PyGILState_Release）——该崩溃仅发生在所有文件
已写完之后，不影响数据，但会让退出码异常。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from modelscope.hub.api import HubApi
from datasets import load_dataset
from transformers import AutoTokenizer

# ============ 常量 ============
DATASET_NAME = "starcoderdata"
NAMESPACE = "AI-ModelScope"
REPO_ID = f"{NAMESPACE}/{DATASET_NAME}"
REVISION = "master"

# 排除子集。
# 前 4 个是 StarCoderData 的非编程语言类别；后 10 个是 Code-focused 口径下
# 的文档 / 结构化数据 / Web markup，不纳入 coding corpus。
EXCLUDE_SUBSETS = {
    "git-commits-cleaned",
    "github-issues-filtered-structured",
    "jupyter-scripts-dedup-filtered",
    "jupyter-structured-clean-dedup",
    "markdown",
    "rmarkdown",
    "restructuredtext",
    "tex",
    "json",
    "yaml",
    "html",
    "css",
    "xslt",
    "java-server-pages",
}

DEFAULT_TOKENIZER = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/Qwen3-1.7B-Base"
DEFAULT_OUTPUT = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/coding_1b"
SEED = 42

# 基础清洗阈值（尽量轻，不改变 StarCoderData 原始分布）
MIN_CHARS = 40
MAX_CHARS = 5_000_000
MAX_CTRL_RATIO = 0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build coding-domain JSONL corpus from StarCoderData")
    p.add_argument("--target_tokens", type=int, default=1_000_000_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--refresh_cache", action="store_true",
                   help="忽略 language_files.json，强制重新列文件")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=MIN_CHARS)
    p.add_argument("--max_chars", type=int, default=MAX_CHARS)
    p.add_argument("--max_ctrl_ratio", type=float, default=MAX_CTRL_RATIO)
    p.add_argument("--checkpoint_every_rows", type=int, default=20_000)
    p.add_argument("--hf_cache_dir", type=str, default=None)
    return p.parse_args()


# ============ ModelScope 文件访问 ============
def _list_files(api: HubApi, path: str) -> list[dict]:
    all_files: list[dict] = []
    page, page_size = 1, 100
    while True:
        batch = api.get_dataset_files(
            REPO_ID, revision=REVISION, root_path=path, recursive=True,
            page_number=page, page_size=page_size,
        )
        all_files.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return sorted(all_files, key=lambda f: f.get("Path", ""))


def _cache_signature() -> str:
    return f"{REPO_ID}@{REVISION}|{','.join(sorted(EXCLUDE_SUBSETS))}"


def ensure_language_files(api: HubApi, output_dir: Path, refresh: bool = False) -> dict[str, list[dict]]:
    """返回 {language: [{"Path":..., "Size":...}, ...]}，并缓存到 language_files.json。

    缓存 key 为 (repo, revision, exclude_subsets) 的签名：一旦排除清单变化，
    自动失效并重新列文件，保证正确性。
    """
    cache_path = output_dir / "language_files.json"
    sig = _cache_signature()
    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("signature") == sig:
            print("[cache] 复用 language_files.json（跳过逐语言列文件）", flush=True)
            return {k: v for k, v in data["lang_files"].items()}

    print("[cache] 未命中，重新列语言文件（可能需要数分钟）...", flush=True)
    top = api.get_dataset_files(REPO_ID, revision=REVISION, recursive=False, page_size=500)
    langs = sorted(
        f["Name"] for f in top
        if f["Type"] == "tree" and f["Name"] not in EXCLUDE_SUBSETS
    )
    lang_files: dict[str, list[dict]] = {}
    for lang in langs:
        files = _list_files(api, lang)
        lang_files[lang] = [
            {"Path": f.get("Path", ""), "Size": int(f.get("Size") or 0)}
            for f in files
        ]

    payload = {
        "signature": sig,
        "dataset": REPO_ID,
        "revision": REVISION,
        "exclude_subsets": sorted(EXCLUDE_SUBSETS),
        "languages": langs,
        "lang_files": lang_files,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "language_files.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(cache_path)
    return lang_files


def list_language_sizes(lang_files: dict[str, list[dict]]) -> dict[str, int]:
    return {
        lang: sum(f["Size"] for f in files)
        for lang, files in lang_files.items()
    }


def compute_quotas(sizes: dict[str, int], target: int) -> dict[str, int]:
    total = sum(sizes.values())
    if total <= 0:
        raise RuntimeError("language sizes sum is zero")
    quotas = {lang: int(round(target * s / total)) for lang, s in sizes.items()}
    diff = target - sum(quotas.values())
    order = sorted(quotas, key=lambda k: quotas[k])
    i = 0
    while diff != 0:
        lang = order[i % len(order)]
        step = 1 if diff > 0 else -1
        if quotas[lang] + step > 0:
            quotas[lang] += step
            diff -= step
        i += 1
    assert sum(quotas.values()) == target
    return quotas


def get_shard_dataset(api: HubApi, lang: str, lang_files: dict[str, list[dict]]):
    files = lang_files.get(lang, [])
    urls = [
        api.get_dataset_file_url(
            file_name=f["Path"], dataset_name=DATASET_NAME,
            namespace=NAMESPACE, revision=REVISION,
        )
        for f in files
    ]
    if not urls:
        return iter(())
    return iter(load_dataset("parquet", data_files=urls, split="train", streaming=True))


# ============ 清洗 ============
def clean_text(s: str, min_chars: int, max_chars: int, max_ctrl_ratio: float) -> bool:
    if not s:
        return False
    s = s.strip()
    if not s:
        return False
    n = len(s)
    if n < min_chars or n > max_chars:
        return False
    if "\x00" in s:
        return False
    n_ctrl = 0
    for ch in s:
        o = ord(ch)
        if (o < 32 and ch not in "\t\n\r") or o == 0xFFFD:
            n_ctrl += 1
    return (n_ctrl / n) <= max_ctrl_ratio


def extract_content(row) -> str:
    c = row.get("content") if isinstance(row, dict) else None
    if c is None:
        return ""
    if isinstance(c, bytes):
        return c.decode("utf-8", errors="replace")
    return str(c)


# ============ checkpoint ============
def load_checkpoint(output_dir: Path) -> dict | None:
    cp = output_dir / "checkpoint.json"
    if not cp.exists():
        return None
    return json.loads(cp.read_text(encoding="utf-8"))


def save_checkpoint(output_dir: Path, state: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "checkpoint.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output_dir / "checkpoint.json")


# ============ 主流程 ============
def main() -> None:
    args = parse_args()
    target = args.target_tokens
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.hf_cache_dir or str(output_dir / ".hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)

    api = HubApi()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("=" * 70, flush=True)
    print("Build coding corpus from ModelScope StarCoderData", flush=True)
    print(f"  target_tokens : {target:,}", flush=True)
    print(f"  tokenizer     : {args.tokenizer_path}", flush=True)
    print(f"  output_dir    : {output_dir}", flush=True)
    print(f"  resume        : {args.resume}", flush=True)
    print("=" * 70, flush=True)

    # 列文件（优先用缓存）
    lang_files = ensure_language_files(api, output_dir, refresh=args.refresh_cache)

    cp = load_checkpoint(output_dir) if args.resume else None
    if args.resume and cp is None:
        print("[WARN] --resume 但未找到 checkpoint.json，将从零开始", flush=True)

    if cp is not None:
        lang_order = list(cp["lang_order"])
        quotas = {k: int(v) for k, v in cp["quotas"].items()}
        lang_sizes = {k: int(v) for k, v in cp["lang_sizes_bytes"].items()}
        done_langs = list(cp["done_langs"])
        current_lang = cp.get("current_lang")
        current_lang_tokens = int(cp.get("current_lang_tokens") or 0)
        current_lang_rows = int(cp.get("current_lang_rows") or 0)
        total_tokens = int(cp.get("total_tokens") or 0)
        samples_written = int(cp.get("samples_written") or 0)
        assert int(cp["target_tokens"]) == target, "target_tokens 与 checkpoint 不一致"
        missing = [l for l in lang_order if l not in lang_files]
        if missing:
            raise RuntimeError(f"checkpoint 中的语言未出现在缓存里: {missing}")
        print(f"[resume] done_langs={len(done_langs)}, current={current_lang}, "
              f"total_tokens={total_tokens:,}", flush=True)
    else:
        lang_sizes = list_language_sizes(lang_files)
        lang_order = sorted(lang_sizes.keys())
        quotas = compute_quotas(lang_sizes, target)
        done_langs = []
        current_lang = None
        current_lang_tokens = 0
        current_lang_rows = 0
        total_tokens = 0
        samples_written = 0
        for lang in lang_order:
            print(f"  {lang:<22} size={lang_sizes[lang]/1e9:6.2f}GB -> "
                  f"quota={quotas[lang]/1e6:8.2f}M tok", flush=True)
    print(f"  total languages: {len(lang_order)}  sum(quotas)={sum(quotas.values()):,}",
          flush=True)

    train_path = output_dir / "train.jsonl"
    fout = open(train_path, "a" if (cp is not None and train_path.exists()) else "w",
                encoding="utf-8")

    lang_tokens = current_lang_tokens
    lang_rows = current_lang_rows

    try:
        for lang in lang_order:
            if lang in done_langs:
                continue
            quota = quotas[lang]

            if lang != current_lang:
                lang_tokens = 0
                lang_rows = 0
                current_lang = lang

            print(f"\n[{lang}] quota={quota:,} tokens, resume@"
                  f"tokens={lang_tokens:,} rows={lang_rows:,}", flush=True)
            ds_iter = get_shard_dataset(api, lang, lang_files)

            if lang_rows > 0:
                to_skip = lang_rows
                skipped = 0
                for _ in ds_iter:
                    skipped += 1
                    if skipped >= to_skip:
                        break
                print(f"  skipped {skipped:,} rows", flush=True)

            batch: list[str] = []
            stop = False

            for row in ds_iter:
                lang_rows += 1
                content = extract_content(row)
                if clean_text(content, args.min_chars, args.max_chars, args.max_ctrl_ratio):
                    batch.append(content)

                if len(batch) >= args.batch_size:
                    lang_tokens, stop, n_written, n_added = process_batch(
                        fout, tok, lang, batch, lang_tokens, quota)
                    total_tokens += n_added
                    samples_written += n_written
                    batch = []
                    print(f"  [{lang}] rows={lang_rows:,} lang_tokens={lang_tokens:,} "
                          f"total_tokens={total_tokens:,} docs={samples_written:,}",
                          flush=True)
                    if stop:
                        break

                if lang_rows % args.checkpoint_every_rows == 0:
                    save_checkpoint(output_dir, make_state(
                        target, total_tokens, samples_written, lang_order, quotas,
                        lang_sizes, done_langs, current_lang, lang_tokens, lang_rows))

            if batch and not stop:
                lang_tokens, stop, n_written, n_added = process_batch(
                    fout, tok, lang, batch, lang_tokens, quota)
                total_tokens += n_added
                samples_written += n_written
                batch = []

            if not stop:
                print(f"  [WARN] {lang} exhausted before quota: "
                      f"{lang_tokens:,}/{quota:,}", flush=True)

            done_langs.append(lang)
            current_lang = None
            lang_tokens = 0
            lang_rows = 0

            save_checkpoint(output_dir, make_state(
                target, total_tokens, samples_written, lang_order, quotas,
                lang_sizes, done_langs, None, 0, 0))
    finally:
        fout.close()

    write_stats_and_meta(output_dir, target, total_tokens, samples_written,
                         lang_order, quotas, lang_sizes, args)
    print("\nDone.", flush=True)
    print(f"  total_tokens    = {total_tokens:,}", flush=True)
    print(f"  samples_written = {samples_written:,}", flush=True)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def process_batch(fout, tok, lang, texts, lang_tokens, quota):
    enc = tok(texts, add_special_tokens=False, truncation=False)
    ids_list = enc["input_ids"]
    n_written = 0
    start = lang_tokens
    for text, ids in zip(texts, ids_list):
        L = len(ids)
        remaining = quota - lang_tokens
        if L <= remaining:
            fout.write(json.dumps({"text": text, "language": lang}, ensure_ascii=False) + "\n")
            lang_tokens += L
            n_written += 1
        else:
            truncated = tok.decode(ids[:remaining], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
            fout.write(json.dumps({"text": truncated, "language": lang}, ensure_ascii=False) + "\n")
            lang_tokens = quota
            n_written += 1
            return lang_tokens, True, n_written, lang_tokens - start
    return lang_tokens, False, n_written, lang_tokens - start


def make_state(target, total_tokens, samples_written, lang_order, quotas,
               lang_sizes, done_langs, current_lang, current_lang_tokens, current_lang_rows):
    return {
        "schema_version": 1,
        "dataset": REPO_ID,
        "tokenizer": DEFAULT_TOKENIZER,
        "target_tokens": int(target),
        "total_tokens": int(total_tokens),
        "samples_written": int(samples_written),
        "lang_order": lang_order,
        "quotas": {k: int(v) for k, v in quotas.items()},
        "lang_sizes_bytes": {k: int(v) for k, v in lang_sizes.items()},
        "done_langs": done_langs,
        "current_lang": current_lang,
        "current_lang_tokens": int(current_lang_tokens),
        "current_lang_rows": int(current_lang_rows),
        "sampling_strategy": "proportional_by_parquet_bytes",
        "seed": SEED,
    }


def write_stats_and_meta(output_dir, target, total_tokens, samples_written,
                         lang_order, quotas, lang_sizes, args):
    total_bytes = sum(lang_sizes.values())
    total_quota = sum(quotas.values()) or 1
    languages = {
        l: {
            "size_bytes": lang_sizes[l],
            "tokens": quotas[l],
            "tokens_pct": round(quotas[l] / total_quota * 100, 4),
            "bytes_pct": round(lang_sizes[l] / total_bytes * 100, 4) if total_bytes else 0,
        }
        for l in lang_order
    }
    stats = {
        "dataset": REPO_ID,
        "tokenizer": DEFAULT_TOKENIZER,
        "target_tokens": target,
        "actual_tokens": int(total_tokens),
        "num_documents": int(samples_written),
        "languages": languages,
        "tokens_per_language": {l: quotas[l] for l in lang_order},
        "seed": SEED,
        "sampling_strategy": "proportional_by_parquet_bytes",
        "filtering_strategy": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "max_ctrl_ratio": args.max_ctrl_ratio,
            "exclude_subsets": sorted(EXCLUDE_SUBSETS),
        },
    }
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "dataset": REPO_ID,
        "dataset_version": REVISION,
        "tokenizer": DEFAULT_TOKENIZER,
        "seed": SEED,
        "sampling_strategy": "proportional_by_parquet_bytes",
        "language_list": lang_order,
        "filtering_rules": stats["filtering_strategy"],
        "target_token_count": int(target),
        "actual_token_count": int(total_tokens),
        "num_documents": int(samples_written),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote dataset_stats.json / metadata.json under {output_dir}", flush=True)


if __name__ == "__main__":
    main()