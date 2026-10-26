"""
从 KbsdJames/Omni-MATH 下载并转换为预训练 JSONL 格式（与 megamath_web_pro 一致）。
输出格式: {"text": "Problem: ...\nSolution: ..."}

用法:
    conda activate opmd
    python OPMD/scripts/download_omni_math.py

可选: 通过环境变量控制输出路径和格式
    OMNI_MATH_OUTPUT=/path/to/output.jsonl python OPMD/scripts/download_omni_math.py
"""

import json
import os
import sys
from pathlib import Path

# 使用 HF 镜像加速下载（国内网络）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset

# ============ 配置 ============
DATASET_ID = "KbsdJames/Omni-MATH"
OUTPUT_DIR = Path(os.environ.get(
    "OMNI_MATH_OUTPUT_DIR",
    "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets"
))
OUTPUT_FILE = Path(os.environ.get(
    "OMNI_MATH_OUTPUT",
    str(OUTPUT_DIR / "omni_math.jsonl")
))
# 文本拼接模板: problem + solution
# 可自定义: 设为 "problem_only" 则只保留 problem
TEXT_MODE = os.environ.get("OMNI_MATH_TEXT_MODE", "problem_solution")  # problem_solution | problem_only
# =============================


def build_text(row: dict, mode: str) -> str:
    """将 Omni-MATH 的一条记录转换为预训练文本"""
    problem = row["problem"].strip()
    solution = row["solution"].strip()

    if mode == "problem_only":
        return problem
    elif mode == "problem_solution":
        return f"Problem:\n{problem}\n\nSolution:\n{solution}"
    else:
        raise ValueError(f"Unknown TEXT_MODE: {mode}")


def main():
    print("=" * 60)
    print("Omni-MATH → 预训练 JSONL 格式转换")
    print(f"数据集: {DATASET_ID}")
    print(f"文本模式: {TEXT_MODE}")
    print(f"输出: {OUTPUT_FILE}")
    print("=" * 60)

    # 加载数据集
    # Omni-MATH 是单文件 JSONL（只有 test split），没有 config
    print("\n[1/2] 加载数据集 ...")
    try:
        ds = load_dataset(DATASET_ID, split="test")
        print(f"  从 HuggingFace 加载成功: {len(ds)} 条记录")
    except Exception as e:
        print(f"  load_dataset 失败 ({e})，尝试直接读取本地文件...")
        local_file = OUTPUT_DIR / "Omni-MATH_test.jsonl"
        if local_file.exists():
            data = []
            with open(local_file, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
            print(f"  从本地文件读取: {len(data)} 条记录")
        else:
            print(f"  错误: 本地文件 {local_file} 不存在，且无法从 HuggingFace 加载")
            print(f"  请手动下载: curl -L -o {local_file} https://huggingface.co/datasets/{DATASET_ID}/resolve/main/test.jsonl")
            sys.exit(1)
        ds = data

    # 转换并写入
    print("\n[2/2] 转换为 JSONL 格式 ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            text = build_text(row, TEXT_MODE)
            obj = {"text": text}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total_chars += len(text)

            if (i + 1) % 500 == 0:
                print(f"  已处理: {i + 1}/{len(ds)} 条")

    # 最终统计
    file_size = OUTPUT_FILE.stat().st_size
    print(f"\n完成!")
    print(f"  总记录数: {len(ds):,}")
    print(f"  总字符数: {total_chars:,}")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  文件大小: {file_size / 1024:.1f} KB ({file_size / 1024 / 1024:.2f} MB)")

    # 验证前 3 行
    print("\n验证前 3 行:")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            obj = json.loads(line)
            preview = obj["text"][:120].replace("\n", "\\n")
            print(f"  [{i}] text 长度: {len(obj['text'])} chars, 预览: {preview}...")


if __name__ == "__main__":
    main()