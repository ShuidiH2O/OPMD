"""
OPMD/NTP — Comparison Visualization
====================================
对比 Base Model 与 NTP 训练后模型的正确率。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def plot_comparison(base_results, ntp_results, output_path):
    """绘制 base vs NTP 对比柱状图。"""
    task_names = [Path(r["file"]).stem.split("_")[0].upper() for r in base_results]

    base_pass1 = [r["pass@1"] * 100 for r in base_results]
    ntp_pass1 = [r["pass@1"] * 100 for r in ntp_results]

    x = np.arange(len(task_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, base_pass1, width, label="Base Model (qwen3-4b)", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, ntp_pass1, width, label="NTP Mid-Trained", color="#DD8452", edgecolor="white")

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f"{height:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f"{height:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Benchmark", fontsize=13)
    ax.set_ylabel("pass@1 Accuracy (%)", fontsize=13)
    ax.set_title("Phase 1 Baseline: Base Model vs NTP Mid-Training", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=12)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_ylim(0, max(max(base_pass1), max(ntp_pass1)) + 10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_summary", type=str, required=True)
    parser.add_argument("--ntp_summary", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.base_summary, "r", encoding="utf-8") as f:
        base_results = json.load(f)
    with open(args.ntp_summary, "r", encoding="utf-8") as f:
        ntp_results = json.load(f)

    plot_comparison(base_results, ntp_results, args.output_path)


if __name__ == "__main__":
    main()