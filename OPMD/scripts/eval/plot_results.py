"""
OPMD/NTP Evaluation — Visualization
====================================
读取 grading_summary.json，生成正确率柱状图。
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def plot_results(results, output_path, title, model_name=""):
    """绘制各任务 pass@1 柱状图。"""
    tasks = []
    pass1_values = []
    avgk_values = []

    for r in results:
        task_name = Path(r["file"]).stem.split("_")[0].upper()
        tasks.append(task_name)
        pass1_values.append(r["pass@1"] * 100)
        avgk_values.append(r["avg@k"] * 100)

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, pass1_values, width, label="pass@1", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, avgk_values, width, label="avg@k", color="#DD8452", edgecolor="white")

    # 在柱子上标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f"{height:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f"{height:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Benchmark", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"{title}\n{model_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_ylim(0, max(max(pass1_values), max(avgk_values)) + 10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", type=str, required=True,
                        help="Path to grading_summary.json")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the figure (PNG)")
    parser.add_argument("--title", type=str, default="NTP Mid-Training Evaluation",
                        help="Chart title")
    parser.add_argument("--model_name", type=str, default="",
                        help="Model name for subtitle")
    args = parser.parse_args()

    with open(args.summary_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    plot_results(results, args.output_path, args.title, args.model_name)


if __name__ == "__main__":
    main()