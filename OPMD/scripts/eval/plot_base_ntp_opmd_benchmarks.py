"""
Qwen3-1.7B-Base — Base vs NTP vs OPMD benchmark comparison.

Compares pass@1 and avg@k on MATH-500 and Minerva for the Qwen3-1.7B-Base
mid-training experiments on MegaMath-Web-Pro:

  * Base            — untrained Qwen3-1.7B-Base
  * NTP             — standard next-token-prediction mid-training (collapse)
  * OPMD (high-entropy / random / low-entropy) — OPD-style anchor selection arms

The three OPMD arms correspond to the checkpoint runs:
  high-entropy : OPMD-0.05        (default)
  random       : OPMD-rand-0.05
  low-entropy  : OPMD-low-0.05

Data sources: pass@1 / avg@k, 16 rollouts, t=0.7 / top-p=0.95, chat template.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path("/mnt/cpfs/epic-user/zhengruize-20260612")

rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

BASE_COLOR      = "#AEB6BF"   # gray
NTP_COLOR       = "#356C9E"   # blue
OPMD_HIGH_COLOR = "#1E7B4A"   # green (high-entropy arm)
OPMD_RAND_COLOR = "#D99028"   # amber (random arm)
OPMD_LOW_COLOR  = "#8E44AD"   # purple (low-entropy arm)
TEXT_COLOR      = "#1A1A1A"
GAIN_COLOR      = "#1E7B4A"
LOSS_COLOR      = "#B73239"

BENCHMARKS = ["MATH-500", "Minerva"]
METRICS = [("pass@1", 1), ("avg@k", 2)]  # (json key, label decimals)

SERIES = [
    ("Base", BASE_COLOR, {
        "MATH-500": ROOT / "OPMD/eval_results/qwen3_1.7b_baseline/grading_summary.json",
        "Minerva":  ROOT / "OPMD/eval_results/qwen3_1.7b_baseline/grading_summary.json",
    }),
    ("NTP", NTP_COLOR, {
        "MATH-500": ROOT / "NTP/eval_results/ntp_qwen3_1.7b_megamath/grading_summary.json",
        "Minerva":  ROOT / "NTP/eval_results/ntp_qwen3_1.7b_megamath/grading_summary.json",
    }),
    ("OPMD\n(high-entropy)", OPMD_HIGH_COLOR, {
        "MATH-500": ROOT / "OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher/step_1132/grading_summary.json",
        "Minerva":  ROOT / "OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher/step_1132/grading_summary.json",
    }),
    ("OPMD\n(random)", OPMD_RAND_COLOR, {
        "MATH-500": ROOT / "OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher_rand/step_1132/grading_summary.json",
        "Minerva":  ROOT / "OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher_rand/step_1132_minerva/grading_summary.json",
    }),
    ("OPMD\n(low-entropy)", OPMD_LOW_COLOR, {
        "MATH-500": ROOT / "OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher_low/step_1699_math500_minerva/grading_summary.json",
        "Minerva":  ROOT / "OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher_low/step_1699_math500_minerva/grading_summary.json",
    }),
]


def get_metric(path, bench, key):
    data = json.loads(path.read_text(encoding="utf-8"))
    for rec in data:
        if bench.lower() in Path(rec["file"]).name.lower():
            return rec[key] * 100.0
    raise KeyError(f"{bench} not found in {path}")


def main():
    tick_labels = [s[0] for s in SERIES]
    legend_labels = [s[0].replace("\n", " ") for s in SERIES]
    colors = [s[1] for s in SERIES]
    n = len(SERIES)
    width = 0.55
    x = list(range(n))

    # metrics x benchmarks -> values
    vals = {}
    for key, _dec in METRICS:
        vals[key] = {b: [get_metric(s[2][b], b, key) for s in SERIES] for b in BENCHMARKS}

    print("=== data used ===")
    for key, _dec in METRICS:
        for b in BENCHMARKS:
            print(f"{key:7s} {b:9s}",
                  {k: round(vv, 2) for k, vv in zip(legend_labels, vals[key][b])})

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6))

    for row, (key, dec) in enumerate(METRICS):
        for col, bench in enumerate(BENCHMARKS):
            ax = axs[row][col]
            v = vals[key][bench]
            base_val = v[0]
            ymax = max(v) * 1.30
            ax.set_ylim(0, ymax)

            bars = ax.bar(x, v, width, color=colors, edgecolor="#555555",
                          linewidth=0.6, zorder=3)

            # value labels
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.{dec}f}",
                            (bar.get_x() + bar.get_width() / 2, h + ymax * 0.02),
                            ha="center", va="bottom", fontsize=10,
                            fontweight="bold", color=TEXT_COLOR)

            # delta vs base
            for j, bar in zip(range(1, n), bars[1:]):
                h = v[j]
                d = h - base_val
                colr = GAIN_COLOR if d >= 0 else LOSS_COLOR
                sign = "+" if d >= 0 else "-"
                ax.annotate(f"{sign}{abs(d):.{dec}f}",
                            (bar.get_x() + bar.get_width() / 2, h + ymax * 0.11),
                            ha="center", va="bottom", fontsize=10,
                            fontweight="bold", color=colr)
                ax.hlines(h, bar.get_x(), bar.get_x() + bar.get_width(),
                          color=colr, linewidth=1.2, linestyle="--", alpha=0.7, zorder=4)

            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, rotation=0)
            ax.set_title(f"{key} — {bench}", fontsize=12.5, fontweight="bold",
                         color=TEXT_COLOR)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    for row, (key, _dec) in enumerate(METRICS):
        axs[row][0].set_ylabel(f"{key} (%)")

    fig.suptitle("Qwen3-1.7B-Base — Base vs NTP vs OPMD (MegaMath-Web-Pro)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.legend(bars, legend_labels, loc="lower center", ncol=n, frameon=True,
               framealpha=0.9, edgecolor="#CCCCCC", fancybox=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    out_pdf = ROOT / "OPMD_graph/base_vs_ntp_vs_opmd_low_benchmarks.pdf"
    out_png = ROOT / "OPMD_graph/base_vs_ntp_vs_opmd_low_benchmarks.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    print("saved:", out_pdf)
    print("saved:", out_png)


if __name__ == "__main__":
    main()