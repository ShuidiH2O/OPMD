"""
Plot the NTP mid-training side experiment (side study for the OPMD paper).

Finding
-------
The same continual-pre-training corpus (MegaMath-Web-Pro, 1B tokens), trained
with standard next-token prediction (NTP) at identical scheduling
(lr=1e-5, effective batch=64, one epoch, 8192 ctx), has *opposite* effects on
base models of different capability, on BOTH benchmark held-out sets:

  * Llama-3.2-3B   (weaker base)    : gains     (MATH-500 +25.8pp, Minerva +9.9pp)
  * Qwen3-1.7B-Base (stronger base) : collapses (MATH-500 -58.4pp, Minerva -23.2pp)

Data sources (pass@1, 16 rollouts, t=0.7 / top-p=0.95).
Within each model family the base and NTP checkpoints use the same decoding
template (Llama: raw/no-chat-template; Qwen3: chat template), so within-model
deltas are protocol-consistent.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "svg.fonttype": "none",
})

BASE_COLOR = "#AEB6BF"
NTP_COLOR = "#356C9E"
GAIN_COLOR = "#1E7B4A"
LOSS_COLOR = "#B73239"
TEXT_COLOR = "#1A1A1A"

ROOT = Path("/mnt/cpfs/epic-user/zhengruize-20260612")

# model -> axis label
MODEL_LABEL = {
    "Llama-3.2-3B": "Llama-3.2-3B\n(raw decoding)",
    "Qwen3-1.7B-Base": "Qwen3-1.7B-Base\n(chat decoding)",
}

# (model, benchmark, base_summary, ntp_summary)
DATA = [
    ("Llama-3.2-3B", "MATH-500",
     ROOT / "OPMD/eval_results/grading_summary.json",
     ROOT / "OPMD/eval_results/ntp_llama_megamath/grading_summary.json"),
    ("Llama-3.2-3B", "Minerva",
     ROOT / "OPMD/eval_results/llama_3.2_3b_baseline/grading_summary.json",
     ROOT / "OPMD/eval_results/ntp_llama_megamath_minerva/grading_summary.json"),
    ("Qwen3-1.7B-Base", "MATH-500",
     ROOT / "OPMD/eval_results/qwen3_1.7b_baseline/grading_summary.json",
     ROOT / "NTP/eval_results/ntp_qwen3_1.7b_megamath/grading_summary.json"),
    ("Qwen3-1.7B-Base", "Minerva",
     ROOT / "OPMD/eval_results/qwen3_1.7b_baseline/grading_summary.json",
     ROOT / "NTP/eval_results/ntp_qwen3_1.7b_megamath/grading_summary.json"),
]

BENCHMARKS = ["MATH-500", "Minerva"]

KEY = {"MATH-500": "math-500", "Minerva": "minerva"}


def load_pass1(path: Path, bench: str) -> float:
    """Extract pass@1 (%) for a benchmark from a grading_summary.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    key = KEY[bench]
    for rec in data:
        stem = Path(rec["file"]).name.lower()
        if key in stem:
            return round(rec["pass@1"] * 100, 2)
    raise ValueError(f"no {bench} entry in {path}")


def main() -> None:
    # Load all numbers.
    values = {}  # (model, bench) -> (base, ntp)
    for model, bench, base_path, ntp_path in DATA:
        b = load_pass1(base_path, bench)
        n = load_pass1(ntp_path, bench)
        values[(model, bench)] = (b, n)
        print(f"{model:<18} {bench:<9} base={b:>6.2f}  ntp={n:>6.2f}  "
              f"delta={n - b:+.2f}pp")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    models = list(MODEL_LABEL.keys())
    width = 0.32
    x = list(range(len(models)))

    for ax, bench in zip(axes, BENCHMARKS):
        b_vals = [values[(m, bench)][0] for m in models]
        n_vals = [values[(m, bench)][1] for m in models]

        top = max(max(b_vals), max(n_vals))
        ymax = top * 1.62
        ax.set_ylim(0, ymax)

        bars_base = ax.bar(
            [i - width / 2 for i in x], b_vals, width,
            color=BASE_COLOR, edgecolor="#555555", linewidth=0.6,
            label="Base model", zorder=3,
        )
        bars_ntp = ax.bar(
            [i + width / 2 for i in x], n_vals, width,
            color=NTP_COLOR, edgecolor="#555555", linewidth=0.6,
            label="NTP mid-training\n(MegaMath-Web-Pro 1B)", zorder=3,
        )

        # Value labels on bars.
        for bars in (bars_base, bars_ntp):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + ymax * 0.02,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=TEXT_COLOR,
                )

        # Delta arrows above each group.
        for i, (b, n) in enumerate(zip(b_vals, n_vals)):
            d = n - b
            color = GAIN_COLOR if d >= 0 else LOSS_COLOR
            arrow = "\u2191" if d >= 0 else "\u2193"
            top_i = max(b, n)
            ax.annotate(
                f"{arrow} {d:+.1f} pp",
                xy=(i, top_i + ymax * 0.20), xycoords="data",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=color,
            )
            ax.plot(
                [i, i], [top_i + ymax * 0.02, top_i + ymax * 0.16],
                color=color, linewidth=1.2, linestyle="--", alpha=0.7, zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in models])
        ax.set_title(bench, fontsize=13, fontweight="bold", color=TEXT_COLOR)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("pass@1 (%)")

    fig.suptitle(
        "Standard NTP mid-training on the same math corpus:\n"
        "gains for the weaker base, collapse for the stronger base",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fancybox=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_pdf = ROOT / "OPMD_graph/ntp_base_vs_ntp_benchmarks.pdf"
    out_png = ROOT / "OPMD_graph/ntp_base_vs_ntp_benchmarks.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()