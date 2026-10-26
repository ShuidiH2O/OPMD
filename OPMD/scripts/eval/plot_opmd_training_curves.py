"""
NTP vs OPMD — Qwen3-1.7B-Base training curves (MegaMath-Web-Pro).

Layout: 2x2 grid.
  row 0: pass@1  (MATH-500 | Minerva)
  row 1: avg@k   (MATH-500 | Minerva)

X axis: raw training data volume, in M tokens.
    OPMD  = step * batch * (context + response) * RAW_RATIO
            context  = 391,939,200 tokens / 435,049 samples = 900.91 / sample
            response = 512 tokens / sample (max_response_length budget)
            high/random arm batch = 384 (= 64 mbs * 6 GPUs)
            low arm batch         = 256 (= 64 mbs * 4 GPUs)
            RAW_RATIO ≈ 1.627 (1B raw pool / full-epoch context+response)
    NTP   = step * 64 * 8192 (raw text, full epoch ≈ 1B tokens)
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
    "svg.fonttype": "none",
})

# --- token 换算 ---
N_SAMPLES = 435_049
TOTAL_CONTEXT = 391_939_200          # 实测 sum of context_length
RESP_LEN = 512                       # max_response_length 预算
CTX_PER_SAMPLE = TOTAL_CONTEXT / N_SAMPLES          # ≈ 900.91
TOKENS_PER_SAMPLE = CTX_PER_SAMPLE + RESP_LEN       # ≈ 1412.91

# 筛选前原始数据量对应：full epoch (context+response) ↔ 1B raw pool
FULL_EPOCH_TOKENS = N_SAMPLES * TOKENS_PER_SAMPLE     # ≈ 614.68M
RAW_POOL = 1_000_000_000
RAW_RATIO = RAW_POOL / FULL_EPOCH_TOKENS              # ≈ 1.627

HIGH_COLOR = "#1E7B4A"   # green (high-entropy arm)
RAND_COLOR = "#D99028"   # amber (random arm)
LOW_COLOR = "#8E44AD"    # purple (low-entropy arm)
NTP_COLOR = "#C0392B"    # red (NTP baseline)
TEXT_COLOR = "#1A1A1A"
ANNOT_COLOR = "#888888"

BENCHMARKS = ["MATH-500", "Minerva"]
METRICS = [("pass@1", "pass@1 (%)"), ("avg@k", "avg@k (%)")]

# 每臂：label, color, 数据目录名, batch, 要绘制的 step 列表
# step 0 表示未训练的 base（共享起点）
ARMS = [
    dict(label="OPMD (high-entropy)", color=HIGH_COLOR, dir="teacher",
         batch=384, steps=[0, 200, 400, 600, 800, 1000, 1132]),
    dict(label="OPMD (random)", color=RAND_COLOR, dir="teacher_rand",
         batch=384, steps=[0, 200, 400, 600, 800, 1000, 1132]),
    dict(label="OPMD (low-entropy)", color=LOW_COLOR, dir="teacher_low",
         batch=256, steps=[0, 300, 600, 900, 1200, 1500, 1699]),
    dict(label="NTP (MegaMath)", color=NTP_COLOR, dir="ntp",
         tokens_per_step=64 * 8192, steps=[0, 400, 800, 1200, 1600, 1902]),
]

BASE_SUMMARY = ROOT / "OPMD/eval_results/qwen3_1.7b_baseline/grading_summary.json"


def resolve_summary(arm_dir: str, step: int, bench: str) -> Path:
    if step == 0:
        return BASE_SUMMARY
    if arm_dir == "ntp":
        return ROOT / f"NTP/eval_results/ntp_qwen3_1.7b_megamath_rerun/step_{step}" / "grading_summary.json"
    base = ROOT / f"OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_{arm_dir}"
    if step == 1132:
        # 随机臂的 Minerva 结果存放在独立目录 step_1132_minerva
        if arm_dir == "teacher_rand" and bench == "Minerva":
            return base / "step_1132_minerva" / "grading_summary.json"
        return base / "step_1132" / "grading_summary.json"
    return base / f"step_{step}_math500_minerva" / "grading_summary.json"


def load_metrics(arm_dir: str, step: int, bench: str):
    path = resolve_summary(arm_dir, step, bench)
    data = json.loads(path.read_text(encoding="utf-8"))
    key = bench.lower()
    for rec in data:
        if key in Path(rec["file"]).name.lower():
            return rec["pass@1"] * 100.0, rec["avg@k"] * 100.0
    raise KeyError(f"{bench} not found in {path}")


def main():
    # 预取每臂各自的数据与 x（统一折算为 raw training tokens，单位 M）
    for arm in ARMS:
        if "tokens_per_step" in arm:
            arm["x"] = [s * arm["tokens_per_step"] / 1e6 for s in arm["steps"]]
        else:
            arm["x"] = [s * arm["batch"] * TOKENS_PER_SAMPLE * RAW_RATIO / 1e6 for s in arm["steps"]]
        arm["data"] = {}
        for bench in BENCHMARKS:
            p1, ak = [], []
            for s in arm["steps"]:
                pp, aa = load_metrics(arm["dir"], s, bench)
                p1.append(pp)
                ak.append(aa)
            arm["data"][bench] = {"pass@1": p1, "avg@k": ak}

    # 打印校验
    for arm in ARMS:
        for bench in BENCHMARKS:
            d = arm["data"][bench]
            print(f"{arm['label']:<20} {bench:<9} "
                  f"pass1={[round(v,2) for v in d['pass@1']]} "
                  f"avgk={[round(v,2) for v in d['avg@k']]}")

    # x 轴范围（所有底轴共享）
    all_x = [v for arm in ARMS for v in arm["x"]]
    x_lo = 0
    x_hi = max(all_x) + (max(all_x) - x_lo) * 0.02

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), sharex=True)

    for row, (mkey, _mlabel) in enumerate(METRICS):
        for col, bench in enumerate(BENCHMARKS):
            ax = axes[row][col]

            for arm in ARMS:
                y = arm["data"][bench][mkey]
                ax.plot(arm["x"], y, marker="o", markersize=4.5, linewidth=1.7,
                        color=arm["color"], label=arm["label"], zorder=3)

            # y 轴收紧到数据范围（先算，供标注定位）
            all_y = [v for arm in ARMS for v in arm["data"][bench][mkey]]
            ymin, ymax = min(all_y), max(all_y)
            rng = ymax - ymin
            ax.set_ylim(ymin - rng * 0.05, ymax + rng * 0.12)

            # base 标注
            base_val = ARMS[0]["data"][bench][mkey][0]
            ax.annotate("Base (untrained)", xy=(0, base_val), xycoords="data",
                        xytext=(12, 14), textcoords="offset points",
                        fontsize=8.5, color="#555555", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=ANNOT_COLOR, lw=0.6))

            # final step 标注：一条注释 + 三条细线分别连到三个终点
            finals = [(arm["x"][-1], arm["data"][bench][mkey][-1]) for arm in ARMS]
            text_x = x_hi - (x_hi - x_lo) * 0.26
            text_y = ymax + rng * 0.06
            ax.text(text_x, text_y, "final step (1 epoch)",
                    ha="left", va="bottom", fontsize=8.5,
                    color="#555555", fontweight="bold")
            for px, py in finals:
                ax.plot([text_x, px], [text_y, py], color=ANNOT_COLOR,
                        linewidth=0.6, alpha=0.85, zorder=2)

            ax.set_title(bench, loc="left", fontsize=12.5, fontweight="bold",
                         color=TEXT_COLOR)
            ax.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)

    # 左列 y 标签
    axes[0][0].set_ylabel("pass@1 (%)")
    axes[1][0].set_ylabel("avg@k (%)")

    # 底轴标签 + 显式范围
    for col in range(2):
        axes[1][col].set_xlabel("Raw training data (M tokens)")
    for ax in axes.flat:
        ax.set_xlim(x_lo, x_hi)

    fig.suptitle(
        "Qwen3-1.7B-Base — NTP vs OPMD training curves (MegaMath-Web-Pro)\n"
        "x = raw training data volume (OPMD pre-filter pool ≈ NTP 1B tokens)",
        fontsize=12.5, fontweight="bold", color=TEXT_COLOR, y=1.00,
    )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
               fancybox=False, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    out_pdf = ROOT / "OPMD_graph/opmd_training_curves.pdf"
    out_png = ROOT / "OPMD_graph/opmd_training_curves.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:")
    print(" ", out_pdf)
    print(" ", out_png)


if __name__ == "__main__":
    main()