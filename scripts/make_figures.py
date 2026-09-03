#!/usr/bin/env python3
"""
Generate report figures from the measured result CSVs:
  - figures/speedups.png   : per-shape median speedup, P100 vs T4 side by side
  - figures/memory_wall.png: baseline attention-score memory per shape (log scale),
                             showing why shape 14 (~20.5 TB) is infeasible

Usage: python scripts/make_figures.py
Requires matplotlib (preinstalled on Colab/Kaggle; `pip install matplotlib` locally).
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_P100 = os.path.join(HERE, "results", "results.csv")
CSV_T4 = os.path.join(HERE, "results", "results_t4.csv")
CSV_T4_FP32 = os.path.join(HERE, "results", "results_t4_fp32.csv")
FIG = os.path.join(HERE, "report", "figures")

# Palette roles. Categorical slots carry series identity; ink carries text; the
# status red is reserved for the one bar that means "cannot run". Both modes are
# selected against their own surface, not flipped from each other.
THEMES = {
    "light": dict(SERIES_1="#2a78d6", SERIES_2="#eb6834", SERIES_3="#b9b7ae", CRITICAL="#d03b3b",
                  SURFACE="#fcfcfb", INK="#0b0b0b", INK_2="#52514e",
                  MUTED="#898781", GRID="#e1e0d9", AXIS="#c3c2b7"),
    "dark":  dict(SERIES_1="#3987e5", SERIES_2="#d95926", SERIES_3="#6c7280", CRITICAL="#e66767",
                  SURFACE="#12171e", INK="#ffffff", INK_2="#c3c2b7",
                  MUTED="#898781", GRID="#2c323b", AXIS="#3d444e"),
}
SERIES_1 = SERIES_2 = SERIES_3 = CRITICAL = SURFACE = INK = INK_2 = MUTED = GRID = AXIS = None


def use_theme(name):
    globals().update(THEMES[name])

# [B, d_model, heads, seq] for the memory-wall chart.
SHAPES = {
    1: (64, 128, 4, 128), 2: (1, 128, 4, 128), 3: (4, 128, 4, 128),
    4: (16, 128, 4, 128), 5: (128, 128, 4, 128), 6: (10000, 128, 4, 128),
    7: (64, 32, 4, 128), 8: (64, 1024, 4, 128), 9: (64, 128, 1, 128),
    10: (64, 128, 2, 128), 11: (64, 128, 16, 128), 12: (64, 128, 4, 32),
    13: (64, 128, 4, 1024), 14: (32, 1024, 16, 100000),
}


def style_axes(ax):
    """Recessive chrome: hairline horizontal grid, no box, muted tick labels."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def read_speedups(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["pass"] == "PASS" and r["speedup"]:
                out[int(r["shape"])] = float(r["speedup"])
    return out


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def speedup_chart():
    p100 = read_speedups(CSV_P100)
    t4 = read_speedups(CSV_T4)
    t4_fp32 = read_speedups(CSV_T4_FP32) if os.path.exists(CSV_T4_FP32) else {}
    ids = sorted(set(p100) | set(t4))
    if not ids:
        print("no PASS speedups yet; skipping speedups.png")
        return

    # (label, short name for the title, data, colour); the shipped series is
    # the saturated one, the fp32-SGEMM reference sits in a muted tone behind it.
    series = [("Tesla P100 (sm_60) - SDPA only, fp32 SGEMM", "P100", p100, SERIES_1)]
    if t4_fp32:
        series.append(("Tesla T4 (sm_75) - fp32 SGEMM (T3_LINEAR=fp32)", "T4 fp32", t4_fp32, SERIES_3))
    if t4:
        series.append(("Tesla T4 (sm_75) - shipped: fp16x3 tensor-core GEMMs", "T4 shipped", t4, SERIES_2))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    n = len(series)
    # 2px-equivalent gap between adjacent bars: total group width < 1 slot.
    group_w = 0.78
    bar_w = group_w / n
    for si, (label, short, data, color) in enumerate(series):
        xs, ys = [], []
        for k, i in enumerate(ids):
            if i in data:
                xs.append(k - group_w / 2 + bar_w * (si + 0.5))
                ys.append(data[i])
        ax.bar(xs, ys, width=bar_w * 0.92, color=color, label=label, zorder=2)

    # Selective direct labels only: each series' best shape, never every bar.
    for label, short, data, color in series:
        if not data or color == SERIES_3:
            continue   # the muted reference series carries no direct label
        best = max(data, key=data.get)
        k = ids.index(best)
        si = [s[0] for s in series].index(label)
        x = k - group_w / 2 + bar_w * (si + 0.5)
        ax.annotate(f"{data[best]:.3f}x", (x, data[best]),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=9, color=INK, fontweight="bold")

    # Parity line. Deliberately unlabelled: at y=1.0 every horizontal position
    # is already occupied by a bar, and the y-axis makes it self-evident.
    ax.axhline(1.0, color=AXIS, ls="--", lw=1.0, zorder=1)

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([str(i) for i in ids])
    ax.set_xlabel("shape #", color=INK_2, fontsize=10)
    ax.set_ylabel("median speedup vs baseline", color=INK_2, fontsize=10)

    med_bits = []
    for label, short, data, _ in series:
        if data:
            med_bits.append(f"{short} median {median(data.values()):.3f}x")
    # Two lines: the medians alone overflow a single title line with three series.
    # Two lines: three series of medians overflow a single title line.
    ax.set_title("Optimized Transformer speedup, 13/13 shapes PASS, three runs per cell"
                 + "\nmedians:  " + "  |  ".join(med_bits),
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "speedups.png"), dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote speedups.png (%d series, %d shapes)" % (len(series), len(ids)))


def memory_wall_chart():
    ids = sorted(SHAPES)
    gb = []
    for i in ids:
        b, d, h, s = SHAPES[i]
        gb.append(b * h * s * s * 4 / 1e9)  # [B,H,S,S] fp32 bytes -> GB

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    colors = [CRITICAL if i == 14 else SERIES_1 for i in ids]
    ax.bar([str(i) for i in ids], gb, color=colors, width=0.7, zorder=2)
    ax.axhline(16, color=INK_2, ls="--", lw=1.2, zorder=3)
    # Anchored left, where the short bars leave the band empty; on the right it
    # collided with the shape-14 bar.
    ax.annotate("16 GB - what a free T4/P100 has",
                (-0.4, 16), textcoords="offset points",
                xytext=(0, 5), ha="left", fontsize=9, color=INK_2)
    ax.set_yscale("log")

    # One direct label, on the bar that is the whole point of the chart.
    ax.annotate("shape 14: 20,480 GB\n= 20.5 TB, cannot run",
                (12.6, gb[-1]), textcoords="offset points", xytext=(0, 8),
                ha="right", fontsize=9, color=CRITICAL, fontweight="bold")

    ax.set_ylim(top=gb[-1] * 60)
    ax.set_xlabel("shape #", color=INK_2, fontsize=10)
    ax.set_ylabel("baseline attention-score memory (GB, log)",
                  color=INK_2, fontsize=10)
    ax.set_title("The memory wall: the baseline materializes an [B,H,S,S] score matrix",
                 color=INK, fontsize=11, loc="left", pad=12)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "memory_wall.png"), dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote memory_wall.png (shape 14 = %.0f GB)" % gb[ids.index(14)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="light", choices=sorted(THEMES))
    ap.add_argument("--outdir", default=None,
                    help="where to write the PNGs (default report/figures)")
    args = ap.parse_args()

    global FIG
    use_theme(args.theme)
    if args.outdir:
        FIG = args.outdir if os.path.isabs(args.outdir) else os.path.join(HERE, args.outdir)
    os.makedirs(FIG, exist_ok=True)
    memory_wall_chart()   # does not need any results file
    speedup_chart()


if __name__ == "__main__":
    main()
