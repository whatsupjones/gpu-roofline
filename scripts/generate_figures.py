#!/usr/bin/env python3
"""Generate publication-quality figures for the GPU waste study — v2."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --------------------------------------------------------------------------
# Global style — clean, modern, high contrast
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Noto Serif", "Georgia", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
})

OUTDIR = Path("D:/gpu-tools/figures")
OUTDIR.mkdir(exist_ok=True)

# Palette
BLUE = "#1e40af"
BLUE_LIGHT = "#3b82f6"
RED = "#b91c1c"
RED_LIGHT = "#ef4444"
GREEN = "#15803d"
GREEN_LIGHT = "#22c55e"
PURPLE = "#7e22ce"
PURPLE_LIGHT = "#a855f7"
ORANGE = "#c2410c"
GRAY = "#64748b"
GRAY_LIGHT = "#e2e8f0"
BG = "#fafbfc"

# Data
CATEGORIES_SHORT = ["Ghost\nAlloc.", "Contention\nSqueeze", "Provisioning\nOverhead",
                     "Burst-Sust.\nGap", "Straggler\nTax", "Over-\nsubscription"]
DETECT_ROOFLINE = [99.9, 100.0, 100.0, 56.5, 94.7, 100.0]
EFFECT_SIZES = [2.46, 8.55, 2.01, 0.73, 2.46, 3.97]
EFFECT_TYPES = ["d", "d", r"$d_z$", r"$d_z$", "d", "d"]
BUCKET_COLORS = [GREEN, PURPLE, PURPLE, PURPLE, GREEN, RED]
BUCKET_NAMES = ["A", "B", "B", "B", "A", "C"]


def save(fig, name):
    fig.savefig(OUTDIR / f"{name}.png", facecolor="white")
    fig.savefig(OUTDIR / f"{name}.svg", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------------------
# Figure 1: The Observability Gap
# --------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(CATEGORIES_SHORT))

    # Only show gpu-roofline bars — nvidia-smi/DCGM at 0% shown as annotation
    bars = ax.bar(x, DETECT_ROOFLINE, width=0.55, color=BLUE, edgecolor="white",
                  linewidth=1, zorder=3)

    # Value labels on top of each bar
    for bar, val in zip(bars, DETECT_ROOFLINE):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=BLUE)

    # Red zero line for nvidia-smi/DCGM
    ax.axhline(y=0, color=RED, linewidth=2, zorder=2)
    ax.text(len(CATEGORIES_SHORT) - 0.5, 3,
            "nvidia-smi & DCGM: 0% detection across all categories",
            ha="right", va="bottom", fontsize=7, color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef2f2", edgecolor=RED, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES_SHORT)
    ax.set_ylabel("Detection Rate (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Figure 1.  The Observability Gap\ngpu-roofline detection rate vs. nvidia-smi / DCGM (0%)")

    # Light grid
    ax.yaxis.grid(True, alpha=0.3, linestyle="-", color=GRAY_LIGHT, zorder=0)

    save(fig, "fig1_detection_rates")


# --------------------------------------------------------------------------
# Figure 2: Effect Sizes (horizontal lollipop)
# --------------------------------------------------------------------------
def fig2():
    cats = ["Ghost Allocations", "Contention Squeeze", "Provisioning Overhead",
            "Burst-Sustained Gap", "Straggler Tax", "Oversubscription"]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.patch.set_facecolor("white")

    y = np.arange(len(cats))[::-1]  # top to bottom

    # Horizontal bars
    for i, (val, color, etype) in enumerate(zip(EFFECT_SIZES, BUCKET_COLORS, EFFECT_TYPES)):
        ax.barh(y[i], val, height=0.5, color=color, edgecolor="white", linewidth=1, zorder=3)
        ax.text(val + 0.2, y[i], f"{etype} = {val:.2f}", va="center", fontsize=7, fontweight="bold")

    # Benchmark lines
    for thresh, label in [(0.2, "Small (0.2)"), (0.5, "Medium (0.5)"), (0.8, "Large (0.8)")]:
        ax.axvline(x=thresh, color=GRAY, linestyle=":", linewidth=1, alpha=0.5, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("Standardized Effect Size (Cohen's d or d_z)")
    ax.set_xlim(0, 10.5)
    ax.set_title("Figure 2.  Effect Sizes by Waste Category")

    ax.xaxis.grid(True, alpha=0.2, linestyle="-", color=GRAY_LIGHT, zorder=0)

    # Bucket legend
    patches = [
        mpatches.Patch(color=GREEN, label="Bucket A: Directly Recoverable"),
        mpatches.Patch(color=PURPLE, label="Bucket B: Decision Support"),
        mpatches.Patch(color=RED, label="Bucket C: Risk Prevention"),
    ]
    ax.legend(handles=patches, loc="lower right", framealpha=0.95, edgecolor=GRAY_LIGHT)

    save(fig, "fig2_effect_sizes")


# --------------------------------------------------------------------------
# Figure 3: Three-Bucket Summary (vertical stacked layout, not triptych)
# --------------------------------------------------------------------------
def fig3():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.patch.set_facecolor("white")

    # Data organized by bucket
    buckets = [
        ("A: DIRECTLY RECOVERABLE", GREEN, [
            ("Ghost Allocations", "512 MiB trapped per teardown", 99.9),
            ("Straggler Tax", "19.1% fleet throughput lost per bad GPU", 94.7),
        ]),
        ("B: DECISION SUPPORT", PURPLE, [
            ("Contention Squeeze", "50–75% BW loss at 2–4 tenants (time-sliced)", 100.0),
            ("Burst-Sustained Gap", "1.7% below spec (H100); up to 16% by workload", 56.5),
            ("Provisioning Overhead", "246 ms per MIG provision (negligible cost)", 100.0),
        ]),
        ("C: RISK PREVENTION", RED, [
            ("Oversubscription", "33% degradation at 1.5× overcommit; crash at 2×", 100.0),
        ]),
    ]

    y_pos = 0
    y_positions = []
    bar_colors = []
    bar_labels_left = []
    bar_labels_right = []
    bar_values = []
    group_spans = []

    for bucket_label, color, items in buckets:
        start_y = y_pos
        for name, desc, detect in items:
            y_positions.append(y_pos)
            bar_colors.append(color)
            bar_labels_left.append(name)
            bar_labels_right.append(desc)
            bar_values.append(detect)
            y_pos += 1
        group_spans.append((bucket_label, color, start_y, y_pos - 1))
        y_pos += 0.6  # gap between groups

    y_arr = np.array(y_positions)

    # Bars
    bars = ax.barh(y_arr, bar_values, height=0.6, color=bar_colors,
                   edgecolor="white", linewidth=1, zorder=3, alpha=0.85)

    # Detection rate labels
    for yp, val in zip(y_arr, bar_values):
        ax.text(val + 1, yp, f"{val:.1f}%", va="center", fontsize=7, fontweight="bold")

    # Impact description labels (right-aligned inside bar or to the right)
    for yp, desc, val in zip(y_arr, bar_labels_right, bar_values):
        x_text = min(val - 2, 55)
        if val > 30:
            ax.text(x_text, yp, desc, va="center", ha="right", fontsize=6,
                    color="white", fontstyle="italic", zorder=4)
        else:
            ax.text(val + 8, yp, desc, va="center", fontsize=6, color=GRAY)

    ax.set_yticks(y_arr)
    ax.set_yticklabels(bar_labels_left, fontsize=7, fontweight="bold")
    ax.set_xlabel("gpu-roofline Detection Rate (%)")
    ax.set_xlim(0, 125)
    ax.invert_yaxis()
    ax.set_title("Figure 3.  Operational Action Buckets — What Visibility Enables")

    # Bucket group labels on the right margin
    for label, color, y_start, y_end in group_spans:
        mid = (y_start + y_end) / 2
        ax.annotate(label, xy=(120, mid), fontsize=6, fontweight="bold", color=color,
                    va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, alpha=0.9))

    ax.xaxis.grid(True, alpha=0.2, linestyle="-", color=GRAY_LIGHT, zorder=0)

    save(fig, "fig3_three_buckets")


# --------------------------------------------------------------------------
# Figure 4: Straggler Fleet Impact
# --------------------------------------------------------------------------
def fig4():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.patch.set_facecolor("white")

    fleet_sizes = np.array([8, 16, 32, 64, 128, 256])
    tax_pct = 19.1
    wasted = (fleet_sizes - 1) * (tax_pct / 100)

    bars = ax.bar(range(len(fleet_sizes)), wasted, color=GREEN, edgecolor="white",
                  linewidth=1, width=0.6, zorder=3, alpha=0.85)

    for i, (bar, val, n) in enumerate(zip(bars, wasted, fleet_sizes)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=GREEN)

    ax.set_xticks(range(len(fleet_sizes)))
    ax.set_xticklabels([f"{n} GPUs" for n in fleet_sizes])
    ax.set_ylabel("GPU-Equivalents Wasted per Straggler")
    ax.set_xlabel("Training Fleet Size")
    ax.set_title(f"Figure 4.  Straggler Tax Scales with Fleet Size\n"
                 f"One degraded GPU (median tax = {tax_pct}%) blocks the entire fleet at sync barriers")

    ax.yaxis.grid(True, alpha=0.3, linestyle="-", color=GRAY_LIGHT, zorder=0)

    # Callout annotation
    ax.annotate(
        "At 128 GPUs:\none straggler wastes\n24.3 GPU-equivalents\nat every barrier",
        xy=(4, wasted[4]),
        xytext=(2.2, wasted[4] + 10),
        fontsize=7, color=GRAY,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2, connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GRAY_LIGHT, edgecolor=GRAY, alpha=0.8),
    )

    save(fig, "fig4_straggler_scaling")


# --------------------------------------------------------------------------
# Figure 5: Contention — Time-Sliced vs MIG
# --------------------------------------------------------------------------
def fig5():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.patch.set_facecolor("white")

    tenants = [1, 2, 3, 4]
    ts_bw = [100, 50, 33.3, 25]
    mig_bw = [100, 100, 100, 100]

    x = np.arange(len(tenants))
    width = 0.32

    bars_mig = ax.bar(x - width/2, mig_bw, width, label="MIG (hardware isolated)",
                      color=GREEN, edgecolor="white", linewidth=1, zorder=3, alpha=0.85)
    bars_ts = ax.bar(x + width/2, ts_bw, width, label="Time-Sliced (shared)",
                     color=PURPLE, edgecolor="white", linewidth=1, zorder=3, alpha=0.85)

    # Value labels
    for bar, val in zip(bars_ts, ts_bw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=7,
                fontweight="bold", color=PURPLE)

    # Loss arrows between MIG and time-sliced
    for i, (mig, ts) in enumerate(zip(mig_bw, ts_bw)):
        if ts < mig:
            mid_x = x[i] + 0.01
            ax.annotate("", xy=(mid_x, ts + 2), xytext=(mid_x, mig - 2),
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
            ax.text(mid_x + 0.22, (mig + ts) / 2, f"−{mig - ts:.0f}%",
                    fontsize=7, color=RED, fontweight="bold", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} tenant{'s' if n > 1 else ''}" for n in tenants])
    ax.set_ylabel("Per-Tenant Bandwidth (% of baseline)")
    ax.set_xlabel("Concurrent Tenants on One GPU")
    ax.set_ylim(0, 120)
    ax.set_title("Figure 5.  Contention Squeeze: Time-Sliced vs. MIG\n"
                 "This bandwidth degradation is invisible to nvidia-smi and DCGM")

    ax.yaxis.grid(True, alpha=0.3, linestyle="-", color=GRAY_LIGHT, zorder=0)

    # "Invisible" annotation
    ax.fill_between([-0.5, 3.5], 0, 25, alpha=0.06, color=RED, zorder=0)
    ax.text(3.4, 12, "Invisible zone:\nbelow 25% per-tenant\nbut GPU shows 100% utilized",
            fontsize=6, color=RED, ha="right", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor=RED_LIGHT, alpha=0.8))

    ax.legend(loc="upper right", framealpha=0.95, edgecolor=GRAY_LIGHT)

    save(fig, "fig5_contention_tenants")


# --------------------------------------------------------------------------
def main():
    print("Generating figures v2...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print(f"Done -> {OUTDIR}")


if __name__ == "__main__":
    main()
