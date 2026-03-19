#!/usr/bin/env python3
"""Generate publication-quality figures for the GPU waste study."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

OUTDIR = Path("D:/gpu-tools/figures")
OUTDIR.mkdir(exist_ok=True)

# Colors
C_ROOFLINE = "#2563eb"   # blue
C_NVIDIA = "#dc2626"     # red
C_DCGM = "#f97316"       # orange
C_BUCKET_A = "#16a34a"   # green
C_BUCKET_B = "#7c3aed"   # purple
C_BUCKET_C = "#dc2626"   # red
C_GRAY = "#94a3b8"

# --------------------------------------------------------------------------
# Data (from simulation results)
# --------------------------------------------------------------------------
CATEGORIES = [
    "Ghost\nAllocations",
    "Contention\nSqueeze",
    "Provisioning\nOverhead",
    "Burst-Sustained\nGap",
    "Straggler\nTax",
    "Over-\nsubscription",
]
SHORT_CATS = [
    "Ghost",
    "Contention",
    "Provisioning",
    "Burst-Sustained",
    "Straggler",
    "Oversubscription",
]

DETECT_ROOFLINE = [99.9, 100.0, 100.0, 56.5, 94.7, 100.0]
DETECT_NVIDIA = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DETECT_DCGM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

EFFECT_SIZES = [2.46, 8.55, 2.01, 0.73, 2.46, 3.97]
EFFECT_LABELS = ["d", "d", "d_z", "d_z", "d", "d"]
DESIGNS = ["indep", "indep", "paired", "paired", "indep", "indep"]

BUCKET_COLORS = [C_BUCKET_A, C_BUCKET_B, C_BUCKET_B, C_BUCKET_B, C_BUCKET_A, C_BUCKET_C]
BUCKET_LABELS = ["A: Recoverable", "B: Decision Support", "B: Decision Support",
                 "B: Decision Support", "A: Recoverable", "C: Risk Prevention"]

PER_EVENT = {
    "Ghost\nAllocations": "512 MiB\ntrapped",
    "Contention\nSqueeze": "66.7%\nBW loss",
    "Provisioning\nOverhead": "246 ms\nlatency",
    "Burst-Sustained\nGap": "1.7%\ngap",
    "Straggler\nTax": "19.1%\nfleet loss",
    "Over-\nsubscription": "33.3%\ndegradation",
}


# --------------------------------------------------------------------------
# Figure 1: Detection Rate Comparison
# --------------------------------------------------------------------------
def fig_detection():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(CATEGORIES))
    width = 0.28

    bars_nv = ax.bar(x - width, DETECT_NVIDIA, width, label="nvidia-smi",
                     color=C_NVIDIA, edgecolor="white", linewidth=0.5)
    bars_dc = ax.bar(x, DETECT_DCGM, width, label="DCGM",
                     color=C_NVIDIA, alpha=0.5, edgecolor="white", linewidth=0.5)
    bars_rf = ax.bar(x + width, DETECT_ROOFLINE, width, label="gpu-roofline",
                     color=C_ROOFLINE, edgecolor="white", linewidth=0.5)

    # Add value labels on gpu-roofline bars
    for bar, val in zip(bars_rf, DETECT_ROOFLINE):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=C_ROOFLINE)

    # Add "0%" labels on nvidia-smi bars
    for bar in bars_nv:
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                "0%", ha="center", va="bottom", fontsize=7, color=C_NVIDIA, fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Figure 1. Waste Event Detection by Monitoring Tool\n"
                 "(nvidia-smi and DCGM: 0% across all categories)")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Horizontal reference line at 0
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.savefig(OUTDIR / "fig1_detection_rates.png")
    fig.savefig(OUTDIR / "fig1_detection_rates.svg")
    plt.close(fig)
    print("  fig1_detection_rates")


# --------------------------------------------------------------------------
# Figure 2: Effect Sizes by Category (horizontal bar with bucket coloring)
# --------------------------------------------------------------------------
def fig_effect_sizes():
    fig, ax = plt.subplots(figsize=(8, 4))

    y = np.arange(len(SHORT_CATS))
    bars = ax.barh(y, EFFECT_SIZES, color=BUCKET_COLORS, edgecolor="white", linewidth=0.5, height=0.6)

    # Value labels
    for bar, val, label in zip(bars, EFFECT_SIZES, EFFECT_LABELS):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{label} = {val:.2f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(SHORT_CATS)
    ax.set_xlabel("Standardized Effect Size (Cohen's d or d_z)")
    ax.set_title("Figure 2. Effect Sizes Across Six Waste Categories\n"
                 "(colored by operational action bucket)")
    ax.set_xlim(0, 10.5)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reference lines for effect size benchmarks
    for threshold, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline(x=threshold, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.6)
        ax.text(threshold + 0.05, len(SHORT_CATS) - 0.3, label, fontsize=7, color=C_GRAY)

    # Legend
    patches = [
        mpatches.Patch(color=C_BUCKET_A, label="A: Directly Recoverable"),
        mpatches.Patch(color=C_BUCKET_B, label="B: Decision Support"),
        mpatches.Patch(color=C_BUCKET_C, label="C: Risk Prevention"),
    ]
    ax.legend(handles=patches, loc="lower right", framealpha=0.9)

    fig.savefig(OUTDIR / "fig2_effect_sizes.png")
    fig.savefig(OUTDIR / "fig2_effect_sizes.svg")
    plt.close(fig)
    print("  fig2_effect_sizes")


# --------------------------------------------------------------------------
# Figure 3: Three-Bucket Operational Model
# --------------------------------------------------------------------------
def fig_three_buckets():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))

    bucket_data = [
        {
            "title": "A: Directly Recoverable",
            "color": C_BUCKET_A,
            "categories": ["Ghost\nAllocations", "Straggler\nTax"],
            "magnitudes": [512, 19.1],
            "units": ["MiB/teardown", "% fleet loss"],
            "detect": [99.9, 94.7],
        },
        {
            "title": "B: Decision Support",
            "color": C_BUCKET_B,
            "categories": ["Contention\nSqueeze", "Burst-Sustained\nGap", "Provisioning\nOverhead"],
            "magnitudes": [66.7, 1.7, 246],
            "units": ["% BW loss", "% below spec", "ms latency"],
            "detect": [100.0, 56.5, 100.0],
        },
        {
            "title": "C: Risk Prevention",
            "color": C_BUCKET_C,
            "categories": ["Over-\nsubscription"],
            "magnitudes": [33.3],
            "units": ["% degradation"],
            "detect": [100.0],
        },
    ]

    for ax, bucket in zip(axes, bucket_data):
        n = len(bucket["categories"])
        y = np.arange(n)

        ax.barh(y, bucket["detect"], color=bucket["color"], alpha=0.8,
                edgecolor="white", linewidth=0.5, height=0.5)

        for i, (mag, unit, det) in enumerate(zip(bucket["magnitudes"], bucket["units"], bucket["detect"])):
            ax.text(det + 1, i, f"{det:.0f}%", va="center", fontsize=8, fontweight="bold")
            ax.text(2, i - 0.28, f"{mag} {unit}", va="center", fontsize=7, color="#555")

        ax.set_yticks(y)
        ax.set_yticklabels(bucket["categories"], fontsize=9)
        ax.set_xlim(0, 120)
        ax.set_xlabel("Detection Rate (%)", fontsize=8)
        ax.set_title(bucket["title"], fontsize=10, fontweight="bold", color=bucket["color"])
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Figure 3. Three Operational Action Buckets with Detection Rates and Per-Event Impact",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_three_buckets.png")
    fig.savefig(OUTDIR / "fig3_three_buckets.svg")
    plt.close(fig)
    print("  fig3_three_buckets")


# --------------------------------------------------------------------------
# Figure 4: Straggler Fleet Scaling
# --------------------------------------------------------------------------
def fig_straggler_scaling():
    fig, ax = plt.subplots(figsize=(7, 4))

    fleet_sizes = [8, 16, 32, 64, 128, 256]
    tax_pct = 19.1  # median straggler tax from simulation

    # Wasted GPU-hours per straggler event (N-1 GPUs idle * tax%)
    wasted_gpus = [(n - 1) * (tax_pct / 100) for n in fleet_sizes]

    bars = ax.bar(range(len(fleet_sizes)), wasted_gpus, color=C_BUCKET_A,
                  edgecolor="white", linewidth=0.5, width=0.6)

    for bar, val, n in zip(bars, wasted_gpus, fleet_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(fleet_sizes)))
    ax.set_xticklabels([str(n) for n in fleet_sizes])
    ax.set_xlabel("Fleet Size (GPUs)")
    ax.set_ylabel("Equivalent GPUs Wasted Per Straggler Event")
    ax.set_title("Figure 4. Straggler Tax Scales with Fleet Size\n"
                 f"(median straggler tax = {tax_pct}%, one degraded GPU)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation
    ax.annotate("At 128 GPUs, one bad GPU\nwastes 24 GPU-equivalents\nat every sync barrier",
                xy=(4, wasted_gpus[4]), xytext=(2.5, wasted_gpus[4] + 8),
                fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

    fig.savefig(OUTDIR / "fig4_straggler_scaling.png")
    fig.savefig(OUTDIR / "fig4_straggler_scaling.svg")
    plt.close(fig)
    print("  fig4_straggler_scaling")


# --------------------------------------------------------------------------
# Figure 5: Contention Bandwidth by Tenant Count
# --------------------------------------------------------------------------
def fig_contention_tenants():
    fig, ax = plt.subplots(figsize=(7, 4))

    tenants = [1, 2, 3, 4]
    # Time-sliced: each tenant gets ~1/N
    ts_bw = [100, 50, 33.3, 25]
    # MIG: each tenant gets dedicated share (no degradation)
    mig_bw = [100, 100, 100, 100]

    x = np.arange(len(tenants))
    width = 0.35

    ax.bar(x - width/2, ts_bw, width, label="Time-Sliced", color=C_BUCKET_B,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, mig_bw, width, label="MIG (hardware isolated)", color=C_BUCKET_A,
           edgecolor="white", linewidth=0.5)

    for i, (ts, mig) in enumerate(zip(ts_bw, mig_bw)):
        if ts < mig:
            ax.annotate(f"−{mig - ts:.0f}%", xy=(i - width/2, ts),
                        xytext=(i - width/2, ts + 5),
                        ha="center", fontsize=8, color=C_BUCKET_B, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} tenant{'s' if n > 1 else ''}" for n in tenants])
    ax.set_ylabel("Per-Tenant Bandwidth (% of baseline)")
    ax.set_xlabel("Number of Concurrent Tenants")
    ax.set_title("Figure 5. Contention Squeeze: Time-Sliced vs. MIG Partitioning\n"
                 "(visibility enables informed partitioning decisions)")
    ax.set_ylim(0, 120)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shade the "invisible zone"
    ax.fill_between([-0.5, 3.5], 0, 100, alpha=0.03, color=C_NVIDIA)
    ax.text(3.3, 8, "This degradation is invisible\nto nvidia-smi and DCGM",
            fontsize=7, color=C_NVIDIA, ha="right", fontstyle="italic")

    fig.savefig(OUTDIR / "fig5_contention_tenants.png")
    fig.savefig(OUTDIR / "fig5_contention_tenants.svg")
    plt.close(fig)
    print("  fig5_contention_tenants")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Generating figures...")
    fig_detection()
    fig_effect_sizes()
    fig_three_buckets()
    fig_straggler_scaling()
    fig_contention_tenants()
    print(f"Done. Output: {OUTDIR}")


if __name__ == "__main__":
    main()
