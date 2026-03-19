#!/usr/bin/env python3
"""Analyze GPU waste study simulation output and emit markdown tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires numpy, pandas, and scipy. "
        "Install them before running analyze_study.py."
    ) from exc


H100_VRAM_BYTES = 80 * 1024**3
GPU_HOURLY_RATE = 2.50

CATEGORY_ORDER = [
    "ghost_allocation",
    "contention_squeeze",
    "provisioning_overhead",
    "burst_sustained_gap",
    "straggler_tax",
    "oversubscription",
]

CATEGORY_LABELS = {
    "ghost_allocation": "Ghost allocations",
    "contention_squeeze": "Contention squeeze",
    "provisioning_overhead": "Provisioning overhead",
    "burst_sustained_gap": "Burst-to-sustained gap",
    "straggler_tax": "Straggler tax",
    "oversubscription": "Oversubscription",
}

SCALE_PARAMS = {
    "small_8gpu": {
        "gpu_count": 8,
        "teardowns_per_day": 5.0,
        "provisions_per_day": 5.0,
        "avg_tenants": 2.0,
        "active_tenant_hours_per_day": 16.0,
        "sustained_hours_per_day": 12.0,
        "training_job_hours": 8.0,
        "jobs_per_day": 1.0,
        "oversubscribed_hours_per_day": 4.0,
        "ghost_lifetime_hours": 24.0,
    },
    "medium_100gpu": {
        "gpu_count": 100,
        "teardowns_per_day": 20.0,
        "provisions_per_day": 20.0,
        "avg_tenants": 4.0,
        "active_tenant_hours_per_day": 20.0,
        "sustained_hours_per_day": 18.0,
        "training_job_hours": 16.0,
        "jobs_per_day": 2.0,
        "oversubscribed_hours_per_day": 12.0,
        "ghost_lifetime_hours": 24.0,
    },
    "large_10000gpu": {
        "gpu_count": 10_000,
        "teardowns_per_day": 50.0,
        "provisions_per_day": 50.0,
        "avg_tenants": 7.0,
        "active_tenant_hours_per_day": 22.0,
        "sustained_hours_per_day": 20.0,
        "training_job_hours": 20.0,
        "jobs_per_day": 4.0,
        "oversubscribed_hours_per_day": 16.0,
        "ghost_lifetime_hours": 12.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="docs/study-results/simulation-raw.json",
        help="Path to the raw JSON emitted by study_sim.",
    )
    parser.add_argument(
        "--output",
        default="docs/study-results/summary.md",
        help="Path to the markdown summary to write.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Bootstrap resamples for BCa confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    trials = pd.DataFrame(raw["trials"])
    if trials.empty:
        raise SystemExit(f"No trials found in {path}")
    return raw, trials


def category_samples(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "control" in set(frame["arm"]):
        treatment = frame.loc[frame["arm"] == "treatment", "primary_metric_value"].to_numpy(dtype=float)
        control = frame.loc[frame["arm"] == "control", "primary_metric_value"].to_numpy(dtype=float)
    else:
        treatment = frame["primary_metric_value"].to_numpy(dtype=float)
        control = frame["control_metric_value"].dropna().to_numpy(dtype=float)
    return treatment, control


def cohens_d(treatment: np.ndarray, control: np.ndarray) -> float:
    n1 = treatment.size
    n2 = control.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1 = treatment.var(ddof=1)
    var2 = control.var(ddof=1)
    pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled <= 0:
        diff = treatment.mean() - control.mean()
        return math.copysign(math.inf, diff) if diff else 0.0
    return (treatment.mean() - control.mean()) / math.sqrt(pooled)


def median_difference(sample_a: np.ndarray, sample_b: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.median(sample_a, axis=axis) - np.median(sample_b, axis=axis)


def bootstrap_ci(
    treatment: np.ndarray,
    control: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    point_estimate = float(np.median(treatment) - np.median(control))
    rng = np.random.default_rng(seed)
    n_treatment = treatment.size
    n_control = control.size
    batch_size = 128
    diffs: list[np.ndarray] = []

    completed = 0
    while completed < resamples:
        batch = min(batch_size, resamples - completed)
        treatment_idx = rng.integers(0, n_treatment, size=(batch, n_treatment))
        control_idx = rng.integers(0, n_control, size=(batch, n_control))
        treatment_medians = np.median(treatment[treatment_idx], axis=1)
        control_medians = np.median(control[control_idx], axis=1)
        diffs.append(treatment_medians - control_medians)
        completed += batch

    all_diffs = np.concatenate(diffs)
    ci_low, ci_high = np.quantile(all_diffs, [0.025, 0.975])
    return point_estimate, float(ci_low), float(ci_high)


def safe_pct(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.astype(bool).mean() * 100.0)


def format_number(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def format_pvalue(value: float) -> str:
    if value < 1e-300:
        return "<1e-300"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def build_stats_table(
    trials: pd.DataFrame,
    bootstrap_resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        frame = trials.loc[trials["category_name"] == category].copy()
        treatment, control = category_samples(frame)
        mw = mannwhitneyu(treatment, control, alternative="greater", method="auto")
        point_estimate, ci_low, ci_high = bootstrap_ci(
            treatment, control, bootstrap_resamples, seed + len(rows)
        )

        positive_events = frame.loc[frame["ground_truth_waste"] == True]

        rows.append(
            {
                "Category": CATEGORY_LABELS[category],
                "Treatment N": int(treatment.size),
                "Control N": int(control.size),
                "Metric": frame["primary_metric_name"].iloc[0],
                "Treatment Median": float(np.median(treatment)),
                "Control Median": float(np.median(control)),
                "Mann-Whitney U": float(mw.statistic),
                "p-value": float(mw.pvalue),
                "Cohen's d": float(cohens_d(treatment, control)),
                "Bootstrap Point": point_estimate,
                "CI Low": ci_low,
                "CI High": ci_high,
                "gpu-roofline Detect %": safe_pct(positive_events["gpu_roofline_detected"]),
                "nvidia-smi Detect %": safe_pct(positive_events["nvidia_smi_detected"]),
                "DCGM Detect %": safe_pct(positive_events["dcgm_detected"]),
            }
        )
    return pd.DataFrame(rows)


def extract_cost_inputs(trials: pd.DataFrame) -> dict[str, float]:
    inputs: dict[str, float] = {}

    ghost = trials[(trials["category_name"] == "ghost_allocation") & (trials["arm"] == "treatment")]
    inputs["ghost_rate"] = float((ghost["primary_metric_value"] > 1024 * 1024).mean())
    inputs["avg_ghost_frac"] = float(np.clip(np.median(ghost["primary_metric_value"]) / H100_VRAM_BYTES, 0.0, 1.0))

    contention = trials[
        (trials["category_name"] == "contention_squeeze") & (trials["arm"] == "treatment")
    ]
    inputs["contention_drop_pct"] = float(np.median(contention["primary_metric_value"]))

    provisioning = trials[trials["category_name"] == "provisioning_overhead"]
    inputs["avg_spin_up_secs"] = float(np.median(provisioning["primary_metric_value"]) / 1000.0)

    burst = trials[trials["category_name"] == "burst_sustained_gap"]
    inputs["burst_sustained_gap_pct"] = float(np.median(burst["primary_metric_value"]))

    straggler = trials[(trials["category_name"] == "straggler_tax") & (trials["arm"] == "treatment")]
    inputs["straggler_tax_pct"] = float(np.median(straggler["primary_metric_value"]))

    oversub = trials[(trials["category_name"] == "oversubscription") & (trials["arm"] == "treatment")]
    inputs["oversub_waste_pct"] = float(np.median(oversub["primary_metric_value"]))

    return inputs


def project_costs(inputs: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scale_name, params in SCALE_PARAMS.items():
        fleet_factor = (params["gpu_count"] - 1.0) / params["gpu_count"]
        w_ghost = (
            inputs["ghost_rate"]
            * inputs["avg_ghost_frac"]
            * params["teardowns_per_day"]
            * 365.0
            * params["ghost_lifetime_hours"]
            * GPU_HOURLY_RATE
        )
        w_contention = (
            (inputs["contention_drop_pct"] / 100.0)
            * params["active_tenant_hours_per_day"]
            * 365.0
            * GPU_HOURLY_RATE
            * (1.0 - 1.0 / params["avg_tenants"])
        )
        w_provisioning = (
            params["provisions_per_day"]
            * (inputs["avg_spin_up_secs"] / 3600.0)
            * GPU_HOURLY_RATE
            * 365.0
        )
        w_burst_gap = (
            (inputs["burst_sustained_gap_pct"] / 100.0)
            * params["sustained_hours_per_day"]
            * 365.0
            * GPU_HOURLY_RATE
        )
        w_straggler = (
            (inputs["straggler_tax_pct"] / 100.0)
            * params["training_job_hours"]
            * params["jobs_per_day"]
            * 365.0
            * fleet_factor
            * GPU_HOURLY_RATE
        )
        w_oversub = (
            (inputs["oversub_waste_pct"] / 100.0)
            * params["oversubscribed_hours_per_day"]
            * 365.0
            * GPU_HOURLY_RATE
        )
        total_per_gpu = w_ghost + w_contention + w_provisioning + w_burst_gap + w_straggler + w_oversub
        rows.append(
            {
                "Scale": scale_name,
                "GPUs": int(params["gpu_count"]),
                "Ghost": w_ghost,
                "Contention": w_contention,
                "Provisioning": w_provisioning,
                "Burst gap": w_burst_gap,
                "Straggler": w_straggler,
                "Oversubscription": w_oversub,
                "Total / GPU / year": total_per_gpu,
                "Total fleet / year": total_per_gpu * params["gpu_count"],
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, divider, *body])


def build_summary(
    raw: dict[str, Any],
    stats_table: pd.DataFrame,
    cost_table: pd.DataFrame,
    input_path: Path,
) -> str:
    stats_view = stats_table.copy()
    stats_view["Treatment Median"] = stats_view["Treatment Median"].map(lambda v: format_number(v, 2))
    stats_view["Control Median"] = stats_view["Control Median"].map(lambda v: format_number(v, 2))
    stats_view["Mann-Whitney U"] = stats_view["Mann-Whitney U"].map(lambda v: format_number(v, 0))
    stats_view["p-value"] = stats_view["p-value"].map(format_pvalue)
    stats_view["Cohen's d"] = stats_view["Cohen's d"].map(lambda v: format_number(v, 2))
    stats_view["95% CI"] = stats_view.apply(
        lambda row: f"[{format_number(row['CI Low'], 2)}, {format_number(row['CI High'], 2)}]",
        axis=1,
    )
    stats_view["gpu-roofline Detect %"] = stats_view["gpu-roofline Detect %"].map(lambda v: format_number(v, 1))
    stats_view["nvidia-smi Detect %"] = stats_view["nvidia-smi Detect %"].map(lambda v: format_number(v, 1))
    stats_view["DCGM Detect %"] = stats_view["DCGM Detect %"].map(lambda v: format_number(v, 1))
    stats_view = stats_view[
        [
            "Category",
            "Treatment N",
            "Control N",
            "Metric",
            "Treatment Median",
            "Control Median",
            "Mann-Whitney U",
            "p-value",
            "Cohen's d",
            "95% CI",
            "gpu-roofline Detect %",
            "nvidia-smi Detect %",
            "DCGM Detect %",
        ]
    ]

    cost_view = cost_table.copy()
    for column in [
        "Ghost",
        "Contention",
        "Provisioning",
        "Burst gap",
        "Straggler",
        "Oversubscription",
        "Total / GPU / year",
        "Total fleet / year",
    ]:
        cost_view[column] = cost_view[column].map(format_currency)

    return "\n".join(
        [
            "# The Hidden Cost of GPU Virtualization",
            "",
            "## Simulation Dataset",
            "",
            f"- Input: `{input_path.as_posix()}`",
            f"- Phase: `{raw['phase']}`",
            f"- Protocol version: `{raw['protocol_version']}`",
            f"- Seed: `{raw['seed']}`",
            f"- Total trials: `{raw['total_trials']}`",
            f"- Target trials/category: `{raw['target_trials_per_category']}`",
            "",
            "## Table 1. Category-Level Statistical Results",
            "",
            markdown_table(stats_view),
            "",
            "## Table 2. Annual Cost Projection at $2.50/GPU/hr",
            "",
            markdown_table(cost_view),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    raw, trials = load_dataset(input_path)
    stats_table = build_stats_table(trials, args.bootstrap_resamples, args.seed)
    cost_inputs = extract_cost_inputs(trials)
    cost_table = project_costs(cost_inputs)
    summary = build_summary(raw, stats_table, cost_table, input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
