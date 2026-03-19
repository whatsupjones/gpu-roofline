#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import (
    binomtest,
    friedmanchisquare,
    kruskal,
    linregress,
    mannwhitneyu,
    ttest_rel,
    wilcoxon,
)

GPU_HOURLY_RATE = 2.50
H100_VRAM_BYTES = 80 * 1024**3

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

TOOL_COLUMNS = {
    "gpu-roofline": "gpu_roofline_detected",
    "nvidia-smi": "nvidia_smi_detected",
    "DCGM": "dcgm_detected",
}

SOURCE_ARTIFACTS = [
    Path("docs/study-protocol-gpu-waste.md"),
    Path("docs/study-simulation-manuscript.md"),
    Path("docs/study-submission-plan.md"),
    Path("crates/gpu-harness/src/bin/study_sim.rs"),
    Path("crates/gpu-harness/src/study/mod.rs"),
    Path("crates/gpu-harness/src/study/runner.rs"),
    Path("crates/gpu-harness/src/study/noise.rs"),
    Path("crates/gpu-harness/src/study/stats.rs"),
    Path("crates/gpu-harness/src/study/cost_model.rs"),
    Path("crates/gpu-harness/src/study/scenarios.rs"),
]

SCALE_PARAMS = {
    "small_8gpu": {
        "label": "8-GPU Fleet",
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
        "label": "100-GPU Fleet",
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
        "label": "10,000-GPU Fleet",
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


@dataclass
class Layout:
    root: Path
    raw_dir: Path
    derived_dir: Path
    metadata_dir: Path
    source_dir: Path
    raw_json: Path
    summary: Path
    supplement: Path
    analysis_json: Path
    provenance_json: Path
    provenance_md: Path
    readme: Path
    sha256: Path
    source_manifest: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the GPU waste simulation dataset and build an archival study bundle.")
    parser.add_argument("--input", required=True, help="Path to the raw simulation JSON.")
    parser.add_argument("--output-root", required=True, help="Root directory for the archival package.")
    parser.add_argument("--repo-root", default=".", help="Path to the source repository for metadata capture.")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--cost-bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def layout(output_root: Path) -> Layout:
    return Layout(
        root=output_root,
        raw_dir=output_root / "raw",
        derived_dir=output_root / "derived",
        metadata_dir=output_root / "metadata",
        source_dir=output_root / "source_snapshot",
        raw_json=output_root / "raw" / "simulation-raw.json",
        summary=output_root / "derived" / "summary.md",
        supplement=output_root / "derived" / "supplement.md",
        analysis_json=output_root / "derived" / "analysis-results.json",
        provenance_json=output_root / "derived" / "provenance.json",
        provenance_md=output_root / "derived" / "PROVENANCE.md",
        readme=output_root / "README.md",
        sha256=output_root / "SHA256SUMS.txt",
        source_manifest=output_root / "metadata" / "source-manifest.json",
    )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    rel = path.relative_to(relative_to).as_posix() if relative_to is not None else path.as_posix()
    return {"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def format_number(value: float | str, digits: int = 2) -> str:
    if value == "":
        return ""
    if value is None:
        return "NA"
    value = float(value)
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def format_pvalue(value: float | str) -> str:
    if value == "":
        return ""
    value = float(value)
    if math.isnan(value):
        return "NA"
    if value < 1e-300:
        return "<1e-300"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def format_currency(value: float | str) -> str:
    if value == "":
        return ""
    return f"${float(value):,.0f}"


def mean_of(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def median_of(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def sd_of(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def iqr_of(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(q3 - q1)


def cohens_d(treatment: np.ndarray, control: np.ndarray) -> float:
    if treatment.size < 2 or control.size < 2:
        return float("nan")
    var_t = np.var(treatment, ddof=1)
    var_c = np.var(control, ddof=1)
    pooled = ((treatment.size - 1) * var_t + (control.size - 1) * var_c) / (treatment.size + control.size - 2)
    diff = float(np.mean(treatment) - np.mean(control))
    if pooled <= 0:
        return 0.0 if diff == 0 else math.copysign(math.inf, diff)
    return diff / math.sqrt(pooled)


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = np.std(diff, ddof=1)
    mean_diff = float(np.mean(diff))
    if sd <= 0:
        return 0.0 if mean_diff == 0 else math.copysign(math.inf, mean_diff)
    return mean_diff / float(sd)


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    return (2.0 * u_stat) / (n1 * n2) - 1.0


def paired_rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    abs_diff = np.abs(diff)
    ranks = pd.Series(abs_diff).rank(method="average").to_numpy()
    pos = float(np.sum(ranks[diff > 0]))
    neg = float(np.sum(ranks[diff < 0]))
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_val) in enumerate(indexed, start=1):
        candidate = (m - rank + 1) * p_val
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_median_diff(treatment: np.ndarray, control: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    point = float(np.median(treatment) - np.median(control))
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    batch_size = 128
    completed = 0
    while completed < resamples:
        batch = min(batch_size, resamples - completed)
        t_idx = rng.integers(0, treatment.size, size=(batch, treatment.size))
        c_idx = rng.integers(0, control.size, size=(batch, control.size))
        diffs.append(np.median(treatment[t_idx], axis=1) - np.median(control[c_idx], axis=1))
        completed += batch
    all_diffs = np.concatenate(diffs)
    ci_low, ci_high = np.quantile(all_diffs, [0.025, 0.975])
    return point, float(ci_low), float(ci_high)


def parse_condition_tokens(condition: str) -> dict[str, Any]:
    tokens: dict[str, Any] = {}
    if condition.startswith("ratio"):
        left, right = condition.split("_instances")
        tokens["overcommit_ratio"] = float(left.replace("ratio", ""))
        tokens["instance_count"] = int(right)
    elif condition.startswith("control_fleet"):
        tokens["fleet_size"] = int(condition.replace("control_fleet", ""))
        tokens["severity"] = 0
        tokens["degradation_type"] = "none"
    elif "_fleet" in condition and "_sev" in condition:
        prefix, fleet = condition.rsplit("_fleet", 1)
        degrade, severity = prefix.rsplit("_sev", 1)
        tokens["fleet_size"] = int(fleet)
        tokens["severity"] = int(severity)
        tokens["degradation_type"] = degrade
    return tokens


def load_dataset(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(raw["trials"])
    if frame.empty:
        raise SystemExit(f"No trials found in {path}")
    indep = pd.json_normalize(frame["independent_variables"]).add_prefix("iv.")
    dep = pd.json_normalize(frame["dependent_variables"]).add_prefix("dv.")
    frame = pd.concat([frame.drop(columns=["independent_variables", "dependent_variables"]), indep, dep], axis=1)
    parsed = pd.DataFrame([parse_condition_tokens(value) for value in frame["condition"]])
    for column in parsed.columns:
        frame[f"parsed.{column}"] = parsed[column]
    return raw, frame


def condition_frame(trials: pd.DataFrame, category: str, arm: str | None = None) -> pd.DataFrame:
    frame = trials.loc[trials["category_name"] == category].copy()
    if arm is not None:
        frame = frame.loc[frame["arm"] == arm].copy()
    return frame.sort_values("trial_id").reset_index(drop=True)


def treatment_control_samples(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "control" in set(frame["arm"]):
        treatment = frame.loc[frame["arm"] == "treatment", "primary_metric_value"].to_numpy(float)
        control = frame.loc[frame["arm"] == "control", "primary_metric_value"].to_numpy(float)
    else:
        treatment = frame["primary_metric_value"].to_numpy(float)
        control = frame["control_metric_value"].dropna().to_numpy(float)
    return treatment, control

def bootstrap_paired_median_diff(a: np.ndarray, b: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap 95% CI for the median of paired differences."""
    diffs = a - b
    point = float(np.median(diffs))
    rng = np.random.default_rng(seed)
    boot_medians: list[np.ndarray] = []
    batch_size = 128
    completed = 0
    while completed < resamples:
        batch = min(batch_size, resamples - completed)
        idx = rng.integers(0, diffs.size, size=(batch, diffs.size))
        boot_medians.append(np.median(diffs[idx], axis=1))
        completed += batch
    all_medians = np.concatenate(boot_medians)
    ci_low, ci_high = np.quantile(all_medians, [0.025, 0.975])
    return point, float(ci_low), float(ci_high)


# Categories whose simulation design is paired (each trial produces both
# the real measurement and the tool-reported / ideal baseline value).
PAIRED_CATEGORIES = {"provisioning_overhead", "burst_sustained_gap"}


def omnibus_stats(trials: pd.DataFrame, bootstrap_resamples: int, seed: int) -> pd.DataFrame:
    """Build the headline cross-category table using the correct test per design.

    Independent-group categories (1, 2, 5, 6) → Mann-Whitney U, rank-biserial r, Cohen's d.
    Paired categories (3, 4)                  → Wilcoxon signed-rank, matched-pairs r, Cohen's d_z.
    All 6 p-values are then Holm-Bonferroni corrected for multiplicity.
    """
    rows: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []

    for idx, category in enumerate(CATEGORY_ORDER):
        frame = condition_frame(trials, category)
        is_paired = category in PAIRED_CATEGORIES

        if is_paired:
            # Paired design: primary_metric_value vs control_metric_value within each trial
            a = frame["primary_metric_value"].to_numpy(float)
            b = frame["control_metric_value"].to_numpy(float)
            stat_result = wilcoxon(a, b, alternative="greater")
            p_val = float(stat_result.pvalue)
            statistic = float(stat_result.statistic)
            test_name = "Wilcoxon signed-rank"
            effect = paired_cohens_d(a, b)
            r_effect = paired_rank_biserial(a, b)
            point, ci_low, ci_high = bootstrap_paired_median_diff(
                a, b, bootstrap_resamples, seed + idx
            )
            n_a = int(a.size)
            n_b = int(b.size)
            med_a = median_of(a)
            med_b = median_of(b)
        else:
            # Independent groups: treatment arm vs control arm
            treatment, control = treatment_control_samples(frame)
            stat_result = mannwhitneyu(treatment, control, alternative="greater", method="auto")
            p_val = float(stat_result.pvalue)
            statistic = float(stat_result.statistic)
            test_name = "Mann-Whitney U"
            effect = cohens_d(treatment, control)
            r_effect = rank_biserial_from_u(statistic, treatment.size, control.size)
            point, ci_low, ci_high = bootstrap_median_diff(
                treatment, control, bootstrap_resamples, seed + idx
            )
            n_a = int(treatment.size)
            n_b = int(control.size)
            med_a = median_of(treatment)
            med_b = median_of(control)

        raw_pvalues.append(p_val)
        rows.append(
            {
                "Category": CATEGORY_LABELS[category],
                "N (a)": n_a,
                "N (b)": n_b,
                "Design": "paired" if is_paired else "independent",
                "Primary Test": test_name,
                "Metric": frame["primary_metric_name"].iloc[0],
                "Median (a)": med_a,
                "Median (b)": med_b,
                "Statistic": statistic,
                "Raw p-value": p_val,
                "Holm p-value": 0.0,  # filled below
                "Cohen's d / d_z": float(effect),
                "Rank-biserial r": float(r_effect),
                "95% CI Low": ci_low,
                "95% CI High": ci_high,
                "Bootstrap Point": point,
            }
        )

    # Apply Holm-Bonferroni across all 6 omnibus p-values
    adjusted = holm_adjust(raw_pvalues)
    for row, adj_p in zip(rows, adjusted):
        row["Holm p-value"] = adj_p

    return pd.DataFrame(rows)


def build_detection_table(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for category in [*CATEGORY_ORDER, "overall"]:
        frame = trials if category == "overall" else condition_frame(trials, category)
        label = "Overall" if category == "overall" else CATEGORY_LABELS[category]
        gt = frame["ground_truth_waste"].astype(bool).to_numpy()
        roof = frame[TOOL_COLUMNS["gpu-roofline"]].astype(bool).to_numpy()
        smi = frame[TOOL_COLUMNS["nvidia-smi"]].astype(bool).to_numpy()
        dcgm = frame[TOOL_COLUMNS["DCGM"]].astype(bool).to_numpy()
        row: dict[str, Any] = {"Category": label}
        for tool_name, col in TOOL_COLUMNS.items():
            pred = frame[col].astype(bool).to_numpy()
            tp = int(np.sum(gt & pred))
            fp = int(np.sum((~gt) & pred))
            tn = int(np.sum((~gt) & (~pred)))
            fn = int(np.sum(gt & (~pred)))
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
            row[f"{tool_name} Detection Rate"] = sensitivity * 100.0
            confusion_rows.append(
                {
                    "Category": label,
                    "Tool": tool_name,
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                    "Sensitivity %": sensitivity * 100.0,
                    "Specificity %": specificity * 100.0,
                    "Precision %": precision * 100.0,
                    "F1": f1,
                }
            )
        for compare_name, left in [("smi", smi), ("dcgm", dcgm)]:
            b = int(np.sum(left & (~roof)))
            c = int(np.sum((~left) & roof))
            p_val = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) else 1.0
            row[f"McNemar p ({compare_name} vs roofline)"] = p_val
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


def extract_cost_inputs(trials: pd.DataFrame) -> dict[str, float]:
    ghost = condition_frame(trials, "ghost_allocation", "treatment")
    contention = condition_frame(trials, "contention_squeeze", "treatment")
    provisioning = condition_frame(trials, "provisioning_overhead")
    burst = condition_frame(trials, "burst_sustained_gap")
    straggler = condition_frame(trials, "straggler_tax", "treatment")
    oversub = condition_frame(trials, "oversubscription", "treatment")
    return {
        "ghost_rate": float((ghost["primary_metric_value"] > 1024 * 1024).mean()),
        "avg_ghost_frac": float(np.clip(np.median(ghost["primary_metric_value"]) / H100_VRAM_BYTES, 0.0, 1.0)),
        "contention_drop_pct": float(np.median(contention["primary_metric_value"])),
        "avg_spin_up_secs": float(np.median(provisioning["primary_metric_value"] - provisioning["control_metric_value"]) / 1000.0),
        "burst_sustained_gap_pct": float(np.median(burst["primary_metric_value"])),
        "straggler_tax_pct": float(np.median(straggler["primary_metric_value"])),
        "oversub_waste_pct": float(np.median(oversub["primary_metric_value"])),
    }


def project_scale(scale_name: str, inputs: dict[str, float]) -> dict[str, float]:
    params = SCALE_PARAMS[scale_name]
    fleet_factor = (params["gpu_count"] - 1.0) / params["gpu_count"]
    ghost = inputs["ghost_rate"] * inputs["avg_ghost_frac"] * params["teardowns_per_day"] * 365.0 * params["ghost_lifetime_hours"] * GPU_HOURLY_RATE
    contention = (inputs["contention_drop_pct"] / 100.0) * params["active_tenant_hours_per_day"] * 365.0 * GPU_HOURLY_RATE * (1.0 - 1.0 / params["avg_tenants"])
    provisioning = params["provisions_per_day"] * inputs["avg_spin_up_secs"] / 3600.0 * GPU_HOURLY_RATE * 365.0
    burst = (inputs["burst_sustained_gap_pct"] / 100.0) * params["sustained_hours_per_day"] * 365.0 * GPU_HOURLY_RATE
    straggler = (inputs["straggler_tax_pct"] / 100.0) * params["training_job_hours"] * params["jobs_per_day"] * 365.0 * fleet_factor * GPU_HOURLY_RATE
    oversub = (inputs["oversub_waste_pct"] / 100.0) * params["oversubscribed_hours_per_day"] * 365.0 * GPU_HOURLY_RATE
    return {
        "ghost": ghost,
        "contention": contention,
        "provisioning": provisioning,
        "burst": burst,
        "straggler": straggler,
        "oversubscription": oversub,
        "total_per_gpu": ghost + contention + provisioning + burst + straggler + oversub,
        "total_fleet": (ghost + contention + provisioning + burst + straggler + oversub) * params["gpu_count"],
    }


def waste_per_event(category: str, trials: pd.DataFrame) -> str:
    frame = condition_frame(trials, category)
    treatment, control = treatment_control_samples(frame)
    delta = float(np.median(treatment) - np.median(control))
    if category == "ghost_allocation":
        return f"{delta / 1024**2:.1f} MiB"
    if category == "provisioning_overhead":
        return f"{delta:.2f} ms"
    return f"{delta:.2f}%"


def events_per_day_medium(category: str) -> float:
    medium = SCALE_PARAMS["medium_100gpu"]
    mapping = {
        "ghost_allocation": medium["teardowns_per_day"],
        "contention_squeeze": medium["active_tenant_hours_per_day"] / 24.0,
        "provisioning_overhead": medium["provisions_per_day"],
        "burst_sustained_gap": medium["sustained_hours_per_day"] / 24.0,
        "straggler_tax": medium["jobs_per_day"],
        "oversubscription": medium["oversubscribed_hours_per_day"] / 24.0,
    }
    return mapping[category]


def build_cost_tables(trials: pd.DataFrame, omnibus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = extract_cost_inputs(trials)
    by_scale = {name: project_scale(name, inputs) for name in SCALE_PARAMS}
    main_rows: list[dict[str, Any]] = []
    category_map = {
        "ghost_allocation": "ghost",
        "contention_squeeze": "contention",
        "provisioning_overhead": "provisioning",
        "burst_sustained_gap": "burst",
        "straggler_tax": "straggler",
        "oversubscription": "oversubscription",
    }
    d_lookup = {row["Category"]: row["Cohen's d / d_z"] for _, row in omnibus.iterrows()}
    for category in CATEGORY_ORDER:
        label = CATEGORY_LABELS[category]
        key = category_map[category]
        main_rows.append(
            {
                "Waste Category": label,
                "Effect Size (d/d_z)": d_lookup[label],
                "Waste per Event": waste_per_event(category, trials),
                "Events/Day/GPU": events_per_day_medium(category),
                "Annual $/GPU": by_scale["medium_100gpu"][key],
                "8-GPU Fleet": by_scale["small_8gpu"][key] * SCALE_PARAMS["small_8gpu"]["gpu_count"],
                "100-GPU Fleet": by_scale["medium_100gpu"][key] * SCALE_PARAMS["medium_100gpu"]["gpu_count"],
                "10,000-GPU Fleet": by_scale["large_10000gpu"][key] * SCALE_PARAMS["large_10000gpu"]["gpu_count"],
            }
        )
    main_rows.append(
        {
            "Waste Category": "Total",
            "Effect Size (d/d_z)": "",
            "Waste per Event": "",
            "Events/Day/GPU": "",
            "Annual $/GPU": by_scale["medium_100gpu"]["total_per_gpu"],
            "8-GPU Fleet": by_scale["small_8gpu"]["total_fleet"],
            "100-GPU Fleet": by_scale["medium_100gpu"]["total_fleet"],
            "10,000-GPU Fleet": by_scale["large_10000gpu"]["total_fleet"],
        }
    )
    scale_rows = []
    for name, payload in by_scale.items():
        scale_rows.append({"Scale": SCALE_PARAMS[name]["label"], **payload})
    return pd.DataFrame(main_rows), pd.DataFrame(scale_rows)


def bootstrap_cost_summary(trials: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in SCALE_PARAMS}
    for _ in range(resamples):
        sample_inputs = {}
        ghost = condition_frame(trials, "ghost_allocation", "treatment")["primary_metric_value"].to_numpy(float)
        contention = condition_frame(trials, "contention_squeeze", "treatment")["primary_metric_value"].to_numpy(float)
        provisioning = condition_frame(trials, "provisioning_overhead")
        burst = condition_frame(trials, "burst_sustained_gap")["primary_metric_value"].to_numpy(float)
        straggler = condition_frame(trials, "straggler_tax", "treatment")["primary_metric_value"].to_numpy(float)
        oversub = condition_frame(trials, "oversubscription", "treatment")["primary_metric_value"].to_numpy(float)
        ghost_sample = ghost[rng.integers(0, ghost.size, ghost.size)]
        contention_sample = contention[rng.integers(0, contention.size, contention.size)]
        prov_idx = rng.integers(0, len(provisioning), len(provisioning))
        prov_sample = provisioning.iloc[prov_idx]
        burst_sample = burst[rng.integers(0, burst.size, burst.size)]
        straggler_sample = straggler[rng.integers(0, straggler.size, straggler.size)]
        oversub_sample = oversub[rng.integers(0, oversub.size, oversub.size)]
        sample_inputs["ghost_rate"] = float((ghost_sample > 1024 * 1024).mean())
        sample_inputs["avg_ghost_frac"] = float(np.clip(np.median(ghost_sample) / H100_VRAM_BYTES, 0.0, 1.0))
        sample_inputs["contention_drop_pct"] = float(np.median(contention_sample))
        sample_inputs["avg_spin_up_secs"] = float(np.median(prov_sample["primary_metric_value"] - prov_sample["control_metric_value"]) / 1000.0)
        sample_inputs["burst_sustained_gap_pct"] = float(np.median(burst_sample))
        sample_inputs["straggler_tax_pct"] = float(np.median(straggler_sample))
        sample_inputs["oversub_waste_pct"] = float(np.median(oversub_sample))
        for scale_name in SCALE_PARAMS:
            values[scale_name].append(project_scale(scale_name, sample_inputs)["total_per_gpu"])
    rows = []
    point_inputs = extract_cost_inputs(trials)
    for scale_name, samples in values.items():
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "Scale": SCALE_PARAMS[scale_name]["label"],
                "Point estimate / GPU / year": project_scale(scale_name, point_inputs)["total_per_gpu"],
                "95% CI low / GPU / year": float(low),
                "95% CI high / GPU / year": float(high),
            }
        )
    return pd.DataFrame(rows)


def build_descriptive_table(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        frame = condition_frame(trials, category)
        for (arm, condition), group in frame.groupby(["arm", "condition"], sort=True):
            values = group["primary_metric_value"].to_numpy(float)
            rows.append(
                {
                    "Category": CATEGORY_LABELS[category],
                    "Arm": arm,
                    "Condition": condition,
                    "N": len(group),
                    "Mean": mean_of(values),
                    "Median": median_of(values),
                    "SD": sd_of(values),
                    "IQR": iqr_of(values),
                    "Min": float(np.min(values)),
                    "Max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)

def ghost_tests(trials: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    frame = condition_frame(trials, "ghost_allocation")
    rows: list[dict[str, Any]] = []
    raw_ps: list[float] = []
    pending: list[dict[str, Any]] = []
    for method in ["clean", "under_load", "rapid_churn"]:
        for profile in ["1g.10gb", "2g.20gb", "3g.40gb"]:
            treatment = frame.loc[frame["condition"] == f"{method}_{profile}", "primary_metric_value"].to_numpy(float)
            control = frame.loc[frame["condition"] == f"control_{profile}", "primary_metric_value"].to_numpy(float)
            mw = mannwhitneyu(treatment, control, alternative="greater", method="auto")
            point, ci_low, ci_high = bootstrap_median_diff(treatment, control, resamples, seed + len(pending))
            raw_ps.append(float(mw.pvalue))
            pending.append(
                {
                    "Category": "Ghost allocations",
                    "Comparison": f"{method}_{profile} vs control_{profile}",
                    "Primary test": "Mann-Whitney U",
                    "Statistic": float(mw.statistic),
                    "Raw p": float(mw.pvalue),
                    "Adjusted p": 0.0,
                    "Effect size": rank_biserial_from_u(float(mw.statistic), treatment.size, control.size),
                    "CI low": ci_low,
                    "CI high": ci_high,
                    "Point estimate": point,
                }
            )
    adjusted = holm_adjust(raw_ps)
    for row, adj in zip(pending, adjusted):
        row["Adjusted p"] = adj
        rows.append(row)
    return pd.DataFrame(rows)


def contention_tests(trials: pd.DataFrame) -> pd.DataFrame:
    frame = condition_frame(trials, "contention_squeeze")
    rows: list[dict[str, Any]] = []
    time_conditions = ["timesliced_1_tenants", "timesliced_2_tenants", "timesliced_3_tenants", "timesliced_4_tenants"]
    paired = [frame.loc[frame["condition"] == cond, "primary_metric_value"].to_numpy(float) for cond in time_conditions]
    min_n = min(len(x) for x in paired)
    paired = [x[:min_n] for x in paired]
    friedman = friedmanchisquare(*paired)
    rows.append({
        "Category": "Contention squeeze",
        "Comparison": "1-4 time-sliced tenants",
        "Primary test": "Friedman chi-square",
        "Statistic": float(friedman.statistic),
        "Raw p": float(friedman.pvalue),
        "Adjusted p": float(friedman.pvalue),
        "Effect size": "",
        "CI low": "",
        "CI high": "",
        "Point estimate": "",
    })
    pair_specs = [("1 vs 2 tenants", paired[0], paired[1]), ("2 vs 3 tenants", paired[1], paired[2]), ("3 vs 4 tenants", paired[2], paired[3])]
    raw_ps: list[float] = []
    pending: list[dict[str, Any]] = []
    for label, left, right in pair_specs:
        stat = wilcoxon(right, left, alternative="greater")
        raw_ps.append(float(stat.pvalue))
        pending.append({
            "Category": "Contention squeeze",
            "Comparison": label,
            "Primary test": "Wilcoxon signed-rank",
            "Statistic": float(stat.statistic),
            "Raw p": float(stat.pvalue),
            "Adjusted p": 0.0,
            "Effect size": paired_rank_biserial(right, left),
            "CI low": "",
            "CI high": "",
            "Point estimate": float(np.median(right - left)),
        })
    adjusted = holm_adjust(raw_ps)
    for row, adj in zip(pending, adjusted):
        row["Adjusted p"] = adj
        rows.append(row)
    for tenant in [1, 2, 3, 4]:
        ts = frame.loc[frame["condition"] == f"timesliced_{tenant}_tenants", "primary_metric_value"].to_numpy(float)
        mig = frame.loc[frame["condition"] == f"mig_{tenant}_tenants", "primary_metric_value"].to_numpy(float)
        n = min(len(ts), len(mig))
        ts = ts[:n]
        mig = mig[:n]
        stat = wilcoxon(ts, mig, alternative="greater")
        rows.append({
            "Category": "Contention squeeze",
            "Comparison": f"timesliced vs MIG at {tenant} tenants",
            "Primary test": "Wilcoxon signed-rank",
            "Statistic": float(stat.statistic),
            "Raw p": float(stat.pvalue),
            "Adjusted p": float(stat.pvalue),
            "Effect size": paired_rank_biserial(ts, mig),
            "CI low": "",
            "CI high": "",
            "Point estimate": float(np.median(ts - mig)),
        })
    return pd.DataFrame(rows)


def provisioning_tests(trials: pd.DataFrame) -> pd.DataFrame:
    frame = condition_frame(trials, "provisioning_overhead")
    actual = frame["primary_metric_value"].to_numpy(float)
    reported = frame["control_metric_value"].to_numpy(float)
    paired_t = ttest_rel(actual, reported, alternative="greater")
    rows = [{
        "Category": "Provisioning overhead",
        "Comparison": "Spin-up latency vs nvidia-smi reported latency",
        "Primary test": "Paired t-test",
        "Statistic": float(paired_t.statistic),
        "Raw p": float(paired_t.pvalue),
        "Adjusted p": float(paired_t.pvalue),
        "Effect size": paired_cohens_d(actual, reported),
        "CI low": "",
        "CI high": "",
        "Point estimate": float(np.median(actual - reported)),
    }]
    load_groups = [group["primary_metric_value"].to_numpy(float) for _, group in frame.groupby("iv.load_state")]
    load_test = kruskal(*load_groups)
    rows.append({
        "Category": "Provisioning overhead",
        "Comparison": "Latency across load states",
        "Primary test": "Kruskal-Wallis",
        "Statistic": float(load_test.statistic),
        "Raw p": float(load_test.pvalue),
        "Adjusted p": float(load_test.pvalue),
        "Effect size": "",
        "CI low": "",
        "CI high": "",
        "Point estimate": "",
    })
    reg = linregress(frame["iv.concurrent_partitions"].to_numpy(float), actual)
    rows.append({
        "Category": "Provisioning overhead",
        "Comparison": "Latency vs concurrent partitions",
        "Primary test": "Linear regression",
        "Statistic": float(reg.slope),
        "Raw p": float(reg.pvalue),
        "Adjusted p": float(reg.pvalue),
        "Effect size": float(reg.rvalue),
        "CI low": "",
        "CI high": "",
        "Point estimate": float(reg.intercept),
    })
    return pd.DataFrame(rows)


def burst_tests(trials: pd.DataFrame) -> pd.DataFrame:
    frame = condition_frame(trials, "burst_sustained_gap")
    actual = frame["primary_metric_value"].to_numpy(float)
    ideal = frame["control_metric_value"].to_numpy(float)
    paired_t = ttest_rel(actual, ideal, alternative="greater")
    rows = [{
        "Category": "Burst-to-sustained gap",
        "Comparison": "Observed gap vs ideal 0% gap",
        "Primary test": "Paired t-test",
        "Statistic": float(paired_t.statistic),
        "Raw p": float(paired_t.pvalue),
        "Adjusted p": float(paired_t.pvalue),
        "Effect size": paired_cohens_d(actual, ideal),
        "CI low": "",
        "CI high": "",
        "Point estimate": float(np.median(actual - ideal)),
    }]
    workload_groups = [group["primary_metric_value"].to_numpy(float) for _, group in frame.groupby("iv.workload_type")]
    kw = kruskal(*workload_groups)
    rows.append({
        "Category": "Burst-to-sustained gap",
        "Comparison": "Gap across workload types",
        "Primary test": "Kruskal-Wallis",
        "Statistic": float(kw.statistic),
        "Raw p": float(kw.pvalue),
        "Adjusted p": float(kw.pvalue),
        "Effect size": "",
        "CI low": "",
        "CI high": "",
        "Point estimate": "",
    })
    return pd.DataFrame(rows)


def straggler_threshold_table(frame: pd.DataFrame) -> pd.DataFrame:
    thresholds = np.quantile(frame["primary_metric_value"].to_numpy(float), np.linspace(0.1, 0.9, 9))
    rows = []
    gt = frame["ground_truth_waste"].astype(bool).to_numpy()
    score = frame["primary_metric_value"].to_numpy(float)
    for threshold in thresholds:
        pred = score >= threshold
        tp = int(np.sum(gt & pred))
        fp = int(np.sum((~gt) & pred))
        tn = int(np.sum((~gt) & (~pred)))
        fn = int(np.sum(gt & (~pred)))
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append({"Threshold": float(threshold), "Sensitivity %": sensitivity * 100.0, "Specificity %": specificity * 100.0})
    return pd.DataFrame(rows)


def straggler_tests(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = condition_frame(trials, "straggler_tax")
    treatment = frame.loc[frame["arm"] == "treatment", "primary_metric_value"].to_numpy(float)
    control = frame.loc[frame["arm"] == "control", "primary_metric_value"].to_numpy(float)
    mw = mannwhitneyu(treatment, control, alternative="greater", method="auto")
    rows = [{
        "Category": "Straggler tax",
        "Comparison": "Treatment vs control",
        "Primary test": "Mann-Whitney U",
        "Statistic": float(mw.statistic),
        "Raw p": float(mw.pvalue),
        "Adjusted p": float(mw.pvalue),
        "Effect size": rank_biserial_from_u(float(mw.statistic), treatment.size, control.size),
        "CI low": "",
        "CI high": "",
        "Point estimate": float(np.median(treatment) - np.median(control)),
    }]
    treat_frame = frame.loc[frame["arm"] == "treatment"].copy()
    fleet_groups = [group["primary_metric_value"].to_numpy(float) for _, group in treat_frame.groupby("iv.fleet_size")]
    fleet_kw = kruskal(*fleet_groups)
    rows.append({
        "Category": "Straggler tax",
        "Comparison": "Treatment across fleet sizes",
        "Primary test": "Kruskal-Wallis",
        "Statistic": float(fleet_kw.statistic),
        "Raw p": float(fleet_kw.pvalue),
        "Adjusted p": float(fleet_kw.pvalue),
        "Effect size": "",
        "CI low": "",
        "CI high": "",
        "Point estimate": "",
    })
    reg = linregress(treat_frame["iv.severity"].to_numpy(float), treat_frame["primary_metric_value"].to_numpy(float))
    rows.append({
        "Category": "Straggler tax",
        "Comparison": "Tax vs severity",
        "Primary test": "Linear regression",
        "Statistic": float(reg.slope),
        "Raw p": float(reg.pvalue),
        "Adjusted p": float(reg.pvalue),
        "Effect size": float(reg.rvalue),
        "CI low": "",
        "CI high": "",
        "Point estimate": float(reg.intercept),
    })
    return pd.DataFrame(rows), straggler_threshold_table(frame)


def oversubscription_tests(trials: pd.DataFrame) -> pd.DataFrame:
    frame = condition_frame(trials, "oversubscription")
    treatment = frame.loc[frame["arm"] == "treatment", "primary_metric_value"].to_numpy(float)
    control = frame.loc[frame["arm"] == "control", "primary_metric_value"].to_numpy(float)
    mw = mannwhitneyu(treatment, control, alternative="greater", method="auto")
    rows = [{
        "Category": "Oversubscription",
        "Comparison": "Treatment vs control",
        "Primary test": "Mann-Whitney U",
        "Statistic": float(mw.statistic),
        "Raw p": float(mw.pvalue),
        "Adjusted p": float(mw.pvalue),
        "Effect size": rank_biserial_from_u(float(mw.statistic), treatment.size, control.size),
        "CI low": "",
        "CI high": "",
        "Point estimate": float(np.median(treatment) - np.median(control)),
    }]
    treat = frame.loc[frame["arm"] == "treatment"].copy()
    reg = linregress(treat["iv.overcommit_ratio"].to_numpy(float), treat["primary_metric_value"].to_numpy(float))
    rows.append({
        "Category": "Oversubscription",
        "Comparison": "Degradation vs overcommit ratio",
        "Primary test": "Linear regression",
        "Statistic": float(reg.slope),
        "Raw p": float(reg.pvalue),
        "Adjusted p": float(reg.pvalue),
        "Effect size": float(reg.rvalue),
        "CI low": "",
        "CI high": "",
        "Point estimate": float(reg.intercept),
    })
    groups = [group["primary_metric_value"].to_numpy(float) for _, group in treat.groupby("iv.instance_count")]
    kw = kruskal(*groups)
    rows.append({
        "Category": "Oversubscription",
        "Comparison": "Degradation across instance counts",
        "Primary test": "Kruskal-Wallis",
        "Statistic": float(kw.statistic),
        "Raw p": float(kw.pvalue),
        "Adjusted p": float(kw.pvalue),
        "Effect size": "",
        "CI low": "",
        "CI high": "",
        "Point estimate": "",
    })
    return pd.DataFrame(rows)


def build_cost_sensitivity(trials: pd.DataFrame) -> pd.DataFrame:
    inputs = extract_cost_inputs(trials)
    rows = []
    for key, value in inputs.items():
        p10 = value * 0.9
        p90 = value * 1.1
        low_inputs = dict(inputs)
        high_inputs = dict(inputs)
        low_inputs[key] = p10
        high_inputs[key] = p90
        low_total = project_scale("medium_100gpu", low_inputs)["total_fleet"]
        high_total = project_scale("medium_100gpu", high_inputs)["total_fleet"]
        rows.append(
            {
                "Parameter": key,
                "P10": p10,
                "P90": p90,
                "100-GPU Total @ P10": low_total,
                "100-GPU Total @ P90": high_total,
                "Range": high_total - low_total,
            }
        )
    return pd.DataFrame(rows)


def full_statistical_tests(trials: pd.DataFrame, bootstrap_resamples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = [
        ghost_tests(trials, bootstrap_resamples, seed),
        contention_tests(trials),
        provisioning_tests(trials),
        burst_tests(trials),
    ]
    straggler_table, roc_table = straggler_tests(trials)
    tables.append(straggler_table)
    tables.append(oversubscription_tests(trials))
    return pd.concat(tables, ignore_index=True), roc_table


def style_main_tables(table1: pd.DataFrame, table2: pd.DataFrame, omnibus: pd.DataFrame, cost_ci: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_view = table1.copy()
    cost_view["Effect Size (d/d_z)"] = cost_view["Effect Size (d/d_z)"].apply(lambda x: x if x == "" else format_number(x, 2))
    for column in ["Annual $/GPU", "8-GPU Fleet", "100-GPU Fleet", "10,000-GPU Fleet"]:
        cost_view[column] = cost_view[column].apply(lambda v: "" if v == "" else format_currency(v))
    cost_view["Events/Day/GPU"] = cost_view["Events/Day/GPU"].apply(lambda v: "" if v == "" else format_number(v, 2))
    detect_view = table2.copy()[["Category", "nvidia-smi Detection Rate", "DCGM Detection Rate", "gpu-roofline Detection Rate", "McNemar p (smi vs roofline)", "McNemar p (dcgm vs roofline)"]]
    for column in ["nvidia-smi Detection Rate", "DCGM Detection Rate", "gpu-roofline Detection Rate"]:
        detect_view[column] = detect_view[column].map(lambda v: format_number(v, 1))
    for column in ["McNemar p (smi vs roofline)", "McNemar p (dcgm vs roofline)"]:
        detect_view[column] = detect_view[column].map(format_pvalue)
    omnibus_view = omnibus.copy()
    omnibus_view["Median (a)"] = omnibus_view["Median (a)"].map(lambda v: format_number(v, 2))
    omnibus_view["Median (b)"] = omnibus_view["Median (b)"].map(lambda v: format_number(v, 2))
    omnibus_view["Statistic"] = omnibus_view["Statistic"].map(lambda v: format_number(v, 0))
    omnibus_view["Raw p-value"] = omnibus_view["Raw p-value"].map(format_pvalue)
    omnibus_view["Holm p-value"] = omnibus_view["Holm p-value"].map(format_pvalue)
    omnibus_view["Cohen's d / d_z"] = omnibus_view["Cohen's d / d_z"].map(lambda v: format_number(v, 2))
    omnibus_view["Rank-biserial r"] = omnibus_view["Rank-biserial r"].map(lambda v: format_number(v, 2))
    omnibus_view["95% CI"] = omnibus.apply(lambda row: f"[{format_number(row['95% CI Low'], 2)}, {format_number(row['95% CI High'], 2)}]", axis=1)
    omnibus_view = omnibus_view[[
        "Category", "N (a)", "N (b)", "Design", "Primary Test", "Metric",
        "Median (a)", "Median (b)", "Statistic",
        "Raw p-value", "Holm p-value",
        "Cohen's d / d_z", "Rank-biserial r", "95% CI",
    ]]
    cost_ci_view = cost_ci.copy()
    for column in cost_ci_view.columns[1:]:
        cost_ci_view[column] = cost_ci_view[column].map(format_currency)
    return cost_view, detect_view, omnibus_view, cost_ci_view


def build_summary_markdown(raw: dict[str, Any], input_path: Path, table1: pd.DataFrame, table2: pd.DataFrame, omnibus: pd.DataFrame, cost_ci: pd.DataFrame) -> str:
    t1, t2, ov, ci = style_main_tables(table1, table2, omnibus, cost_ci)
    return "\n".join([
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
        "- Claim framing: simulation evidence, modeled tool visibility, and scenario-based economic impact. Hardware validation remains future work.",
        "",
        "## Table 1. Annual Cost of Invisible GPU Waste by Category and Fleet Scale",
        "",
        markdown_table(t1),
        "",
        "## Table 2. Detection Rates by Tool Across All 6 Waste Categories",
        "",
        markdown_table(t2),
        "",
        "## Table 3. Cross-Category Omnibus Statistical Tests",
        "",
        markdown_table(ov),
        "",
        "## Cost Model Bootstrap Confidence Intervals",
        "",
        markdown_table(ci),
        "",
        "## Interpretation Guardrails",
        "",
        "- These costs are scenario-model outputs derived from simulation, not direct hardware measurements.",
        "- The raw data, derived tables, hashes, and execution provenance are preserved in this package for auditability.",
        "- Category-specific primary tests and sensitivity analyses are reported in `derived/supplement.md`.",
        "",
    ])


def build_supplement_markdown(descriptive: pd.DataFrame, tests: pd.DataFrame, roc_table: pd.DataFrame, sensitivity: pd.DataFrame, confusion: pd.DataFrame) -> str:
    desc_view = descriptive.copy()
    for column in ["Mean", "Median", "SD", "IQR", "Min", "Max"]:
        desc_view[column] = desc_view[column].map(lambda v: format_number(v, 2))
    stats_view = tests.copy()
    for column in ["Statistic", "Effect size", "CI low", "CI high", "Point estimate"]:
        stats_view[column] = stats_view[column].apply(lambda v: "" if v == "" else format_number(v, 4))
    stats_view["Raw p"] = stats_view["Raw p"].apply(format_pvalue)
    stats_view["Adjusted p"] = stats_view["Adjusted p"].apply(format_pvalue)
    roc_view = roc_table.copy()
    for column in ["Threshold", "Sensitivity %", "Specificity %"]:
        roc_view[column] = roc_view[column].map(lambda v: format_number(v, 2))
    sensitivity_view = sensitivity.copy()
    for column in ["P10", "P90"]:
        sensitivity_view[column] = sensitivity_view[column].map(lambda v: format_number(v, 4))
    for column in ["100-GPU Total @ P10", "100-GPU Total @ P90", "Range"]:
        sensitivity_view[column] = sensitivity_view[column].map(format_currency)
    confusion_view = confusion.copy()
    for column in ["Sensitivity %", "Specificity %", "Precision %", "F1"]:
        confusion_view[column] = confusion_view[column].map(lambda v: format_number(v, 2))
    return "\n".join([
        "# Supplementary Tables",
        "",
        "## Table S1. Descriptive Statistics by Category and Condition",
        "",
        markdown_table(desc_view),
        "",
        "## Table S2. Full Statistical Test Results",
        "",
        markdown_table(stats_view),
        "",
        "## Table S4. Cost Model Parameter Sensitivity",
        "",
        markdown_table(sensitivity_view),
        "",
        "## Table S5. Tool Detection Confusion Matrices",
        "",
        markdown_table(confusion_view),
        "",
        "## Straggler Threshold Sweep",
        "",
        markdown_table(roc_view),
        "",
        "- Table S3 remains reserved for hardware-phase calibration results.",
        "",
    ])


def build_analysis_payload(raw: dict[str, Any], omnibus: pd.DataFrame, table1: pd.DataFrame, table2: pd.DataFrame, scale_costs: pd.DataFrame, tests: pd.DataFrame, sensitivity: pd.DataFrame) -> dict[str, Any]:
    return {
        "protocol_version": raw["protocol_version"],
        "phase": raw["phase"],
        "seed": raw["seed"],
        "generated_at_utc": now_utc_iso(),
        "omnibus_results": omnibus.to_dict(orient="records"),
        "main_cost_table": table1.to_dict(orient="records"),
        "detection_table": table2.to_dict(orient="records"),
        "scale_costs": scale_costs.to_dict(orient="records"),
        "full_statistical_tests": tests.to_dict(orient="records"),
        "cost_sensitivity": sensitivity.to_dict(orient="records"),
    }

def run_git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"unavailable\n{exc}\n"


def capture_git_metadata(repo_root: Path, metadata_dir: Path) -> dict[str, str]:
    payloads = {
        "git-head.txt": run_git(repo_root, "rev-parse", "HEAD"),
        "git-status.txt": run_git(repo_root, "status", "--short", "--untracked-files=all"),
        "git-log.txt": run_git(repo_root, "log", "--oneline", "--decorate", "-n", "20"),
        "git-diff.patch": run_git(repo_root, "diff"),
    }
    paths: dict[str, str] = {}
    for name, content in payloads.items():
        path = metadata_dir / name
        path.write_text(content, encoding="utf-8")
        paths[name] = path.as_posix()
    return paths


def copy_source_artifacts(repo_root: Path, source_dir: Path, self_path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    self_target = source_dir / "analysis" / "analyze_study.py"
    self_target.parent.mkdir(parents=True, exist_ok=True)
    if self_target.resolve() != self_path.resolve():
        shutil.copy2(self_path, self_target)
    manifest["analysis/analyze_study.py"] = file_manifest(self_target, source_dir)
    for relative in SOURCE_ARTIFACTS:
        src = repo_root / relative
        if not src.exists():
            continue
        dst = source_dir / "repo" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[(Path("repo") / relative).as_posix()] = file_manifest(dst, source_dir)
    return manifest


def write_readme(path: Path, raw_hash: str) -> None:
    path.write_text("\n".join([
        "# Study Archive",
        "",
        "This archive is the canonical simulation-phase package for The Hidden Cost of GPU Virtualization.",
        "",
        "## Contents",
        "",
        "- `raw/simulation-raw.json`: immutable raw trial-level dataset copied into the archive.",
        "- `derived/summary.md`: main-paper tables and framing.",
        "- `derived/supplement.md`: protocol-specific and reviewer-facing supporting analyses.",
        "- `derived/analysis-results.json`: machine-readable derived statistics.",
        "- `derived/provenance.json` and `derived/PROVENANCE.md`: audit trail, hashes, commands, and software versions.",
        "- `metadata/`: git metadata, source manifest, and environment capture.",
        "- `source_snapshot/`: exact copies of analysis and simulation source files used to build this archive.",
        "- `SHA256SUMS.txt`: recursive checksums for the entire archive.",
        "",
        "## Raw Dataset Lock",
        "",
        f"- SHA256: `{raw_hash}`",
        "- Treat the archived raw JSON as immutable once cited in a manuscript.",
        "",
    ]), encoding="utf-8")


def recursive_sha256(root: Path, destination: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != destination):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_provenance(raw: dict[str, Any], archive_root: Path, input_original: Path, input_copy: Path, source_manifest: dict[str, Any], git_paths: dict[str, str], derived_paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generated_at_utc": now_utc_iso(),
        "study": "The Hidden Cost of GPU Virtualization",
        "protocol_version": raw["protocol_version"],
        "phase": raw["phase"],
        "raw_dataset_metadata": {
            "generator": raw.get("generator"),
            "seed": raw.get("seed"),
            "target_trials_per_category": raw.get("target_trials_per_category"),
            "total_trials": raw.get("total_trials"),
        },
        "archive_root": archive_root.as_posix(),
        "input_dataset": {
            "original_path": input_original.as_posix(),
            "archived_copy": input_copy.as_posix(),
            "sha256": sha256_file(input_copy),
            "bytes": input_copy.stat().st_size,
        },
        "analysis_command": " ".join([
            sys.executable,
            Path(__file__).as_posix(),
            "--input",
            args.input,
            "--output-root",
            args.output_root,
            "--repo-root",
            args.repo_root,
            "--bootstrap-resamples",
            str(args.bootstrap_resamples),
            "--cost-bootstrap-resamples",
            str(args.cost_bootstrap_resamples),
            "--seed",
            str(args.seed),
        ]),
        "recommended_simulation_command": (
            "CARGO_TARGET_DIR=D:/cargo-target/gpu-tools TMPDIR=D:/tmp "
            "cargo run -p gpu-harness --release --bin study_sim -- "
            "--out docs/study-results/simulation-raw.json"
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "analysis_parameters": {
            "bootstrap_resamples": args.bootstrap_resamples,
            "cost_bootstrap_resamples": args.cost_bootstrap_resamples,
            "seed": args.seed,
        },
        "source_snapshot": source_manifest,
        "git_metadata_files": git_paths,
        "derived_outputs": {path.name: file_manifest(path, archive_root) for path in derived_paths},
    }


def build_provenance_markdown(provenance: dict[str, Any]) -> str:
    return "\n".join([
        "# Provenance",
        "",
        f"- Generated at: `{provenance['generated_at_utc']}`",
        f"- Protocol version: `{provenance['protocol_version']}`",
        f"- Phase: `{provenance['phase']}`",
        f"- Archive root: `{provenance['archive_root']}`",
        f"- Original input path: `{provenance['input_dataset']['original_path']}`",
        f"- Archived raw copy: `{provenance['input_dataset']['archived_copy']}`",
        f"- Input SHA256: `{provenance['input_dataset']['sha256']}`",
        f"- Input bytes: `{provenance['input_dataset']['bytes']}`",
        f"- Raw generator: `{provenance['raw_dataset_metadata']['generator']}`",
        f"- Raw dataset seed: `{provenance['raw_dataset_metadata']['seed']}`",
        f"- Raw total trials: `{provenance['raw_dataset_metadata']['total_trials']}`",
        f"- Bootstrap resamples: `{provenance['analysis_parameters']['bootstrap_resamples']}`",
        f"- Cost bootstrap resamples: `{provenance['analysis_parameters']['cost_bootstrap_resamples']}`",
        f"- Seed: `{provenance['analysis_parameters']['seed']}`",
        "",
        "## Commands",
        "",
        f"- Analysis: `{provenance['analysis_command']}`",
        f"- Recommended simulation rerun: `{provenance['recommended_simulation_command']}`",
        "",
        "## Software",
        "",
        f"- Python: `{provenance['software']['python'].splitlines()[0]}`",
        f"- Platform: `{provenance['software']['platform']}`",
        f"- numpy: `{provenance['software']['numpy']}`",
        f"- pandas: `{provenance['software']['pandas']}`",
        f"- scipy: `{provenance['software']['scipy']}`",
        "",
        "## Source Snapshot",
        "",
        *[f"- `{path}` sha256 `{entry['sha256']}` ({entry['bytes']} bytes)" for path, entry in provenance['source_snapshot'].items()],
        "",
        "## Derived Outputs",
        "",
        *[f"- `{entry['path']}` sha256 `{entry['sha256']}` ({entry['bytes']} bytes)" for entry in provenance['derived_outputs'].values()],
        "",
        "## Best-Practice Retention",
        "",
        "- Keep `raw/simulation-raw.json` immutable once cited.",
        "- Preserve this provenance file, `provenance.json`, and `SHA256SUMS.txt` together with any journal submission package.",
        "- Treat markdown tables as derived artifacts that are reproducible from the archived raw JSON and the archived analysis script.",
        "",
    ])


def main() -> None:
    args = parse_args()
    archive_root = Path(args.output_root)
    repo_root = Path(args.repo_root)
    src_input = Path(args.input)
    paths = layout(archive_root)
    for directory in [paths.root, paths.raw_dir, paths.derived_dir, paths.metadata_dir, paths.source_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_input, paths.raw_json)
    raw, trials = load_dataset(paths.raw_json)
    omnibus = omnibus_stats(trials, args.bootstrap_resamples, args.seed)
    table1, scale_costs = build_cost_tables(trials, omnibus)
    table2, confusion = build_detection_table(trials)
    descriptive = build_descriptive_table(trials)
    tests, roc_table = full_statistical_tests(trials, args.bootstrap_resamples, args.seed)
    sensitivity = build_cost_sensitivity(trials)
    cost_ci = bootstrap_cost_summary(trials, args.cost_bootstrap_resamples, args.seed)
    summary = build_summary_markdown(raw, paths.raw_json, table1, table2, omnibus, cost_ci)
    supplement = build_supplement_markdown(descriptive, tests, roc_table, sensitivity, confusion)
    analysis_payload = build_analysis_payload(raw, omnibus, table1, table2, scale_costs, tests, sensitivity)
    paths.summary.write_text(summary, encoding="utf-8")
    paths.supplement.write_text(supplement, encoding="utf-8")
    paths.analysis_json.write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")
    source_manifest = copy_source_artifacts(repo_root, paths.source_dir, Path(__file__))
    paths.source_manifest.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    git_paths = capture_git_metadata(repo_root, paths.metadata_dir)
    env_path = paths.metadata_dir / "environment.txt"
    env_path.write_text("\n".join([
        f"generated_at_utc={now_utc_iso()}",
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"scipy={scipy.__version__}",
    ]) + "\n", encoding="utf-8")
    git_paths["environment.txt"] = env_path.as_posix()
    provenance = build_provenance(raw, archive_root, src_input, paths.raw_json, source_manifest, git_paths, [paths.summary, paths.supplement, paths.analysis_json], args)
    paths.provenance_json.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    paths.provenance_md.write_text(build_provenance_markdown(provenance), encoding="utf-8")
    write_readme(paths.readme, provenance["input_dataset"]["sha256"])
    recursive_sha256(paths.root, paths.sha256)
    print(paths.root)
    for path in [paths.raw_json, paths.summary, paths.supplement, paths.analysis_json, paths.provenance_json, paths.provenance_md, paths.readme, paths.sha256, paths.source_manifest]:
        print(path)


if __name__ == "__main__":
    main()
