#!/usr/bin/env python3
"""
Compare GPU benchmark results across cloud providers.

Usage:
    python compare-providers.py results/lambda-labs_*/  results/runpod_*/  results/vastai_*/

Reads JSON output from benchmark-cloud.sh and generates a markdown comparison table.
"""

import json
import sys
import os
from pathlib import Path


def load_result(result_dir: str) -> dict:
    """Load benchmark results from a provider's output directory."""
    result_dir = Path(result_dir)
    provider = result_dir.name.rsplit("_", 2)[0]  # strip timestamp

    data = {"provider": provider}

    # Load burst results
    burst_path = result_dir / "burst.json"
    if burst_path.exists():
        with open(burst_path) as f:
            burst = json.load(f)
        data["device_name"] = burst.get("device_name", "Unknown")
        data["burst_bandwidth_gbps"] = burst.get("peak_bandwidth_gbps", 0)
        data["burst_gflops"] = burst.get("peak_gflops", 0)

    # Load dynamic results
    dynamic_path = result_dir / "dynamic.json"
    if dynamic_path.exists():
        with open(dynamic_path) as f:
            dynamic = json.load(f)
        data["sustained_bandwidth_gbps"] = dynamic.get("sustained_bandwidth_gbps",
                                            dynamic.get("peak_bandwidth_gbps", 0))
        data["sustained_gflops"] = dynamic.get("sustained_gflops",
                                    dynamic.get("peak_gflops", 0))
        data["tension_pct"] = dynamic.get("tension_pct", 0)

    # Load diagnostic findings
    diag_path = result_dir / "diagnose.json"
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)
        findings = diag.get("findings", [])
        if findings:
            data["issues"] = ", ".join(f.get("probe", "Unknown") for f in findings)
        else:
            data["issues"] = "None"

    # Load variance data from repeated burst runs
    variances = []
    for i in range(1, 4):
        run_path = result_dir / f"burst_run{i}.json"
        if run_path.exists():
            with open(run_path) as f:
                run = json.load(f)
            variances.append(run.get("peak_bandwidth_gbps", 0))

    if len(variances) >= 2:
        mean_bw = sum(variances) / len(variances)
        if mean_bw > 0:
            std_bw = (sum((v - mean_bw) ** 2 for v in variances) / len(variances)) ** 0.5
            data["cv_pct"] = (std_bw / mean_bw) * 100
        else:
            data["cv_pct"] = 0
    else:
        data["cv_pct"] = 0

    # Compute gaps
    burst_bw = data.get("burst_bandwidth_gbps", 0)
    sustained_bw = data.get("sustained_bandwidth_gbps", 0)
    if burst_bw > 0 and sustained_bw > 0:
        data["bw_gap_pct"] = ((sustained_bw - burst_bw) / burst_bw) * 100

    burst_gf = data.get("burst_gflops", 0)
    sustained_gf = data.get("sustained_gflops", 0)
    if burst_gf > 0 and sustained_gf > 0:
        data["gflops_gap_pct"] = ((sustained_gf - burst_gf) / burst_gf) * 100

    return data


def format_table(results: list[dict]) -> str:
    """Generate a markdown comparison table."""
    lines = []
    lines.append(
        "| Provider | GPU | Burst BW | Sustained BW | BW Gap "
        "| Burst FP32 | Sustained FP32 | FP32 Gap | CV% | Issues |"
    )
    lines.append(
        "|----------|-----|----------|--------------|--------"
        "|------------|----------------|----------|-----|--------|"
    )

    for r in results:
        provider = r.get("provider", "?")
        gpu = r.get("device_name", "?")
        burst_bw = r.get("burst_bandwidth_gbps", 0)
        sust_bw = r.get("sustained_bandwidth_gbps", 0)
        bw_gap = r.get("bw_gap_pct", 0)
        burst_gf = r.get("burst_gflops", 0)
        sust_gf = r.get("sustained_gflops", 0)
        gf_gap = r.get("gflops_gap_pct", 0)
        cv = r.get("cv_pct", 0)
        issues = r.get("issues", "?")

        lines.append(
            f"| {provider} | {gpu} "
            f"| {burst_bw:,.0f} GB/s | {sust_bw:,.0f} GB/s | {bw_gap:+.1f}% "
            f"| {burst_gf/1000:,.1f} TFLOPS | {sust_gf/1000:,.1f} TFLOPS | {gf_gap:+.1f}% "
            f"| {cv:.2f}% | {issues} |"
        )

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare-providers.py <result-dir-1> [result-dir-2] ...")
        print("Example: python compare-providers.py results/lambda-labs_*/ results/runpod_*/")
        sys.exit(1)

    results = []
    for path in sys.argv[1:]:
        if os.path.isdir(path):
            results.append(load_result(path))
        else:
            print(f"Warning: {path} is not a directory, skipping")

    if not results:
        print("No results loaded.")
        sys.exit(1)

    print(format_table(results))
    print()
    print(f"{len(results)} providers compared.")


if __name__ == "__main__":
    main()
