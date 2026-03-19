# A Reproducible Simulation Study of Invisible Waste in Multi-Tenant GPU Infrastructure

**Authors:** [Author list TBD]
**Target Venue:** The Journal of Supercomputing (Phase 1: Simulation)
**Protocol Version:** 1.0
**Data Availability:** Trial-level data, analysis scripts, and provenance artifacts archived with SHA-256 verification at `docs/study-results/`

---

## Abstract

Multi-tenant GPU infrastructure introduces forms of economic waste that are structurally invisible to conventional device-centric monitoring tools. We present a reproducible simulation study of six waste categories in virtualized GPU environments: (1) ghost allocations persisting after teardown, (2) contention squeeze on tenant arrival, (3) provisioning overhead during partition spin-up, (4) burst-to-sustained performance gaps under thermal throttling, (5) straggler tax in distributed training, and (6) oversubscription blind spots from silent resource exhaustion. We implement a parameterized simulation framework in Rust (`gpu-harness`) and execute 120,000 deterministic trials across the six categories with realistic noise injection matching NVML instrumentation characteristics. The simulation is fully reproducible: re-execution with seed 42 produces byte-identical output (SHA-256 verified).

Across all categories, simulated effects are statistically significant after Holm-Bonferroni correction (all adjusted p < 1e-300). Effect sizes range from Cohen's d_z = 0.73 (burst-to-sustained gap) to Cohen's d = 8.55 (contention squeeze). The `gpu-roofline` measurement framework detects 56.5-100% of waste events across categories; modeled `nvidia-smi` and DCGM baselines detect 0% (McNemar p < 1e-300 for all pairwise comparisons). Scenario-based cost projections estimate $8,171-$33,999 per GPU per year of invisible waste depending on fleet scale, with bootstrap 95% CIs of [$8,149, $8,192] to [$33,809, $34,193].

This paper contributes a structured taxonomy, a reproducible measurement framework, and a simulation benchmark. It does not claim that simulated magnitudes are final real-world measurements; hardware validation on bare-metal H100 systems is reserved for a follow-on study.

## 1. Introduction

The rapid expansion of GPU-accelerated computing has made multi-tenant GPU virtualization a critical infrastructure layer. Technologies such as NVIDIA Multi-Instance GPU (MIG), GRID vGPU, and time-slicing enable cloud providers to partition expensive accelerators among multiple tenants. However, this partitioning introduces forms of economic waste that existing monitoring tools cannot observe.

The fundamental limitation is architectural: `nvidia-smi`, NVIDIA DCGM, and similar tools report *instantaneous state* per device or per instance. They do not compute deltas across lifecycle transitions, cross-tenant comparisons, temporal degradation curves, fleet-level aggregates exceeding physical capacity, or distributed synchronization overhead. These are not bugs—they are structural constraints of tools designed for single-device health monitoring, not multi-tenant economic analysis.

We identify and formalize six categories of invisible waste, each with a distinct mechanism and a distinct reason why current tools miss it:

1. **Ghost allocations:** VRAM not reclaimed after vGPU destruction. `nvidia-smi` reports the instance as gone but does not verify that physical memory returned to baseline.
2. **Contention squeeze:** Existing tenants lose bandwidth when a new tenant provisions on a time-sliced GPU. Per-GPU utilization may remain at 100% while individual tenants experience degradation proportional to co-tenant count.
3. **Provisioning overhead:** MIG partition creation takes 100-500ms of wall-clock time during which capacity is neither free nor usable. nvidia-smi reports instant state transitions because it queries after the command returns.
4. **Burst-to-sustained gap:** GPUs are priced at peak (burst) specifications, but sustained performance under thermal throttling is 1-20% lower depending on workload and cooling configuration.
5. **Straggler tax:** In data-parallel distributed training, one degraded GPU blocks N-1 others at synchronization barriers. Per-device monitoring provides no fleet-wide barrier analysis.
6. **Oversubscription blind spots:** Hypervisors can allocate more aggregate vGPU VRAM than physical capacity. Per-instance tools show each allocation succeeding individually, but no tool sums allocations against physical limits.

### Contributions

1. A six-category taxonomy of invisible waste in virtualized GPU systems, with formal definitions and mechanisms.
2. A reproducible, deterministic simulation harness producing 120,000 parameterized trials with realistic noise injection.
3. Protocol-aligned statistical analyses using design-appropriate tests: Mann-Whitney U for independent groups, Wilcoxon signed-rank for paired measurements, with Holm-Bonferroni correction across omnibus comparisons.
4. A provenance-preserving results package with SHA-256 hashes, software versions, and byte-identical rerun capability.
5. A scenario-based cost model with bootstrap confidence intervals that prioritizes which waste categories merit hardware validation first.

### Scope and Claim Boundary

This paper makes simulation-backed claims about the *detectability* and *relative magnitude* of each waste category under explicit modeling assumptions. It does not claim that the simulated magnitudes are final real-world fleet measurements. The cost model outputs are scenario analyses under explicit parameters ($2.50/GPU/hr, specific fleet utilization assumptions), not direct market measurements. Hardware validation is deferred to a follow-on study.

## 2. Background and Related Work

### 2.1 GPU Virtualization Technologies

NVIDIA Multi-Instance GPU (MIG), introduced with the A100 architecture, provides hardware-isolated partitions with dedicated compute, memory, and cache resources. NVIDIA GRID/vGPU enables time-sliced sharing where tenants alternate access to the full GPU. Both approaches are widely deployed in cloud infrastructure [1].

### 2.2 GPU Monitoring and Observability

The standard GPU monitoring stack consists of `nvidia-smi` (NVML command-line interface) and NVIDIA Data Center GPU Manager (DCGM). Both tools provide per-device and per-instance metrics including utilization, memory consumption, temperature, and power. Neither tool computes lifecycle deltas, cross-tenant comparisons, or fleet-wide aggregates—capabilities that would be necessary to observe the waste categories identified in this study.

### 2.3 Performance Measurement

The roofline model [2] provides an analytical framework for understanding compute and memory bandwidth limits. Our `gpu-roofline` tool extends this with a dynamic roofline that captures burst versus sustained performance under thermal constraints, and a fleet-level extension for distributed training analysis.

### 2.4 GPU Power and Thermal Modeling

Thermal throttling in modern GPUs follows well-characterized physics. Our simulation uses Newton's law of cooling with empirically-calibrated parameters for each GPU profile (H100 SXM5, H200, RTX 5090, RTX 4090, MI300X). The power model maps workload intensity and thermal throttle factors to operating clock frequency and board power.

### 2.5 Distributed Training Synchronization

Data-parallel training requires all-reduce synchronization at gradient aggregation barriers. The slowest GPU determines effective throughput for the entire fleet [3]. Our straggler detection framework (`gpu-fleet::straggler`) measures per-GPU throughput, identifies outliers, and diagnoses root causes across five degradation types.

## 3. Waste Taxonomy and Measurement Model

### 3.1 Category Definitions

Each waste category is defined by: (a) the physical mechanism producing waste, (b) why existing tools miss it, and (c) what measurement capability is required for detection.

**Category 1: Ghost Allocations.** After vGPU teardown, VRAM may not be fully reclaimed by the driver. Detection requires capturing pre-teardown `nvmlDeviceGetMemoryInfo().used`, waiting for the instance to disappear, and polling post-teardown memory with a stabilization protocol until readings converge within 1 MiB. In the simulation, ghost bytes are injected as `Uniform(0, 1024 MiB)` for treatment trials and 0 for controls.

**Category 2: Contention Squeeze.** Under time-slicing, each tenant receives approximately 1/N of the GPU's bandwidth, where N is the tenant count. Under MIG (hardware partitioning), each tenant receives a dedicated share with no contention. The simulation models per-tenant bandwidth ratios with Gaussian jitter (sigma = 2% of baseline).

**Category 3: Provisioning Overhead.** MIG partition creation incurs wall-clock latency that scales with profile size (120-450ms base) plus load-state penalties (0-80ms) and concurrent-partition penalties (15ms per existing partition). The simulation adds log-normal jitter (sigma = 0.3) to model real-world latency variation. This category uses a paired design: each trial measures both true spin-up latency and nvidia-smi reported latency (~0.5ms).

**Category 4: Burst-to-Sustained Gap.** The simulation uses a physics-based thermal model with per-GPU profile parameters. Burst performance is measured at ambient temperature; sustained performance is measured at thermal equilibrium (t = 3*tau, where tau is the thermal time constant). This category uses a paired within-subjects design: each trial compares burst and sustained measurements on the same simulated GPU.

**Category 5: Straggler Tax.** Fleet simulations inject degradation into one GPU (out of 8, 32, or 128) across five hardware failure modes at three severity levels. Effective fleet throughput equals min(all GPU throughputs) * N, and the straggler tax percentage quantifies the gap between effective and ideal throughput.

**Category 6: Oversubscription Blind Spots.** The simulation allocates vGPU instances at overcommit ratios of 1.0x (control), 1.25x, 1.5x, and 2.0x across 2-16 instances. Performance degradation is modeled as (1 - 1/ratio) * 100% when ratio > 1.0.

### 3.2 Noise Model

All simulations include realistic measurement noise matching real-world NVML instrumentation characteristics:

| Noise Source | Distribution | Parameters | Justification |
|-------------|-------------|------------|--------------|
| NVML memory readings | Gaussian | sigma = 512 KiB | NVML quantization and timing jitter |
| Spin-up latency | Log-normal | sigma = 0.3 | Right-skewed latency distributions |
| Bandwidth measurement | Gaussian | sigma = 2% of baseline | Memory subsystem variation |
| Background memory spikes | Poisson × Uniform | lambda = 0.01, range 1-64 MiB | Kernel driver allocations |
| Thermal readings | Gaussian | sigma = 1.0°C | Sensor precision |
| Fleet performance | Gaussian | sigma = 2% of peak | Manufacturing variation |

### 3.3 Detection Model

For each trial, three detection outcomes are recorded:

- **gpu-roofline detected:** The purpose-built detection mechanism fires (e.g., ghost bytes > 1 MiB threshold, contention drop > 5%, spin-up latency > 10ms, gap > 1%, straggler below 90th percentile threshold, or overcommit ratio > 1.0).
- **nvidia-smi detected:** Modeled as 0% for all categories except oversubscription (which is also 0% since nvidia-smi reports per-instance only).
- **DCGM detected:** Modeled identically to nvidia-smi for these waste categories.

The tool visibility model reflects the architectural limitations of device-centric monitoring, not implementation bugs. A reviewer should note that these detection rates are *modeled* based on tool capabilities, not measured against live tool output.

## 4. Simulation Infrastructure and Experimental Design

### 4.1 Implementation

The simulation is implemented in Rust as part of the `gpu-harness` crate. The binary `study_sim` accepts three parameters: `--scale` (default 1.0, producing 120,000 trials), `--seed` (default 42), and `--out` (output path). The simulation runs in 0.4 seconds on commodity hardware and produces byte-identical JSON output for a given seed.

Key implementation modules:

| Module | Responsibility | Size |
|--------|---------------|------|
| `study::runner` | Orchestrates all 6 categories, emits per-trial JSON | 838 lines |
| `study::scenarios` | Parameterized trial generators per category | 872 lines |
| `study::noise` | Noise injection matching protocol Section 14.3 | 150 lines |
| `study::stats` | Mann-Whitney U, bootstrap BCa, Holm-Bonferroni | 511 lines |
| `study::cost_model` | Parametric economic projections at 3 fleet scales | 246 lines |
| `sim::gpu_model` | GPU profile definitions (H100, H200, RTX 5090, etc.) | — |
| `sim::power` | Power and clock frequency model | — |
| `sim::thermal` | Newton's law of cooling with throttle factors | — |
| `sim::fleet` | Multi-GPU fleet with degradation injection | — |

### 4.2 Trial Structure

Each trial record contains:

```json
{
  "trial_id": 1,
  "category": 1,
  "category_name": "ghost_allocation",
  "arm": "treatment",
  "condition": "under_load_2g.20gb",
  "primary_metric_name": "ghost_bytes_measured",
  "primary_metric_value": 536350426.50,
  "control_metric_name": null,
  "control_metric_value": null,
  "independent_variables": { "teardown_method": "under_load", ... },
  "dependent_variables": { "ghost_bytes_injected": ..., ... },
  "ground_truth_waste": true,
  "gpu_roofline_detected": true,
  "nvidia_smi_detected": false,
  "dcgm_detected": false
}
```

### 4.3 Experimental Design Per Category

| Category | N | Design | Arms | Conditions | Primary Metric |
|----------|---|--------|------|-----------|---------------|
| Ghost allocations | 20,000 | Independent groups | Treatment / Control | 3 teardown × 3 MIG + 3 control | ghost_bytes_measured |
| Contention squeeze | 20,000 | Independent groups | Treatment / Control | 3 tenant counts + 1 baseline + 4 MIG | bandwidth_loss_pct |
| Provisioning overhead | 20,000 | Paired (within-trial) | Paired | 5 profiles × 3 loads × up to 4 concurrent | spin_up_latency_ms vs nvidia_smi_reported_ms |
| Burst-sustained gap | 20,000 | Paired (within-trial) | Paired | 5 GPU profiles × 3 workloads | gap_pct vs ideal 0% |
| Straggler tax | 20,000 | Independent groups | Treatment / Control | 5 degrade types × 3 severity × 3 fleet sizes + 3 controls | straggler_tax_pct |
| Oversubscription | 20,000 | Independent groups | Treatment / Control | 3 ratios × 4 instances + 4 controls | performance_degradation_pct |

### 4.4 Statistical Analysis Plan

The primary statistical test is selected based on the experimental design of each category:

- **Independent groups** (Categories 1, 2, 5, 6): Mann-Whitney U test (one-sided, treatment > control). Effect size: rank-biserial correlation *r* and Cohen's *d*.
- **Paired measurements** (Categories 3, 4): Wilcoxon signed-rank test (one-sided, actual > reported/ideal). Effect size: matched-pairs rank-biserial *r* and Cohen's *d_z* (standardized by SD of within-pair differences).

All six omnibus p-values are corrected for multiplicity using the Holm-Bonferroni step-down procedure (family-wise alpha = 0.05).

Supplementary per-category analyses include:
- **Category 1:** Per-condition Mann-Whitney U with Holm correction across 9 comparisons.
- **Category 2:** Friedman test across tenant counts; pairwise Wilcoxon signed-rank with Holm correction; Wilcoxon time-sliced vs MIG at each tenant count.
- **Category 3:** Paired t-test (actual vs reported latency); Kruskal-Wallis across load states; linear regression of latency on concurrent partition count.
- **Category 4:** Paired t-test (gap vs ideal); Kruskal-Wallis across workload types.
- **Category 5:** Mann-Whitney U treatment vs control; Kruskal-Wallis across fleet sizes; linear regression of tax on severity.
- **Category 6:** Mann-Whitney U treatment vs control; linear regression of degradation on overcommit ratio; Kruskal-Wallis across instance counts.

Bootstrap 95% confidence intervals (10,000 resamples, BCa-adjusted for the Rust implementation, percentile for the Python analysis) are computed for the median difference in each category.

Detection performance comparison across tools uses McNemar's exact binomial test for paired proportions on the same waste events.

## 5. Results

### 5.1 Cross-Category Omnibus Results

All six waste categories produce statistically significant effects after Holm-Bonferroni correction (Table 3). Effect sizes are uniformly large, ranging from d_z = 0.73 (burst-to-sustained gap, the most subtle category) to d = 8.55 (contention squeeze, the most dramatic).

**Table 3. Cross-Category Omnibus Statistical Tests**

| Category | N (a) | N (b) | Design | Primary Test | Statistic | Raw p | Holm p | d / d_z | r | 95% CI |
|----------|-------|-------|--------|-------------|-----------|-------|--------|---------|---|--------|
| Ghost allocations | 10,000 | 10,000 | independent | Mann-Whitney U | 99,937,086 | <1e-300 | <1e-300 | 2.46 | 1.00 | [527M, 545M] bytes |
| Contention squeeze | 10,000 | 10,000 | independent | Mann-Whitney U | 100,000,000 | <1e-300 | <1e-300 | 8.55 | 1.00 | [66.61, 66.68]% |
| Provisioning overhead | 20,000 | 20,000 | paired | Wilcoxon signed-rank | 200,010,000 | <1e-300 | <1e-300 | 2.01 | 1.00 | [244.3, 248.3] ms |
| Burst-sustained gap | 20,000 | 20,000 | paired | Wilcoxon signed-rank | 87,021,028 | <1e-300 | <1e-300 | 0.73 | 1.00 | [1.63, 1.77]% |
| Straggler tax | 10,000 | 10,000 | independent | Mann-Whitney U | 99,805,998 | <1e-300 | <1e-300 | 2.46 | 1.00 | [14.56, 15.09]% |
| Oversubscription | 10,000 | 10,000 | independent | Mann-Whitney U | 100,000,000 | <1e-300 | <1e-300 | 3.97 | 1.00 | [33.33, 33.33]% |

*Note: p-values are extremely small due to large sample sizes (N=10,000-20,000 per group). The effect sizes and confidence intervals are the more informative measures.*

### 5.2 Hypothesis Testing Summary

**H1 (Waste exists across all categories):** Supported. All six categories show statistically significant treatment effects with large standardized effect sizes (all d/d_z > 0.73). The weakest category (burst-to-sustained gap, d_z = 0.73) still qualifies as a "medium-to-large" effect by conventional benchmarks.

**H2 (Existing tools miss >80% of waste):** Supported. Across all categories, modeled `nvidia-smi` and DCGM detection rates are 0.0%, while `gpu-roofline` achieves 56.5-100% detection rates. McNemar's test confirms the difference is significant (p < 1e-300) for all tool comparisons. The weakest detection rate for `gpu-roofline` is 56.5% for the burst-to-sustained gap, where small gaps (< 1% threshold) are missed.

**H3 (Economic cost > $0):** Supported. Bootstrap 95% CI lower bounds for annual waste per GPU are $8,149 (8-GPU fleet), $21,215 (100-GPU fleet), and $33,809 (10,000-GPU fleet), all well above zero.

### 5.3 Detection Performance

**Table 2. Detection Rates by Tool Across All 6 Waste Categories**

| Category | nvidia-smi | DCGM | gpu-roofline | McNemar p (smi vs roofline) |
|----------|-----------|------|-------------|---------------------------|
| Ghost allocations | 0.0% | 0.0% | 99.9% | <1e-300 |
| Contention squeeze | 0.0% | 0.0% | 100.0% | <1e-300 |
| Provisioning overhead | 0.0% | 0.0% | 100.0% | <1e-300 |
| Burst-sustained gap | 0.0% | 0.0% | 56.5% | <1e-300 |
| Straggler tax | 0.0% | 0.0% | 94.7% | <1e-300 |
| Oversubscription | 0.0% | 0.0% | 100.0% | <1e-300 |
| **Overall** | **0.0%** | **0.0%** | **88.4%** | — |

### 5.4 Economic Impact

**Table 1. Annual Cost of Invisible GPU Waste by Category and Fleet Scale**

| Waste Category | d / d_z | Waste/Event | Annual $/GPU (100-GPU) | 8-GPU Fleet | 100-GPU Fleet | 10,000-GPU Fleet |
|---------------|---------|-------------|----------------------|-------------|---------------|-----------------|
| Ghost allocations | 2.46 | 511.5 MiB | $2,733 | $5,466 | $273,321 | $34,165,064 |
| Contention squeeze | 8.55 | 66.66% | $9,124 | $38,929 | $912,394 | $114,700,996 |
| Provisioning overhead | 2.01 | 246.09 ms | $1 | $2 | $125 | $31,190 |
| Burst-sustained gap | 0.73 | 1.69% | $278 | $1,481 | $27,762 | $3,084,702 |
| Straggler tax | 2.46 | 14.81% | $5,518 | $9,755 | $551,828 | $139,336,568 |
| Oversubscription | 3.97 | 33.33% | $3,650 | $9,733 | $365,000 | $48,666,667 |
| **Total** | — | — | **$21,304** | **$65,366** | **$2,130,430** | **$339,985,187** |

The cost model reveals a clear priority ordering: contention squeeze and straggler tax dominate fleet-level waste, contributing 69% of the total at the 100-GPU scale. Provisioning overhead is economically negligible (<$1/GPU/year). Ghost allocations and oversubscription occupy the middle tier. This ordering directly informs which categories should be prioritized for hardware validation.

**Cost Model Bootstrap 95% CIs:**

| Fleet Scale | Point Estimate ($/GPU/yr) | 95% CI |
|------------|--------------------------|--------|
| 8-GPU | $8,171 | [$8,149, $8,192] |
| 100-GPU | $21,304 | [$21,215, $21,389] |
| 10,000-GPU | $33,999 | [$33,809, $34,193] |

The narrow confidence intervals reflect the large sample size (120,000 trials) and the deterministic structure of the cost model. They should not be interpreted as precision about real-world costs—the uncertainty in fleet-scale operating parameters (teardown frequency, tenant count, utilization hours) likely dominates.

## 6. Sensitivity Analysis and Threats to Validity

### 6.1 Parameter Sensitivity

A one-at-a-time sensitivity analysis varies each cost model input parameter by ±10% and reports the impact on the 100-GPU fleet total. The most sensitive parameters are contention bandwidth loss, straggler tax percentage, and oversubscription waste percentage. Full results are reported in Table S4 of the supplement.

### 6.2 Internal Validity

**Simulation fidelity.** The noise model parameters (memory jitter sigma, latency log-sigma, bandwidth relative sigma) are based on instrumentation characteristics reported in NVIDIA documentation, but have not been empirically calibrated against hardware measurements. The thermal model uses Newton's law of cooling with published TDP and thermal specifications but has not been validated against measured temperature trajectories.

**Detection model.** The 0% detection rate for nvidia-smi and DCGM is based on architectural analysis of tool capabilities, not empirical measurement. A more nuanced evaluation against live tool output may reveal edge cases where these tools provide partial visibility.

### 6.3 External Validity

**Generalizability.** Results are specific to the simulated GPU profiles (H100, H200, RTX 5090, RTX 4090, MI300X) and the modeled workload characteristics. Real-world workloads exhibit more diversity than the three workload types simulated (compute-bound, memory-bound, mixed).

**Fleet parameters.** The cost model assumes specific operational parameters (teardowns per day, average tenants, utilization hours, etc.) that vary across providers and deployment contexts. The sensitivity analysis partially addresses this, but users should substitute their own parameters for fleet-specific projections.

### 6.4 Construct Validity

**Waste definition.** We define waste as GPU capacity that is economically allocated but not productively utilized due to virtualization mechanisms. This definition excludes intentional overprovisioning, scheduling headroom, and maintenance windows. A broader definition might yield different magnitudes.

### 6.5 Statistical Conclusion Validity

**Effect size inflation.** With 10,000-20,000 observations per group, even trivially small effects achieve statistical significance. We therefore emphasize effect sizes and confidence intervals over p-values throughout. Readers should evaluate the practical significance of each category's median effect against their operational context.

**Multiple comparisons.** The six omnibus comparisons are Holm-Bonferroni corrected. Per-category supplementary analyses use category-appropriate corrections. The family-wise error rate is controlled at alpha = 0.05 for each analysis level.

## 7. Reproducibility Package

### 7.1 Artifact Inventory

| Artifact | Path | SHA-256 |
|----------|------|---------|
| Raw simulation data (120K trials) | `raw/simulation-raw.json` | `e386a3d5...` |
| Summary tables | `derived/summary.md` | `47d1d322...` |
| Supplementary tables | `derived/supplement.md` | `f62be1cb...` |
| Machine-readable results | `derived/analysis-results.json` | `4a254a35...` |
| Provenance record | `derived/provenance.json` | verified |
| Source snapshot | `source_snapshot/` | per-file hashes |
| SHA-256 manifest | `SHA256SUMS.txt` | self-verifying |

### 7.2 Rerun Commands

**Simulation (produces byte-identical JSON):**
```bash
CARGO_TARGET_DIR=D:/cargo-target/gpu-tools \
  cargo run -p gpu-harness --release --bin study_sim -- \
  --out docs/study-results/simulation-raw.json
```

**Analysis (requires Python 3.11+, numpy, pandas, scipy):**
```bash
python scripts/analyze_study.py \
  --input docs/study-results/simulation-raw.json \
  --output-root D:/study-output \
  --repo-root . \
  --bootstrap-resamples 10000 \
  --seed 42
```

### 7.3 Software Versions

| Component | Version |
|-----------|---------|
| Rust | 1.93.0 (cargo) |
| Python | 3.11.4 |
| numpy | 2.4.3 |
| pandas | 3.0.1 |
| scipy | 1.17.1 |
| Platform | Windows 10 Pro 10.0.19045 |

## 8. Conclusion and Hardware Validation Roadmap

We have presented a reproducible simulation study demonstrating that six categories of invisible waste exist in multi-tenant GPU virtualized environments, that these categories are measurable with purpose-built instrumentation, and that existing monitoring tools provide no visibility into them. The economic impact, while model-dependent, is substantial: even the most conservative estimate ($8,149/GPU/year for a small 8-GPU fleet) suggests that invisible waste is a meaningful operational cost that merits active monitoring.

### Priority Ordering for Hardware Validation

The simulation results suggest the following priority order for bare-metal validation:

1. **Contention squeeze** — Largest per-GPU cost ($9,124/yr), extremely large effect size (d=8.55), 100% detection rate. Validate by measuring per-tenant bandwidth before and after tenant arrival on time-sliced H100.
2. **Straggler tax** — Second-largest cost ($5,518/yr), strong detection (94.7%). Validate by injecting degradation into one GPU in a real distributed training cluster.
3. **Oversubscription** — Third-largest cost ($3,650/yr), 100% detection. Validate by overcommitting vGPU VRAM allocations and measuring performance degradation.
4. **Ghost allocations** — Moderate cost ($2,733/yr), 99.9% detection. Validate by cycling MIG teardowns and measuring NVML memory deltas.
5. **Burst-sustained gap** — Moderate cost ($278/yr), lowest detection rate (56.5%). Validate by measuring H100 TFLOPS at cold start vs thermal equilibrium.
6. **Provisioning overhead** — Negligible cost (<$1/yr). Low priority for hardware validation, though latency measurements are straightforward.

### Hardware Phase Design

The follow-on study will execute 1,200 hardware trials on bare-metal H100 SXM5 systems following the same measurement protocol. Table S3 (simulation-vs-hardware effect size comparison) is reserved for that publication. The target venue for the hardware validation paper is IEEE Transactions on Parallel and Distributed Systems (TPDS).

### Open-Source Availability

The `gpu-roofline` measurement framework, simulation harness, and analysis pipeline are available as open-source Rust and Python code. The framework produces a single statically-linked binary (<10 MB) suitable for deployment alongside existing GPU monitoring infrastructure.

---

## References

[1] NVIDIA, "Multi-Instance GPU User Guide," NVIDIA Documentation, 2024.

[2] S. Williams, A. Waterman, and D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65-76, 2009.

[3] A. Harlap et al., "Addressing the straggler problem for iterative convergent parallel ML," *SoCC*, 2016.

---

*Supplementary tables (S1-S5) are available in `docs/study-results/supplement.md`. Full trial-level data, provenance chain, and SHA-256 manifest are archived in the study results package.*
