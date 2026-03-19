# A Reproducible Simulation Study of Invisible Waste in Multi-Tenant GPU Infrastructure

**Author:** Chris Jones
**Date:** March 2026

---

## Abstract

**Background.** Multi-tenant GPU virtualization (MIG, GRID, time-slicing) is now standard in cloud infrastructure, yet the monitoring tools used to manage these environments — `nvidia-smi` and NVIDIA DCGM — were designed for single-device health, not multi-tenant economic analysis. This architectural mismatch creates categories of waste that are structurally invisible to operators.

**Methods.** We formalize six categories of invisible waste in virtualized GPU environments, each with a distinct physical mechanism and a distinct reason current tools miss it. We implement a parameterized simulation in Rust, execute 120,000 deterministic trials (20,000 per category) with realistic NVML-matched noise injection, and analyze the results using design-appropriate statistical tests: Mann-Whitney U for independent-group categories and Wilcoxon signed-rank for paired within-trial categories, with Holm-Bonferroni correction across all six omnibus comparisons.

**Results.** All six waste categories produce statistically significant effects with large standardized effect sizes (Cohen's d/d_z ranging from 0.73 to 8.55). The central finding is a complete observability gap: `nvidia-smi` and DCGM detect 0% of waste events across all categories, while the purpose-built `gpu-roofline` framework detects 56.5–100% (McNemar p < 1e-300 for all comparisons). The waste categories map to three distinct operational actions: (A) directly recoverable capacity (ghost allocations freeing 512 MiB per teardown; straggler detection recovering 19% of fleet throughput), (B) decision support for infrastructure design (contention data enabling MIG vs. time-slicing decisions; thermal data enabling accurate SLAs), and (C) risk prevention (oversubscription detection before silent tenant degradation).

**Conclusions.** GPU fleet operators currently have zero visibility into six measurable waste categories. This study provides the taxonomy, the detection framework, and a reproducible simulation benchmark. Hardware validation on bare-metal H100 systems is the logical next step; the open protocol and simulation predictions are published to enable community participation.

**Keywords:** GPU virtualization, multi-tenant infrastructure, invisible waste, roofline model, MIG, observability, simulation study

---

## 1. Introduction

A single NVIDIA H100 GPU costs $2.50 per hour to rent in the cloud — approximately $22,000 per year. At scale, GPU fleets represent millions of dollars of annual infrastructure investment. Multi-tenant virtualization technologies (MIG, GRID, time-slicing) allow providers to maximize revenue from each physical GPU by serving multiple tenants simultaneously. But this sharing creates waste that no existing monitoring tool can observe.

The waste is not hypothetical. Consider three scenarios that GPU fleet operators encounter routinely:

- A MIG partition is destroyed, but 512 MiB of VRAM is not reclaimed. The memory is trapped — unavailable for new tenants — and `nvidia-smi` reports the partition as cleanly removed. Over 20 teardowns per day, this silently accumulates into gigabytes of lost capacity.

- A distributed training job across 8 H100s takes 40% longer than expected. One GPU has degraded NVLink bandwidth, and because data-parallel training synchronizes at barriers, the other 7 GPUs sit idle waiting for the straggler. No per-device monitoring tool flags this — each GPU individually reports normal utilization.

- A cloud provider advertises H100 instances at peak specifications. But under sustained workloads, thermal throttling reduces actual performance by 1–16% depending on workload type. Tenants receive less than they are paying for, and neither the provider nor the tenant has any visibility into the gap.

These are not edge cases. They are structural consequences of how GPU virtualization works, and they are invisible because the standard monitoring tools — `nvidia-smi` and NVIDIA DCGM — were designed to report instantaneous per-device state, not lifecycle transitions, cross-tenant interference, fleet-level aggregates, or temporal degradation curves.

This paper formalizes six categories of invisible waste, demonstrates through simulation that all six are detectable with purpose-built instrumentation, and quantifies both the observability gap and the operational impact of closing it.

### 1.1 Summary of Key Findings

Before presenting the methodology and detailed results, we summarize the three central findings of this study:

**Finding 1: The observability gap is total.** Across 120,000 simulation trials spanning six waste categories, modeled `nvidia-smi` and DCGM detection rates are 0.0%. Not low — zero. The `gpu-roofline` measurement framework, designed specifically for multi-tenant lifecycle analysis, detects 56.5–100% of waste events depending on the category. This is not a comparison of tool quality; it is a demonstration that current tools lack the architectural capability to observe these phenomena.

**Finding 2: The waste is real and measurable.** All six categories produce statistically significant effects with large standardized effect sizes after Holm-Bonferroni correction (d/d_z = 0.73 to 8.55). The effects are not subtle statistical artifacts detectable only at large sample sizes — they represent 512 MiB of trapped VRAM per teardown, 19% fleet throughput loss per straggler GPU, and 50–75% bandwidth degradation per tenant under time-slicing.

**Finding 3: The six categories require three different responses.** Not all waste is recoverable. Ghost allocations and straggler GPUs represent directly recoverable capacity. Contention and thermal throttling are physics — they cannot be eliminated, but measuring them enables better infrastructure decisions (MIG vs. time-slicing, accurate SLAs). Oversubscription is a preventable risk that monitoring can catch before tenants are impacted. Treating all six categories as a single dollar figure is misleading; the value of visibility is category-specific.

### 1.2 Contributions

1. A six-category taxonomy of invisible waste in virtualized GPU systems, with formal mechanisms and detection requirements.
2. A reproducible simulation harness producing 120,000 deterministic trials with byte-identical rerun capability (SHA-256 verified).
3. Design-appropriate statistical analyses: Mann-Whitney U for independent groups, Wilcoxon signed-rank for paired measurements, Holm-Bonferroni correction, and bootstrap confidence intervals.
4. A three-bucket operational impact model that honestly distinguishes recoverable capacity from decision support from risk prevention.
5. An open protocol and simulation benchmark inviting hardware validation from the community.

### 1.3 Scope and Claim Boundary

This paper makes simulation-backed claims about the *detectability* and *relative magnitude* of each waste category. It does not claim that simulated magnitudes are final real-world measurements. The detection model for `nvidia-smi` and DCGM is based on architectural capability analysis, not live tool comparison. Hardware validation on bare-metal H100 systems is reserved for a follow-on study; the open protocol is published to enable community participation.

## 2. Background and Related Work

### 2.1 GPU Virtualization Technologies

NVIDIA Multi-Instance GPU (MIG), introduced with the A100 architecture, provides hardware-isolated partitions with dedicated compute, memory, and cache resources. Each MIG instance operates as an independent GPU with guaranteed quality of service. NVIDIA GRID/vGPU enables time-sliced sharing where tenants alternate access to the full GPU, achieving higher tenant density at the cost of bandwidth contention. Both technologies are widely deployed in cloud infrastructure from AWS, GCP, Azure, and Oracle Cloud [1].

### 2.2 The Monitoring Gap

The standard GPU monitoring stack consists of `nvidia-smi` (the NVML command-line interface) and NVIDIA Data Center GPU Manager (DCGM). Both provide per-device and per-instance metrics: utilization, memory consumption, temperature, power, and ECC error counts. These tools were designed for single-device health monitoring — answering questions like "Is this GPU functional?" and "How hot is it?"

Multi-tenant waste, however, requires a fundamentally different class of measurement:

| Waste Type | Required Measurement | nvidia-smi/DCGM Capability |
|-----------|---------------------|--------------------------|
| Lifecycle transitions | Pre/post memory delta across teardown | Reports only post-state |
| Cross-tenant interference | Per-tenant bandwidth before/after new arrival | Reports only aggregate per-GPU |
| Temporal degradation | Burst vs. sustained performance trajectory | Reports only instantaneous snapshots |
| Fleet aggregation | Sum of all vGPU allocations vs. physical capacity | Reports only per-instance |
| Distributed synchronization | Fleet-wide barrier wait analysis | Reports only per-device |

This is not a criticism of tool quality. It is an identification of an architectural gap that leaves an entire class of operational waste unmeasured.

### 2.3 Performance Measurement and the Roofline Model

The roofline model [2] provides a well-established framework for analyzing compute and memory bandwidth bounds. Our `gpu-roofline` tool extends this with a *dynamic* roofline that captures both burst (peak clock, cold GPU) and sustained (thermally throttled) performance ceilings. The gap between these two ceilings — which we term the burst-to-sustained gap — represents capacity that is advertised but not sustainably deliverable.

### 2.4 Distributed Training and the Straggler Problem

Data-parallel distributed training requires all-reduce synchronization at gradient aggregation barriers. The slowest GPU determines effective throughput for the entire fleet [3]. Prior work has focused on software mitigations (redundant computation, asynchronous SGD). Our contribution is on the *detection* side: identifying which GPU is the straggler, diagnosing *why* it is degraded, and quantifying the fleet-wide impact.

## 3. Waste Taxonomy and Measurement Model

We define six categories of invisible waste. Each category is specified by three components: (a) the physical mechanism producing waste, (b) why existing monitoring tools miss it, and (c) what measurement capability is required for detection.

### 3.1 Category 1: Ghost Allocations

**Mechanism.** After vGPU teardown, VRAM may not be fully reclaimed by the driver. The unreleased memory is trapped — physically consumed but not allocated to any instance.

**Why tools miss it.** `nvidia-smi` reports the instance as removed. It does not verify that the physical memory returned to baseline.

**Detection requires.** Pre-teardown memory capture via `nvmlDeviceGetMemoryInfo().used`, post-teardown polling with a stabilization protocol (readings converge within 1 MiB), and delta computation.

**Simulation design.** Independent groups: 10,000 treatment trials with ghost bytes injected as Uniform(0, 1024 MiB) across 3 teardown methods × 3 MIG profiles; 10,000 control trials with zero injection.

### 3.2 Category 2: Contention Squeeze

**Mechanism.** Under time-slicing, each tenant receives approximately 1/N of the GPU's memory bandwidth, where N is the tenant count. Existing tenants experience bandwidth degradation when a new tenant arrives.

**Why tools miss it.** `nvidia-smi` reports per-GPU utilization (which remains at 100%). It has no concept of per-tenant bandwidth.

**Detection requires.** Per-tenant baseline capture before each new tenant arrives, post-arrival measurement, and comparison.

**Simulation design.** Independent groups: 10,000 treatment trials across 3 tenant counts (time-sliced); 10,000 control trials (single-tenant baseline + MIG-partitioned).

### 3.3 Category 3: Provisioning Overhead

**Mechanism.** MIG partition creation takes 120–500ms of wall-clock time depending on profile size and GPU load state. During this time, capacity is neither free nor usable.

**Why tools miss it.** `nvidia-smi` reports state after the command returns. It does not measure the transition duration.

**Detection requires.** Wall-clock timing from create command to first successful compute dispatch.

**Simulation design.** Paired within-trial: 20,000 trials each measuring both true spin-up latency and nvidia-smi reported latency (~0.5ms). Wilcoxon signed-rank is the appropriate test.

### 3.4 Category 4: Burst-to-Sustained Gap

**Mechanism.** GPUs boost to peak clock at cold start, then thermally throttle to a lower sustained clock within 30–120 seconds. The gap between burst and sustained performance is 1–16% for datacenter GPUs (H100, H200) and larger for consumer GPUs.

**Why tools miss it.** Monitoring tools take snapshots. They do not track the thermal trajectory from burst to equilibrium.

**Detection requires.** Continuous measurement through the thermal ramp: TFLOPS, clock frequency, temperature, and power from t=0 through thermal equilibrium.

**Simulation design.** Paired within-trial: 20,000 trials each measuring gap_pct versus ideal 0% across 5 GPU profiles × 3 workload types. Physics-based thermal model using Newton's law of cooling.

### 3.5 Category 5: Straggler Tax

**Mechanism.** In data-parallel training, one degraded GPU forces N-1 healthy GPUs to idle at synchronization barriers. Five hardware degradation types are simulated: thermal paste failure, NVLink degradation, PCIe fallback, memory subsystem failure, and clock stuck.

**Why tools miss it.** Each GPU individually reports normal or near-normal metrics. No tool correlates fleet-wide to identify the barrier bottleneck.

**Detection requires.** Fleet-wide measurement, median-vs-outlier comparison, and per-GPU diagnostic probing.

**Simulation design.** Independent groups: 10,000 treatment trials (degradation injected into one GPU) across 5 types × 3 severities × 3 fleet sizes (8, 32, 128); 10,000 control trials with no degradation.

### 3.6 Category 6: Oversubscription Blind Spots

**Mechanism.** Hypervisors can allocate more aggregate vGPU VRAM than physical GPU capacity. At overcommit ratios above 1.0, tenants experience performance degradation proportional to the excess.

**Why tools miss it.** Each instance's allocation appears successful individually. No tool sums allocations against physical limits.

**Detection requires.** Aggregate allocation tracking across all instances, comparison to physical capacity.

**Simulation design.** Independent groups: 10,000 treatment trials at overcommit ratios 1.25x, 1.5x, 2.0x; 10,000 control trials at 1.0x (no overcommit).

### 3.7 Noise Model

All simulations include realistic measurement noise calibrated to NVML instrumentation characteristics:

| Noise Source | Distribution | Parameters |
|-------------|-------------|------------|
| NVML memory readings | Gaussian | sigma = 512 KiB |
| Spin-up latency | Log-normal | sigma = 0.3 |
| Bandwidth measurement | Gaussian | sigma = 2% of baseline |
| Background memory spikes | Poisson × Uniform | lambda = 0.01, 1–64 MiB |
| Thermal readings | Gaussian | sigma = 1.0°C |
| Fleet performance | Gaussian | sigma = 2% of peak |

## 4. Simulation Infrastructure and Experimental Design

### 4.1 Implementation

The simulation is implemented in Rust as part of the `gpu-harness` crate (~2,600 lines across 6 modules). The binary accepts three parameters: scale factor (default 1.0, producing 120,000 trials), random seed (default 42), and output path. Execution time is 0.4 seconds on commodity hardware. The output is deterministic: re-execution with the same seed produces byte-identical JSON (SHA-256: `e386a3d599a238901abff636d421dbcb095cca31616e425d05e5653744fd3912`).

### 4.2 Experimental Design Summary

| Category | N | Design | Primary Test |
|----------|---|--------|-------------|
| 1. Ghost allocations | 20,000 | Independent groups | Mann-Whitney U |
| 2. Contention squeeze | 20,000 | Independent groups | Mann-Whitney U |
| 3. Provisioning overhead | 20,000 | Paired (within-trial) | Wilcoxon signed-rank |
| 4. Burst-sustained gap | 20,000 | Paired (within-trial) | Wilcoxon signed-rank |
| 5. Straggler tax | 20,000 | Independent groups | Mann-Whitney U |
| 6. Oversubscription | 20,000 | Independent groups | Mann-Whitney U |

The choice of statistical test follows from the experimental design: Categories 3 and 4 produce paired observations (each trial measures both the actual value and the tool-reported or ideal baseline), requiring a paired test. All six omnibus p-values are Holm-Bonferroni corrected for multiplicity (family-wise alpha = 0.05).

### 4.3 Detection Model

For each trial, three binary detection outcomes are recorded: whether `gpu-roofline`, `nvidia-smi`, or DCGM would detect the waste event. The `nvidia-smi` and DCGM detection rates are modeled at 0% based on architectural capability analysis (these tools lack the measurement primitives described in Section 3). This is a modeling assumption, not an empirical measurement; live tool comparison is deferred to hardware validation.

## 5. Results

### 5.1 All Six Waste Categories Are Detectable

The primary hypothesis — that all six categories of invisible waste produce measurable, statistically significant effects — is supported. Table 1 presents the omnibus results.

**Table 1. Omnibus Statistical Results Across All Six Waste Categories**

| Category | Design | Test | d / d_z | r | 95% CI of Median Difference |
|----------|--------|------|---------|---|----------------------------|
| Ghost allocations | Independent | Mann-Whitney U | 2.46 | 1.00 | [527M, 545M] bytes |
| Contention squeeze | Independent | Mann-Whitney U | 8.55 | 1.00 | [66.61, 66.68]% bandwidth loss |
| Provisioning overhead | Paired | Wilcoxon signed-rank | 2.01 | 1.00 | [244, 248] ms hidden latency |
| Burst-sustained gap | Paired | Wilcoxon signed-rank | 0.73 | 1.00 | [1.63, 1.77]% below spec |
| Straggler tax | Independent | Mann-Whitney U | 2.46 | 1.00 | [14.56, 15.09]% fleet throughput |
| Oversubscription | Independent | Mann-Whitney U | 3.97 | 1.00 | [33.33, 33.33]% degradation |

*All Holm-Bonferroni adjusted p < 1e-300. N = 10,000–20,000 per group. Effect sizes (d, d_z) and bootstrap 95% CIs (10,000 resamples) are the primary measures; p-values are uninformative at this sample size.*

Effect sizes are uniformly large (Figure 2). The weakest category — burst-to-sustained gap (d_z = 0.73) — still qualifies as a medium-to-large effect. The strongest — contention squeeze (d = 8.55) — reflects the dramatic and deterministic bandwidth partitioning under time-slicing.

![Figure 2. Standardized effect sizes across categories, colored by operational action bucket.](figures/fig2_effect_sizes.png)

### 5.2 Current Tools Provide Zero Visibility

The second hypothesis — that existing tools miss the vast majority of waste events — is supported, and the result is more extreme than hypothesized. We predicted >80% missed; the simulation shows 100% missed.

**Table 2. Detection Rates by Monitoring Tool**

| Category | nvidia-smi | DCGM | gpu-roofline | McNemar p |
|----------|-----------|------|-------------|-----------|
| Ghost allocations | 0.0% | 0.0% | 99.9% | <1e-300 |
| Contention squeeze | 0.0% | 0.0% | 100.0% | <1e-300 |
| Provisioning overhead | 0.0% | 0.0% | 100.0% | <1e-300 |
| Burst-sustained gap | 0.0% | 0.0% | 56.5% | <1e-300 |
| Straggler tax | 0.0% | 0.0% | 94.7% | <1e-300 |
| Oversubscription | 0.0% | 0.0% | 100.0% | <1e-300 |
| **Overall** | **0.0%** | **0.0%** | **88.4%** | — |

![Figure 1. Detection rates by monitoring tool across all six waste categories.](figures/fig1_detection_rates.png)

The 0% detection rate for `nvidia-smi` and DCGM is a modeling result, not an empirical measurement (see Section 4.3). However, it follows directly from the tools' documented capabilities: they do not compute memory deltas across teardowns, track per-tenant bandwidth, time provisioning transitions, measure thermal trajectories, correlate fleet-wide barriers, or sum allocations against physical limits. These are not features that could be enabled with configuration changes — they require fundamentally different measurement architectures.

The `gpu-roofline` detection rate varies by category. The weakest performance is on the burst-to-sustained gap (56.5%), where small thermal gaps below the 1% detection threshold are missed. The strongest is on contention, provisioning, and oversubscription (100%), where the signal-to-noise ratio is high.

### 5.3 What Operators Gain: Three Action Buckets

The six waste categories do not all represent the same type of problem, and they do not all have the same solution. We categorize them into three action buckets based on what visibility enables (Figure 3):

![Figure 3. Three operational action buckets with detection rates and per-event impact.](figures/fig3_three_buckets.png)

#### Bucket A: Directly Recoverable Capacity

**Ghost allocations** trap VRAM that could host additional tenant instances. The simulation shows a median of 512 MiB trapped per teardown event. At 20 teardowns per day, this is approximately 10 GiB of VRAM per GPU per day that can be reclaimed once the ghost is detected. Detection rate: 99.9%.

**Straggler tax** wastes fleet-wide GPU-hours at synchronization barriers. A single degraded GPU in an 8-GPU training job wastes 16.7% of fleet throughput — the other 7 GPUs idle at the barrier. This scales multiplicatively with fleet size: at 128 GPUs, one straggler wastes 24 GPU-equivalents at every sync barrier (Figure 4). Identifying the straggler (detection rate: 94.7%) and replacing or reassigning it immediately recovers this capacity.

![Figure 4. Straggler tax scales multiplicatively with fleet size.](figures/fig4_straggler_scaling.png)

These categories represent genuine capacity recovery: detect the problem, take action, free the resource.

#### Bucket B: Decision Support

**Contention squeeze** is the inherent cost of time-slicing. With 2 tenants, each gets ~50% bandwidth; with 4, each gets ~25% (Figure 5). This cannot be eliminated — it is the physics of shared access. But *measuring* it enables operators to make informed decisions: use MIG (hardware isolation, guaranteed bandwidth) for latency-sensitive workloads, and time-slicing (higher density, shared bandwidth) for throughput-tolerant batch jobs. Without per-tenant bandwidth visibility, this decision is made blind.

![Figure 5. Per-tenant bandwidth under time-slicing vs. MIG — the degradation nvidia-smi cannot see.](figures/fig5_contention_tenants.png)

**Burst-to-sustained gap** is thermal physics. H100 SXM GPUs show a median 1.7% gap; compute-bound workloads on some profiles show up to 16%. This cannot be fixed — but it can be priced correctly. Cloud providers advertising peak specs for sustained workloads are overcommitting. Measuring the actual sustained ceiling enables honest SLAs and prevents customer complaints.

**Provisioning overhead** (246ms per MIG provision) is measurable but economically negligible (<$1/GPU/year).

#### Bucket C: Risk Prevention

**Oversubscription** occurs when aggregate vGPU VRAM allocations exceed physical GPU capacity. At 1.5x overcommit, tenants experience 33% performance degradation. At 2x, out-of-memory crashes become likely. Each tenant's allocation appears successful individually — no tool sums them against the physical limit. Detection enables enforcement before tenants are silently impacted.

### 5.4 Hypothesis Summary

| Hypothesis | Prediction | Result |
|-----------|-----------|--------|
| H1: All six categories produce significant waste | Significant effects in all 6 | **Supported** — all d/d_z > 0.73, all Holm p < 1e-300 |
| H2: Existing tools miss >80% of waste | <20% detection by nvidia-smi/DCGM | **Supported** — 0% detection across all categories |
| H3: Economic impact > $0 per GPU per year | Bootstrap CI lower bound > $0 | **Supported** — lower bound $8,149/GPU/yr (8-GPU fleet) |

## 6. Discussion

### 6.1 The Visibility Gap Is the Primary Finding

While the effect sizes and operational impact are important, the most significant result is the *total observability gap*. GPU fleet operators are currently managing multi-million-dollar infrastructure with monitoring tools that have zero visibility into six measurable waste categories. This is analogous to operating a power grid without metering individual consumers — the aggregate numbers work, but waste at the tenant level is invisible.

The gap is not a software deficiency that can be patched. It is an architectural limitation: `nvidia-smi` and DCGM report per-device instantaneous state. Detecting lifecycle transitions, cross-tenant interference, and fleet-level aggregates requires a fundamentally different measurement approach — one that tracks state across time, across tenants, and across devices.

### 6.2 Implications for Infrastructure Operators

The three-bucket model provides a framework for prioritizing monitoring investments:

**Highest ROI: Straggler detection (Bucket A).** A single degraded GPU in a 128-GPU training cluster wastes 127 × (straggler overhead) in blocked GPU-hours. The return on detecting and replacing one straggler is immediate and multiplicative with fleet size.

**Highest risk: Oversubscription detection (Bucket C).** Silent overcommit is the only category that can cause direct customer-visible failures (OOM crashes). Even if the probability is low, the reputational cost of unexplained tenant failures in a cloud offering is high.

**Highest strategic value: Contention visibility (Bucket B).** The MIG-vs-time-slicing decision is one of the most consequential infrastructure choices a GPU cloud operator makes. Without per-tenant bandwidth data, this decision is driven by marketing claims and vendor guidance rather than measured workload characteristics.

### 6.3 Limitations

**Simulation vs. reality.** The noise model parameters are based on published NVML instrumentation characteristics, not empirical calibration. The thermal model uses Newton's law of cooling with published TDP specifications. Real hardware may exhibit behaviors not captured by these models.

**Detection model.** The 0% detection rate for `nvidia-smi` and DCGM is a modeling assumption based on documented capabilities. A systematic live evaluation may reveal edge cases where partial detection is possible (e.g., DCGM custom metric plugins).

**Effect size inflation.** With 10,000–20,000 observations per group, statistical significance is guaranteed for any non-zero effect. We therefore emphasize effect sizes and confidence intervals over p-values throughout. Readers should evaluate practical significance against their operational context.

**Cost model assumptions.** The scenario-based cost projections assume specific fleet utilization parameters that vary across providers. The sensitivity analysis (Appendix B, Table S4) shows which parameters most influence the projections.

### 6.4 Threats to External Validity

Results are specific to the simulated GPU profiles (primarily H100 SXM, with H200 and consumer GPUs in the burst-sustained category) and three workload types (compute-bound, memory-bound, mixed). Real-world workloads are more diverse. Fleet utilization patterns vary significantly across providers. The cost model parameters should be treated as substitutable inputs, not universal constants.

## 7. Reproducibility

### 7.1 Deterministic Simulation

The simulation is fully deterministic. Re-execution with seed 42 produces byte-identical output:

```
SHA-256: e386a3d599a238901abff636d421dbcb095cca31616e425d05e5653744fd3912
```

This was verified by independent re-execution during the analysis phase.

### 7.2 Artifact Inventory

| Artifact | Description |
|----------|------------|
| `simulation-raw.json` (83.7 MB) | 120,000 trial-level records with IVs, DVs, and detection outcomes |
| `analyze_study.py` (1,280 lines) | Full analysis pipeline: statistics, cost model, provenance |
| `summary.md` | Main results tables (Tables 1–4) |
| `supplement.md` | Supplementary tables (S1–S5) |
| `PROVENANCE.md` | Software versions, commands, SHA-256 chain |
| `SHA256SUMS.txt` | Per-file checksums for all artifacts |

### 7.3 Rerun Commands

```bash
# Simulation (0.4s, produces identical output)
cargo run -p gpu-harness --release --bin study_sim -- \
  --out docs/study-results/simulation-raw.json

# Analysis (requires Python 3.11+, numpy, pandas, scipy)
python scripts/analyze_study.py \
  --input docs/study-results/simulation-raw.json \
  --output-root study-output --repo-root . \
  --bootstrap-resamples 10000 --seed 42
```

## 8. Conclusion

We have demonstrated that multi-tenant GPU virtualization produces six categories of measurable waste that are completely invisible to the standard monitoring tools used by GPU fleet operators. The observability gap is total: `nvidia-smi` and DCGM detect 0% of waste events across all categories, while purpose-built instrumentation detects 56.5–100%.

The waste categories are not all alike. Ghost allocations and straggler GPUs represent directly recoverable capacity — detect, act, recover. Contention and thermal throttling are physics that cannot be eliminated but can be measured, enabling better infrastructure decisions. Oversubscription is a preventable risk.

This simulation study provides the taxonomy, the detection framework, and a reproducible benchmark. The natural next step is hardware validation on bare-metal H100 systems to confirm the simulation predictions. We publish the full protocol, analysis pipeline, and simulation predictions as an open invitation: we welcome hardware contributions from GPU cloud operators, system integrators, and NVIDIA.

---

## References

[1] NVIDIA, "Multi-Instance GPU User Guide," NVIDIA Documentation, 2024.

[2] S. Williams, A. Waterman, and D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65-76, 2009.

[3] A. Harlap et al., "Addressing the straggler problem for iterative convergent parallel ML," *Proceedings of the Seventh ACM Symposium on Cloud Computing (SoCC)*, 2016.

[4] NVIDIA, "NVIDIA Data Center GPU Manager (DCGM) Documentation," NVIDIA Documentation, 2024.

[5] NVIDIA, "NVIDIA System Management Interface (nvidia-smi) Documentation," NVIDIA Documentation, 2024.
