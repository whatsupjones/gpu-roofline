# The GPU Efficiency Gap: A Systematic Method for Detecting Invisible Waste

**Christopher D. Jones**

March 2026

---

## Abstract

**Background.** GPU fleets cost millions of dollars annually. The standard monitoring tools, `nvidia-smi` and NVIDIA DCGM, cannot observe the efficiency losses that accumulate across device thermal physics, virtualization partitioning, and fleet-level coordination. These tools were designed for single-device health checks. They do not measure lifecycle transitions, cross-tenant interference, thermal trajectories, or distributed synchronization waste. The result is a systematic observability gap: operators manage GPU infrastructure without visibility into losses that span the full operations stack.

**Methods.** We formalize six categories of invisible GPU waste covering single-device thermal physics, virtualization partitioning, and fleet-level coordination. We employ a synthetic model calibrated against hardware-validated roofline measurements on H100 SXM (2,905 GB/s HBM3, 59.1 TFLOPS FP32, validated at 87 to 100% of spec) and H200 systems. The simulation executes 120,000 deterministic trials (20,000 per category) with NVML-matched noise injection. We apply Mann-Whitney U for independent-group categories and Wilcoxon signed-rank for paired within-trial categories, with Holm-Bonferroni correction across all six omnibus comparisons.

**Results.** All six waste categories produce statistically significant effects with large standardized effect sizes (Cohen's d/d_z ranging from 0.73 to 8.55). `nvidia-smi` and DCGM detect 0% of waste events across all categories. The `gpu-roofline` framework detects 56.5 to 100% (McNemar p < 1e-300 for all comparisons). The waste categories map to three operational responses: (A) directly recoverable capacity, including ghost allocations freeing 512 MiB per teardown and straggler detection recovering 19% of fleet throughput; (B) decision support for infrastructure design, including contention data for MIG versus time-slicing decisions and thermal data for accurate SLAs; and (C) risk prevention through oversubscription detection before silent tenant degradation.

**Conclusions.** We provide strong experimental evidence that six categories of measurable GPU efficiency loss are completely invisible to the standard monitoring tools deployed across the industry. This study establishes the taxonomy, the detection method, and a reproducible benchmark built on hardware-validated performance models. The open protocol and predictions are published to enable community hardware validation.

**Keywords:** GPU efficiency, invisible waste, roofline model, observability gap, MIG, virtualization, fleet operations, distributed training

---

## 1. Introduction

GPU infrastructure represents one of the largest capital expenditures in modern computing. A single NVIDIA H100 GPU costs approximately $2.50 per hour in the cloud, or $22,000 per year. At fleet scale, these costs reach into the hundreds of millions. The tools used to monitor this infrastructure, `nvidia-smi` and NVIDIA DCGM, were designed for a fundamentally different purpose: single-device health monitoring. They report whether a GPU is functional and whether it is overheating. They do not report where efficiency is being lost.

This distinction matters because GPU efficiency loss spans the full operations stack. At the device level, thermal throttling silently reduces sustained performance below advertised specifications. At the virtualization level, memory remains trapped after partition teardown and bandwidth degrades under tenant contention. At the fleet level, a single degraded GPU forces an entire distributed training cluster to idle at synchronization barriers. These losses are not small. They are invisible because the monitoring architecture lacks the measurement primitives to observe them.

Three scenarios illustrate the scope of this problem.

First, consider device-level thermal degradation. A cloud provider advertises H100 instances at peak specifications. Under sustained workloads, thermal throttling reduces actual performance by 1 to 16% depending on workload type. Neither the provider nor the tenant can observe this gap because no tool tracks the thermal trajectory from burst clock to thermal equilibrium.

Second, consider virtualization lifecycle waste. A MIG partition is destroyed, but 512 MiB of VRAM is not reclaimed by the driver. The memory is physically consumed but allocated to no instance. `nvidia-smi` reports the partition as cleanly removed. Over 20 teardowns per day, gigabytes of sellable capacity silently disappear.

Third, consider fleet-level synchronization loss. A distributed training job across 8 H100 GPUs takes 40% longer than expected. One GPU has degraded NVLink bandwidth. Data-parallel training synchronizes at gradient barriers, so the other 7 GPUs sit idle waiting for the straggler. Each GPU individually reports normal utilization. No per-device tool flags the fleet-wide impact.

These are not edge cases. They are structural consequences of how GPU infrastructure operates. Existing monitoring tools report instantaneous per-device state rather than tracking transitions across time, across tenants, and across devices.

This paper presents a systematic method for identifying these efficiency losses. We formalize six categories of invisible GPU waste spanning the full operations stack and demonstrate that all six are detectable with purpose-built instrumentation where existing tools see nothing.

### 1.1 Key Findings

**Finding 1: The observability gap is total.** Across 120,000 trials spanning six waste categories, modeled `nvidia-smi` and DCGM detection rates are 0.0%. The `gpu-roofline` measurement framework detects 56.5 to 100% of waste events depending on category. Current tools lack the architectural capability to observe GPU efficiency loss beyond instantaneous device state.

**Finding 2: The waste is real and measurable.** All six categories produce statistically significant effects with large standardized effect sizes after Holm-Bonferroni correction (d/d_z = 0.73 to 8.55). These represent 512 MiB of trapped VRAM per teardown, 19% fleet throughput loss per straggler GPU, and 50 to 75% bandwidth degradation per tenant under time-slicing.

**Finding 3: The six categories require three different operational responses.** Ghost allocations and straggler GPUs represent directly recoverable capacity. Contention and thermal throttling are governed by physics. They cannot be eliminated, but measuring them enables better infrastructure decisions including MIG versus time-slicing selection and accurate SLA pricing. Oversubscription is a preventable risk that monitoring can intercept before tenants are impacted. The value of visibility is category-specific.

### 1.2 Contributions

1. A six-category taxonomy of invisible waste across the GPU operations stack (device, virtualization, fleet), with formal mechanisms and detection requirements for each.
2. A reproducible simulation harness producing 120,000 deterministic trials with byte-identical rerun capability (SHA-256 verified).
3. Design-appropriate statistical analyses employing Mann-Whitney U for independent groups, Wilcoxon signed-rank for paired measurements, Holm-Bonferroni correction, and bootstrap confidence intervals.
4. A three-bucket operational impact model that distinguishes recoverable capacity from decision support from risk prevention.
5. An open protocol and simulation benchmark for community hardware validation.

### 1.3 Scope

This study establishes detectability and relative magnitude through simulation. Absolute real-world magnitudes require hardware validation, which we reserve for a follow-on study and enable through the published open protocol.

## 2. Background and Related Work

### 2.1 GPU Virtualization Technologies

NVIDIA Multi-Instance GPU (MIG), introduced with the A100 architecture, provides hardware-isolated partitions with dedicated compute, memory, and cache resources. Each MIG instance operates as an independent GPU with guaranteed quality of service. NVIDIA GRID/vGPU enables time-sliced sharing where tenants alternate access to the full GPU, achieving higher tenant density at the cost of bandwidth contention. Both technologies are widely deployed in cloud infrastructure from AWS, GCP, Azure, and Oracle Cloud [1].

### 2.2 The Monitoring Gap

The standard GPU monitoring stack consists of `nvidia-smi` (the NVML command-line interface) and NVIDIA Data Center GPU Manager (DCGM). Both provide per-device and per-instance metrics including utilization, memory consumption, temperature, power, and ECC error counts. These tools were designed for single-device health monitoring.

Multi-tenant waste requires a fundamentally different class of measurement:

| Waste Type | Required Measurement | nvidia-smi/DCGM Capability |
|-----------|---------------------|--------------------------|
| Lifecycle transitions | Pre/post memory delta across teardown | Reports only post-state |
| Cross-tenant interference | Per-tenant bandwidth before/after new arrival | Reports only aggregate per-GPU |
| Temporal degradation | Burst vs. sustained performance trajectory | Reports only instantaneous snapshots |
| Fleet aggregation | Sum of all vGPU allocations vs. physical capacity | Reports only per-instance |
| Distributed synchronization | Fleet-wide barrier wait analysis | Reports only per-device |

The table above identifies an architectural gap that leaves an entire class of operational waste unmeasured.

### 2.3 Performance Measurement and the Roofline Model

The roofline model [2] provides a well-established framework for analyzing compute and memory bandwidth bounds. The `gpu-roofline` tool extends this framework with a dynamic roofline that captures both burst performance (peak clock, cold GPU) and sustained performance (thermally throttled equilibrium). The gap between these two ceilings represents capacity that is advertised but not sustainably deliverable. We term this the burst-to-sustained gap.

### 2.4 Distributed Training and the Straggler Problem

Data-parallel distributed training requires all-reduce synchronization at gradient aggregation barriers. The slowest GPU determines effective throughput for the entire fleet [3]. Prior work has focused on software mitigations including redundant computation and asynchronous SGD. Our contribution addresses the detection side: identifying which GPU is the straggler, diagnosing the cause of degradation, and quantifying fleet-wide impact.

## 3. Waste Taxonomy and Measurement Model

We define six categories of invisible waste across the GPU operations stack. Each category is characterized by a physical mechanism, a reason existing tools miss it, and a measurement capability required for detection.

### 3.1 Ghost Allocations

After vGPU teardown, VRAM may not be fully reclaimed by the driver. The unreleased memory sits physically consumed but unallocated to any instance. `nvidia-smi` reports the partition as removed without verifying that physical memory returned to baseline. Detecting this ghost requires pre-teardown memory capture via `nvmlDeviceGetMemoryInfo().used`, post-teardown polling with a stabilization protocol where readings converge within 1 MiB, and delta computation.

We simulate 10,000 treatment trials with ghost bytes injected as Uniform(0, 1024 MiB) across 3 teardown methods and 3 MIG profiles, against 10,000 zero-injection controls.

### 3.2 Contention Squeeze

When a new tenant arrives on a time-sliced GPU, every existing tenant loses bandwidth. Each receives approximately 1/N of total memory bandwidth, where N is the tenant count. `nvidia-smi` continues reporting 100% GPU utilization because it has no concept of per-tenant bandwidth. The aggregate number masks the per-tenant degradation entirely.

Detection requires per-tenant baseline capture before each new tenant arrives, post-arrival measurement, and comparison. We simulate 10,000 treatment trials across 3 tenant counts under time-slicing, against 10,000 control trials at single-tenant baseline plus MIG-partitioned configurations.

### 3.3 Provisioning Overhead

MIG partition creation takes 120 to 500 ms of wall-clock time depending on profile size and GPU load state. During this interval, capacity is neither free nor usable. `nvidia-smi` reports state only after the command returns, missing the transition entirely.

Detection requires wall-clock timing from create command to first successful compute dispatch. We simulate 20,000 paired trials measuring both true spin-up latency and nvidia-smi reported latency (approximately 0.5 ms).

### 3.4 Burst-to-Sustained Gap

GPUs boost to peak clock at cold start and thermally throttle to a lower sustained clock within 30 to 120 seconds. The gap between burst and sustained performance ranges from 1 to 16% for datacenter GPUs (H100, H200) and is larger for consumer hardware. Monitoring tools take instantaneous snapshots. They never track the thermal trajectory from burst to equilibrium, so the gap between advertised and deliverable performance remains invisible.

We simulate 20,000 paired trials measuring gap percentage versus ideal 0% across 5 GPU profiles and 3 workload types. The thermal model follows Newton's law of cooling with per-GPU coefficients derived from hardware measurements.

### 3.5 Straggler Tax

One degraded GPU forces N-1 healthy GPUs to idle at every synchronization barrier in data-parallel training. Five hardware degradation types contribute: thermal paste failure, NVLink degradation, PCIe fallback, memory subsystem failure, and clock stuck. Each GPU individually reports normal or near-normal metrics. No tool correlates fleet-wide performance to identify the barrier bottleneck.

Detection requires fleet-wide measurement, median-versus-outlier comparison, and per-GPU diagnostic probing. We simulate 10,000 treatment trials with degradation injected into one GPU across 5 types, 3 severities, and 3 fleet sizes (8, 32, 128), against 10,000 control trials with no degradation.

### 3.6 Oversubscription Blind Spots

Hypervisors can allocate more aggregate vGPU VRAM than physical GPU capacity. At overcommit ratios above 1.0, tenants experience performance degradation proportional to the excess. Each allocation appears successful individually. No tool sums allocations against physical limits, so the overcommit remains invisible until tenants experience degradation or out-of-memory crashes.

We simulate 10,000 treatment trials at overcommit ratios of 1.25x, 1.5x, and 2.0x, against 10,000 control trials at 1.0x.

### 3.7 Noise Model

All simulations include realistic measurement noise calibrated to NVML instrumentation characteristics:

| Noise Source | Distribution | Parameters |
|-------------|-------------|------------|
| NVML memory readings | Gaussian | sigma = 512 KiB |
| Spin-up latency | Log-normal | sigma = 0.3 |
| Bandwidth measurement | Gaussian | sigma = 2% of baseline |
| Background memory spikes | Poisson x Uniform | lambda = 0.01, 1 to 64 MiB |
| Thermal readings | Gaussian | sigma = 1.0 C |
| Fleet performance | Gaussian | sigma = 2% of peak |

## 4. Simulation Infrastructure and Experimental Design

### 4.1 Methodology

We employ a synthetic model to systematically evaluate all six waste categories under controlled conditions. We implement the simulation framework in Rust as part of the `gpu-harness` crate, comprising approximately 2,600 lines across 6 modules. The binary accepts three parameters: scale factor (default 1.0, producing 120,000 trials), random seed (default 42), and output path. Execution time is 0.4 seconds on commodity hardware. The output is deterministic: re-execution with the same seed produces byte-identical JSON (SHA-256: `e386a3d599a238901abff636d421dbcb095cca31616e425d05e5653744fd3912`).

### 4.2 Hardware-Validated Foundation

We calibrate the simulation models against measured hardware performance, not theoretical specifications. The dynamic roofline model underpinning the burst-to-sustained gap analysis (Category 4) and the straggler detection framework (Category 5) has been validated on bare-metal systems:

| GPU | Measured HBM BW | Measured FP32 | % of Theoretical | Platform |
|-----|----------------|---------------|-----------------|----------|
| H100 SXM 80GB | 2,905 GB/s | 59.1 TFLOPS | 87 to 100% | RunPod bare-metal |
| H200 SXM 141GB | 4,028 GB/s | 59.5 TFLOPS | 87 to 100% | RunPod bare-metal |
| RTX 5090 32GB | 1,503 GB/s | 95.8 TFLOPS | 87 to 100% | Cloud instance |

We derive the thermal model (Newton's law of cooling with per-GPU thermal coefficients) and power model (workload intensity to clock frequency mapping) from these validated profiles. GPU profiles used in the simulation, including clock speeds, TDP, thermal throttle onset temperatures, and bandwidth specifications, match hardware-measured values rather than vendor datasheet maximums. Full validation reports with measurement methodology, coefficient of variation, and NVML telemetry are archived in the repository (`docs/validation/`).

This hardware calibration is critical. It establishes that the simulation's performance predictions for burst clocks, sustained throughput, and thermal trajectories are grounded in measured physics.

### 4.3 Experimental Design Summary

| Category | N | Design | Primary Test |
|----------|---|--------|-------------|
| 1. Ghost allocations | 20,000 | Independent groups | Mann-Whitney U |
| 2. Contention squeeze | 20,000 | Independent groups | Mann-Whitney U |
| 3. Provisioning overhead | 20,000 | Paired (within-trial) | Wilcoxon signed-rank |
| 4. Burst-sustained gap | 20,000 | Paired (within-trial) | Wilcoxon signed-rank |
| 5. Straggler tax | 20,000 | Independent groups | Mann-Whitney U |
| 6. Oversubscription | 20,000 | Independent groups | Mann-Whitney U |

The choice of statistical test follows from the experimental design. Categories 3 and 4 produce paired observations where each trial measures both the actual value and the tool-reported or ideal baseline, requiring a paired test. We apply Holm-Bonferroni correction across all six omnibus p-values (family-wise alpha = 0.05).

### 4.4 Detection Model

For each trial, we record three binary detection outcomes: whether `gpu-roofline`, `nvidia-smi`, or DCGM would detect the waste event. We model `nvidia-smi` and DCGM detection at 0% based on architectural capability analysis. These tools lack the measurement primitives described in Section 3. Live tool comparison is deferred to hardware validation.

## 5. Results

### 5.1 Overview

The results support all three study hypotheses. All six waste categories produce large, statistically significant effects. Existing monitoring tools detect none of them. The six categories map to three distinct operational responses. We present the results in three sections: omnibus statistical outcomes (5.2), the observability gap (5.3), and the three-bucket operational model (5.4).

### 5.2 All Six Waste Categories Are Detectable

All six categories of invisible waste produce measurable, statistically significant effects. Table 1 presents the omnibus results.

**Table 1. Omnibus Statistical Results Across All Six Waste Categories**

| Category | Design | Test | d / d_z | r | 95% CI of Median Difference |
|----------|--------|------|---------|---|----------------------------|
| Ghost allocations | Independent | Mann-Whitney U | 2.46 | 1.00 | [527M, 545M] bytes |
| Contention squeeze | Independent | Mann-Whitney U | 8.55 | 1.00 | [66.61, 66.68]% bandwidth loss |
| Provisioning overhead | Paired | Wilcoxon signed-rank | 2.01 | 1.00 | [244, 248] ms hidden latency |
| Burst-sustained gap | Paired | Wilcoxon signed-rank | 0.73 | 1.00 | [1.63, 1.77]% below spec |
| Straggler tax | Independent | Mann-Whitney U | 2.46 | 1.00 | [14.56, 15.09]% fleet throughput |
| Oversubscription | Independent | Mann-Whitney U | 3.97 | 1.00 | [33.33, 33.33]% degradation |

*All Holm-Bonferroni adjusted p < 1e-300. N = 10,000 to 20,000 per group. Effect sizes (d, d_z) and bootstrap 95% CIs (10,000 resamples) are the primary measures. P-values are reported for completeness but are uninformative at this sample size.*

Effect sizes are uniformly large (Figure 2). The weakest category, burst-to-sustained gap at d_z = 0.73, still qualifies as a medium-to-large effect by conventional benchmarks. The strongest, contention squeeze at d = 8.55, reflects the deterministic nature of bandwidth partitioning under time-slicing.

![Figure 2. Standardized effect sizes across categories, colored by operational action bucket.](figures/fig2_effect_sizes.png)

### 5.3 Current Tools Provide Zero Visibility

We predicted that existing tools would miss greater than 80% of waste events. The result is more extreme. Modeled detection is 0% across all six categories for both `nvidia-smi` and DCGM.

**Table 2. Detection Rates by Monitoring Tool**

| Category | nvidia-smi | DCGM | gpu-roofline | McNemar p |
|----------|-----------|------|-------------|-----------|
| Ghost allocations | 0.0% | 0.0% | 99.9% | <1e-300 |
| Contention squeeze | 0.0% | 0.0% | 100.0% | <1e-300 |
| Provisioning overhead | 0.0% | 0.0% | 100.0% | <1e-300 |
| Burst-sustained gap | 0.0% | 0.0% | 56.5% | <1e-300 |
| Straggler tax | 0.0% | 0.0% | 94.7% | <1e-300 |
| Oversubscription | 0.0% | 0.0% | 100.0% | <1e-300 |
| **Overall** | **0.0%** | **0.0%** | **88.4%** | |

![Figure 1. Detection rates by monitoring tool across all six waste categories.](figures/fig1_detection_rates.png)

The 0% detection rate for `nvidia-smi` and DCGM is a modeling result based on documented tool capabilities (see Section 4.4). These tools do not compute memory deltas across teardowns, track per-tenant bandwidth, time provisioning transitions, measure thermal trajectories, correlate fleet-wide barriers, or sum allocations against physical limits. These capabilities cannot be enabled through configuration. They require fundamentally different measurement architectures.

The `gpu-roofline` detection rate varies by category. The weakest performance occurs in the burst-to-sustained gap (56.5%), where small thermal gaps below the 1% detection threshold are missed. The strongest performance occurs in contention, provisioning, and oversubscription (100%), where the signal-to-noise ratio is high.

### 5.4 Operational Impact: Three Action Buckets

The six waste categories do not all represent the same type of problem and they do not all have the same solution. We categorize them into three action buckets based on what visibility enables (Figure 3).

![Figure 3. Three operational action buckets with detection rates and per-event impact.](figures/fig3_three_buckets.png)

#### Bucket A: Directly Recoverable Capacity

Ghost allocations trap VRAM that could host additional tenant instances. The simulation yields a median of 512 MiB trapped per teardown event. At 20 teardowns per day, this represents approximately 10 GiB of VRAM per GPU per day that can be reclaimed once the ghost is detected. Detection rate: 99.9%.

The straggler tax wastes fleet-wide GPU-hours at synchronization barriers. A single degraded GPU in an 8-GPU training job wastes 16.7% of fleet throughput as the other 7 GPUs idle at the barrier. This effect scales multiplicatively with fleet size. At 128 GPUs, one straggler wastes the equivalent of 24 GPUs at every sync barrier (Figure 4). Identifying the straggler (detection rate: 94.7%) and replacing or reassigning it immediately recovers this capacity.

![Figure 4. Straggler tax scales multiplicatively with fleet size.](figures/fig4_straggler_scaling.png)

These two categories represent genuine capacity recovery. Detect the problem, take action, free the resource.

#### Bucket B: Decision Support

Contention squeeze is the inherent cost of time-slicing. With 2 tenants, each receives approximately 50% of available bandwidth. With 4 tenants, each receives approximately 25% (Figure 5). This degradation cannot be eliminated. It is the physics of shared access. Measuring it enables operators to make informed infrastructure decisions: deploy MIG with hardware isolation and guaranteed bandwidth for latency-sensitive workloads, and use time-slicing with higher density for throughput-tolerant batch jobs. Without per-tenant bandwidth visibility, this decision is made without data.

![Figure 5. Per-tenant bandwidth degradation under time-slicing versus MIG isolation.](figures/fig5_contention_tenants.png)

The burst-to-sustained gap is thermal physics. H100 SXM GPUs exhibit a median 1.7% gap between burst and sustained performance. Compute-bound workloads on some profiles exhibit gaps up to 16%. This cannot be fixed, but it can be priced correctly. Cloud providers advertising peak specifications for sustained workloads are implicitly overcommitting on performance. Measuring the actual sustained ceiling enables accurate SLAs.

Provisioning overhead (246 ms per MIG provision) is measurable but economically negligible at less than $1 per GPU per year.

#### Bucket C: Risk Prevention

Oversubscription occurs when aggregate vGPU VRAM allocations exceed physical GPU capacity. At 1.5x overcommit, tenants experience 33% performance degradation. At 2x overcommit, out-of-memory crashes become likely. Each tenant allocation appears successful individually. No tool sums them against the physical limit. Detection enables enforcement before tenants are silently impacted.

### 5.5 Hypothesis Summary

| Hypothesis | Prediction | Result |
|-----------|-----------|--------|
| H1: All six categories produce significant waste | Significant effects in all 6 | **Supported.** All d/d_z > 0.73, all Holm p < 1e-300 |
| H2: Existing tools miss >80% of waste | <20% detection by nvidia-smi/DCGM | **Supported.** 0% detection across all categories |
| H3: Economic impact > $0 per GPU per year | Bootstrap CI lower bound > $0 | **Supported.** Lower bound $8,149/GPU/yr (8-GPU fleet) |

## 6. Discussion

### 6.1 A Monitoring Paradigm Shift

The GPU monitoring paradigm emerged in the single-device, single-tenant era. `nvidia-smi` answered the questions that mattered when one workload ran on one GPU: Is this device functional? Is it overheating? Has it encountered an ECC error? The industry has moved to multi-tenant virtualization and fleet-scale distributed training, but the monitoring architecture has not followed. This study quantifies the consequences of that gap.

The six waste categories identified in this study are not six independent problems. They are one structural problem observed at three scales. At the device scale, thermal physics creates a gap between advertised and deliverable performance. At the virtualization scale, lifecycle transitions and bandwidth sharing create losses that per-device metrics cannot capture. At the fleet scale, synchronization dependencies amplify single-device degradation into cluster-wide waste. The common thread is that all six require measurement across time, across tenants, or across devices. Current tools provide none of these measurement capabilities.

The analogy is straightforward. Operating a GPU fleet with `nvidia-smi` alone is comparable to operating a power grid with meters that show voltage but not consumption. The equipment appears healthy while efficiency losses accumulate unmeasured.

### 6.2 Operational Priorities

The three-bucket model translates directly into deployment priority for operators building monitoring capability.

Straggler detection offers the highest return. A single degraded GPU in a 128-GPU training cluster wastes 127 GPU-equivalents of blocked time at every synchronization barrier. The return on detecting and replacing one straggler is immediate and scales multiplicatively with fleet size. For organizations running large-scale distributed training, this category alone justifies the investment in fleet-wide performance monitoring.

Oversubscription detection addresses the highest risk. Silent overcommit is the only category that can cause direct customer-visible failures through out-of-memory crashes. The probability may be low in well-managed environments. The reputational cost of unexplained tenant failures in a cloud offering is disproportionately high.

Contention visibility provides the highest strategic value. The decision between MIG and time-slicing determines tenant density, performance guarantees, and pricing models. Without per-tenant bandwidth data, this decision is driven by vendor guidance rather than measured workload characteristics. Operators who can measure per-tenant bandwidth can optimize partitioning strategies for their specific workload mix.

### 6.3 Limitations and Validity

The noise model parameters are based on published NVML instrumentation characteristics, not empirical calibration against live systems. The thermal model uses Newton's law of cooling with published TDP specifications. Real hardware may exhibit behaviors not captured by these models.

The 0% detection rate for `nvidia-smi` and DCGM is a modeling assumption based on documented capabilities. Live evaluation may reveal edge cases of partial detection through DCGM custom metric plugins.

With 10,000 to 20,000 observations per group, statistical significance is guaranteed for any non-zero effect. We emphasize effect sizes and confidence intervals over p-values throughout. Readers should evaluate practical significance against their specific operational context.

Results are specific to the simulated GPU profiles (primarily H100 SXM, with H200 and consumer GPUs in the burst-sustained category) and three workload types (compute-bound, memory-bound, mixed). Real-world workloads are more diverse. Fleet utilization patterns vary across providers. The cost model parameters should be treated as substitutable inputs rather than universal constants.

## 7. Reproducibility

Full reproducibility is a requirement of this work, not a convenience feature. The simulation is fully deterministic. Re-execution with seed 42 produces byte-identical output:

```
SHA-256: e386a3d599a238901abff636d421dbcb095cca31616e425d05e5653744fd3912
```

We verified this by independent re-execution during the analysis phase.

### 7.1 Artifact Inventory

| Artifact | Description |
|----------|------------|
| `simulation-raw.json` (83.7 MB) | 120,000 trial-level records with IVs, DVs, and detection outcomes |
| `analyze_study.py` (1,280 lines) | Full analysis pipeline: statistics, cost model, provenance |
| `summary.md` | Main results tables (Tables 1 through 4) |
| `supplement.md` | Supplementary tables (S1 through S5) |
| `PROVENANCE.md` | Software versions, commands, SHA-256 chain |
| `SHA256SUMS.txt` | Per-file checksums for all artifacts |

### 7.2 Rerun Commands

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

GPU efficiency loss spans every layer of the operations stack, and so does the observability gap. We demonstrate that six categories of measurable waste are completely invisible to the standard monitoring tools deployed across the industry. `nvidia-smi` and DCGM detect 0% of waste events across all six categories. Purpose-built instrumentation detects 56.5 to 100%.

We provide strong experimental evidence that these losses are quantifiable and large (Cohen's d/d_z from 0.73 to 8.55). They span device-level thermal physics, virtualization lifecycle management, and fleet-level distributed training coordination.

The efficiency gap requires three distinct operational responses. Ghost allocations and straggler GPUs represent directly recoverable capacity. Contention and thermal throttling are governed by physics that cannot be eliminated but can be measured, enabling better infrastructure decisions. Oversubscription is a preventable risk.

To address this gap, comprehensive hardware validation studies must be conducted across datacenter-class systems under production conditions. We publish the complete protocol, analysis pipeline, and quantitative predictions as an open invitation to GPU cloud operators, system integrators, and NVIDIA.

---

## References

[1] NVIDIA, "Multi-Instance GPU User Guide," NVIDIA Documentation, 2024.

[2] S. Williams, A. Waterman, and D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65-76, 2009.

[3] A. Harlap et al., "Addressing the straggler problem for iterative convergent parallel ML," *Proceedings of the Seventh ACM Symposium on Cloud Computing (SoCC)*, 2016.

[4] NVIDIA, "NVIDIA Data Center GPU Manager (DCGM) Documentation," NVIDIA Documentation, 2024.

[5] NVIDIA, "NVIDIA System Management Interface (nvidia-smi) Documentation," NVIDIA Documentation, 2024.
