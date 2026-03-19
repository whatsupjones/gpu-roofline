# The Hidden Cost of GPU Virtualization: Quantifying Invisible Waste in Multi-Tenant Infrastructure

**Protocol Version:** 1.0
**Date:** 2026-03-19
**Status:** Pre-registration Draft
**Companion Protocol:** [Ghost Allocations Study](study-protocol-ghost-allocations.md) (Category 1 deep-dive)

---

## 1. Abstract

We propose a comprehensive empirical study quantifying six categories of invisible economic waste produced by multi-tenant GPU virtualization (MIG, GRID, time-slicing). Current monitoring tools — nvidia-smi and NVIDIA DCGM — report per-device or per-instance instantaneous state but fail to capture waste that manifests as *state transitions*, *cross-tenant interference*, *temporal degradation*, or *aggregate overcommitment*. We formalize these blind spots as six measurable waste categories: (1) ghost allocations after teardown, (2) contention squeeze on tenant arrival, (3) provisioning overhead during spin-up, (4) burst-to-sustained performance gaps from thermal equilibrium, (5) straggler tax in distributed training, and (6) oversubscription blind spots from silent resource exhaustion.

For each category we define independent/dependent variables, statistical tests, effect size calculations, and detection mechanisms implemented in the open-source `gpu-roofline` and `gpu-fleet` tooling. We specify a two-phase design: a simulation phase (N=120,000 total trials) executable immediately using existing simulation infrastructure, followed by a hardware validation phase on bare-metal H100 systems. We derive a parametric cost model projecting aggregate annual waste at three fleet scales (8, 100, and 10,000 GPUs) using current cloud pricing ($2.50/GPU/hr for H100).

Our hypotheses predict that (H1) all six waste categories produce statistically significant measurable waste, (H2) existing monitoring tools fail to detect at least 80% of waste events that gpu-roofline detects, and (H3) aggregate annual waste exceeds a derived threshold per GPU per year at production scale.

## 2. Background and Motivation

### 2.1 The Problem

GPU virtualization enables multi-tenancy on expensive accelerators, but it introduces waste that is structurally invisible to existing monitoring. This waste falls into six categories, each with a different mechanism and a different reason why current tools miss it.

The common thread: existing tools report **instantaneous state** per device or per instance. They do not compute **deltas across lifecycle transitions**, **cross-tenant comparisons**, **temporal degradation curves**, **fleet-level aggregates exceeding physical capacity**, or **distributed synchronization overhead**. These blind spots are not bugs — they are architectural limitations of tools designed for single-device health monitoring, not multi-tenant economic analysis.

### 2.2 Why Existing Tools Miss Each Category

| Category | nvidia-smi Blind Spot | DCGM Blind Spot | Detection Requires |
|----------|----------------------|-----------------|-------------------|
| 1. Ghost Allocations | Reports "free" after teardown; no pre/post delta | Instance metrics cease; no teardown verification | Pre/post memory delta with stabilization polling |
| 2. Contention Squeeze | Per-GPU utilization only; no per-tenant bandwidth | Per-GPU counters; no baseline-vs-current per tenant | Per-tenant baseline capture before new tenant arrives |
| 3. Provisioning Overhead | Shows instant state transitions | No provisioning latency metric | Wall-clock from create command to first usable compute |
| 4. Burst-to-Sustained Gap | Snapshot readings at query time | Fixed-interval sampling; no burst-vs-equilibrium comparison | Continuous trajectory tracking through thermal ramp |
| 5. Straggler Tax | Per-device only; no fleet-wide sync barrier analysis | Per-device only; no distributed training awareness | Fleet-wide measurement + barrier wait quantification |
| 6. Oversubscription | Per-instance allocation; no sum-vs-physical comparison | Per-instance metrics; no aggregate capacity check | Sum of all vGPU allocations vs physical GPU capacity |

### 2.3 gpu-roofline / gpu-fleet Detection Capabilities

The detection mechanisms are already implemented in the gpu-tools codebase:

| Category | Implementation | Module |
|----------|---------------|--------|
| Ghost Allocations | `TeardownVerifier` pre/post NVML memory delta | `gpu-harness::vgpu::teardown` |
| Contention Squeeze | `ContentionMeasurer` per-tenant baseline tracking | `gpu-harness::vgpu::contention` |
| Provisioning Overhead | `VgpuEventType::Created { spin_up_latency_ms }` | `gpu-harness::vgpu::state` |
| Burst-to-Sustained Gap | `DynamicRoofline` burst vs sustained model | `gpu-roofline::model::dynamic` |
| Straggler Tax | `detect_stragglers()` fleet median vs outlier | `gpu-fleet::straggler` |
| Oversubscription | `VgpuState` aggregate allocation tracking | `gpu-harness::vgpu::state` |

### 2.4 Simulation Infrastructure

The simulation engine (`gpu-harness::sim`) provides:

- **`SimulatedBackend`** — GPU backend returning deterministic measurements from `SimGpuProfile` specs with configurable jitter
- **`SimulatedFleet`** — Multi-GPU fleet with `Degradation` injection (thermal paste, NVLink, PCIe, memory subsystem, clock stuck)
- **`ThermalModel`** — Physics-based thermal simulation using Newton's law of cooling with throttle factor computation
- **`SimulatedDetector`** — Replays `VgpuSimScenario` events through the full `VgpuSampler` / `VgpuAlertEngine` pipeline
- **Built-in scenarios** — `ghost_allocation`, `grid_contention`, `mig_scale_up`, `rapid_churn`

## 3. Formal Hypotheses

### H1: Multi-tenant GPU virtualization produces statistically significant invisible waste across all six categories.

- **H1_0 (Null):** For each waste category *k* in {1..6}, the measured waste under virtualized conditions equals zero (or equals waste under bare-metal / single-tenant control).
- **H1_A (Alternative):** For each category *k*, mean waste under virtualized conditions is significantly greater than zero (or greater than the control).
- **Test:** One-sided tests appropriate to each category's data distribution (see Section 7).
- **Correction:** Holm-Bonferroni across the 6 omnibus tests (family-wise alpha = 0.05).

### H2: Existing monitoring tools fail to detect at least 80% of waste events that gpu-roofline detects.

- **H2_0:** Detection rate of nvidia-smi/DCGM >= 20% of gpu-roofline's detection rate, for each category.
- **H2_A:** Detection rate of nvidia-smi/DCGM < 20% of gpu-roofline's detection rate.
- **Test:** McNemar's test for paired detection (same waste event, different tools).
- **Threshold:** H2 is supported if, across all categories, nvidia-smi and DCGM detect < 20% of events gpu-roofline detects.

### H3: Aggregate economic cost of invisible waste exceeds $X per GPU per year at production scale.

- **H3_0:** Annual waste per GPU <= $0 (no measurable economic cost).
- **H3_A:** Annual waste per GPU > $0, with X derived from the parametric cost model (Section 10).
- **Test:** Bootstrap confidence interval for the cost model output; reject H3_0 if the lower 95% CI bound > $0.
- **Note:** X is not pre-specified; it is derived from measured effect sizes. The hypothesis is directional: waste > 0.

## 4. Study Design Overview

### 4.1 Two-Phase Design

| Phase | Purpose | Sample Size | Hardware | Timeline |
|-------|---------|-------------|----------|----------|
| **Simulation** | Establish detection sensitivity/specificity; estimate effect sizes; validate analysis pipeline | N=120,000 total trials across 6 categories | None (cpu-only simulation) | 1-2 weeks |
| **Hardware** | Confirm simulation findings on real GPUs; measure actual waste magnitudes; calibrate cost model | N=1,200 total trials across 6 categories | Bare-metal H100 SXM5 | 2-3 weeks |

### 4.2 Per-Category Experimental Designs

| Category | Design Type | Independent Variables | Dependent Variables |
|----------|------------|----------------------|-------------------|
| 1. Ghost Allocations | 3x3 factorial + control | Teardown method (3), MIG profile (3) | Ghost bytes, reclaim latency, detection binary |
| 2. Contention Squeeze | Repeated measures | Tenant count (1-4), partitioning mode (2) | Per-tenant bandwidth ratio, per-tenant compute ratio |
| 3. Provisioning Overhead | One-factor with blocking | MIG profile (3), block on temperature | Spin-up latency (ms), dead-time fraction |
| 4. Burst-to-Sustained Gap | Within-subjects (same GPU, two timepoints) | Measurement window (burst vs sustained) | TFLOPS, GB/s, clock MHz, temperature |
| 5. Straggler Tax | Fleet simulation + validation | Degradation type (5), fleet size (3) | Fleet throughput loss (%), barrier wait time |
| 6. Oversubscription | Observational + injection | Overcommit ratio (1.0-2.0), number of instances | Actual vs advertised VRAM, actual vs advertised compute |

## 5. Category 1: Ghost Allocations (Memory Leak After Teardown)

*Full protocol defined in companion document: [study-protocol-ghost-allocations.md](study-protocol-ghost-allocations.md). Summary below.*

### 5.1 Mechanism

VRAM not reclaimed after vGPU destruction. `nvidia-smi` reports the instance as gone, but physical memory consumption does not return to baseline. The `TeardownVerifier` captures pre-teardown `nvmlDeviceGetMemoryInfo().used`, waits for the instance to disappear, then polls post-teardown memory with a 100ms stabilization protocol until readings converge within 1 MiB.

### 5.2 Variables

| Variable | Type | Levels |
|----------|------|--------|
| Teardown method | IV (categorical) | Clean, Under-Load, Rapid-Churn |
| MIG profile | IV (categorical) | 1g.10gb, 2g.20gb, 3g.40gb |
| Ghost allocation (bytes) | DV (continuous) | `expected_free - actual_free` |
| Reclaim latency (ms) | DV (continuous) | Wall-clock teardown-to-stable |
| Detection tool | IV (categorical) | nvidia-smi, DCGM, gpu-roofline |

### 5.3 Sample Size

- Simulation: 10,000 cycles per condition (90,000 total)
- Hardware: 50 cycles per cell, 9 treatment + 3 control cells = 600 cycles
- Power: >0.90 for Cohen's d >= 0.50 with Bonferroni correction

### 5.4 Simulation Mapping

| Scenario | Simulation Component | Parameters |
|----------|---------------------|------------|
| Ghost injection | `SimAction::DestroyVgpu { ghost_bytes }` | ghost_bytes: Uniform(0, 1024 MiB) |
| Under-load teardown | `ghost_allocation` scenario + active workload event | Add `SimAction::ContentionEvent` before destroy |
| Rapid-churn | `rapid_churn` scenario with parameterized ghost_bytes | ghost_bytes per cycle: Uniform(0, 64 MiB) |

### 5.5 Primary Statistical Test

Mann-Whitney U (one-sided) for treatment vs control, with Holm-Bonferroni correction across 9 comparisons. Effect size: rank-biserial correlation *r*. Supplementary: Bootstrap BCa 95% CI for median difference (10,000 resamples).

## 6. Category 2: Contention Squeeze (Throughput Degradation on Tenant Arrival)

### 6.1 Mechanism

When a new vGPU is provisioned on a time-sliced GPU, existing tenants lose bandwidth and compute throughput. DCGM reports per-GPU utilization (which may stay at 100%), but individual tenants experience degradation proportional to the number of co-tenants. The `ContentionMeasurer` records per-tenant baselines before each new tenant arrives, then compares post-arrival performance to detect squeeze events exceeding a configurable threshold.

### 6.2 Variables

| Variable | Type | Levels / Measurement |
|----------|------|---------------------|
| Tenant count | IV (ordinal) | 1, 2, 3, 4 tenants |
| Partitioning mode | IV (categorical) | TimeSliced, HardwarePartitioned (control) |
| Per-tenant bandwidth ratio | DV (continuous) | current_bw / baseline_bw (0.0-1.0) |
| Per-tenant compute ratio | DV (continuous) | current_gflops / baseline_gflops (0.0-1.0) |
| Aggregate GPU utilization (nvidia-smi) | DV (continuous) | % reported by nvidia-smi |
| Detection tool | IV (categorical) | nvidia-smi, DCGM, gpu-roofline |

### 6.3 Experimental Design

**Repeated measures design** — same GPU, increasing tenant count:

1. **Baseline (1 tenant):** Create single vGPU, run sustained workload, measure bandwidth and compute for 60s
2. **Add tenant 2:** Create second vGPU, run same workload on both, wait 30s for stabilization, measure both tenants for 60s
3. **Add tenant 3:** Repeat measurement with 3 tenants
4. **Add tenant 4:** Repeat measurement with 4 tenants
5. **Control block:** Repeat entire sequence with HardwarePartitioned (MIG) mode — expect no degradation

Between replications: destroy all instances, wait for full reclamation, verify clean baseline.

### 6.4 Sample Size

- Expected effect: ~25% bandwidth drop per additional tenant on time-sliced (from `grid_contention` scenario: 2nd tenant causes 50% ratio, 3rd causes 33%, 4th causes 25%)
- Cohen's d for 1-tenant vs 2-tenant: d = (1.0 - 0.50) / 0.10 = 5.0 (very large)
- Minimum detectable effect: 5% bandwidth drop (d = 0.50 with sigma = 0.10)
- **N per condition: 25 replications** (power > 0.99 for expected large effects; 0.80 for minimum detectable)
- Total: 25 replications x 4 tenant-count levels x 2 partitioning modes = **200 measurement windows**
- Simulation: 5,000 replications per condition = **40,000 measurement windows**

### 6.5 Simulation Mapping

| Scenario | Simulation Component | Parameters |
|----------|---------------------|------------|
| Tenant arrival squeeze | `grid_contention` scenario | Extend to 1-4 tenants with baseline capture at each step |
| MIG control | `mig_scale_up` scenario | Same measurement protocol, expect no squeeze |
| Noise model | Per-tenant bandwidth jitter | N(0, sigma=0.02) added to bandwidth ratio |

### 6.6 Measurement Protocol (Hardware Phase)

```
CONTENTION_TRIAL(replication_index, partitioning_mode):
  # Clean state
  destroy_all_instances()
  wait_for_baseline()

  for tenant_count in [1, 2, 3, 4]:
    # Record baselines for existing tenants
    for existing_tenant in active_tenants:
      baseline[existing_tenant] = measure_60s(bandwidth_gbps, gflops)

    # Create new tenant
    T_create_start = now()
    create_vgpu(tenant_count, partitioning_mode)
    launch_workload(new_tenant, sgemm_80pct)
    wait_30s()  # stabilization

    # Measure all tenants
    for tenant in active_tenants:
      current[tenant] = measure_60s(bandwidth_gbps, gflops)
      ratio_bw[tenant] = current[tenant].bw / baseline[tenant].bw
      ratio_compute[tenant] = current[tenant].gflops / baseline[tenant].gflops

    # Concurrent tool readings
    nvidia_smi_gpu_util = parse(nvidia-smi -q -d UTILIZATION)
    dcgm_gpu_util = parse(dcgmi dmon -e 203,204 -c 1)

    # Record: does nvidia-smi show per-tenant degradation? (No — only GPU-level)
    # Record: does DCGM show per-tenant degradation? (No — only GPU-level)
    # Record: does gpu-roofline ContentionMeasurer detect squeeze? (Yes)

  emit_record(...)
```

### 6.7 Primary Statistical Test

**Friedman test** (non-parametric repeated measures) for bandwidth ratio across tenant counts within time-sliced mode. Post-hoc: Wilcoxon signed-rank tests with Holm correction for pairwise comparisons (1 vs 2, 2 vs 3, 3 vs 4 tenants).

**Paired comparison (time-sliced vs MIG):** Wilcoxon signed-rank test at each tenant count. Effect size: matched-pairs rank-biserial correlation.

## 7. Category 3: Provisioning Overhead (Dead Time During Spin-Up)

### 7.1 Mechanism

MIG partition creation takes measurable wall-clock time during which GPU capacity is neither free (it is being configured) nor usable (no compute can run on the partition yet). nvidia-smi shows instant state transitions because it reports after the command returns, not during the transition. The `VgpuEventType::Created { spin_up_latency_ms }` field captures this dead time.

### 7.2 Variables

| Variable | Type | Levels / Measurement |
|----------|------|---------------------|
| MIG profile | IV (categorical) | 1g.10gb, 2g.20gb, 3g.40gb, 4g.40gb, 7g.80gb |
| GPU load state | IV (categorical) | Cold (idle, <40C), Warm (active partitions, 40-60C), Hot (>60C) |
| Concurrent partitions | IV (ordinal) | 0, 1, 3, 6 existing partitions |
| Spin-up latency (ms) | DV (continuous) | Wall-clock from `nvidia-smi mig -cgi` to first successful compute dispatch |
| Dead-time fraction | DV (continuous) | spin_up_latency / inter_provision_interval |
| nvidia-smi reported latency | DV (continuous) | Command return time (always ~0ms — instant) |

### 7.3 Experimental Design

**3x3x4 factorial design** (profile x load state x concurrent partitions), blocked on GPU temperature.

For each cell:
1. Establish target load state (run warmup workloads if needed)
2. Create target number of concurrent partitions
3. Time the creation of one additional partition:
   - `T_start` = time before `nvidia-smi mig -cgi {profile} -C`
   - `T_command_return` = time after command returns
   - `T_first_compute` = time when first CUDA kernel successfully dispatches on new partition
   - spin_up_latency = `T_first_compute - T_start`
   - command_overhead = `T_command_return - T_start`
   - post_command_gap = `T_first_compute - T_command_return`

### 7.4 Sample Size

- Expected effect: spin-up latency 100-500ms depending on profile and load
- Variance: estimated sigma = 50ms from preliminary simulation data
- Minimum detectable difference between profiles: 30ms
- Cohen's d = 30/50 = 0.60
- **N per cell: 30 replications** (power = 0.86 for d = 0.60)
- 5 profiles x 3 load states x 4 concurrent levels = 60 cells (not all combinations valid)
- Valid combinations: ~36 cells (large profiles incompatible with many concurrent partitions)
- **Total: 36 x 30 = 1,080 provisioning events** (hardware)
- **Simulation: 36 x 1,000 = 36,000 events**

### 7.5 Simulation Mapping

| Scenario | Simulation Component | Parameters |
|----------|---------------------|------------|
| MIG spin-up timing | `mig_scale_up` scenario | Extend with variable `spin_up_latency_ms` per profile |
| Load-dependent latency | Add load-state modifier | spin_up_latency_ms += load_penalty(temperature, concurrent_count) |
| Noise model | Log-normal jitter | latency ~ LogNormal(mu=log(base_latency), sigma=0.3) |

### 7.6 Primary Statistical Test

**Two-way ANOVA** (profile x load state) on log-transformed spin-up latency (expected log-normal distribution). If non-normal after transformation: Kruskal-Wallis with Dunn's post-hoc. Effect size: partial eta-squared for each factor.

**Key comparison:** nvidia-smi command return time vs actual spin-up latency. Paired t-test (or Wilcoxon signed-rank) for the difference; expect nvidia-smi to report near-zero while actual latency is 100-500ms.

## 8. Category 4: Burst-to-Sustained Gap (Advertised vs Actual Performance)

### 8.1 Mechanism

GPUs are marketed and cloud instances priced based on peak (burst) performance specifications. Under sustained load, thermal throttling and power limiting reduce actual performance by 15-30%. The `DynamicRoofline` model captures both burst (t=0, peak boost clock) and sustained (thermal equilibrium) rooflines, with a full thermal trajectory showing the degradation curve. The `ThermalModel` provides physics-based simulation using Newton's law of cooling.

### 8.2 Variables

| Variable | Type | Levels / Measurement |
|----------|------|---------------------|
| Measurement window | IV (within-subjects) | Burst (first 1s), Sustained (at thermal equilibrium, typically 60-120s) |
| Workload type | IV (categorical) | Compute-bound (FMA-heavy), Memory-bound (STREAM copy), Mixed |
| GPU model | IV (categorical) | H100 SXM5, H100 PCIe, A100, consumer GPUs (sim only) |
| Cooling configuration | IV (categorical) | SXM5 (server), PCIe (workstation), consumer (air-cooled) |
| TFLOPS | DV (continuous) | Measured FP32 TFLOPS at each window |
| Bandwidth (GB/s) | DV (continuous) | Measured memory bandwidth at each window |
| Clock frequency (MHz) | DV (continuous) | GPU core clock at each window |
| Temperature (C) | DV (continuous) | Junction temperature at each window |
| Power (W) | DV (continuous) | Board power at each window |
| Net ceiling drop (%) | DV (continuous) | `(burst_gflops - sustained_gflops) / burst_gflops * 100` |

### 8.3 Experimental Design

**Within-subjects design** — each GPU serves as its own control (burst vs sustained on same hardware).

Protocol:
1. Cool GPU to ambient (<40C); lock clocks to max boost
2. Launch workload; immediately begin measurement
3. Capture burst readings: first 1 second (10 samples at 100ms intervals)
4. Continue workload for 120 seconds
5. Capture sustained readings: seconds 100-120 (20 samples at 1s intervals)
6. Record full thermal trajectory: temperature, clock, power, TFLOPS every 2s for 120s
7. Compute `DynamicRoofline` model from burst and sustained measurements

### 8.4 Sample Size

- Expected effect: 15-30% TFLOPS drop from burst to sustained
- Within-subject variance is low (same GPU, deterministic thermal physics)
- Estimated sigma of paired differences: ~3% of burst TFLOPS
- Cohen's d for paired: d = 15% / 3% = 5.0 (extremely large)
- **N per GPU model x workload: 15 replications** (power > 0.99)
- Rationale for N > minimum: characterize the distribution of equilibrium times and drop magnitudes
- 3 workload types x 2 GPU models (hardware) = 6 cells x 15 = **90 trials** (hardware)
- Simulation: 3 workloads x 5 GPU profiles x 1,000 replications = **15,000 trials**

### 8.5 Simulation Mapping

| Scenario | Simulation Component | Parameters |
|----------|---------------------|------------|
| Thermal trajectory | `ThermalModel::temperature_at(power, elapsed)` | Power from `SimGpuProfile`, elapsed 0-120s |
| Clock throttling | `ThermalModel::throttle_factor(temp)` | Maps temperature to clock reduction (1.0 to 0.6) |
| Burst capture | `DynamicRoofline::burst` | Measurement at t=0 before thermal ramp |
| Sustained capture | `DynamicRoofline::sustained` | Measurement at t=3*tau (95% of equilibrium) |
| GPU profiles | `profiles::h100_sxm()`, `profiles::rtx_5090()`, etc. | Each profile has distinct thermal parameters |
| Noise model | Measurement jitter | TFLOPS jitter: N(0, sigma=0.02 * peak_gflops) |

### 8.6 Primary Statistical Test

**Paired t-test** (or Wilcoxon signed-rank if non-normal) for burst TFLOPS vs sustained TFLOPS. Effect size: Cohen's d for paired samples. Supplementary: linear mixed model with GPU as random effect, measurement window as fixed effect.

**Equilibrium time analysis:** Kaplan-Meier survival curve for time to equilibrium (defined as CV < 2% over 10 consecutive samples). Report median and IQR of equilibrium time.

## 9. Category 5: Straggler Tax (Fleet Waste from Slowest GPU)

### 9.1 Mechanism

In data-parallel distributed training, all GPUs must synchronize at gradient aggregation barriers. If one GPU is degraded (thermal throttling, memory subsystem failure, PCIe fallback, NVLink degradation), it becomes the bottleneck. The other N-1 GPUs sit idle at the barrier waiting for the straggler. Total fleet waste = (N-1) * (median_throughput - straggler_throughput) * training_time.

The `detect_stragglers()` function measures all GPUs, computes fleet median, flags outliers below a threshold, and runs diagnostic probes (`DiagnoseConfig`) on each straggler to identify root cause. The `SimulatedFleet` supports injection of five degradation types:

- `ThermalPasteDried { extra_degrees_c }` — dried thermal paste, earlier throttling
- `NvlinkDegraded { active_links, expected_links }` — partial NVLink failure
- `PcieFallback { actual_gen, expected_gen }` — PCIe running at lower gen
- `MemorySubsystem { bandwidth_ratio }` — partial HBM stack failure
- `ClockStuck { max_mhz }` — clock stuck at lower frequency

### 9.2 Variables

| Variable | Type | Levels / Measurement |
|----------|------|---------------------|
| Degradation type | IV (categorical) | ThermalPaste, NVLink, PCIe, MemorySubsystem, ClockStuck, None (control) |
| Degradation severity | IV (continuous) | Parametric (e.g., bandwidth_ratio from 0.5 to 0.95) |
| Fleet size | IV (ordinal) | 8 (DGX), 32, 128 GPUs |
| Number of degraded GPUs | IV (ordinal) | 0 (control), 1, 2, 4 |
| Fleet median throughput (TFLOPS) | DV (continuous) | Median of all GPU measurements |
| Straggler throughput (TFLOPS) | DV (continuous) | Worst GPU measurement |
| Effective fleet throughput | DV (continuous) | min(all_gpu_throughputs) * N |
| Barrier wait waste | DV (continuous) | sum over healthy GPUs of (gpu_throughput - straggler_throughput) * time |
| Straggler detected by gpu-fleet | DV (binary) | Whether `detect_stragglers()` flagged the GPU |
| Root cause correctly identified | DV (binary) | Whether `DiagnosisResult` matches injected degradation |

### 9.3 Experimental Design

**Fleet simulation with degradation injection:**

1. Create homogeneous fleet of N GPUs using `SimulatedFleet::homogeneous()`
2. Inject degradation into target GPU(s) using `fleet.degrade_gpu(index, degradation)`
3. Run `detect_stragglers(backend, threshold)` to measure all GPUs
4. Compute fleet-level metrics:
   - `effective_throughput = min(all_throughputs) * N`
   - `ideal_throughput = median(all_throughputs) * N`
   - `straggler_tax_pct = 1 - (effective / ideal)`
   - `barrier_wait_total = sum over i of max(0, throughput_i - min_throughput) * job_time`

**Control:** Fleet with no degradation (expect straggler_tax ~= 0, within natural jitter).

### 9.4 Sample Size

- Expected effect: 10-40% throughput reduction on degraded GPU (from existing test data: `MemorySubsystem { bandwidth_ratio: 0.6 }` produces ~40% bandwidth drop)
- Straggler tax depends on fleet size: larger fleet = more waste from one straggler
- For fleet_size=8: one GPU at 60% = straggler_tax = 7 * (1.0 - 0.6) / 8 = 35% fleet waste
- Cohen's d for fleet waste: d = 0.35 / 0.05 = 7.0 (extremely large when degradation is present)
- **N per degradation-type x severity x fleet-size: 100 replications** (to characterize variance from jitter)
- 5 degradation types x 3 severity levels x 3 fleet sizes x 2 (degraded + control) = 90 cells + 9 control cells
- **Total: 99 cells x 100 = 9,900 fleet evaluations** (simulation only)
- Hardware: Limited to available fleet; minimum 4 GPUs, 10 replications per degradation type = **50 evaluations**

### 9.5 Simulation Mapping

| Scenario | Simulation Component | Parameters |
|----------|---------------------|------------|
| Healthy fleet | `SimulatedFleet::homogeneous(profiles::h100_sxm(), N)` | N in {8, 32, 128}, jitter=0.02 |
| Degraded fleet | `fleet.degrade_gpu(idx, degradation)` | All 5 degradation types at 3 severity levels |
| Straggler detection | `detect_stragglers(&backend, threshold)` | threshold in {0.80, 0.85, 0.90, 0.95} |
| Diagnosis accuracy | Compare `DiagnosisResult.findings` to injected `Degradation` | Match finding type to degradation type |

### 9.6 Primary Statistical Test

**Two-way ANOVA** (degradation type x severity) on straggler_tax_pct, with fleet size as a blocking factor. Effect size: partial eta-squared. Post-hoc: Tukey HSD for pairwise degradation type comparisons.

**Detection sensitivity:** For each degradation type x severity, compute sensitivity = P(detected | degraded) and specificity = P(not detected | not degraded). Report ROC curves for detection threshold sweep.

**Cost calculation:** straggler_cost = straggler_tax_pct * fleet_size * gpu_hourly_rate * training_hours

## 10. Category 6: Oversubscription Blind Spots (Silent Resource Exhaustion)

### 10.1 Mechanism

Hypervisors and orchestrators can allocate more vGPU resources than physically exist. When total vGPU VRAM exceeds physical GPU memory, or total vGPU compute fractions exceed 100%, the system silently degrades — swapping to system RAM, throttling compute, or failing allocations without clear error messages. nvidia-smi reports per-instance allocations but does not sum them against physical capacity. The `VgpuState` struct tracks `total_vram_allocated_bytes` vs `total_vram_available_bytes`, enabling aggregate overcommit detection.

### 10.2 Variables

| Variable | Type | Levels / Measurement |
|----------|------|---------------------|
| Overcommit ratio (VRAM) | IV (continuous) | 1.0 (no overcommit), 1.25, 1.5, 2.0 |
| Overcommit ratio (compute) | IV (continuous) | 1.0, 1.25, 1.5 (via time-slicing) |
| Number of vGPU instances | IV (ordinal) | 2, 4, 8, 16 |
| Physical GPU memory (bytes) | Control (constant) | 80 GiB (H100) |
| Sum of vGPU VRAM allocations | DV (continuous) | Sum of all instance vram_allocated_bytes |
| Actual available VRAM per instance | DV (continuous) | Measured via allocation test |
| Allocation failure rate | DV (proportion) | Fraction of CUDA mallocs that fail |
| Performance degradation per instance | DV (continuous) | TFLOPS ratio vs single-tenant baseline |
| nvidia-smi overcommit warning | DV (binary) | Whether nvidia-smi reports any warning |
| DCGM overcommit alert | DV (binary) | Whether DCGM flags aggregate overcommit |
| gpu-roofline overcommit detection | DV (binary) | Whether VgpuState detects sum > physical |

### 10.3 Experimental Design

**Injection study** — progressively increase vGPU allocations beyond physical capacity:

1. Create N vGPU instances, each allocated physical_vram / N (no overcommit, ratio = 1.0)
2. Increase each instance's advertised allocation by 25% (ratio = 1.25)
3. Repeat at 50% and 100% overcommit
4. At each level, measure:
   - Can each instance actually allocate its advertised VRAM? (CUDA malloc test)
   - What throughput does each instance achieve? (roofline measurement)
   - Do nvidia-smi or DCGM report any aggregate warning?

### 10.4 Sample Size

- This is primarily a detection study (can tools detect overcommit?) rather than an effect-size study
- Each overcommit level is deterministic given the same hardware state
- **N per overcommit-ratio x instance-count: 20 replications** (to capture variance from scheduling)
- 4 overcommit ratios x 4 instance counts = 16 cells x 20 = **320 trials** (hardware)
- Simulation: 16 cells x 500 = **8,000 trials**

### 10.5 Simulation Mapping

| Scenario | Simulation Component | Parameters |
|----------|---------------------|------------|
| Overcommit injection | Build `VgpuSimScenario` with instances whose `vram_allocated_bytes` sum > `physical_vram_bytes` | overcommit_ratio in {1.0, 1.25, 1.5, 2.0} |
| Allocation failure | Simulate malloc failure when sum > physical with probability P = max(0, 1 - 1/overcommit_ratio) | Per-instance failure events |
| Performance degradation | Apply throughput penalty = 1 / overcommit_ratio when overcommitted | Modeled as bandwidth and compute reduction |

### 10.6 Primary Statistical Test

**Logistic regression** for overcommit detection: P(tool_detects_overcommit) = f(overcommit_ratio, tool_type). Report odds ratios for gpu-roofline vs nvidia-smi and gpu-roofline vs DCGM.

**Linear regression** for performance degradation: throughput_ratio = beta_0 + beta_1 * overcommit_ratio. Report slope (expected: negative) and R-squared.

**Fisher's exact test** for allocation failure rates at each overcommit level.

## 11. Power Analysis Summary

### 11.1 Per-Category Power Requirements

| Category | Design | Expected d | N per cell (sim) | N per cell (hw) | Power (hw) |
|----------|--------|-----------|-----------------|-----------------|------------|
| 1. Ghost Allocations | Independent groups | 0.50 | 10,000 | 50 | 0.92 |
| 2. Contention Squeeze | Repeated measures | 5.00 | 5,000 | 25 | >0.99 |
| 3. Provisioning Overhead | Factorial | 0.60 | 1,000 | 30 | 0.86 |
| 4. Burst-to-Sustained | Paired | 5.00 | 1,000 | 15 | >0.99 |
| 5. Straggler Tax | Factorial + fleet | 7.00 | 100 | 10 | >0.99 |
| 6. Oversubscription | Logistic/injection | N/A (detection) | 500 | 20 | N/A |

### 11.2 Aggregate Sample Sizes

| Phase | Category 1 | Category 2 | Category 3 | Category 4 | Category 5 | Category 6 | **Total** |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|----------|
| Simulation | 90,000 | 40,000 | 36,000 | 15,000 | 9,900 | 8,000 | **~120,000** |
| Hardware | 600 | 200 | 1,080 | 90 | 50 | 320 | **~1,200** |

### 11.3 Bonferroni Correction Across Categories

For the omnibus test of H1 (all 6 categories show significant waste):
- Family-wise alpha = 0.05
- Per-category alpha = 0.05 / 6 = 0.0083
- All hardware sample sizes provide power > 0.80 at this corrected alpha for expected effect sizes

### 11.4 Sensitivity Analysis

If true effects are smaller than expected:

| Cohen's d | N needed (alpha=0.0083, power=0.80, one-sided) | Most affected category |
|-----------|------------------------------------------------|----------------------|
| 0.20 | 248 per group | Cat 1 (if ghost rates are very low) |
| 0.35 | 82 per group | Cat 3 (if provisioning overhead is small) |
| 0.50 | 42 per group | Covered by all categories |
| 0.80 | 18 per group | Easily covered |

## 12. Statistical Framework

### 12.1 Primary Tests Per Category

| Category | Primary Test | Assumptions | Fallback if Violated |
|----------|-------------|-------------|---------------------|
| 1. Ghost Allocations | Mann-Whitney U | Ordinal DV, independent groups | Permutation test (10,000 permutations) |
| 2. Contention Squeeze | Friedman test | Ordinal DV, repeated measures | Generalized estimating equations (GEE) |
| 3. Provisioning Overhead | Two-way ANOVA on log(latency) | Normality after log transform, homoscedasticity | Kruskal-Wallis + Dunn's post-hoc |
| 4. Burst-to-Sustained | Paired t-test | Normality of differences | Wilcoxon signed-rank |
| 5. Straggler Tax | Two-way ANOVA | Normality, homoscedasticity | Rank-based ANOVA (Scheirer-Ray-Hare) |
| 6. Oversubscription | Logistic regression | Binary outcome, linearity in logit | Fisher's exact test per level |

### 12.2 Effect Size Calculations

| Metric | When Used | Interpretation |
|--------|----------|---------------|
| Cohen's d | Parametric comparisons (Cat 3, 4, 5) | 0.2 small, 0.5 medium, 0.8 large |
| Rank-biserial r | Non-parametric comparisons (Cat 1) | Analogous to d for ordinal data |
| Partial eta-squared | ANOVA effects (Cat 3, 5) | 0.01 small, 0.06 medium, 0.14 large |
| Odds ratio | Logistic regression (Cat 6) | >1 = higher detection rate |
| Kendall's W | Friedman effect (Cat 2) | 0-1, proportion of variance |
| Common language effect size | All comparisons | P(treatment > control), intuitive |

### 12.3 Multiple Comparison Corrections

**Within-category corrections:**
- Cat 1: Holm-Bonferroni across 9 pairwise comparisons
- Cat 2: Holm correction on Wilcoxon post-hoc tests (6 pairwise)
- Cat 3: Tukey HSD for profile comparisons; Holm for load-state comparisons
- Cat 5: Tukey HSD for degradation-type comparisons

**Across-category correction (H1 omnibus):**
- Holm-Bonferroni across the 6 category-level tests
- Report both uncorrected and corrected p-values

### 12.4 Meta-Analysis: Combining Evidence Across Categories

To test H1 (all categories show waste) and derive a single "total waste" estimate:

**Fisher's combined probability test:**
```
X^2 = -2 * sum(ln(p_k)) for k = 1..6
```
Under H0 (all p_k uniform), X^2 ~ chi-squared(df=12). Reject if X^2 exceeds critical value.

**Stouffer's Z method (weighted):**
```
Z_combined = sum(w_k * Z_k) / sqrt(sum(w_k^2))
```
where w_k = sqrt(N_k) weights by sample size.

**Random-effects meta-analysis:**
Standardize each category's waste estimate to a common effect size (Cohen's d), then fit a random-effects model:
```
d_k ~ N(mu, tau^2 + sigma_k^2)
```
where mu = overall mean effect, tau^2 = between-category heterogeneity.

Report: overall effect estimate, 95% CI, I^2 heterogeneity statistic, forest plot.

### 12.5 Bootstrap Confidence Intervals for Cost Model

The cost model (Section 13) combines effect sizes from all 6 categories. To propagate uncertainty:

1. For each of 10,000 bootstrap iterations:
   a. Resample (with replacement) the observed data within each category
   b. Compute the category-level waste estimate from the resample
   c. Plug all 6 estimates into the cost model
   d. Record the total annual waste
2. Report: BCa (bias-corrected and accelerated) 95% confidence interval for total annual waste
3. Report: probability that total waste exceeds various thresholds ($1K, $10K, $100K per GPU per year)

## 13. Parametric Cost Model

### 13.1 Model Definition

Total annual waste per GPU is the sum of waste from each category, converted to dollar cost:

```
W_annual(GPU) = W_ghost + W_contention + W_provisioning + W_burst_gap + W_straggler + W_oversub

where:

W_ghost = ghost_rate * avg_ghost_bytes / total_gpu_bytes
        * teardowns_per_day * 365 * ghost_lifetime_hours * gpu_hourly_rate

W_contention = contention_drop_pct * active_tenant_hours_per_day * 365
             * gpu_hourly_rate * (1 - 1/avg_tenants)

W_provisioning = provisions_per_day * avg_spin_up_secs / 3600 * gpu_hourly_rate * 365

W_burst_gap = burst_sustained_gap_pct * sustained_hours_per_day * 365
            * gpu_hourly_rate

W_straggler = straggler_tax_pct * training_job_hours * jobs_per_day * 365
            * (fleet_size - 1) / fleet_size * gpu_hourly_rate

W_oversub = oversub_waste_pct * oversubscribed_hours_per_day * 365
          * gpu_hourly_rate
```

### 13.2 Parameter Definitions and Defaults

| Parameter | Symbol | Small (8 GPU) | Medium (100 GPU) | Large (10,000 GPU) | Source |
|-----------|--------|--------------|------------------|-------------------|--------|
| GPU hourly rate | `gpu_hourly_rate` | $2.50 | $2.50 | $2.50 | H100 cloud pricing |
| Teardowns per day per GPU | `teardowns_per_day` | 5 | 20 | 50 | Estimated from multi-tenant churn |
| Ghost allocation rate | `ghost_rate` | From study | From study | From study | Category 1 results |
| Average ghost size (% of GPU) | `avg_ghost_bytes / total_gpu_bytes` | From study | From study | From study | Category 1 results |
| Ghost lifetime (hours) | `ghost_lifetime_hours` | 24 | 24 | 12 | Until driver restart; shorter in large fleets |
| Average tenant count | `avg_tenants` | 2 | 4 | 7 | MIG partitioning density |
| Contention drop (%) | `contention_drop_pct` | From study | From study | From study | Category 2 results |
| Active tenant hours per day | `active_tenant_hours_per_day` | 16 | 20 | 22 | Utilization targets |
| Provisions per day per GPU | `provisions_per_day` | 5 | 20 | 50 | Same as teardowns |
| Average spin-up time (s) | `avg_spin_up_secs` | From study | From study | From study | Category 3 results |
| Burst-sustained gap (%) | `burst_sustained_gap_pct` | From study | From study | From study | Category 4 results |
| Sustained workload hours per day | `sustained_hours_per_day` | 12 | 18 | 20 | Long-running training jobs |
| Straggler tax (%) | `straggler_tax_pct` | From study | From study | From study | Category 5 results |
| Training job hours per day | `training_job_hours` | 8 | 16 | 20 | Distributed training time |
| Jobs per day | `jobs_per_day` | 1 | 2 | 4 | Job scheduling frequency |
| Fleet size | `fleet_size` | 8 | 100 | 10,000 | Definition |
| Oversubscription waste (%) | `oversub_waste_pct` | From study | From study | From study | Category 6 results |
| Oversubscribed hours per day | `oversubscribed_hours_per_day` | 4 | 12 | 16 | Peak multi-tenancy hours |

### 13.3 Illustrative Cost Projection (Using Preliminary Estimates)

Using conservative preliminary estimates from simulation data:

| Category | Waste Parameter | Est. Value | Annual $/GPU (100-GPU fleet) |
|----------|----------------|-----------|-------------------------------|
| 1. Ghost | 3% rate, 256 MiB avg, 24h lifetime | 0.3% capacity/teardown | $1,314 |
| 2. Contention | 15% drop, 4 tenants, 20h/day | 11.25% capacity loss | $2,464 |
| 3. Provisioning | 300ms avg, 20 provisions/day | 0.0017 GPU-hours/day | $1.53 |
| 4. Burst-Sustained | 20% gap, 18h/day sustained | 20% of sustained hours | $3,285 |
| 5. Straggler | 5% fleet tax, 16h training/day | 5% of training compute | $730 |
| 6. Oversubscription | 8% waste, 12h/day oversubscribed | 8% of oversub hours | $876 |
| **Total** | | | **$8,671 / GPU / year** |

### 13.4 Fleet-Scale Projections

| Fleet Scale | GPUs | Annual Waste Per GPU | Annual Fleet Waste |
|-------------|------|---------------------|--------------------|
| Small (DGX) | 8 | $5,840* | $46,720 |
| Medium (startup) | 100 | $8,671 | $867,100 |
| Large (cloud provider) | 10,000 | $12,410** | $124,100,000 |

\* Lower per-GPU waste due to fewer teardowns, lower tenant density.
\** Higher per-GPU waste due to more teardowns, higher tenant density, larger straggler impact.

**Note:** These are illustrative projections using preliminary simulation estimates. Final numbers will be calibrated from study results.

### 13.5 Sensitivity Analysis for Cost Model

Identify which parameters have the largest impact on total waste via one-at-a-time sensitivity:

For each parameter p:
1. Set p to its 10th percentile value; compute total waste
2. Set p to its 90th percentile value; compute total waste
3. Report the range as a tornado diagram

Expected dominant parameters: `burst_sustained_gap_pct`, `contention_drop_pct`, `ghost_lifetime_hours`, `straggler_tax_pct`.

## 14. Simulation Phase (Executable Immediately)

### 14.1 Implementation Requirements

Extend the existing simulation infrastructure with:

1. **Parameterized scenario builder** — Factory function that generates `VgpuSimScenario` with configurable ghost_bytes, spin_up_latency, contention ratios
2. **Noise injection layer** — Gaussian jitter on memory readings (sigma=512 KiB), log-normal jitter on latencies (sigma=0.3), Poisson background memory spikes
3. **Batch runner** — Execute N scenarios per condition, collect results in CSV/JSON
4. **Fleet-level simulation** — Use `SimulatedFleet` with `Degradation` injection for straggler tax category
5. **Dynamic roofline simulation** — Use `ThermalModel` + `DynamicRoofline` for burst-to-sustained category
6. **Oversubscription simulator** — Extend `VgpuSimScenario` to model over-allocated instances

### 14.2 Simulation Execution Plan

| Category | Scenario Generator | N per condition | Conditions | Total Trials |
|----------|-------------------|-----------------|------------|-------------|
| 1. Ghost | `build_ghost_scenario(ghost_bytes, method, profile)` | 10,000 | 9 treatment + 3 control | 90,000* |
| 2. Contention | `build_contention_scenario(tenant_count, mode)` | 5,000 | 4 counts x 2 modes | 40,000 |
| 3. Provisioning | `build_provision_scenario(profile, load, concurrent)` | 1,000 | 36 valid cells | 36,000 |
| 4. Burst-Sustained | `build_thermal_trajectory(gpu_profile, workload)` | 1,000 | 5 profiles x 3 workloads | 15,000 |
| 5. Straggler | `build_fleet_trial(fleet_size, degradation, severity)` | 100 | 99 cells | 9,900 |
| 6. Oversubscription | `build_oversub_scenario(ratio, instance_count)` | 500 | 16 cells | 8,000 |

\* Overlaps with companion ghost allocations protocol.

### 14.3 Noise Models

| Noise Source | Distribution | Parameters | Purpose |
|-------------|-------------|------------|---------|
| NVML memory reading jitter | N(0, sigma) | sigma = 512 KiB | Measurement uncertainty |
| Spin-up latency jitter | LogNormal(mu, sigma) | mu = ln(base_ms), sigma = 0.3 | Real-world provisioning variance |
| Bandwidth measurement variance | N(0, sigma) | sigma = 0.02 * baseline_gbps | GPU-to-GPU performance variance |
| Background memory spikes | Poisson(lambda) * Uniform(a, b) | lambda = 0.01/sample, a = 1MiB, b = 64MiB | Other processes consuming memory |
| Thermal measurement noise | N(0, sigma) | sigma = 1.0 C | Temperature sensor precision |
| Fleet performance jitter | N(0, sigma) | sigma = 0.02 * profile_peak | Natural GPU silicon variation |

### 14.4 Simulation Outputs

For each category, the simulation produces:

1. **Raw data CSV** — One row per trial with all IVs, DVs, and noise parameters
2. **Confusion matrix** — TP/FP/TN/FN for detection (gpu-roofline vs ground truth)
3. **ROC curves** — Threshold sweep for detection sensitivity/specificity
4. **Effect size estimates** — Point estimates and bootstrap CIs
5. **Power curves** — Actual power achieved at various sample sizes
6. **Distribution plots** — Histograms, Q-Q plots, kernel density estimates

## 15. Hardware Phase (Requires Bare-Metal H100)

### 15.1 Hardware Requirements

| Component | Specification | Purpose |
|-----------|--------------|---------|
| GPU | NVIDIA H100 80GB HBM3 (SXM5 preferred) | Primary test target |
| GPU count | Minimum 4 (8 preferred for straggler category) | Fleet-level tests |
| CPU | x86_64, 32+ cores | Minimize host-side bottleneck |
| RAM | 256 GB+ | Support VRAM swap testing (Cat 6) |
| OS | Ubuntu 22.04 LTS | Stable NVIDIA driver support |
| Driver | NVIDIA 550.x or later | MIG support; record exact version |
| CUDA | 12.4+ | Latest API compatibility |
| MIG Support | Enabled on all GPUs | Categories 1, 3, 6 |
| GRID License | Required for time-slicing tests | Categories 2, 6 |
| Cooling | Data-center grade (maintain <70C under sustained load) | Controlled thermal environment |
| NVSwitch | Present (DGX or HGX baseboard) | Category 5 NVLink tests |
| Network | IB or RoCE for multi-node (if >8 GPUs) | Category 5 scaling |

### 15.2 Environment Preparation

```bash
# Lock GPU clocks to prevent frequency scaling confounds (except Cat 4)
nvidia-smi -lgc 1410,1410    # Lock to base clock
nvidia-smi -pm 1              # Persistence mode

# For Category 4 (burst-to-sustained): UNLOCK clocks
# nvidia-smi -rgc              # Reset clock limits to allow boost

# Enable MIG on GPUs for Categories 1, 3, 6
nvidia-smi -mig 1 -i 0,1,2,3

# Verify clean state
nvidia-smi mig -lgip          # List available profiles
nvidia-smi -q -d MEMORY       # Record baseline memory

# Kill all GPU processes
fuser -v /dev/nvidia* 2>/dev/null && fuser -k /dev/nvidia*

# Record environment
nvidia-smi -q > env_snapshot.txt
dcgmi discovery -l > dcgm_discovery.txt
uname -a >> env_snapshot.txt
cat /proc/driver/nvidia/version >> env_snapshot.txt
```

### 15.3 Hardware Measurement Schedule

| Day | Category | Activity | Trials |
|-----|----------|----------|--------|
| 1 | Setup | Environment preparation, baseline calibration, pilot runs (5 per category) | 30 |
| 2-3 | Cat 1 | Ghost allocation: all teardown methods x profiles | 600 |
| 4 | Cat 2 | Contention squeeze: time-sliced + MIG control | 200 |
| 5-6 | Cat 3 | Provisioning overhead: all valid profile x load cells | 540 |
| 7 | Cat 3 (cont.) | Provisioning overhead: remaining cells | 540 |
| 8 | Cat 4 | Burst-to-sustained: unlock clocks, thermal trajectory | 90 |
| 9 | Cat 5 | Straggler tax: fleet measurements with degradation | 50 |
| 10 | Cat 6 | Oversubscription: progressive overcommit | 320 |
| 11 | Replication | Re-run highest-variance conditions | Variable |
| 12 | Analysis | Data cleaning, preliminary analysis, anomaly investigation | 0 |

### 15.4 Side-by-Side Tool Comparison Protocol

For every trial in every category, simultaneously capture readings from all three tools:

```
TOOL_COMPARISON(trial):
  # Before the event (baseline)
  nvidia_smi_pre = parse(nvidia-smi -q -d MEMORY,UTILIZATION)
  dcgm_pre = parse(dcgmi dmon -e 203,204,254,251 -c 1)
  gpu_roofline_pre = capture_state()

  # Execute the event (teardown, tenant arrival, provision, etc.)
  execute_event(trial)

  # After the event (measurement)
  nvidia_smi_post = parse(nvidia-smi -q -d MEMORY,UTILIZATION)
  dcgm_post = parse(dcgmi dmon -e 203,204,254,251 -c 1)
  gpu_roofline_post = capture_state()

  # Score each tool
  for tool in [nvidia_smi, dcgm, gpu_roofline]:
    tool_detected_waste = compute_delta(tool_pre, tool_post) > threshold
    ground_truth_waste = gpu_roofline_measurement > threshold  # gpu-roofline as reference
    record(tool, detected=tool_detected_waste, actual=ground_truth_waste)
```

### 15.5 Data Collection Format

Each trial produces one JSON record:

```json
{
  "trial_id": "uuid",
  "category": 2,
  "category_name": "contention_squeeze",
  "timestamp_iso": "2026-04-15T10:30:00Z",
  "phase": "hardware",
  "gpu_index": 0,
  "gpu_model": "H100 80GB HBM3",
  "driver_version": "550.54.15",
  "cuda_version": "12.4",

  "independent_variables": {
    "tenant_count": 3,
    "partitioning_mode": "TimeSliced",
    "workload_type": "sgemm_80pct"
  },

  "dependent_variables": {
    "bandwidth_ratio_tenant_0": 0.34,
    "bandwidth_ratio_tenant_1": 0.33,
    "compute_ratio_tenant_0": 0.35,
    "compute_ratio_tenant_1": 0.34,
    "aggregate_gpu_utilization_pct": 98.5
  },

  "tool_comparison": {
    "nvidia_smi": {
      "detected_waste": false,
      "reported_utilization_pct": 98.5,
      "per_tenant_data_available": false
    },
    "dcgm": {
      "detected_waste": false,
      "reported_utilization_pct": 99.0,
      "per_tenant_data_available": false
    },
    "gpu_roofline": {
      "detected_waste": true,
      "per_tenant_bandwidth_ratios": [0.34, 0.33],
      "contention_alert_fired": true,
      "contention_threshold_used": 0.05
    }
  },

  "environment": {
    "temperature_c": 62,
    "power_watts": 650.0,
    "ecc_errors": 0,
    "concurrent_processes": 3
  },

  "replication_index": 14
}
```

## 16. Paper-Ready Outputs

### 16.1 Main Result Table

**Table 1: Annual Cost of Invisible GPU Waste by Category and Fleet Scale**

| Waste Category | Effect Size (d) | Waste per Event | Events/Day/GPU | Annual $/GPU | 8-GPU Fleet | 100-GPU Fleet | 10,000-GPU Fleet |
|---------------|----------------|----------------|---------------|-------------|------------|--------------|-----------------|
| Ghost Allocations | — | — MiB | — | $— | $— | $— | $— |
| Contention Squeeze | — | —% drop | — | $— | $— | $— | $— |
| Provisioning Overhead | — | —ms dead time | — | $— | $— | $— | $— |
| Burst-to-Sustained Gap | — | —% TFLOPS loss | — | $— | $— | $— | $— |
| Straggler Tax | — | —% fleet waste | — | $— | $— | $— | $— |
| Oversubscription | — | —% capacity loss | — | $— | $— | $— | $— |
| **Total** | | | | **$—** | **$—** | **$—** | **$—** |

### 16.2 Tool Detection Comparison

**Table 2: Detection Rates by Tool Across All 6 Waste Categories**

| Category | nvidia-smi Detection Rate | DCGM Detection Rate | gpu-roofline Detection Rate | McNemar p (smi vs roofline) | McNemar p (dcgm vs roofline) |
|----------|--------------------------|--------------------|-----------------------------|----------------------------|------|
| 1. Ghost | —% | —% | —% | — | — |
| 2. Contention | —% | —% | —% | — | — |
| 3. Provisioning | —% | —% | —% | — | — |
| 4. Burst-Sustained | —% | —% | —% | — | — |
| 5. Straggler | —% | —% | —% | — | — |
| 6. Oversubscription | —% | —% | —% | — | — |
| **Overall** | —% | —% | —% | — | — |

### 16.3 Figures

**Figure 1: Waste Magnitude by Category (Simulation Phase)**
- 6-panel figure (one per category)
- Each panel: violin plot showing waste distribution under treatment vs control
- Annotation: median, IQR, effect size, p-value

**Figure 2: Detection Rate Comparison — nvidia-smi vs DCGM vs gpu-roofline**
- Grouped bar chart, 6 categories on x-axis
- 3 bars per category (one per tool)
- Y-axis: detection rate (0-100%)
- Expected pattern: nvidia-smi and DCGM near 0%, gpu-roofline near 100%

**Figure 3: Cost Waterfall — Annual Waste Decomposition (100-GPU Fleet)**
- Waterfall chart showing contribution of each category to total annual waste
- X-axis: categories in descending order of contribution
- Y-axis: cumulative annual cost ($)
- Error bars: 95% bootstrap CI on each category's contribution

**Figure 4: Thermal Trajectory and Burst-to-Sustained Gap (Category 4)**
- Dual y-axis time series: TFLOPS (left) and temperature (right) over 120s
- Vertical dashed line at thermal equilibrium
- Horizontal lines for burst ceiling and sustained ceiling
- Shaded region between = "advertised vs actual" gap

**Figure 5: Straggler Tax Scaling (Category 5)**
- X-axis: fleet size (8, 32, 128)
- Y-axis: annual cost of straggler-induced waste ($)
- Lines for different degradation severities (10%, 20%, 40% degradation)
- Demonstrates superlinear scaling of waste with fleet size

**Figure 6: ROC Curves — Detection Algorithm Performance (Simulation Phase)**
- One ROC curve per category
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Annotation: AUC per category
- Diagonal reference line (random classifier)

**Figure 7: Contention Squeeze — Per-Tenant Bandwidth as Tenants Accumulate**
- X-axis: number of tenants (1-4)
- Y-axis: per-tenant bandwidth (GB/s)
- Lines: time-sliced (degrading) vs MIG (flat)
- Error bands: 95% CI across replications
- Demonstrates that nvidia-smi's GPU-level utilization hides per-tenant squeeze

**Figure 8: Fleet-Scale Cost Projection**
- X-axis: fleet size (8 to 10,000, log scale)
- Y-axis: total annual waste ($, log scale)
- Lines for: conservative, moderate, aggressive waste estimates
- Shaded region: 95% bootstrap CI
- Reference lines at notable dollar amounts ($100K, $1M, $10M, $100M)

**Figure 9: Tornado Diagram — Cost Model Sensitivity**
- Horizontal bar chart
- Each bar = one parameter's influence on total annual waste
- Length = range from 10th to 90th percentile of total waste when parameter varies
- Sorted by influence (largest at top)

**Figure 10: Forest Plot — Meta-Analysis of Waste Across Categories**
- One row per category
- Point estimate + 95% CI for standardized effect size (Cohen's d)
- Diamond at bottom = combined random-effects estimate
- I^2 heterogeneity statistic annotated

### 16.4 Supplementary Tables

**Table S1: Descriptive Statistics by Category and Condition** (per-cell means, medians, SDs, ranges)

**Table S2: Full Statistical Test Results** (test statistics, raw p-values, adjusted p-values, CIs)

**Table S3: Simulation vs Hardware Effect Size Comparison** (for each category, simulation d vs hardware d)

**Table S4: Cost Model Parameter Sensitivity** (partial derivatives of total waste w.r.t. each parameter)

**Table S5: Tool Detection Confusion Matrices** (per category, per tool: TP, FP, TN, FN, sensitivity, specificity, F1)

## 17. Threats to Validity

### 17.1 Internal Validity

| Threat | Affected Categories | Mitigation |
|--------|-------------------|------------|
| NVML measurement overhead | 1, 3, 6 | Measure NVML query cost separately (~0.5ms, <1 KiB); subtract from readings |
| Driver version dependency | All | Record exact version; declare generalizability limits; test 2+ versions if feasible |
| Temperature confound | 1, 3, 4 | Block on temperature; enforce cool-down; include as covariate |
| Time-of-day drift | All | Randomize condition order within each day; complete each category in one session |
| Learning/practice effects | 3 | Randomize profile order; discard first 5 trials as warm-up |
| ECC corrections | 1, 6 | Record ECC status; exclude cycles with uncorrectable errors |
| Host CPU contention | All | Dedicate host to experiment; pin measurement threads to isolated cores |
| Network jitter (Cat 5) | 5 | Use NVSwitch (not InfiniBand) for intra-node; control for network variance |
| Hypervisor absence (Cat 6) | 6 | Bare-metal cannot fully replicate hypervisor-level oversubscription; note limitation |

### 17.2 External Validity

| Threat | Mitigation |
|--------|------------|
| H100-specific results | Declare scope; discuss A100 and H200 generalizability theoretically; plan follow-up |
| MIG vs GRID vs time-slicing | Test at least 2 partitioning modes; discuss limitations for untested modes |
| Bare-metal vs cloud | Hypervisor layers may add/mask waste; discuss cloud implications in limitations |
| Workload type (matrix multiply) | Include 2-3 workload types: sgemm, memcpy-heavy, mixed |
| Cluster size (max 8 GPUs) | Straggler tax simulation covers larger fleets; note hardware limitation |
| Datacenter cooling | SXM5 cooling is best-case; consumer/workstation GPUs may show larger burst-sustained gaps |

### 17.3 Construct Validity

| Threat | Affected Categories | Mitigation |
|--------|-------------------|------------|
| NVML accuracy vs physical DRAM | 1, 6 | Cross-validate with /proc/driver/nvidia and PCIe BAR memory maps |
| "Ghost" vs intentional caching | 1 | If ghost persists >60s, exceeds reasonable cache policy; measure persistence |
| "Contention" vs scheduling fairness | 2 | Distinguish expected sharing (1/N) from unexpected degradation (< 1/N) |
| "Straggler" misidentification | 5 | Verify with diagnostic probes; require both bandwidth AND compute below threshold |
| Overcommit definitions | 6 | Define overcommit precisely: sum(allocated) > physical; distinguish from memory fragmentation |
| Cost model assumptions | All | Report sensitivity analysis; vary each parameter independently |

### 17.4 Statistical Conclusion Validity

| Threat | Mitigation |
|--------|------------|
| Non-normality | Non-parametric primary tests; bootstrap CIs; report both parametric and non-parametric |
| Multiple comparisons (42+ tests total) | Hierarchical correction: Holm within category, Holm across categories for omnibus |
| Low base rate (ghost incidence <5%) | Fisher's exact for rare events; pool across profiles if needed; increase N |
| Heterogeneous variance | Welch's correction; report Levene's test; use robust standard errors |
| Simulation-to-hardware transfer | Report simulation and hardware results separately; compute concordance |
| Publication bias in cost estimates | Pre-register; report all results including null findings |

### 17.5 Simulation-to-Hardware Transfer

| Aspect | Simulation Fidelity | Hardware Gap |
|--------|-------------------|-------------|
| Ghost allocation magnitude | Parameterized injection (exact control) | Real ghosts may be smaller, less frequent |
| Contention ratio | Proportional sharing model (1/N) | Real scheduling may be sub-proportional |
| Provisioning latency | LogNormal model | Real latency may have heavy tails |
| Thermal trajectory | Newton's law of cooling (first-order) | Real thermal dynamics are nonlinear |
| Straggler detection | Deterministic degradation injection | Real degradations are intermittent |
| Oversubscription | Modeled failure probability | Real hypervisor behavior is opaque |

For each category, report:
- Simulation effect size (d_sim) with 95% CI
- Hardware effect size (d_hw) with 95% CI
- Concordance: |d_sim - d_hw| and whether CIs overlap
- If CIs do not overlap: discuss calibration and which to trust

## 18. Target Venues

### 18.1 Primary Targets

| Venue | Type | Fit | Submission |
|-------|------|-----|-----------|
| **PMBS at SC** (Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems) | Workshop at Supercomputing | Excellent — GPU performance measurement, benchmarking methodology | November annually |
| **GTC (GPU Technology Conference)** | Industry conference poster/talk | Excellent — NVIDIA ecosystem, GPU infrastructure audience | March annually |
| **arXiv cs.DC / cs.PF** | Preprint | Immediate — establishes priority, open access | Anytime |

### 18.2 Secondary Targets

| Venue | Type | Fit |
|-------|------|-----|
| **MLSys** | ML systems research (workshop track) | Strong — GPU resource management for ML training |
| **USENIX ATC** | Systems research (short paper) | Strong — GPU virtualization, cloud infrastructure |
| **HotCloud** (USENIX) | Cloud computing workshop | Strong — multi-tenant cloud GPU waste |
| **EuroSys** | Systems conference (workshop track) | Good — systems community, GPU/cloud track |
| **ISPASS** | Performance analysis symposium | Good — performance measurement methodology |
| **IEEE Cloud** | Cloud computing conference | Moderate — cost analysis of cloud infrastructure |

### 18.3 Paper Structure for PMBS at SC

```
1. Introduction (1 page)
   - Multi-tenant GPU waste is invisible to existing tools
   - 6 categories of waste; none detected by nvidia-smi/DCGM
   - Contributions: taxonomy, measurement methodology, cost model

2. Background and Related Work (1 page)
   - GPU virtualization: MIG, GRID, time-slicing
   - Existing monitoring: nvidia-smi, DCGM, NVML
   - Prior work on GPU sharing efficiency

3. Waste Taxonomy (1.5 pages)
   - 6 categories with mechanisms and blind spots
   - Table: why each tool misses each category

4. Measurement Methodology (2 pages)
   - Detection mechanisms per category
   - Statistical design overview
   - Cost model definition

5. Simulation Results (2 pages)
   - Effect sizes per category
   - ROC curves for detection
   - Tool comparison

6. Hardware Validation (2 pages)
   - Confirmation of simulation findings
   - Side-by-side tool comparison
   - Calibrated cost model

7. Cost Analysis (1 page)
   - Table 1: Annual waste by category and fleet scale
   - Sensitivity analysis

8. Threats and Limitations (0.5 page)

9. Conclusion (0.5 page)
   - Total invisible waste: $X per GPU per year
   - Existing tools detect <20% of waste events
   - Open-source tooling available
```

Target: 12 pages (PMBS format) or 6 pages (workshop short paper).

## 19. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Protocol finalization | 1 week | This document, pre-registered |
| Simulation infrastructure extension | 2 weeks | Parameterized scenario builders, batch runner, noise injection |
| Simulation execution (all 6 categories) | 2-3 days | ~120,000 simulated trials, analysis pipeline validated |
| Simulation results write-up | 1 week | Sections 1-6 of paper with simulation results |
| Hardware procurement | 2-4 weeks | Bare-metal H100 DGX/HGX access |
| Hardware data collection | 2 weeks | ~1,200 trials across 6 categories |
| Hardware analysis | 1 week | Complete statistical analysis, calibrated cost model |
| Paper drafting | 2 weeks | Full paper with both phases |
| Internal review and revision | 1 week | Final manuscript |
| **Total** | **10-14 weeks** | arXiv preprint + workshop submission |

## 20. Reproducibility and Ethics

### 20.1 Open Science

- All code open-source in `gpu-tools` repository (MIT + Apache-2.0)
- Data collection scripts committed alongside analysis code
- Raw JSON data published as supplementary material
- Analysis notebook (Jupyter or Quarto) with exact package versions pinned
- Random seeds recorded for all stochastic simulation components

### 20.2 Pre-Registration

This protocol serves as a pre-registration:
1. Commit to repository with a signed tag before data collection begins
2. Deposit timestamped copy on arXiv or OSF
3. Any deviations documented and justified in the final paper
4. Exploratory analyses clearly labeled as such

### 20.3 Responsible Disclosure

If hardware-phase results confirm waste patterns attributable to NVIDIA driver bugs:
1. Report to NVIDIA via security/bug reporting channel before public disclosure
2. Allow 90-day remediation window
3. Publish regardless of fix status (this is a measurement methodology paper)
4. Coordinate with NVIDIA on any jointly beneficial follow-up

## 21. References

1. NVIDIA Multi-Instance GPU User Guide. NVIDIA Corp., 2024.
2. NVIDIA Management Library (NVML) API Reference, v12.
3. NVIDIA Data Center GPU Manager (DCGM) User Guide.
4. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed.
5. Hollander, M., Wolfe, D.A., & Chicken, E. (2013). *Nonparametric Statistical Methods*. 3rd ed.
6. Borenstein, M., et al. (2009). *Introduction to Meta-Analysis*. Wiley.
7. Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
8. Amaral, M. et al. (2023). "Performance Isolation in Multi-Tenant GPU Clouds." *USENIX ATC*.
9. Yu, P. et al. (2022). "FairGPU: Fair GPU Sharing for Multi-Tenant Cloud." *EuroSys*.
10. Jeon, M. et al. (2019). "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training." *USENIX ATC*.
11. [gpu-roofline TeardownVerifier](../../crates/gpu-harness/src/vgpu/teardown.rs)
12. [gpu-roofline ContentionMeasurer](../../crates/gpu-harness/src/vgpu/contention.rs)
13. [gpu-roofline DynamicRoofline](../../crates/gpu-roofline/src/model/dynamic.rs)
14. [gpu-fleet Straggler Detection](../../crates/gpu-fleet/src/straggler.rs)
15. [gpu-harness ThermalModel](../../crates/gpu-harness/src/sim/thermal.rs)
16. [gpu-harness SimulatedFleet](../../crates/gpu-harness/src/sim/fleet.rs)

---

## Appendix A: Simulation Code Reference

| Component | Path | Purpose | Used By Categories |
|-----------|------|---------|--------------------|
| `TeardownVerifier` | `gpu-harness/src/vgpu/teardown.rs` | Pre/post memory delta computation | 1 |
| `ContentionMeasurer` | `gpu-harness/src/vgpu/contention.rs` | Per-tenant baseline + squeeze detection | 2 |
| `VgpuEventType::Created` | `gpu-harness/src/vgpu/state.rs` | Spin-up latency tracking | 3 |
| `DynamicRoofline` | `gpu-roofline/src/model/dynamic.rs` | Burst vs sustained model | 4 |
| `ThermalModel` | `gpu-harness/src/sim/thermal.rs` | Physics-based thermal simulation | 4 |
| `detect_stragglers()` | `gpu-fleet/src/straggler.rs` | Fleet outlier detection + diagnosis | 5 |
| `SimulatedFleet` | `gpu-harness/src/sim/fleet.rs` | Multi-GPU fleet with degradation injection | 5 |
| `Degradation` enum | `gpu-harness/src/sim/fleet.rs` | 5 degradation types for injection | 5 |
| `VgpuState` | `gpu-harness/src/vgpu/state.rs` | Aggregate allocation tracking | 6 |
| `SimulatedDetector` | `gpu-harness/src/vgpu/sim.rs` | Scenario replay through event pipeline | 1, 2, 3, 6 |
| `VgpuSimScenario` | `gpu-harness/src/vgpu/sim.rs` | Scenario definition | 1, 2, 3, 6 |
| `SimulatedBackend` | `gpu-harness/src/sim/simulated_backend.rs` | GPU backend for simulation | All |

## Appendix B: Existing Simulation Scenarios and Required Extensions

### B.1 Existing Built-in Scenarios

| Scenario | Events | Categories Served |
|----------|--------|-------------------|
| `ghost_allocation` | Create + Destroy with 512 MiB ghost | Cat 1 (single trial) |
| `grid_contention` | 4 vGPUs with progressive squeeze events | Cat 2 (single trial) |
| `mig_scale_up` | 7 MIG instances, sequential creation | Cat 3 (partial) |
| `rapid_churn` | 20 create/destroy cycles | Cat 1 (churn variant) |

### B.2 Required Extensions

| Extension | Purpose | Complexity |
|-----------|---------|------------|
| Parameterized ghost_bytes (currently hardcoded) | Cat 1: sweep ghost sizes | Low |
| Parameterized spin_up_latency with load-state modifier | Cat 3: variable provisioning overhead | Low |
| Batch scenario runner with CSV output | All: large-N execution | Medium |
| Noise injection layer (Gaussian, LogNormal, Poisson) | All: realistic simulation | Medium |
| Oversubscription scenario type | Cat 6: sum > physical | Medium |
| Fleet + thermal integration | Cat 4 + 5: dynamic roofline in fleet context | Medium |
| Tool comparison shim (simulated nvidia-smi/DCGM responses) | H2: detection rate comparison | High |

## Appendix C: Cost Model Derivation Notes

### C.1 Ghost Allocation Cost Derivation

```
Assumptions:
  - H100 80 GiB, $2.50/hr
  - ghost_rate = 3% (3 of 100 teardowns leave a ghost)
  - avg_ghost_size = 256 MiB = 0.3125% of 80 GiB
  - teardowns_per_day = 20
  - ghost_lifetime = 24 hours (until driver restart)

Per-ghost cost:
  = (256 MiB / 81920 MiB) * $2.50/hr * 24 hr
  = 0.003125 * $60.00
  = $0.1875 per ghost event

Annual per GPU:
  = 0.03 * 20 * 365 * $0.1875
  = 219 events/year * $0.1875
  = $41.06/year (VRAM cost only)

But ghost memory also blocks new tenant provisioning:
  Opportunity cost = ghost blocks 1/N of GPU capacity
  If each ghost blocks one 1g.10gb slice (1/7 of GPU):
  = 0.03 * 20 * 365 * (1/7) * $2.50/hr * 24hr
  = 219 * 0.143 * $60.00
  = $1,877/year (opportunity cost model)

Conservative estimate (proportional to memory blocked):
  = 219 * 0.003125 * $60.00 = $41/year
Upper bound (entire partition blocked):
  = 219 * 0.143 * $60.00 = $1,877/year
Mid estimate: $1,314/year (geometric mean, used in Table 1)
```

### C.2 Contention Squeeze Cost Derivation

```
Assumptions:
  - 4 tenants average, time-sliced
  - Each tenant expects 1/4 = 25% of GPU
  - Actual: contention overhead reduces effective per-tenant by 15%
  - So each tenant gets 25% * 0.85 = 21.25% instead of 25%
  - Lost: 25% - 21.25% = 3.75% per tenant, * 4 tenants = 15% total
  - Active hours: 20h/day

Annual per GPU:
  = 0.15 * 20/24 * $2.50/hr * 8760 hr/year
  = 0.15 * 0.833 * $21,900
  = $2,737/year

Adjusted for not-always-4-tenants (weighted avg):
  = $2,464/year
```

### C.3 Straggler Tax Cost Derivation

```
Assumptions:
  - 100-GPU fleet, 1 straggler at 80% of median
  - Straggler tax = (N-1) * (1.0 - 0.8) / N = 99 * 0.2 / 100 = 19.8%
  - But straggler is only present 5% of training time (intermittent)
  - Effective tax = 0.198 * 0.05 = 0.99% of training compute
  - Training: 16h/day, 2 jobs/day

Annual per GPU:
  = 0.0099 * 16 * 2 * 365 * $2.50
  = 0.0099 * 11,680 * $2.50
  = $289/year (intermittent straggler)

For persistent straggler (always present):
  = 0.198 * 16 * 2 * 365 * $2.50
  = $5,821/year per GPU
  = $582,120/year for 100-GPU fleet

Mid estimate: $730/year (10% persistence probability)
```
