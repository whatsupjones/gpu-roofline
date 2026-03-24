# Hardware Validation of nvidia-smi Monitoring Limitations Under MIG

**Protocol Version:** 1.0
**Date:** 2026-03-24
**Status:** Pre-registered (commit tagged before data collection)
**GPU Target:** NVIDIA GH200 480GB (Lambda Labs, bare metal)
**Driver:** 580.105.08 | CUDA 12.8

---

## 1. Objectives

This study validates whether nvidia-smi accurately reports GPU resource state under MIG partitioning. We test five hypotheses spanning provisioning latency, memory reporting divergence, ghost allocations, cross-partition contention, and thermal performance degradation.

Pilot measurements on the same hardware (2026-03-24, ad-hoc) informed effect size estimates for the power analysis below. All empirical results in this protocol are collected prospectively under the pre-registered design.

## 2. Hypotheses

### H1: MIG Provisioning Overhead

MIG instance creation incurs measurable latency that nvidia-smi does not report.

- H1_0: Mean create latency = 0ms
- H1_A: Mean create latency > 0ms
- Test: One-sample t-test (one-sided)
- Alpha: 0.05 (before Holm-Bonferroni correction)
- Pilot: 1,185ms mean, SD 30ms, n=50

### H2: nvidia-smi Memory Reporting Divergence

nvidia-smi memory.free diverges from actual MIG-available capacity as a linear function of partition count n.

- H2_0: Divergence D(n) = 0 for all n
- H2_A: D(n) = (q - u_mean) * n + M_fragmentation
- Test: Linear regression of D on n, F-test for slope != 0
- Alpha: 0.05
- Note: Also provable algebraically from the definitions of nvidia-smi's reporting model

### H3: Ghost Allocation (Capacity-Based)

MIG teardown fails to release partition capacity for re-provisioning.

- H3_0: Re-creation success rate = 100%
- H3_A: Re-creation success rate < 100%
- Test: Exact binomial test (one-sided)
- Alpha: 0.05
- Pilot: 0/10 on this hardware. N=50 detects ghost rate >= 5.8% with power 0.80.

### H4: MIG Cross-Partition Contention

Per-partition bandwidth decreases when co-tenant partitions are under sustained load.

- H4_0: Bandwidth ratio (contention / baseline) = 1.0
- H4_A: Bandwidth ratio < 1.0
- Test: Paired Wilcoxon signed-rank (one-sided)
- Alpha: 0.05
- Expected: MIG provides hardware isolation. Null should hold (no contention). This is the control condition for future GRID time-slicing comparison.

### H5: Burst-to-Sustained Performance Gap

GPU compute throughput under sustained load is lower than cold-start burst throughput.

- H5_0: Sustained GFLOPS = Burst GFLOPS
- H5_A: Sustained GFLOPS < Burst GFLOPS
- Test: Paired Wilcoxon signed-rank (one-sided)
- Alpha: 0.05
- Pilot: 0.0% gap on datacenter-cooled GH200. Null expected for datacenter cooling.

### Multiple Comparison Correction

Holm-Bonferroni across H1-H5 (family-wise alpha = 0.05).

## 3. Power Analysis

| Hypothesis | Pilot effect | Pilot SD | Cohen's d | N (power=0.80) | N planned | Rationale |
|-----------|-------------|---------|----------|----------------|-----------|-----------|
| H1 | 1,185ms | 30ms | 39.5 | 3 | 30 | CI precision |
| H2 | 11,249 MiB/n | ~0 | Algebraic | 7 | 7 | Deterministic |
| H3 | 0% rate | — | — | 50 | 50 | Detect >= 5.8% rate |
| H4 | Unknown | Est 2% CV | Est 0.5 | 27 | 30 | Conservative |
| H5 | 0.0% | 0.1% | ~0 | >1000 | 15 | Confirm null |

## 4. Protocol

### 4.1 Environment

- Single NVIDIA GH200 480GB
- Bare metal access (MIG configurable)
- No other GPU processes during testing
- gpu-roofline 0.1.0 with --features cuda
- Python 3 + PyTorch for VRAM workloads

### 4.2 Execution Order

1. Category 3 (provisioning) — fastest, warms the GPU
2. Category 6 (divergence) — deterministic, validates framework
3. Category 1 (ghost) — most cycles, after GPU is warm
4. Category 2 (contention) — paired design, needs stable baseline
5. Category 4 (tension) — longest per trial, MIG disabled

### 4.3 Category 3: Provisioning Overhead

FOR each profile in [1g.12gb (ID 19), 3g.48gb (ID 9)], 30 cycles:
1. Clean MIG state (destroy all instances, sleep 1s)
2. Record: gpu_temp, gpu_power, timestamp
3. T_create_start = nanosecond timestamp
4. nvidia-smi mig -cgi {profile} -C -i 0
5. T_create_end = nanosecond timestamp
6. create_ms = (T_create_end - T_create_start) / 1e6
7. T_destroy_start = nanosecond timestamp
8. nvidia-smi mig -dci -i 0; nvidia-smi mig -dgi -i 0
9. T_destroy_end = nanosecond timestamp
10. destroy_ms = (T_destroy_end - T_destroy_start) / 1e6
11. Output JSON record

### 4.4 Category 6: Divergence Validation

1. Enable MIG, clean state
2. Record M_overhead = memory.used (MIG enabled, 0 instances)
3. FOR n in [1..7]:
   a. Clean state
   b. Create n instances of 1g.12gb
   c. Sleep 2s
   d. Record: nvidia_free, nvidia_used, nvidia_total
   e. Compute: m_available = nvidia_total - n*11264 - m_overhead
   f. Compute: divergence = nvidia_free - m_available
   g. Output JSON record
4. Clean state

### 4.5 Category 1: Ghost Allocation

**Test A (50 cycles):**
FOR each cycle in [1..50]:
1. Clean state
2. Create 7 instances of 1g.12gb
3. Destroy GI 7: nvidia-smi mig -dci -ci 0 -gi 7 -i 0; nvidia-smi mig -dgi -gi 7 -i 0
4. Sleep 1s
5. Attempt: nvidia-smi mig -cgi 19 -C -i 0
6. Record: recreate_success (true/false)
7. If success: destroy new instance
8. Clean state

**Test B (5 replications):**
FOR each replication in [1..5]:
1. Clean state, record initial_max (create max, count, clean)
2. 20x rapid cycle: create 1g.12gb, destroy
3. Sleep 2s
4. Record final_max (create max, count, clean)
5. capacity_loss = initial_max - final_max

### 4.6 Category 2: MIG Contention

FOR each trial in [1..30]:
**Phase A (baseline):**
1. Clean state
2. Create 1 instance of 1g.12gb
3. Record UUID
4. CUDA_VISIBLE_DEVICES={uuid} gpu-roofline measure --burst --format json
5. Record: bw_baseline, gflops_baseline
6. Destroy instance

**Phase B (co-tenant load):**
1. Clean state
2. Create 7 instances of 1g.12gb
3. Record target UUID (first instance)
4. Launch PyTorch sustained workload on 6 non-target instances
5. Sleep 3s (stabilization)
6. CUDA_VISIBLE_DEVICES={target_uuid} gpu-roofline measure --burst --format json
7. Record: bw_contention, gflops_contention
8. Kill workloads, destroy all instances

Output: bw_ratio = bw_contention / bw_baseline per trial

### 4.7 Category 4: Burst-to-Sustained

1. Disable MIG, sleep 5s
2. FOR each replication in [1..15]:
   a. Record: gpu_temp, gpu_clock, gpu_power
   b. gpu-roofline measure --format json (120s dynamic)
   c. Record: burst_gflops, sustained_gflops, tension_pct
   d. Sleep 30s (thermal reset)

## 5. Analysis Plan

All analyses locked before data collection.

1. **H1:** One-sample t-test, report mean, 95% CI, Cohen's d
2. **H2:** OLS regression D ~ n, report slope (95% CI), intercept, R-squared, F-test p-value
3. **H3:** Exact binomial, report proportion, Clopper-Pearson 95% CI
4. **H4:** Paired Wilcoxon signed-rank on bw_ratio vs 1.0, report median, p-value, rank-biserial r
5. **H5:** Paired Wilcoxon signed-rank on tension, report median, p-value, rank-biserial r
6. **Correction:** Holm-Bonferroni adjusted p-values for H1-H5
7. **Effect sizes:** Report for all hypotheses regardless of significance

## 6. Deviations

Any protocol deviations during execution will be documented here after data collection, with justification.

(None recorded — protocol not yet executed.)

## 7. Data Availability

All raw JSON data, the execution script, and analysis code will be committed to the repository. The pre-registration tag ensures the protocol predates the data.
