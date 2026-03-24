# Category 5: Straggler Tax — Hardware Validation Protocol

**Protocol Version:** 1.0
**Date:** 2026-03-24
**Status:** Pre-registered (commit tagged before data collection)
**GPU Target:** 8x H100 SXM (Lambda Labs, bare metal, NVLink)
**Companion:** study-protocol-hardware-validation.md (Categories 1-4, 6)

---

## 1. Objective

Validate whether a single degraded GPU in a multi-GPU fleet causes measurable throughput loss at the fleet level, and whether nvidia-smi can identify the fleet-wide impact from per-GPU metrics alone.

## 2. Hypothesis

### H6: Straggler Tax

A single GPU with reduced bandwidth causes measurable fleet-level throughput loss that nvidia-smi's per-GPU metrics do not capture.

- H6_0: Effective fleet throughput with one degraded GPU = effective fleet throughput with no degraded GPUs
- H6_A: Effective fleet throughput with one degraded GPU < effective fleet throughput with no degraded GPUs
- Test: Paired Wilcoxon signed-rank (one-sided), alpha = 0.05
- Expected: Straggler at 60% bandwidth on 1 of 8 GPUs causes ~5% effective fleet loss (all GPUs wait at barrier for slowest)

### H6b: nvidia-smi Fleet Visibility

nvidia-smi per-GPU utilization metrics do not reveal fleet-level synchronization loss.

- H6b_0: nvidia-smi reports per-GPU metrics that correctly predict fleet throughput loss
- H6b_A: nvidia-smi per-GPU metrics show normal utilization on healthy GPUs, failing to indicate they are idle at barrier
- Test: Descriptive (compare nvidia-smi utilization on healthy GPUs during straggler condition vs baseline)

### Correction

Holm-Bonferroni with H1-H5 from the companion protocol (family-wise alpha = 0.05 across all 6+ hypotheses).

## 3. Power Analysis

From simulation pilot data:
- One GPU at 60% bandwidth: straggler_tax = 7 * (1.0 - 0.6) / 8 = 35% effective fleet waste
- Expected variance in roofline measurement: CV ~2%
- Cohen's d for fleet throughput difference: d = 0.35 / 0.02 = 17.5 (massive effect)
- N for power 0.80: 3 trials
- **N planned: 20** (10 baseline + 10 straggler) for CI precision and robustness

## 4. Protocol

### 4.1 Environment Verification

1. Confirm 8 GPUs visible: `nvidia-smi -L`
2. Confirm NVLink topology: `nvidia-smi topo -m`
3. Record: GPU models, driver version, CUDA version, NVLink bandwidth
4. No MIG enabled (full GPU per device)
5. No other GPU processes running

### 4.2 Baseline Phase (10 replications)

FOR each replication in [1..10]:
1. Record: per-GPU temperature, clock, power
2. Run gpu-roofline measure --burst on ALL 8 GPUs simultaneously (parallel)
3. Record per-GPU: bandwidth_gbps, gflops
4. Compute: fleet_median = median(all_bw), fleet_min = min(all_bw)
5. Compute: effective_throughput = fleet_min * 8
6. Compute: ideal_throughput = fleet_median * 8
7. Compute: straggler_tax = 1 - (effective / ideal)
8. Record nvidia-smi utilization per GPU
9. Sleep 10s between replications

### 4.3 Straggler Phase (10 replications)

The "straggler" is created by running a competing bandwidth-heavy workload on GPU 0 while measuring roofline on all 8 GPUs.

FOR each replication in [1..10]:
1. Launch sustained memory-bandwidth workload on GPU 0 (STREAM copy, consuming ~50% of bandwidth)
2. Sleep 3s (stabilization)
3. Run gpu-roofline measure --burst on ALL 8 GPUs simultaneously
4. Record per-GPU: bandwidth_gbps, gflops
5. Compute: fleet_median, fleet_min, effective_throughput, ideal_throughput, straggler_tax
6. Record nvidia-smi utilization per GPU (expect GPU 0 shows high utilization but no fleet-level indicator)
7. Kill competing workload
8. Sleep 10s between replications

### 4.4 Straggler Detection Test

After collecting raw data, run gpu-fleet straggler detection:
1. `gpu-fleet straggler` on the baseline data (expect: no straggler flagged)
2. `gpu-fleet straggler` on the straggler data (expect: GPU 0 flagged)
3. Record: detection success/failure, root cause identification accuracy

## 5. Analysis Plan

1. **H6:** Paired Wilcoxon signed-rank comparing effective_throughput (baseline vs straggler). Report median difference, p-value, rank-biserial r.
2. **H6b:** Descriptive comparison of nvidia-smi utilization on GPUs 1-7 during baseline vs straggler. If utilization appears normal despite throughput loss, nvidia-smi fails to capture the straggler's fleet impact.
3. **Effect size:** Straggler tax percentage (effective fleet waste).
4. **Detection metrics:** gpu-fleet sensitivity (P(detected | straggler present)), specificity (P(not detected | no straggler)).

## 6. Estimated Time

| Step | Time |
|------|------|
| Environment verification | 2 min |
| Build gpu-roofline + gpu-fleet from source | 2 min |
| Baseline (10 reps x ~30s each) | 5 min |
| Straggler (10 reps x ~30s each) | 5 min |
| Straggler detection test | 2 min |
| **Total** | **~16 min** |

## 7. Deviations

Any protocol deviations during execution will be documented here after data collection.

(None recorded — protocol not yet executed.)
