# Ghost Allocations After MIG Partition Teardown: A Study Protocol for Detecting Unreported GPU Memory Leaks

**Protocol Version:** 1.0
**Date:** 2026-03-19
**Status:** Pre-registration Draft

---

## 1. Abstract

We propose a controlled study to measure whether NVIDIA Multi-Instance GPU (MIG) partition teardown reliably reclaims all allocated GPU memory, and whether existing monitoring tools (nvidia-smi, NVIDIA DCGM) detect residual "ghost allocations" — memory that remains consumed after the partition reports as destroyed. We introduce a detection mechanism implemented in gpu-roofline's `TeardownVerifier` that captures pre/post teardown memory deltas via direct NVML queries, and demonstrate its detection capabilities first through large-scale simulation (N=1000+ cycles), then through a hardware validation phase on bare-metal H100 systems.

## 2. Background and Motivation

### 2.1 The Problem

Cloud GPU providers and enterprise data centers partition NVIDIA H100/H200 GPUs using MIG to serve multiple tenants. Each MIG partition receives a hardware-isolated slice of GPU memory and compute. When a tenant's workload completes and the partition is destroyed, the expectation is that all resources return to the available pool.

However, anecdotal evidence from production GPU clusters suggests that teardown under specific conditions (GPU under load, rapid sequential create/destroy cycles, concurrent MIG reconfigurations) can leave memory in a state where:

1. The MIG instance reports as destroyed
2. `nvidia-smi` shows the instance no longer exists
3. DCGM metrics no longer reference the instance
4. **Physical GPU memory consumption does not return to the pre-allocation baseline**

This "ghost allocation" represents real resource waste: memory that is neither available to new partitions nor tracked by any monitoring tool.

### 2.2 Why Existing Tools Miss It

| Tool | What It Reports After Teardown | What It Misses |
|------|-------------------------------|----------------|
| `nvidia-smi` | MIG instance no longer listed; per-instance memory counters removed | Does not compare aggregate physical memory before/after; no delta tracking |
| DCGM (`dcgmi dmon`) | Instance metrics cease; no error reported | No teardown verification metric; no ghost allocation counter |
| `nvml` (raw API) | `nvmlDeviceGetMemoryInfo` returns current used/free | Requires caller to implement pre/post delta logic; not built into any standard tool |

The gap is structural: existing tools report *current state*, not *state transitions*. Detecting ghost allocations requires capturing a snapshot before teardown, waiting for teardown completion, then computing the delta. No shipping tool does this.

### 2.3 gpu-roofline's TeardownVerifier

Our `TeardownVerifier` (implemented in `gpu-harness::vgpu::teardown`) captures:
- Pre-teardown: total physical memory used (bytes), VRAM allocated to the target partition
- Post-teardown: total physical memory used (bytes)
- Delta: `expected_free = vram_allocated`, `actual_free = pre_used - post_used`
- Ghost allocation size: `expected_free - actual_free` (when positive)
- Reclaim latency: wall-clock time from teardown signal to memory delta stabilization

## 3. Hypotheses

### H1 (Primary)
MIG partition teardown under stress conditions produces statistically significant ghost allocations (memory not reclaimed) compared to teardown of idle, freshly-created partitions.

- **H1_0 (Null):** Mean ghost allocation size under stress conditions = mean ghost allocation size under control conditions (fresh, idle partition teardown).
- **H1_A (Alternative):** Mean ghost allocation size under stress conditions > mean ghost allocation size under control conditions.

### H2 (Secondary)
Neither nvidia-smi nor DCGM detect ghost allocations that gpu-roofline's TeardownVerifier identifies.

- **H2_0:** The set of ghost allocations detected by nvidia-smi/DCGM = the set detected by TeardownVerifier.
- **H2_A:** TeardownVerifier detects ghost allocations that nvidia-smi and DCGM do not report.

### H3 (Exploratory)
Ghost allocation incidence and magnitude vary by MIG profile size.

## 4. Study Design

### 4.1 Variables

| Variable | Type | Levels / Measurement |
|----------|------|---------------------|
| Teardown Method | Independent (categorical, 3 levels) | Clean, Under-Load, Rapid-Churn |
| MIG Profile | Independent (categorical, 3 levels) | 1g.10gb, 2g.20gb, 3g.40gb |
| Ghost Allocation Size | Dependent (continuous, bytes) | `expected_free - actual_free` |
| Reclaim Latency | Dependent (continuous, ms) | Wall-clock teardown-to-stable |
| Memory Reclaimed | Dependent (binary) | `actual_free >= expected_free` |
| Detection Tool | Independent (categorical, 3 levels) | nvidia-smi, DCGM, gpu-roofline |

### 4.2 Experimental Conditions

**Control (C): Clean Teardown**
1. Create MIG partition (cold GPU, no other partitions active)
2. Wait 5 seconds (no workload)
3. Destroy partition
4. Measure memory delta

**Treatment T1: Under-Load Teardown**
1. Create MIG partition
2. Launch sustained compute workload (matrix multiply filling 80% of partition memory)
3. While workload is running, destroy partition
4. Measure memory delta

**Treatment T2: Rapid-Churn Teardown**
1. Create partition, run 10-second workload, destroy
2. Immediately create new partition in same slot
3. Repeat 20 times in sequence
4. On the final teardown, measure cumulative memory delta vs. pristine baseline

**Treatment T3: Concurrent-Partition Teardown**
1. Create maximum MIG partitions (7x 1g.10gb on H100)
2. Destroy all 7 in rapid sequence (< 1 second between teardowns)
3. Measure aggregate memory delta

### 4.3 Blocking and Randomization

- **Block on:** GPU temperature bin (cold: <40C, warm: 40-60C, hot: >60C), driver version
- **Randomize:** Order of MIG profile sizes within each teardown-method block
- **Counterbalance:** Each profile size appears in each position equally often across replications

## 5. Power Analysis and Sample Size

### 5.1 Effect Size Estimation

From preliminary simulation data (ghost_allocation scenario, 512 MiB ghost on 8 GiB allocation = 6.25% waste):

- Expected control mean: 0 bytes ghost allocation (clean teardown)
- Expected treatment mean: 32-512 MiB ghost allocation (based on industry reports of 0.4-6.25% memory leak per teardown cycle)
- Expected standard deviation: ~128 MiB (estimated from variance in NVML memory reporting)
- Minimum detectable effect: 64 MiB (0.8% of 1g.10gb partition) -- below this, noise dominates

Cohen's d for minimum detectable effect:
```
d = (64 MiB - 0 MiB) / 128 MiB = 0.50 (medium effect)
```

### 5.2 Sample Size Calculation

For a one-sided independent samples t-test (treatment > control):
- alpha = 0.05
- Power (1 - beta) = 0.80
- Cohen's d = 0.50

```
n per group = ((z_alpha + z_beta) / d)^2
n = ((1.645 + 0.842) / 0.50)^2
n = (2.487 / 0.50)^2
n = 4.974^2
n = 24.7 → 25 per group
```

With 3 teardown methods x 3 MIG profiles = 9 cells, and a control group:
- **Minimum: 25 teardown cycles per cell**
- **Recommended: 50 per cell** (for robustness to non-normality)
- **Total cycles: 50 x 9 treatment cells + 50 x 3 control cells = 600 teardown cycles**

For the Bonferroni-corrected comparison (9 pairwise comparisons across MIG profiles):
- alpha_corrected = 0.05 / 9 = 0.0056
- At d=0.50, n per group = 38 → round to 50

### 5.3 Simulation Phase Sample Size

Simulation has no hardware cost. Run at much larger N for precise effect estimation:
- **N = 10,000 teardown cycles per condition** (90,000 total simulated cycles)
- Purpose: establish detection algorithm sensitivity, specificity, and ROC curve with narrow confidence intervals

## 6. Measurement Protocol

### 6.1 Pre-Teardown Capture

```
1. Record GPU index, MIG profile, instance UUID
2. Query nvmlDeviceGetMemoryInfo(physical_device) → total, free, used
3. Query nvmlDeviceGetMigDeviceHandleByIndex → confirm instance exists
4. Record timestamp T_pre
5. Store: {instance_id, memory_used_pre, vram_allocated, T_pre}
```

### 6.2 Teardown Execution

```
1. Issue: nvidia-smi mig -dci -ci {instance} -gi {gpu_instance}
2. Issue: nvidia-smi mig -dgi -gi {gpu_instance}
3. Poll nvmlDeviceGetMigDeviceHandleByIndex until instance disappears
4. Record timestamp T_destroyed
```

### 6.3 Post-Teardown Measurement (Stabilization Protocol)

Memory reclamation is not instantaneous. The protocol must wait for stabilization:

```
1. Set T_start = now()
2. Repeat every 100ms for up to 10 seconds:
   a. Query nvmlDeviceGetMemoryInfo(physical_device) → used_bytes
   b. Record (elapsed_ms, used_bytes)
   c. If last 5 readings within 1 MiB of each other → stabilized
3. Record final memory_used_post and T_stable
4. Compute:
   - actual_free = memory_used_pre - memory_used_post
   - ghost_bytes = vram_allocated - actual_free (if positive, else 0)
   - reclaim_latency_ms = T_stable - T_destroyed
```

### 6.4 Concurrent Tool Comparison

For each teardown cycle, simultaneously capture:

1. **nvidia-smi:** `nvidia-smi -q -d MEMORY` before and after teardown. Parse "Used" field.
2. **DCGM:** `dcgmi dmon -e 254,251` (DCGM_FI_DEV_FB_USED, DCGM_FI_DEV_FB_FREE). Record before/after.
3. **gpu-roofline:** TeardownVerifier pre/post capture (NVML direct).

Record whether each tool:
- Reports memory decrease matching expected_free (within 1 MiB tolerance)
- Reports any anomaly/warning about incomplete reclamation
- Provides any ghost allocation metric

### 6.5 Data Record Format

Each teardown cycle produces one JSON record:

```json
{
  "cycle_id": "uuid",
  "timestamp_iso": "2026-03-19T15:30:00Z",
  "gpu_index": 0,
  "gpu_model": "H100 80GB HBM3",
  "driver_version": "550.54.15",
  "cuda_version": "12.4",
  "mig_profile": "1g.10gb",
  "teardown_method": "under_load",
  "condition": "treatment_T1",
  "temperature_pre_c": 42,
  "temperature_post_c": 45,
  "power_draw_pre_w": 285.0,
  "power_draw_post_w": 120.0,
  "vram_allocated_bytes": 10737418240,
  "memory_used_pre_bytes": 42949672960,
  "memory_used_post_bytes": 32749125632,
  "memory_used_post_stabilized_bytes": 32749125632,
  "stabilization_readings": [
    {"elapsed_ms": 100, "used_bytes": 33285996544},
    {"elapsed_ms": 200, "used_bytes": 32749125632},
    {"elapsed_ms": 300, "used_bytes": 32749125632}
  ],
  "actual_free_bytes": 10200547328,
  "expected_free_bytes": 10737418240,
  "ghost_allocation_bytes": 536870912,
  "reclaim_latency_ms": 215.4,
  "memory_reclaimed": false,
  "nvidia_smi_detects_anomaly": false,
  "dcgm_detects_anomaly": false,
  "gpu_roofline_detects_anomaly": true,
  "workload_type": "sgemm_80pct_fill",
  "concurrent_partitions_active": 0,
  "block_temperature_bin": "warm",
  "replication_index": 17
}
```

## 7. Statistical Analysis Plan

### 7.1 Descriptive Statistics

For each condition (teardown method x MIG profile):
- Mean, median, standard deviation of ghost_allocation_bytes
- Incidence rate (proportion of cycles with ghost_bytes > 0)
- Distribution visualization (histogram + kernel density estimate)
- Q-Q plot against normal distribution

### 7.2 Normality Assessment

Memory allocation data is expected to be non-normal (zero-inflated with a right tail). Test with:
- Shapiro-Wilk test (per condition, N <= 50)
- Anderson-Darling test
- Visual: Q-Q plots, histograms

If non-normal (expected), use non-parametric tests as primary analysis.

### 7.3 Primary Analysis: Treatment vs. Control

**If approximately normal:**
- Welch's t-test (one-sided: treatment > control)
- Report: t-statistic, degrees of freedom, p-value, 95% CI for difference in means

**If non-normal (expected):**
- Mann-Whitney U test (one-sided) — primary
- Report: U statistic, p-value, Hodges-Lehmann estimator of median difference, 95% CI
- Supplementary: Bootstrap BCa confidence interval for difference in medians (10,000 resamples)

**Effect size (both cases):**
- Cohen's d (parametric) or rank-biserial correlation r (non-parametric)
- Common language effect size (probability that a random treatment observation exceeds a random control observation)

### 7.4 Multiple Comparisons

With 3 teardown methods x 3 MIG profiles, we have 9 treatment cells vs. 3 control cells.

**Correction strategy:**
1. First test omnibus: Kruskal-Wallis H test across all conditions
2. If significant (p < 0.05): proceed to pairwise comparisons
3. Pairwise: Mann-Whitney U with Holm-Bonferroni correction
4. Report both corrected and uncorrected p-values

For the MIG profile comparison (H3):
- Kruskal-Wallis across profiles within each teardown method
- Post-hoc: Dunn's test with Bonferroni correction

### 7.5 Detection Tool Comparison (H2)

For each teardown cycle where gpu-roofline detects a ghost allocation (ghost_bytes > threshold):
- Record whether nvidia-smi also detected it (binary)
- Record whether DCGM also detected it (binary)

Analysis:
- McNemar's test (paired comparison of detection rates): gpu-roofline vs nvidia-smi, gpu-roofline vs DCGM
- Sensitivity/specificity of each tool (ground truth = NVML delta measurement)
- Cohen's kappa for inter-tool agreement

### 7.6 Detection Algorithm Performance (Simulation Phase)

Using simulated data with known ground truth:
- **Sensitivity (True Positive Rate):** P(TeardownVerifier detects | ghost exists)
- **Specificity (True Negative Rate):** P(TeardownVerifier does not detect | no ghost)
- **ROC Curve:** Vary the detection threshold from 0 to max_ghost_bytes, plot TPR vs FPR
- **AUC:** Area under the ROC curve (target: > 0.95)
- **Precision-Recall Curve:** More informative when ghost incidence is low
- **F1 Score:** At the operating threshold (ghost_bytes > 0)
- **False Discovery Rate:** Critical for operational use (alerting on non-existent ghosts)

### 7.7 Regression Analysis (Exploratory)

Logistic regression for ghost allocation incidence:
```
logit(P(ghost > 0)) = beta_0 + beta_1*teardown_method + beta_2*mig_profile
                       + beta_3*temperature + beta_4*concurrent_partitions
                       + beta_5*cumulative_churn_count
```

Linear regression for ghost allocation magnitude (on cycles where ghost > 0):
```
log(ghost_bytes) = beta_0 + beta_1*teardown_method + beta_2*mig_profile
                   + beta_3*temperature + beta_4*reclaim_latency_ms
```

Report: coefficients, 95% CIs, model R-squared, residual diagnostics.

## 8. Simulation Phase (Executable Immediately)

### 8.1 Simulation Architecture

The simulation uses `gpu-harness::vgpu::sim::SimulatedDetector` which replays `VgpuSimScenario` events through the full `VgpuSampler` → `VgpuAlertEngine` pipeline. Ghost allocation sizes are injected via `SimAction::DestroyVgpu { ghost_bytes }`.

### 8.2 Simulation Experimental Design

Generate scenarios programmatically:

**Factor 1: Ghost Allocation Size (Ground Truth)**
- 0 bytes (clean, 50% of cycles — for specificity measurement)
- Uniform(1 MiB, 1024 MiB) (50% of cycles — for sensitivity measurement)

**Factor 2: MIG Profile**
- 1g.10gb (vram_allocated = 10 GiB)
- 2g.20gb (vram_allocated = 20 GiB)
- 3g.40gb (vram_allocated = 40 GiB)

**Factor 3: Teardown Method Simulation**
- Clean: single create → destroy, ghost_bytes from Factor 1
- Under-load: create → active → sample → teardown-started → destroy
- Rapid-churn: 20 sequential create/destroy on same slot

**Total simulated cycles:**
- 2 (ghost/clean) x 3 (profiles) x 3 (methods) x ~1,111 = **~20,000 cycles**
- Round to 10,000 per condition for power: **90,000 total simulated cycles**

### 8.3 Simulation Procedure

```
For each condition (method x profile x ghost_present):
  For trial = 1 to N:
    1. Build VgpuSimScenario with parameterized ghost_bytes
    2. Instantiate SimulatedDetector + VgpuSampler
    3. Run scenario through event loop
    4. Collect all VgpuAlerts
    5. Record:
       - ghost_bytes_injected (ground truth)
       - ghost_bytes_detected (from TeardownVerification in Destroyed event)
       - alert_fired (boolean: did GhostAllocation alert trigger?)
       - false_positive (alert fired but ghost_bytes_injected == 0)
       - false_negative (alert did not fire but ghost_bytes_injected > 0)
```

### 8.4 Simulation Outputs

1. **Confusion Matrix** (per threshold):

| | Ghost Injected | No Ghost Injected |
|---|---|---|
| **Alert Fired** | True Positive | False Positive |
| **No Alert** | False Negative | True Negative |

2. **ROC Curve:** Sweep detection threshold from 0 to 1 GiB in 1 MiB steps
3. **Detection Latency Distribution:** Time from teardown event to alert
4. **Ghost Size Distribution:** Histogram of detected sizes vs. injected sizes
5. **Accumulation Analysis:** In rapid-churn, does ghost accumulate linearly or plateau?

### 8.5 Noise Injection for Realism

To prevent the simulation from having trivially perfect detection:
- Add Gaussian noise to memory readings: N(0, sigma) where sigma = 512 KiB (representing NVML measurement jitter)
- Add timing jitter to teardown latency: Exponential(lambda = 1/50ms)
- Occasionally inject memory spikes from other processes: Poisson(lambda = 0.01) x Uniform(1 MiB, 64 MiB)

This creates a detection challenge: the algorithm must distinguish real ghost allocations from measurement noise.

## 9. Hardware Phase

### 9.1 Hardware Requirements

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA H100 80GB HBM3 (SXM5 preferred) |
| Count | Minimum 1 (3 recommended for inter-GPU replication) |
| CPU | Any x86_64, 16+ cores |
| RAM | 128 GB+ |
| OS | Ubuntu 22.04 LTS |
| Driver | NVIDIA 550.x or later (record exact version) |
| CUDA | 12.4+ |
| MIG Support | Enabled (`nvidia-smi -mig 1`) |
| Cooling | Sufficient to maintain GPU < 70C under sustained load |

### 9.2 Environment Preparation

```bash
# Lock GPU clocks to prevent frequency scaling confounds
nvidia-smi -lgc 1410,1410    # Lock to base clock
nvidia-smi -pm 1              # Persistence mode
nvidia-smi -mig 1             # Enable MIG

# Verify clean state
nvidia-smi mig -lgip          # List available profiles
nvidia-smi -q -d MEMORY       # Record baseline memory

# Disable ECC page retirement notifications (they affect memory reporting)
# Record current ECC status as a covariate

# Kill all GPU processes
fuser -v /dev/nvidia* 2>/dev/null && fuser -k /dev/nvidia*
```

### 9.3 Per-Cycle Protocol

```
CYCLE(profile, method, cycle_index):
  # 1. Ensure clean state
  nvidia-smi mig -dci && nvidia-smi mig -dgi  # Destroy all existing
  sleep 2
  BASELINE_MEM = nvmlDeviceGetMemoryInfo().used

  # 2. Create partition
  nvidia-smi mig -cgi {profile} -C
  sleep 1
  PRE_MEM = nvmlDeviceGetMemoryInfo().used
  ALLOCATED = PRE_MEM - BASELINE_MEM

  # 3. Execute workload (if method != clean)
  if method == "under_load":
    launch_sgemm(partition, fill_pct=0.80) &
    WORKLOAD_PID = $!
    sleep 10  # Let workload stabilize
  elif method == "rapid_churn":
    for i in 1..20:
      nvidia-smi mig -dci && nvidia-smi mig -dgi
      nvidia-smi mig -cgi {profile} -C
      launch_sgemm(partition, fill_pct=0.80)
      sleep 2
      kill_workload()
    # Final iteration measured below

  # 4. Capture pre-teardown state
  T_pre = now()
  MEM_PRE_TEARDOWN = nvmlDeviceGetMemoryInfo().used
  TEMP_PRE = nvmlDeviceGetTemperature()

  # 5. Teardown
  kill_workload() if running
  nvidia-smi mig -dci -ci ... -gi ...
  nvidia-smi mig -dgi -gi ...
  T_teardown = now()

  # 6. Stabilization measurement (poll every 100ms, 10s max)
  readings = []
  for t in 0..100:
    sleep 0.1
    mem = nvmlDeviceGetMemoryInfo().used
    readings.append((t*100, mem))
    if last_5_stable(readings, tolerance=1MiB):
      break
  MEM_POST = readings[-1].mem
  T_stable = now()

  # 7. Compute ghost
  ACTUAL_FREE = MEM_PRE_TEARDOWN - MEM_POST
  GHOST = max(0, ALLOCATED - ACTUAL_FREE)
  LATENCY = T_stable - T_teardown

  # 8. Concurrent tool readings
  NVIDIA_SMI_POST = parse(nvidia-smi -q -d MEMORY)
  DCGM_POST = parse(dcgmi dmon -e 254,251 -c 1)

  # 9. Record JSON
  emit_record(...)
```

### 9.4 Cool-Down Protocol

Between conditions (not between cycles within a condition):
- Wait for GPU temperature to drop below 40C
- Verify memory returns to baseline (if not, restart nvidia driver: `nvidia-smi -r`)
- If driver restart needed, record it as a covariate

### 9.5 Hardware Phase Timeline

| Day | Activity | Cycles |
|-----|----------|--------|
| 1 | Environment setup, baseline calibration, pilot run (10 cycles per condition) | 90 |
| 2 | Control condition (clean teardown, all 3 profiles, 50 each) | 150 |
| 3 | T1: Under-load teardown, all profiles | 150 |
| 4 | T2: Rapid-churn teardown, all profiles | 150 |
| 5 | T3: Concurrent-partition teardown, 1g.10gb only | 50 |
| 6 | Replication on second GPU (if available) | 150 |
| 7 | Analysis, anomaly investigation, re-runs for failed cycles | Variable |

## 10. Expected Outputs

### 10.1 Paper-Ready Figures

**Figure 1: Ghost Allocation Distribution by Condition**
- 3x3 grid of violin plots (teardown method x MIG profile)
- Y-axis: ghost allocation size (MiB), log scale
- Overlay: individual data points (jittered)
- Annotation: median, IQR, p-value vs. control

**Figure 2: ROC Curve — Detection Algorithm Performance (Simulation)**
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Curves for: TeardownVerifier at various thresholds
- Annotation: AUC, optimal operating point

**Figure 3: Tool Comparison — Detection Rate**
- Grouped bar chart
- X-axis: ghost allocation size bins (0, 1-64 MiB, 64-256 MiB, 256+ MiB)
- Y-axis: detection rate (%)
- Bars: nvidia-smi, DCGM, gpu-roofline
- Expected: nvidia-smi and DCGM at 0% detection; gpu-roofline near 100%

**Figure 4: Cumulative Ghost Allocation Under Rapid Churn**
- X-axis: teardown cycle number (1-20)
- Y-axis: cumulative unreturned memory (MiB)
- Lines: per-trial trajectories (light), mean trajectory (bold)
- Annotation: linear regression slope with 95% CI

**Figure 5: Reclaim Latency Distribution**
- Histogram + KDE for each teardown method
- X-axis: milliseconds
- Vertical line at operational threshold (e.g., 1000ms)

**Figure 6: Cost Impact Projection**
- X-axis: fleet size (100 to 10,000 GPUs)
- Y-axis: annual wasted cost (USD)
- Lines for: 1% ghost rate, 3% ghost rate, 6% ghost rate
- Based on: H100 cloud pricing ($3.50/hr), ghost memory as fraction of total

### 10.2 Tables

**Table 1: Descriptive Statistics by Condition**

| Condition | N | Ghost Incidence (%) | Median Ghost (MiB) | Mean Ghost (MiB) | SD (MiB) | Max (MiB) |
|-----------|---|--------------------|--------------------|-------------------|-----------|-----------|
| Control/1g.10gb | 50 | — | — | — | — | — |
| T1/1g.10gb | 50 | — | — | — | — | — |
| ... | ... | ... | ... | ... | ... | ... |

**Table 2: Statistical Test Results**

| Comparison | Test | Statistic | p-value | p-adj (Holm) | Effect Size | 95% CI |
|------------|------|-----------|---------|--------------|-------------|--------|
| T1 vs C (1g.10gb) | Mann-Whitney U | — | — | — | r = — | [—, —] |
| ... | ... | ... | ... | ... | ... | ... |

**Table 3: Detection Tool Comparison**

| Tool | True Positives | False Negatives | Sensitivity | Specificity | McNemar p vs gpu-roofline |
|------|---------------|-----------------|-------------|-------------|---------------------------|
| nvidia-smi | — | — | — | — | — |
| DCGM | — | — | — | — | — |
| gpu-roofline | — | — | — | — | (reference) |

**Table 4: Cost Projection**

| Ghost Rate | Per-GPU Annual Waste | 1,000-GPU Fleet | 10,000-GPU Fleet |
|------------|---------------------|-----------------|------------------|
| 1% (80 MiB/teardown) | — | — | — |
| 3% (256 MiB/teardown) | — | — | — |
| 6% (512 MiB/teardown) | — | — | — |

### 10.3 Cost Model

```
Annual waste per GPU = ghost_rate * avg_partition_size * teardowns_per_day * 365
                       * (ghost_lifetime / 24hr) * hourly_price

Parameters:
  ghost_rate:          from study results (proportion of teardowns with ghost)
  avg_partition_size:  from MIG profile (10/20/40 GiB)
  teardowns_per_day:   estimate 10-50 for multi-tenant clusters
  ghost_lifetime:      until driver restart or GPU reset (estimate 24hr mean)
  hourly_price:        H100 cloud: ~$3.50/hr; on-demand: ~$32/hr
  memory_fraction:     ghost_bytes / total_gpu_memory

Cost of ghost = (ghost_bytes / total_bytes) * hourly_price * ghost_lifetime_hours

Example: 512 MiB ghost / 80 GiB total = 0.625%
  0.00625 * $3.50/hr * 24hr = $0.53 per ghost event
  At 20 teardowns/day, 5% ghost rate: 1 ghost/day = $0.53/day/GPU
  10,000 GPU fleet: $5,300/day = $1.93M/year
```

## 11. Threats to Validity

### 11.1 Internal Validity

| Threat | Mitigation |
|--------|------------|
| **NVML measurement overhead:** NVML queries themselves may transiently allocate memory | Measure NVML query overhead separately; subtract from readings; include as covariate |
| **Driver version dependency:** Ghost behavior may be driver-specific | Record exact driver version; test on 2+ versions if possible; declare generalizability limits |
| **Temperature confound:** Hot GPU may behave differently during teardown | Block on temperature; enforce cool-down between conditions; include temperature as covariate |
| **Time-of-day effects:** Thermal/power behavior may drift | Randomize condition order; complete each condition within a single temperature block |
| **ECC corrections:** Background ECC operations consume memory transiently | Record ECC status; exclude cycles with ECC events from primary analysis |
| **Kernel driver memory pools:** Driver may retain memory pools for performance | Distinguish driver pools (systematic, constant) from ghost allocations (variable, accumulating) via regression on cycle number |

### 11.2 External Validity

| Threat | Mitigation |
|--------|------------|
| **GPU architecture specificity:** H100 results may not generalize to A100/H200 | Declare scope; plan follow-up on A100 (Ampere) and H200 (Hopper+) |
| **MIG vs GRID:** Ghost behavior in MIG (hardware partitioned) may differ from GRID (time-sliced) | Run supplementary GRID experiments if hardware available; discuss in limitations |
| **Bare-metal vs cloud:** Hypervisor layers in cloud may add/mask ghost allocations | Declare bare-metal scope; discuss cloud implications theoretically |
| **Workload type:** Matrix multiply may not represent all workload patterns | Include 2-3 workload types: sgemm, memcpy-heavy, mixed compute/memory |

### 11.3 Construct Validity

| Threat | Mitigation |
|--------|------------|
| **NVML accuracy:** NVML memory_used may not reflect physical DRAM allocation | Cross-validate with /proc/driver/nvidia/gpus/*/information and /sys/bus/pci memory maps |
| **"Ghost" vs "pool":** What we call ghost may be intentional driver caching | Measure persistence: if "ghost" persists for >60s, it exceeds any reasonable cache policy |
| **Measurement as intervention:** Frequent NVML polling may affect teardown behavior | Test with polling disabled during teardown; compare results |

### 11.4 Statistical Conclusion Validity

| Threat | Mitigation |
|--------|------------|
| **Non-normality:** Memory data likely zero-inflated, right-skewed | Use non-parametric primary analysis; bootstrap CIs |
| **Multiple comparisons:** 9+ tests inflate Type I error | Holm-Bonferroni correction; report both raw and adjusted p-values |
| **Low base rate:** If ghost incidence is < 5%, many cells may have zero events | Use Fisher's exact test for rare events; pool across MIG profiles if needed |
| **Heterogeneous variance:** Treatment conditions may have much higher variance than control | Use Welch's correction; report Levene's test results |

## 12. Ethical and Reproducibility Considerations

### 12.1 Reproducibility

- All code open-source in gpu-tools repository
- Data collection scripts committed alongside analysis code
- Raw JSON data published as supplementary material
- Analysis notebook (Jupyter or Quarto) with exact package versions
- Random seeds recorded for all stochastic components

### 12.2 Pre-Registration

This protocol serves as a pre-registration. Before data collection begins:
1. Commit protocol to repository with a signed tag
2. Deposit timestamped copy on OSF or arXiv
3. Any deviations from protocol documented and justified in the final paper

### 12.3 Responsible Disclosure

If hardware-phase results confirm ghost allocations as a driver bug:
1. Report to NVIDIA via their security/bug reporting channel before public disclosure
2. Allow 90-day remediation window
3. Publish regardless of fix status (this is a measurement methodology paper, not an exploit)

## 13. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Protocol finalization | 1 week | This document, pre-registered |
| Simulation implementation | 1 week | Extended scenario builder, analysis scripts |
| Simulation execution | 1 day | 90,000 simulated cycles, ROC curves |
| Simulation paper draft | 1 week | Sections 1-8 of paper with simulation results |
| Hardware procurement | 2-4 weeks | Bare-metal H100 access (cloud or on-prem) |
| Hardware data collection | 1 week | 600+ teardown cycles, JSON dataset |
| Hardware analysis | 1 week | Complete statistical analysis |
| Paper completion | 2 weeks | Full paper with both phases |
| **Total** | **8-10 weeks** | Arxiv preprint + workshop submission |

## 14. Target Venues

- **MLSys** (Workshop track): Systems for ML, strong fit for GPU resource management
- **USENIX ATC** (Short paper): Systems research, GPU virtualization audience
- **EuroSys** (Workshop): Systems community, strong GPU/cloud infrastructure track
- **HotCloud** (USENIX): Cloud computing workshop, directly relevant
- **arXiv cs.DC / cs.PF**: Immediate open access, establishes priority

## 15. References

1. NVIDIA Multi-Instance GPU User Guide. NVIDIA Corp., 2024.
2. NVIDIA Management Library (NVML) API Reference, v12.
3. NVIDIA Data Center GPU Manager (DCGM) User Guide.
4. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
5. Hollander, M., Wolfe, D.A., & Chicken, E. (2013). Nonparametric Statistical Methods.
6. [gpu-roofline TeardownVerifier source](../../crates/gpu-harness/src/vgpu/teardown.rs)

---

## Appendix A: Simulation Code Reference

The simulation infrastructure is already implemented in the gpu-tools codebase:

| Component | Path | Purpose |
|-----------|------|---------|
| `TeardownVerifier` | `gpu-harness/src/vgpu/teardown.rs` | Pre/post memory delta computation |
| `TeardownVerification` | `gpu-harness/src/vgpu/state.rs` | Data structure for verification results |
| `SimulatedDetector` | `gpu-harness/src/vgpu/sim.rs` | Replays scenario events through pipeline |
| `VgpuSimScenario` | `gpu-harness/src/vgpu/sim.rs` | Scenario definition (events, profiles, timing) |
| `VgpuSampler` | `gpu-roofline/src/monitor/vgpu_sampler.rs` | Event loop + state management |
| `VgpuAlertEngine` | `gpu-roofline/src/monitor/vgpu_alerting.rs` | Ghost allocation alert rule |
| Stress tests | `gpu-roofline/tests/vgpu_stress.rs` | Churn + density scenarios |

The existing `ghost_allocation` scenario (512 MiB ghost on 8 GiB GRID vGPU) and `rapid_churn` scenario (20 cycles, 0-byte ghost) serve as templates. The simulation phase requires extending these with:

1. Parameterized ghost_bytes (currently hardcoded per scenario)
2. Noise injection layer (Gaussian measurement jitter)
3. Large-N batch runner (currently single-scenario execution)
4. CSV/JSON bulk output for statistical analysis

## Appendix B: Power Analysis Sensitivity

If the true effect is smaller than assumed:

| Cohen's d | N per group (power=0.80) | N per group (power=0.90) |
|-----------|--------------------------|--------------------------|
| 0.20 (small) | 155 | 208 |
| 0.35 | 52 | 69 |
| 0.50 (medium) | 25 | 34 |
| 0.65 | 16 | 20 |
| 0.80 (large) | 10 | 13 |

For the Bonferroni-adjusted alpha (0.0056):

| Cohen's d | N per group (power=0.80) | N per group (power=0.90) |
|-----------|--------------------------|--------------------------|
| 0.50 | 38 | 49 |
| 0.80 | 17 | 21 |

Our recommendation of N=50 per cell provides power > 0.90 for medium effects and power > 0.80 for effects as small as d=0.35, even after multiple comparison correction.
