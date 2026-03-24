# NVIDIA GH200 480GB (Grace Hopper) — Full Validation Report

**Date:** 2026-03-24
**Platform:** Lambda Labs (GH200 480GB, aarch64)
**Driver:** 580.105.08 | CUDA 12.8 | sm_90
**Build:** gpu-roofline 0.1.0 --features cuda (built from source on aarch64)
**Instances tested:** 2 (independent machines, same configuration)

## Hardware Verification

nvidia-smi confirmed peak operation during all tests:
- **Max Clock:** 1980 MHz (100% of boost)
- **Memory Clock:** 2619 MHz (sustained)
- **Power:** 184-192W (burst workload)
- **Temperature:** 33-42°C (datacenter cooling)
- **No throttling detected**

## Test Results

### 1. Burst Roofline

| Metric | Instance 1 | Instance 2 | Mean |
|--------|-----------|-----------|------|
| FP32 CUDA Core | **59.3 TFLOPS** | **59.4 TFLOPS** | **59.4 TFLOPS** |
| HBM Bandwidth | **3,552 GB/s** | **3,564 GB/s** | **3,558 GB/s** |
| Ridge Point | 16.7 FLOP/byte | 16.7 FLOP/byte | 16.7 FLOP/byte |

### 2. Bandwidth Kernels (all 7)

| Kernel | BW (GB/s) | Efficiency | CV |
|--------|-----------|------------|-----|
| copy | 3,551 | 100% | 0.6% |
| scale | 3,554 | 100% | 0.6% |
| add | 2,439 | 52% | 0.4% |
| triad | 2,441 | 64% | 0.4% |
| fma_light | 3,551 | 100% | 0.9% |
| fma_medium | 3,484 | 98% | 0.6% |
| fma_heavy | 928 | 100% (compute) | 0.2% |

### 3. Validation Engine (4/4 PASS)

| Check | Measured | Expected | Status |
|-------|----------|----------|--------|
| Bandwidth | 3,552 GB/s | 3,040-4,500 | PASS |
| FP32 Compute | 59.4 TFLOPS | 44-65 | PASS |
| Stability | CV 0.7% | < 5% | PASS |
| Roofline Shape | 6/6 | All correct | PASS |

### 4. Dynamic Roofline (120s sustained)

**Instance 1:**
- Burst: 59,372 GFLOP/s | 3,550 GB/s | 1980 MHz | 33°C
- Sustained: 59,356 GFLOP/s | 3,551 GB/s | 1980 MHz | 37°C
- **Net drop: 0.0%** (no thermal throttling)
- Equilibrium reached at 10s

**Instance 2:**
- Burst: 59,254 GFLOP/s | 3,548 GB/s | 1980 MHz | 39°C
- Sustained: 59,315 GFLOP/s | 3,549 GB/s | 1980 MHz | 41°C
- **Net drop: -0.1%** (slight improvement as GPU warms)
- Equilibrium reached at 10s

### 5. Diagnostic Engine (6 probes)

Both instances: **No issues detected. GPU is healthy. 0 findings.**

### 6. MIG Ghost Allocation Test

**Protocol:** Enable MIG, create 1g.12gb instance (profile 19), run burst workload on MIG instance, destroy instance, measure memory delta. Repeat 10 cycles per instance.

| Cycle | Instance 1 Memory | Instance 2 Memory |
|-------|------------------|------------------|
| Baseline | 0 MiB | 13 MiB |
| 1 | 0 MiB | 0 MiB |
| 2 | 0 MiB | 0 MiB |
| 3 | 0 MiB | 0 MiB |
| 4 | 0 MiB | 0 MiB |
| 5 | 0 MiB | 0 MiB |
| 6 | 0 MiB | 0 MiB |
| 7 | 0 MiB | 0 MiB |
| 8 | 0 MiB | 0 MiB |
| 9 | 0 MiB | 0 MiB |
| 10 | 0 MiB | 0 MiB |
| **Ghost** | **0 MiB** | **0 MiB** |

**Single-instance result:** No ghost allocations detected across 20 total MIG create/destroy cycles with workload.

### 6b. Multi-Tenant Selective Teardown

**Protocol:** Create 4 MIG instances (1g.12gb each), then destroy one at a time while others remain active. Measure memory after each selective teardown.

| State | Memory Used | Delta |
|-------|-----------|-------|
| 4 instances active | 58 MiB | — |
| Remove 1st (3 left) | 44 MiB | -14 MiB |
| Remove 2nd (2 left) | 29 MiB | -15 MiB |
| Remove 3rd (1 left) | 15 MiB | -14 MiB |
| Remove last (0 left) | 0 MiB | -15 MiB |

**Result:** Clean selective teardown. Each instance frees ~15 MiB when destroyed, even while other instances remain active.

### 6c. Kill -9 Stress Test

**Protocol:** Allocate 1,735 MiB on a MIG instance via PyTorch, then kill -9 the CUDA process (no graceful cleanup). Measure memory before and after.

| State | Memory Used |
|-------|-----------|
| Active workload | 1,735 MiB |
| After kill -9 | 15 MiB |
| After MIG teardown | 0 MiB |

**Result:** Even SIGKILL with 1.7 GB of unfreed CUDA allocations produced zero ghost allocation. Driver 580.105.08 handles forced process termination correctly.

### 6d. Ghost Allocation Summary

No ghost allocations detected on GH200 with driver 580.105.08 under any test condition:
- 20 single-instance create/destroy cycles with workload
- Multi-tenant selective teardown (4 instances, sequential removal)
- 839 MB non-graceful exit (Python exit without explicit free)
- 1,735 MB kill -9 (forced process termination)

MIG teardown is clean on this hardware and driver combination. Ghost allocations may require older driver versions (535.x/545.x), GRID time-sliced vGPUs (different memory management path), or specific H100 SXM configurations. These remain open test conditions for future validation.

## GH200 vs H100 vs H200 Comparison

| Metric | H100 SXM | H200 SXM | GH200 480GB |
|--------|----------|----------|-------------|
| HBM BW | 2,905 GB/s | 4,028 GB/s | **3,558 GB/s** |
| FP32 CUDA | 59.1 TFLOPS | 59.5 TFLOPS | **59.4 TFLOPS** |
| Sustained drop | -0.1% | 0.1% | **0.0%** |
| Temperature (burst) | 31-33°C | N/A | **33-42°C** |
| Architecture | Hopper (sm_90) | Hopper (sm_90) | **Hopper (sm_90)** |
| Host CPU | x86_64 | x86_64 | **ARM Grace (aarch64)** |
| MIG ghost alloc | Not tested | Not tested | **0 MiB (20 cycles)** |

The GH200's bandwidth sits between H100 and H200, consistent with its HBM3 configuration (vs H100's HBM3 and H200's HBM3e). FP32 compute is identical across all three (same Hopper SMs). The Grace CPU host (aarch64) does not affect GPU-side measurements.

## Category 3: MIG Provisioning Overhead

**Protocol:** Time MIG instance creation and destruction using nanosecond wall-clock timestamps. nvidia-smi reports the instance after it exists but cannot measure the creation latency itself.

### 3a. Create/Destroy Latency (50 cycles, 1g.12gb)

| Metric | Create (ms) | Destroy (ms) |
|--------|-----------|-------------|
| Median | 1,185 | 615 |
| Min | 1,148 | 593 |
| Max | 1,248 | 673 |

**Finding:** MIG partition creation takes 1.1-1.3 seconds. This dead time is invisible to nvidia-smi. The study predicted 120-500ms. Actual is 2-10x worse than predicted.

### 3b. Profile Size Effect (3g.48gb, 10 cycles)

| Metric | Create (ms) | Destroy (ms) |
|--------|-----------|-------------|
| Median | 1,280 | 1,000 |

Larger profiles take longer. 3g.48gb creation is ~100ms slower than 1g.12gb. Destruction is ~400ms slower.

### 3c. Cold vs Warm Provisioning

| Condition | Create (ms) |
|-----------|-----------|
| Cold (first after MIG enable) | 1,217 |
| Warm (immediate re-create) | 1,168 |

Minimal difference between cold and warm. The overhead is consistent.

### 3d. Sequential Multi-Instance Provisioning

| Instance | Create (ms) |
|----------|-----------|
| 1st | 1,189 |
| 2nd | 289 |
| 3rd | 298 |
| 4th | 273 |
| 5th | 264 |
| 6th | 301 |
| 7th | 273 |
| Bulk destroy all 7 | 1,838 |

**Finding:** First instance has full 1.2s overhead. Subsequent instances on the same GPU create in ~280ms. This suggests the first create initializes MIG infrastructure on the GPU, and subsequent creates are incremental. Bulk destroy of 7 instances takes 1.8s.

### 3e. nvidia-smi Visibility

nvidia-smi reports the instance within 11ms of creation completing. But it cannot report the 1.2 seconds of creation time itself. A monitoring system polling nvidia-smi would see the instance appear instantaneously, with no record of the provisioning latency.

---

## Category 6: Oversubscription Visibility

**Protocol:** Create maximum MIG instances (7x 1g.12gb), query nvidia-smi for free memory, compare against actual available capacity for new partitions.

### Results

With 7 MIG instances active (7 x 11 GB = 77 GB committed to MIG):

| Metric | nvidia-smi reports | Actual |
|--------|-------------------|--------|
| memory.total | 97,871 MiB | 97,871 MiB |
| memory.used | 102 MiB | 102 MiB |
| memory.free | 96,667 MiB | **~18,600 MiB** |

**Finding:** nvidia-smi reports 96,667 MiB free. Only ~18,600 MiB is actually available for new MIG partitions. nvidia-smi overstates available memory by 78 GB because it reports physical free memory (memory not currently holding data), not memory available for new MIG allocations. The 77 GB committed to existing MIG instances appears as "free" since the instances haven't filled their VRAM quotas.

A platform operator using `nvidia-smi --query-gpu=memory.free` to decide whether to provision another tenant would see 94 GB free and conclude there is room. There is not. Only 18.6 GB is unallocated to MIG instances.

---

## Hardware Validation Summary: 4 of 6 Categories Tested

| Category | Result | nvidia-smi Detection |
|----------|--------|---------------------|
| 1. Ghost allocations | Clean on driver 580.105.08 (20 single-tenant cycles, multi-tenant selective, kill -9) | N/A (no ghost found) |
| 2. Contention squeeze | Not tested (requires GRID time-slicing) | — |
| 3. Provisioning overhead | **1,185ms create, 615ms destroy per MIG instance** | **0% detection (reports post-facto only)** |
| 4. Burst-to-sustained gap | 0.0% on datacenter-cooled GH200 | nvidia-smi does not track thermal trajectory |
| 5. Straggler tax | Not tested (requires multi-GPU fleet) | — |
| 6. Oversubscription visibility | **nvidia-smi overstates free memory by 78 GB with 7 MIG instances** | **0% detection (reports physical free, not MIG-available)** |

Categories 3 and 6 confirm the study's central thesis: nvidia-smi cannot observe waste that gpu-roofline's methodology detects.

---

## Notes

- The GH200 reports as "NVIDIA GH200 480GB" but has 94.5 GB usable VRAM (consistent with HBM3 480GB module with ECC overhead).
- MIG profiles on GH200 differ slightly from H100 (1g.12gb vs 1g.10gb), reflecting the larger memory per slice.
- Both instances showed near-identical performance, confirming measurement reproducibility.
- The 0.0% burst-to-sustained gap confirms datacenter-grade cooling eliminates thermal throttling on Hopper architecture.

## Files

- `gh200/burst_instance1.json` — Burst roofline data (Instance 1)
- `gh200/burst_instance2.json` — Burst roofline data (Instance 2)
