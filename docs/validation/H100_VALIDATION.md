# NVIDIA H100 80GB HBM3 — Full Validation Report

**Date:** 2026-03-18
**Platform:** RunPod (H100 SXM, 80GB HBM3)
**Driver:** 580.126.09 | CUDA 12.4 | sm_90
**Build:** gpu-roofline 0.1.0 --features cuda

## Hardware Verification

nvidia-smi confirmed peak operation during all tests:
- **Max Clock:** 1980 MHz (100% of boost)
- **Memory Clock:** 2619 MHz (sustained)
- **Power:** 130-140W (burst workload)
- **Temperature:** 31-33°C (datacenter cooling)
- **No throttling detected**

## Test Results

### 1. Multi-Precision Burst Roofline

| Precision | Measured | Theoretical | % of Spec | CV |
|-----------|----------|-------------|-----------|-----|
| FP16 Tensor Core | **494.7 TFLOPS** | 989 TFLOPS | 50% | 0.3% |
| BF16 Tensor Core | **494.6 TFLOPS** | 989 TFLOPS | 50% | 0.3% |
| FP32 CUDA Core | **59.1 TFLOPS** | 67 TFLOPS | 88% | 0.1% |
| HBM3 Bandwidth | **2,905 GB/s** | 3,350 GB/s | 87% | 0.4% |

### 2. Bandwidth Kernels (all 7)

| Kernel | BW (GB/s) | Efficiency | CV |
|--------|-----------|------------|-----|
| copy | 2,905 | 100% | 0.4% |
| scale | 2,905 | 100% | 0.5% |
| add | 2,005 | 52% | 0.3% |
| triad | 2,004 | 65% | 0.3% |
| fma_light | 2,906 | 100% | 0.8% |
| fma_medium | 2,910 | 100% | 0.4% |
| fma_heavy | 923 | 100% (compute) | 0.1% |

### 3. Validation Engine (4/4 PASS)

| Check | Measured | Expected | Status |
|-------|----------|----------|--------|
| Bandwidth | 2,905 GB/s | 2,160-3,100 | PASS |
| FP32 Compute | 59.1 TFLOPS | 44-65 | PASS |
| Stability | CV 1.2% | < 5% | PASS |
| Roofline Shape | 6/6 | All correct | PASS |

### 4. Dynamic Roofline (120s sustained)

- Burst: 59,115 GFLOP/s | 2,911 GB/s
- Sustained: 59,154 GFLOP/s | 2,914 GB/s
- **Net drop: -0.1%** (slight improvement as GPU warms — boost stable)
- Equilibrium reached at 10s
- Clock held at 1980 MHz throughout

### 5. Continuous Monitor (60s, 6 samples)

All samples status: **normal**
- BW range: 2,892 - 2,910 GB/s (0.6% variance)
- GFLOPS range: 58,679 - 59,107 (0.7% variance)
- All 6/6 samples: stable

## H100 vs H200 Comparison

| Metric | H100 | H200 | H200/H100 |
|--------|------|------|-----------|
| HBM BW | 2,905 GB/s | 4,028 GB/s | **1.39x** |
| FP32 CUDA | 59.1 TFLOPS | 59.5 TFLOPS | 1.01x |
| FP16 Tensor | 494.7 TFLOPS | 684.1 TFLOPS | **1.38x** |
| BF16 Tensor | 494.6 TFLOPS | 685.3 TFLOPS | **1.39x** |
| Sustained drop | -0.1% | 0.1% | Both negligible |

The H200's HBM3e delivers 39% more bandwidth than H100's HBM3.
Tensor Core throughput also improves 38% (likely due to reduced memory stalls).

## Files

- `h100/burst.json` — Full burst roofline data (7 kernels)
- `h100/dynamic.json` — 120s sustained roofline with tension analysis
- `h100/validate_json.json` — Validation engine results
- `h100/monitor.jsonl` — Monitor daemon JSON log (6 samples)
- `h100/tensor_results.txt` — FP16/BF16/FP32 comparison
