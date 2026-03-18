# NVIDIA H200 SXM 141GB — Full Validation Report

**Date:** 2026-03-18
**Platform:** RunPod (H200 SXM, 141GB HBM3e)
**Driver:** 570.124.06 | CUDA 12.4 | sm_90
**Build:** gpu-roofline 0.1.0 --features cuda

## Hardware Verification

nvidia-smi confirmed peak operation during all tests:
- **Max Clock:** 1980 MHz (100% of boost)
- **Memory Clock:** 3201 MHz (sustained)
- **Power:** 114-137W (burst workload, well within 700W TDP)
- **Temperature:** 25-30°C (datacenter cooling)
- **No throttling detected**

## Test Results

### 1. Multi-Precision Burst Roofline

| Precision | Measured | Theoretical | % of Spec | CV |
|-----------|----------|-------------|-----------|-----|
| FP16 Tensor Core | **684.1 TFLOPS** | 989 TFLOPS | 69% | 0.5% |
| BF16 Tensor Core | **685.3 TFLOPS** | 989 TFLOPS | 69% | 0.5% |
| FP32 CUDA Core | **59.5 TFLOPS** | 67 TFLOPS | 89% | 0.1% |
| HBM3e Bandwidth | **4,028 GB/s** | 4,800 GB/s | 84% | 0.8% |

### 2. Bandwidth Kernels (all 7)

| Kernel | BW (GB/s) | Efficiency | CV |
|--------|-----------|------------|-----|
| copy | 4,011 | 100% | 0.8% |
| scale | 4,023 | 100% | 0.8% |
| add | 2,788 | 52% | 0.7% |
| triad | 2,788 | 65% | 0.5% |
| fma_light | 4,025 | 100% | 0.9% |
| fma_medium | 3,753 | 93% | 0.7% |
| fma_heavy | 929 | 100% (compute) | 0.1% |

### 3. Validation Engine (4/4 PASS)

| Check | Measured | Expected | Status |
|-------|----------|----------|--------|
| Bandwidth | 4,008 GB/s | 3,040-4,500 | PASS |
| FP32 Compute | 59.5 TFLOPS | 44-65 | PASS |
| Stability | CV 4.6% | < 5% | PASS |
| Roofline Shape | 6/6 | All correct | PASS |

### 4. Dynamic Roofline (120s sustained)

- Burst: 59,479 GFLOP/s | 4,012 GB/s
- Sustained: 59,448 GFLOP/s | 4,024 GB/s
- **Net drop: 0.1%** (datacenter cooling keeps H200 at peak)
- Equilibrium reached at 10s

### 5. Continuous Monitor (60s, 6 samples)

All samples status: **normal**
- BW range: 3,975 - 4,014 GB/s (0.9% variance)
- GFLOPS range: 59,250 - 59,480 (0.4% variance)
- NVML telemetry: 25-27°C, 114-119W, 1980 MHz throughout

## Simulation Calibration

H200 simulation kernel_efficiency calibrated to 0.84 (4,028 / 4,800).
H100 simulation kernel_efficiency remains 0.86 (2,894 / 3,350).

## Files

- `h200/burst.json` — Full burst roofline data (7 kernels)
- `h200/dynamic.json` — 120s sustained roofline with tension analysis
- `h200/validate_json.json` — Validation engine results
- `h200/monitor.jsonl` — Monitor daemon JSON log (6 samples)
- `h200/tensor_results.txt` — FP16/BF16/FP32 comparison
