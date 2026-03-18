# NVIDIA GeForce RTX 5090 32GB GDDR7 — Full Validation Report

**Date:** 2026-03-18
**Platform:** JarvisLabs (RTX 5090, 32GB GDDR7)
**Driver:** 570.195.03 | CUDA 12.x | sm_120 (Blackwell consumer)
**Build:** gpu-roofline 0.1.0 --features cuda

## Hardware Verification

nvidia-smi dmon captured during burst workload:
- **Max Clock:** 2415 MHz (at or near boost)
- **Memory Clock:** 13801 MHz (GDDR7)
- **Power:** 68-74W (burst workload — well below 575W TDP, cloud instance may be power-limited)
- **Temperature:** 23-25°C (cloud cooling)
- **No throttling detected**

Note: NVML telemetry (temperature/clock/power) is not reported through the gpu-roofline NVML path for this driver — nvidia-smi dmon was used as the authoritative source.

## Test Results

### 1. Multi-Precision Burst Roofline

| Precision | Measured | Theoretical | % of Spec | CV |
|-----------|----------|-------------|-----------|-----|
| FP16 Tensor Core | **246.8 TFLOPS** | 380 TFLOPS | 65% | 0.5% |
| BF16 Tensor Core | **247.5 TFLOPS** | 380 TFLOPS | 65% | 0.9% |
| FP32 CUDA Core | **95.9 TFLOPS** | 96 TFLOPS | 100% | 0.5% |
| GDDR7 Bandwidth | **1,497-1,517 GB/s** | 1,792 GB/s | 84% | 0.6% |

Tensor cores compiled with `--arch=sm_90` (Hopper). The RTX 5090 is sm_120 (Blackwell consumer). Full sm_120 tensor performance likely requires native arch compilation.

### 2. Bandwidth Kernels (all 7)

| Kernel | BW (GB/s) | Efficiency | CV |
|--------|-----------|------------|-----|
| copy | 1,497 | 99% | 0.6% |
| add | 1,034 | 52% ⚠ | 16.5% |
| scale | 1,505 | 100% | 1.0% |
| triad | 1,037 | 64% ⚠ | 0.3% |
| fma_light | 1,501 | 100% | 0.7% |
| fma_medium | 1,503 | 100% | 0.4% |
| fma_heavy | 1,498 | 100% (compute) | 0.5% |

### 3. Validation Engine (4/4 PASS)

| Check | Measured | Expected | Status |
|-------|----------|----------|--------|
| Bandwidth | 1,517 GB/s | 1,200-1,800 | PASS |
| FP32 Compute | 96.7 TFLOPS | 72-110 | PASS |
| Stability | CV 0.4% | < 5% | PASS |
| Roofline Shape | 6/6 | All correct | PASS |

### 4. Dynamic Roofline (120s sustained)

- Burst: 95,367 GFLOP/s | 1,495 GB/s
- Sustained: 96,275 GFLOP/s | 1,507 GB/s
- **Net drop: -1.0%** (no throttling — cloud cooling keeps RTX 5090 well within thermal envelope)
- Equilibrium reached at 10s

### 5. Continuous Monitor (60s, 6 samples)

All samples status: **normal**
- BW range: 1,497 - 1,503 GB/s (0.4% variance)
- GFLOPS range: 95,525 - 96,351 (0.9% variance)
- NVML telemetry: not reported via library (nvidia-smi confirms 23-25°C, 68-74W, 2415 MHz)

## Simulation Calibration

RTX 5090 simulation kernel_efficiency calibrated to 0.84 (1,505 / 1,792).

## Notes

- The RTX 5090 reports as sm_120 (compute capability 12.0), which is Blackwell consumer architecture
- Tensor core results (247T FP16/BF16) are below spec (380T) because kernels are compiled for sm_90 — native sm_120 compilation may improve this significantly
- FP32 CUDA core performance hits 100% of theoretical (95.9T vs 96T spec)
- GDDR7 bandwidth at 84% of theoretical is consistent with achievable kernel throughput vs DMA ceiling
- No thermal throttling observed in any test — burst-to-sustained gap is effectively zero in cloud cooling

## Files

- `rtx5090/burst.json` — Full burst roofline data (7 kernels)
- `rtx5090/dynamic.json` — 120s sustained roofline with tension analysis
- `rtx5090/validate_json.json` — Validation engine results
- `rtx5090/monitor.jsonl` — Monitor daemon JSON log (6 samples)
- `rtx5090/tensor_results.txt` — FP16/BF16/FP32 comparison
- `rtx5090/burst_smi.txt` — nvidia-smi dmon during burst
- `rtx5090/dynamic_smi.txt` — nvidia-smi dmon during dynamic
