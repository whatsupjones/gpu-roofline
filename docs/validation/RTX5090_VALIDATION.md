# NVIDIA GeForce RTX 5090 32GB GDDR7 — Full Validation Report

**Date:** 2026-03-18
**Platform:** JarvisLabs (RTX 5090, 32GB GDDR7)
**Driver:** 570.195.03 | CUDA 12.x | sm_120 (Blackwell consumer)
**Build:** gpu-roofline 0.1.0 --features cuda (includes NVML telemetry)

## Hardware Verification

NVML telemetry confirmed via gpu-roofline and nvidia-smi dmon:
- **Boost Clock:** 2415-2925 MHz
- **Memory Clock:** 13801 MHz (GDDR7)
- **Power:** 58-148W (burst workload — well below 575W TDP, cloud instance may be power-limited)
- **Temperature:** 23-27°C (cloud cooling)
- **No throttling detected**

## Test Results

### 1. Multi-Precision Burst Roofline

| Precision | Measured | Theoretical | % of Spec | CV |
|-----------|----------|-------------|-----------|-----|
| FP16 Tensor Core | **247.2 TFLOPS** | 380 TFLOPS | 65% | 0.6% |
| BF16 Tensor Core | **246.9 TFLOPS** | 380 TFLOPS | 65% | 0.4% |
| FP32 CUDA Core | **95.8 TFLOPS** | 96 TFLOPS | 100% | 0.5% |
| GDDR7 Bandwidth | **1,496-1,515 GB/s** | 1,792 GB/s | 84% | 0.4% |

Tensor cores compiled natively for sm_120 (Blackwell). The 65% tensor utilization reflects WMMA API throughput — Blackwell's native MMA instructions may yield higher results.

### 2. Bandwidth Kernels (all 7)

| Kernel | BW (GB/s) | Efficiency | CV |
|--------|-----------|------------|-----|
| copy | 1,496 | 100% | 0.4% |
| add | 1,033 | 52% ⚠ | 0.3% |
| scale | 1,503 | 100% | 0.4% |
| triad | 1,034 | 64% ⚠ | 0.4% |
| fma_light | 1,495 | 99% | 0.7% |
| fma_medium | 1,495 | 99% | 0.6% |
| fma_heavy | 1,497 | 100% (compute) | 0.5% |

### 3. Validation Engine (4/4 PASS)

| Check | Measured | Expected | Status |
|-------|----------|----------|--------|
| Bandwidth | 1,515 GB/s | 960-1,800 | PASS |
| FP32 Compute | 96.9 TFLOPS | 58-110 | PASS |
| Stability | CV 1.1% | < 5% | PASS |
| Roofline Shape | 6/6 | All correct | PASS |

### 4. Dynamic Roofline (120s sustained)

- Burst: 95,819 GFLOP/s | 1,492 GB/s | 2475 MHz | 25°C | 84W
- Sustained: 95,503 GFLOP/s | 1,497 GB/s | 2415 MHz | 25°C | 67W
- **Net drop: 0.3%** (no throttling — cloud cooling keeps RTX 5090 well within thermal envelope)
- Equilibrium reached at 10s

### 5. Continuous Monitor (60s, 6 samples)

All samples status: **normal**
- BW range: 1,500 - 1,506 GB/s (0.4% variance)
- GFLOPS range: 95,621 - 96,282 (0.7% variance)
- NVML telemetry: 25°C, 2392-2407 MHz, 58-64W throughout

### 6. Validate JSON

Full JSON export: 4/4 checks passed, no diagnostic alerts.

## Simulation Calibration

RTX 5090 simulation kernel_efficiency calibrated to 0.84 (1,503 / 1,792).

## Notes

- The RTX 5090 reports as sm_120 (compute capability 12.0), Blackwell consumer architecture
- Tensor core results (247T FP16/BF16) at 65% of spec (380T) — this is the WMMA API ceiling, not a compilation issue (sm_120 native compilation confirmed, same throughput as sm_90 compat mode)
- FP32 CUDA core performance hits 100% of theoretical (95.8T vs 96T spec)
- GDDR7 bandwidth at 84% of theoretical is consistent with achievable kernel throughput vs DMA ceiling
- No thermal throttling observed in any test — burst-to-sustained gap is 0.3%
- NVML telemetry fully operational after fixing cuda feature to include nvml dependency

## Files

- `rtx5090/burst.json` — Full burst roofline data (7 kernels)
- `rtx5090/dynamic.json` — 120s sustained roofline with tension analysis
- `rtx5090/validate_json.json` — Validation engine results
- `rtx5090/monitor.jsonl` — Monitor daemon JSON log (6 samples)
- `rtx5090/tensor_results.txt` — FP16/BF16/FP32 comparison
- `rtx5090/burst_smi.txt` — nvidia-smi dmon during burst
- `rtx5090/dynamic_smi.txt` — nvidia-smi dmon during dynamic
