# gpu-roofline

**Cross-vendor GPU roofline model with dynamic tension analysis.**

Measure burst vs sustained performance ceilings across NVIDIA, AMD, and Intel GPUs from a single Rust binary. No multi-GB toolkit required.

> Traditional roofline models lie to you. They measure peak performance in a 1-second burst. Your ML training job runs for hours at the *sustained* ceiling — which is 15-30% lower. gpu-roofline measures both.

## What Makes This Different

| Feature | Nsight Compute | Intel Advisor | AMD Omniperf | **gpu-roofline** |
|---------|:---:|:---:|:---:|:---:|
| Cross-vendor | NVIDIA only | Intel only | AMD MI-series only | **NVIDIA + AMD + Intel** |
| Install size | ~5 GB | ~15 GB | ~3 GB | **<10 MB** |
| Dynamic roofline | No | No | No | **Yes** |
| Live TUI monitoring | No | No | No | **Yes** |
| CI-native (JSON, exit codes) | No | No | No | **Yes** |
| Sustained ceiling measurement | No | No | No | **Yes** |
| Tension analysis | No | No | No | **Yes** |
| Degradation alerting | No | No | No | **Yes** |
| Single binary | No | No | No | **Yes** |

## Dynamic Roofline: The Tension Model

Real GPU performance is shaped by competing forces — not a flat line:

```
              Performance (GFLOP/s)
    82.6T  +═══════════════════════ Peak Burst (t=0, 2520 MHz, 62°C)
            ╲
    71.3T  + ╲═════════════════════ Thermal Equilibrium (t=34s, 2280 MHz, 78°C)
            ╲   ╲
    65.1T  +  ╲   ╲═══════════════ Power-Limited Sustained (t=120s, 2100 MHz, 83°C)
              ╲     ╲
              +───+───+───+───+──→ Arithmetic Intensity (FLOP/byte)
              0.1  1   10  82  1K

    Tension Analysis:
    ├─ Thermal Tension:  82.6T → 71.3T (−13.7%) after 34s
    ├─ Power Tension:    71.3T → 65.1T (−8.7%) after 120s
    └─ Net Ceiling Drop: 82.6T → 65.1T (−21.2%) burst-to-sustained
```

The tool measures three roofline modes:
- **Burst** (t=0) — what benchmarks report
- **Sustained** (t=60s) — what production workloads actually see
- **Degraded** — what multi-tenant environments deliver

## Validated Hardware

Measured on real datacenter and consumer GPUs. Bandwidth measures achievable compute kernel throughput (not hardware DMA ceiling).

| GPU | HBM BW | FP32 | FP16 Tensor | BF16 Tensor | Backend |
|-----|--------|------|-------------|-------------|---------|
| **NVIDIA H200 141GB** | **4,028 GB/s** | **59.5 TFLOPS** | **686 TFLOPS** | **686 TFLOPS** | CUDA |
| **NVIDIA H100 80GB** | **2,893 GB/s** | **59.5 TFLOPS** | *pending* | *pending* | CUDA |
| Intel UHD Graphics | 7 GB/s | 0.15 TFLOPS | — | — | Vulkan |

*More GPUs coming: RTX 4090, RTX 5090, MI300X. [Contribute your results!](https://github.com/whatsupjones/gpu-roofline/issues)*

### Multi-Precision Roofline — NVIDIA H200

```
  TFLOP/s  (log scale)
  989T ── ── ── ── ── ── ── ═══════════ FP16/BF16 Tensor Core (spec)
                            ╱
  686T ── ── ── ── ── ═════╪═══════════ FP16/BF16 Tensor Core (measured)
                       ╱   │
   59T ── ═══════════╪═════╪═══════════ FP32 CUDA Core (measured)
              ╱      │     │
  4028 GB/s  ╱       │     │
            ╱────────┼─────┼──────────→ Arithmetic Intensity (FLOP/byte)
           0.1      1     10    100

  Tensor Cores deliver 11.5x the throughput of CUDA Cores.
  ML training runs on the upper ceiling — that's what this tool measures.
```

### H100 Roofline (CUDA Backend)
```
gpu-roofline 0.1.0 | NVIDIA H100 80GB HBM3
  Peak FLOPS:     59.5 TFLOP/s (FP32)
  Peak Bandwidth: 2893 GB/s
  Ridge Point:    20.6 FLOP/byte

┌────────────┬─────────────┬─────────┬───────────┬────────────┬───────────────────┐
│ Kernel     ┆ AI (FLOP/B) ┆ GFLOP/s ┆ BW (GB/s) ┆ Efficiency ┆ Bottleneck        │
╞════════════╪═════════════╪═════════╪═══════════╪════════════╪═══════════════════╡
│ copy       ┆ 0.00        ┆ 0.0     ┆ 2893      ┆ 100%       ┆ Memory (HBM/DRAM) │
│ fma_light  ┆ 1.00        ┆ 2893    ┆ 2893      ┆ 100%       ┆ Memory (HBM/DRAM) │
│ fma_medium ┆ 8.00        ┆ 23169   ┆ 2896      ┆ 100%       ┆ Memory (HBM/DRAM) │
│ fma_heavy  ┆ 64.00       ┆ 59541   ┆ 930       ┆ 100%       ┆ Compute           │
└────────────┴─────────────┴─────────┴───────────┴────────────┴───────────────────┘
```

## Quick Start

```bash
# Install (consumer GPUs — Vulkan/DX12/Metal)
cargo install gpu-roofline

# Install with CUDA support (datacenter H100/H200/A100)
cargo install gpu-roofline --features cuda

# Full install: CUDA + NVML telemetry (real temp/clock/power)
cargo install gpu-roofline --features full

# Quick burst roofline (~10 seconds)
gpu-roofline measure --burst

# Full dynamic roofline with tension analysis (~120 seconds)
gpu-roofline measure

# Force specific backend
gpu-roofline measure --burst --backend cuda    # Datacenter (headless)
gpu-roofline measure --burst --backend vulkan  # Consumer Linux
gpu-roofline measure --burst --backend dx12    # Windows

# Validate GPU against known baselines (preflight health check)
gpu-roofline validate                          # Auto-detect GPU, check against baseline
gpu-roofline validate --strict                 # 90% threshold (default: 80%)
gpu-roofline validate --sim h100_sxm           # Validate simulation accuracy

# CI mode: fail if sustained performance regressed
gpu-roofline check --baseline roofline.json --threshold 0.9

# Save baseline for later comparison
gpu-roofline measure --save-baseline roofline.json
```

## Output Formats

```bash
gpu-roofline measure --format json       # Machine-readable
gpu-roofline measure --format ascii      # Terminal roofline chart
gpu-roofline measure --format table      # Colored table (default)
```

## Simulation Mode (No GPU Required)

The built-in physics-based simulation engine lets you explore roofline behavior without hardware:

```bash
gpu-roofline measure --sim rtx_5090
gpu-roofline measure --sim h100_sxm --json
gpu-roofline measure --sim mi300x --ascii
```

List all profiles:
```bash
gpu-roofline profiles
```

Available: `rtx_5090`, `rtx_4090`, `h100_sxm`, `h200_sxm`, `b200`, `mi300x`, `arc_a770` + degraded variants for testing straggler detection.

## GPU Validation Engine

Preflight health check against per-GPU hardware baselines. Supports 12 GPU models with auto-detection:

```bash
gpu-roofline validate
```
```
┌──────────────┬─────────┬──────────┬──────────┬────────┐
│ Check        ┆ Status  ┆ Measured ┆ Expected ┆ Result │
╞══════════════╪═════════╪══════════╪══════════╪════════╡
│ Bandwidth    ┆ ✓ PASS  ┆ 2893 GB/s┆ 2700-3100┆ 100%   │
│ FP32 Compute ┆ ✓ PASS  ┆ 59.5T   ┆ 55-65T   ┆ 100%   │
│ Stability    ┆ ✓ PASS  ┆ CV 0.3% ┆ <5%      ┆ 100%   │
│ Roofline     ┆ ✓ PASS  ┆ 20.6    ┆ 17-25    ┆ 100%   │
└──────────────┴─────────┴──────────┴──────────┴────────┘
```

Smart diagnosis: distinguishes HBM degradation from thermal throttling from driver issues.

## Continuous Monitoring

Live TUI dashboard — like `htop` for your GPU's performance envelope:

```
gpu-roofline monitor ── NVIDIA H100 SXM5 80GB ── Driver 550.54 ── sm_90
┌ Performance ──────────────────────┐┌ Tension Analysis ─────────────────┐
│ BW    2891 GB/s  99.9%  ▁▂▃▃▃▃▃▃││ Burst:     59.5T (t=0)           │
│ FP32  59.2T      99.4%  ▁▃▅▅▅▅▅▅││ Current:   59.2T (−0.5%)         │
│ CV    0.3%       stable  ▁▁▁▁▁▁▁▁││ Thermal:   −0.3% (72°C)         │
├ Thermals & Power ─────────────────┤│ Power:     −0.2% (685W)          │
│ Temp   72°C         ▁▂▃▄▅▅▅▅▅▅▅▅││ Net drop:  −0.5%                 │
│ Power  685W/700W    ▃▅▆▇▇▇▇▇▇▇▇▇│├ Session ──────────────────────────┤
│ Clock  1980 MHz     ▇▇▇▆▆▆▆▆▆▆▆▆││ Samples: 47  Uptime: 47m 12s     │
│ Throttle: none                    ││ Avg BW: 2889  Min BW: 2871 GB/s  │
├ Memory ───────────────────────────┤│ Max Temp: 74°C  Alerts: 0        │
│ VRAM  12.4 / 80.0 GB  ████░░░░░░│├ Alerts ────────────────────────────┤
│ HBM BW utilization: 86%          ││ (none)                            │
└───────────────────────────────────┘└──────────────────────────────────┘
 q quit  Samples: 47  Uptime: 47m 12s
```

```bash
# Live TUI dashboard (default)
gpu-roofline monitor

# Custom interval and duration
gpu-roofline monitor --interval 10 --duration 3600

# Alert if performance drops below 80% of baseline
gpu-roofline monitor --alert-threshold 0.8

# Daemon mode: JSON lines to file (for Prometheus/Grafana/Datadog)
gpu-roofline monitor --daemon --log monitor.json

# Simulated monitoring (no GPU required)
gpu-roofline monitor --sim h100_sxm --interval 5 --duration 60
```

### What It Detects
- **Sudden degradation** — bandwidth or compute drops below threshold
- **Thermal throttling** — temperature exceeding operating range
- **Gradual decline** — rolling average trending down over time
- **Measurement instability** — increasing CV indicating noisy results

### Daemon Mode (Prometheus/Grafana)
`--daemon` outputs one JSON object per line, scrapable by any log aggregator:
```json
{"timestamp":"2026-03-18T14:00:00Z","bandwidth_gbps":2891,"gflops":59200,"temperature_c":42,"status":"normal"}
{"timestamp":"2026-03-18T14:01:00Z","bandwidth_gbps":2102,"gflops":58900,"temperature_c":67,"status":"warning","alerts":[...]}
```

## gpu-fleet (Coming Soon)

Multi-GPU cluster validation — detects stragglers, NVLink degradation, NUMA misalignment, and GPUs running below their roofline.

## Architecture

```
gpu-roofline/
├── crates/
│   ├── gpu-harness/          # Shared backend abstraction
│   │   ├── sim/              # Physics-based GPU simulation engine
│   │   ├── cuda_backend.rs   # Native CUDA compute (datacenter)
│   │   ├── wgpu_backend.rs   # Vulkan/DX12/Metal/GL (consumer)
│   │   └── nvml_telemetry.rs # Real GPU temp/clock/power via NVML
│   ├── gpu-roofline/         # Roofline engine + CLI
│   │   ├── ceilings/         # Burst + dynamic measurement
│   │   ├── model/            # Roofline model + tension analysis
│   │   ├── monitor/          # Live TUI + alerting + daemon
│   │   ├── validate/         # Preflight GPU health checks
│   │   └── shaders/          # WGSL + CUDA compute kernels
│   └── gpu-fleet/            # Multi-GPU cluster validation (WIP)
```

All code programs against the `GpuBackend` trait — swap between simulated and real GPU backends without changing application logic.

## Simulation Engine

The simulation models GPU behavior as competing physical forces:

- **ThermalModel** — Newton's law of cooling with throttle curve
- **PowerModel** — TDP-limited clock scaling with V/F curve
- **BandwidthModel** — Hierarchical L1/L2/HBM/DRAM with contention

11 pre-built profiles (RTX 5090, H100, H200, B200, MI300X, Arc A770) plus degraded variants for testing straggler detection. Calibrated against real hardware — H100 simulation matches validated results within 0.4%.

## Library Usage

```rust
use gpu_harness::sim::{SimulatedBackend, profiles};
use gpu_harness::GpuBackend;

// Create a simulated H100
let backend = SimulatedBackend::new(profiles::h100_sxm());

// Discover devices
let devices = backend.discover_devices()?;
println!("Found: {}", devices[0].name);

// Query device state (thermal, clock, power)
let state = backend.device_state(0)?;
println!("{}°C, {} MHz, {:.0}W", state.temperature_c, state.clock_mhz, state.power_watts);
```

## Contributing

Contributions welcome! See [open issues](https://github.com/whatsupjones/gpu-roofline/issues) for good starting points:

- Add GPU simulation profiles (RX 9070 XT, RTX 3060/3070/3080)
- Test and report results on your hardware
- Add ROCm-SMI backend for AMD GPU monitoring
- Add Intel Level Zero backend
- Improve ASCII chart rendering
- Add SVG roofline chart export

## License

MIT OR Apache-2.0
