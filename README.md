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
| CI-native (JSON, exit codes) | No | No | No | **Yes** |
| Sustained ceiling measurement | No | No | No | **Yes** |
| Tension analysis | No | No | No | **Yes** |
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

## Quick Start

```bash
cargo install gpu-roofline

# Full dynamic roofline with tension analysis (~120 seconds)
gpu-roofline measure

# Quick burst-only (traditional static roofline, ~10 seconds)
gpu-roofline measure --burst

# All GPUs, overlay comparison
gpu-roofline measure --all

# CI mode: fail if sustained performance regressed
gpu-roofline check --baseline roofline.json --threshold 0.9
```

## Output Formats

```bash
gpu-roofline measure --json              # Machine-readable
gpu-roofline measure --ascii             # Terminal chart
gpu-roofline measure --svg roofline.svg  # For papers and docs
```

## Simulation Mode (No GPU Required)

The built-in physics-based simulation engine lets you explore roofline behavior without hardware:

```bash
gpu-roofline measure --sim rtx_5090
gpu-roofline measure --sim h100_sxm --json
gpu-roofline measure --sim mi300x --ascii
```

Available profiles: `rtx_5090`, `rtx_4090`, `h100_sxm`, `h200_sxm`, `b200`, `mi300x`, `arc_a770`

## gpu-fleet: Multi-GPU Cluster Validation

The workspace includes `gpu-fleet` for validating multi-GPU clusters:

```bash
cargo install gpu-fleet

gpu-fleet topology              # PCIe/NVLink tree view
gpu-fleet validate --roofline   # Per-GPU roofline-based health check
gpu-fleet symmetry              # Flag mismatched configs across fleet
gpu-fleet monitor               # Live TUI dashboard
```

Detects stragglers, NVLink degradation, NUMA misalignment, and GPUs running below their roofline — problems that hardware health checks miss.

## Architecture

```
gpu-roofline/
├── crates/
│   ├── gpu-harness/     # Device discovery, simulation engine, GpuBackend trait
│   ├── gpu-roofline/    # Dynamic roofline model + tension analysis
│   └── gpu-fleet/       # Fleet validation + straggler detection
```

All code programs against the `GpuBackend` trait — swap between simulated and real GPU backends without changing application logic.

## Simulation Engine

The simulation models GPU behavior as competing physical forces:

- **ThermalModel** — Newton's law of cooling with throttle curve
- **PowerModel** — TDP-limited clock scaling with V/F curve
- **BandwidthModel** — Hierarchical L1/L2/HBM/DRAM with contention

11 pre-built profiles (RTX 5090, H100, H200, B200, MI300X, Arc A770) plus degraded variants for testing straggler detection.

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

Contributions welcome! Good first issues:

- Add GPU profile for RX 9070 XT (RDNA 4)
- Add ROCm-SMI backend for AMD monitoring
- Add Intel Level Zero backend
- Improve thermal model accuracy for specific GPU models

## License

MIT OR Apache-2.0
