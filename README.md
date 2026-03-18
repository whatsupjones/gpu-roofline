# gpu-roofline

**GPU lifecycle monitoring and performance analysis for virtualized infrastructure.**

The first tool that monitors vGPU instances from the moment they provision — detecting contention, verifying teardown, catching ghost allocations. Plus cross-vendor roofline measurement, degradation alerting, and CI-native GPU health checks. Single Rust binary, <10 MB.

> Every existing GPU monitoring tool polls a vGPU **after** it exists. Nobody measures from the moment of creation. Spin-up overhead is invisible. Contention impact on existing tenants goes undetected until workloads fail. Ghost allocations after teardown silently leak resources. gpu-roofline fixes this.

## The Problem

DGX Cloud, AWS, GCP, and Azure manage thousands of GPU lifecycles. MIG partitions on H100/H200, GRID time-slicing, SR-IOV, Kubernetes device plugins — all create and destroy virtual GPUs constantly. But:

- **Provisioning latency is invisible** — no tool measures spin-up overhead
- **Contention goes undetected** — when a new vGPU appears, existing tenants get squeezed silently
- **Teardown leaks resources** — ghost allocations persist after vGPU destruction
- **Performance baselines don't exist** — nobody measures the burst-to-sustained gap under virtualization

## vGPU Lifecycle Monitoring

Auto-attaches when a vGPU provisions, detaches when it drops. Zero overhead when idle.

```bash
# Install with vGPU + CUDA support
cargo install gpu-roofline --features vgpu,cuda

# Watch lifecycle events in real time (TUI dashboard)
gpu-roofline vgpu watch --sim grid_contention

# Daemon mode: JSON lines for Prometheus/Grafana/Datadog
gpu-roofline vgpu watch --sim grid_contention --daemon --log vgpu.jsonl

# List current vGPU instances
gpu-roofline vgpu list --sim mig_scale_up --json
```

### What It Catches

| Alert | Trigger | Severity |
|-------|---------|----------|
| **ContentionSqueeze** | Existing tenant bandwidth dropped >5% when new vGPU appeared | Critical |
| **GhostAllocation** | Teardown left unreleased memory on physical GPU | Critical |
| **MemoryOvercommit** | Sum of vGPU VRAM exceeds physical GPU capacity | Critical |
| **SlowProvision** | vGPU spin-up took >500ms | Warning |
| **SlowReclaim** | Resource reclamation >1000ms after teardown | Warning |
| **OverSubscription** | More vGPUs than safe density threshold | Warning |
| **UnderperformingInstance** | vGPU below expected fraction of physical GPU | Warning |

### Supported Technologies

| Technology | Linux Detection | Fallback |
|-----------|----------------|----------|
| **NVIDIA MIG** | procfs + NVML MIG APIs | NVML polling |
| **NVIDIA GRID** | inotify on sysfs mdev + NVML | NVML polling |
| **SR-IOV** | inotify on sysfs VFs | sysfs polling |
| **Cloud Passthrough** | udev device events | device enumeration delta |
| **Kubernetes** | kubelet device-plugins watch | kubelet API polling |
| **Simulated** | Built-in scenarios for testing | — |

### Simulation Scenarios (No Hardware Required)

```bash
gpu-roofline vgpu scenarios
```

| Scenario | What It Tests |
|----------|--------------|
| `mig_scale_up` | 7 MIG instances on H100 — hardware-partitioned, no contention |
| `grid_contention` | 4 GRID vGPUs — each new one squeezes existing tenants |
| `ghost_allocation` | Create + destroy with 512MB not reclaimed |
| `rapid_churn` | 20 create/destroy cycles — stress-tests for state leaks |

---

## Performance Measurement

Beyond lifecycle monitoring, gpu-roofline measures what no other tool does: the **sustained** performance ceiling — not just the burst peak that benchmarks report.

### Dynamic Roofline with Tension Analysis

```
              Performance (GFLOP/s)
    82.6T  +═══════════════════════ Peak Burst (t=0, 2520 MHz, 62°C)
            ╲
    71.3T  + ╲═════════════════════ Thermal Equilibrium (t=34s, 2280 MHz, 78°C)
            ╲   ╲
    65.1T  +  ╲   ╲═══════════════ Power-Limited Sustained (t=120s, 2100 MHz, 83°C)
              ╲     ╲
              +───+───+───+───+──→ Arithmetic Intensity (FLOP/byte)

    Tension: 82.6T → 65.1T (−21.2%) burst-to-sustained
```

Your ML training job runs for hours at the **sustained** ceiling — 15-30% lower than what benchmarks report.

### Validated Hardware

| GPU | HBM BW | FP32 | FP16 Tensor | BF16 Tensor | Backend |
|-----|--------|------|-------------|-------------|---------|
| **NVIDIA H200 141GB** | **4,028 GB/s** | **59.5 TFLOPS** | **686 TFLOPS** | **686 TFLOPS** | CUDA |
| **NVIDIA H100 80GB** | **2,958 GB/s** | **59.0 TFLOPS** | **495 TFLOPS** | **495 TFLOPS** | CUDA |
| **NVIDIA RTX 5090 32GB** | **1,503 GB/s** | **95.8 TFLOPS** | **247 TFLOPS** | **247 TFLOPS** | CUDA |
| Intel UHD Graphics | 7 GB/s | 0.15 TFLOPS | — | — | Vulkan |

GPU-side CUDA Event timestamps for sub-microsecond accuracy. Automatic fallback to CPU timing on non-CUDA backends.

### Quick Start

```bash
# Install (consumer GPUs — Vulkan/DX12/Metal)
cargo install gpu-roofline

# Install with CUDA (datacenter H100/H200/A100)
cargo install gpu-roofline --features cuda

# Burst roofline (~10s)
gpu-roofline measure --burst

# Full dynamic roofline with tension analysis (~120s)
gpu-roofline measure

# Validate GPU against known baselines
gpu-roofline validate

# CI mode: fail if performance regressed
gpu-roofline check --baseline roofline.json --threshold 0.9
```

### Continuous Monitoring

Live TUI dashboard with degradation alerting:

```bash
gpu-roofline monitor                           # Interactive TUI
gpu-roofline monitor --daemon --log monitor.json  # JSON lines for log aggregation
```

Detects: sudden degradation, thermal throttling, gradual decline, measurement instability.

### Simulation Mode

11 pre-built GPU profiles — test without hardware:

```bash
gpu-roofline measure --sim h100_sxm
gpu-roofline profiles     # List all available profiles
```

---

## What Makes This Different

| Feature | Nsight Compute | Intel Advisor | AMD Omniperf | **gpu-roofline** |
|---------|:---:|:---:|:---:|:---:|
| vGPU lifecycle monitoring | No | No | No | **Yes** |
| Cross-vendor | NVIDIA only | Intel only | AMD only | **NVIDIA + AMD + Intel** |
| Install size | ~5 GB | ~15 GB | ~3 GB | **<10 MB** |
| CI-native (JSON, exit codes) | No | No | No | **Yes** |
| Sustained ceiling measurement | No | No | No | **Yes** |
| Contention detection | No | No | No | **Yes** |
| Ghost allocation detection | No | No | No | **Yes** |
| Live TUI monitoring | No | No | No | **Yes** |
| Single binary | No | No | No | **Yes** |

---

## Architecture

```
gpu-roofline/
├── crates/
│   ├── gpu-harness/          # Shared backend abstraction
│   │   ├── vgpu/             # vGPU lifecycle detection + contention + teardown
│   │   ├── sim/              # Physics-based GPU simulation engine
│   │   ├── cuda_backend.rs   # CUDA compute + Event timing (datacenter)
│   │   ├── wgpu_backend.rs   # Vulkan/DX12/Metal/GL (consumer)
│   │   └── nvml_telemetry.rs # Real GPU temp/clock/power via NVML
│   ├── gpu-roofline/         # CLI + measurement + monitoring
│   │   ├── monitor/          # TUI + alerting + vGPU sampler + daemon
│   │   ├── ceilings/         # Burst + dynamic roofline measurement
│   │   ├── validate/         # Preflight GPU health checks
│   │   └── shaders/          # WGSL + CUDA compute kernels
│   │   └── diagnose/          # "Why Is My GPU Slow?" diagnostic engine
│   └── gpu-fleet/            # Multi-GPU cluster validation
│       ├── topology.rs       # PCIe/NVLink tree + P2P bandwidth matrix
│       ├── symmetry.rs       # Fleet config mismatch detection
│       ├── fleet_validate.rs # Per-GPU roofline health checks
│       └── straggler.rs      # Outlier detection + automatic diagnosis
```

## "Why Is My GPU Slow?" Diagnostic Engine

Six targeted probes identify root causes of GPU underperformance:

```bash
# Diagnose a specific GPU
gpu-roofline diagnose --device 0

# Diagnose with simulation (no hardware needed)
gpu-roofline diagnose --sim degraded_h100_memory

# JSON output for automation
gpu-roofline diagnose --sim h100_sxm --json
```

```
gpu-roofline diagnose | NVIDIA H100 SXM 80GB

  Finding: HBM bandwidth at 75% of expected
  Cause:   Partial HBM stack failure (measured 2,512 vs expected 3,350 GB/s)
  Fix:     RMA GPU or verify ECC settings. Run nvidia-smi -q for error counts.

  6 probes run | 1 finding | Severity: Warning
```

| Probe | What It Detects |
|-------|----------------|
| **L2 Cache Thrashing** | Working set exceeds L2 capacity, causing excessive HBM traffic |
| **HBM Degradation** | Partial HBM stack failure or ECC-related bandwidth loss |
| **PCIe Bottleneck** | Host-device transfers saturating PCIe link |
| **Thermal Throttling** | Clock reduction from cooling failure or paste degradation |
| **Clock Stuck** | GPU locked at base clock, boost disabled or broken |
| **Compute Deficit** | Achieved TFLOPS well below expected for the architecture |

---

## gpu-fleet: Multi-GPU Cluster Validation

```bash
# Discover topology: GPU list + NVLink/PCIe P2P bandwidth matrix
gpu-fleet topology --sim h100_sxm --count 8

# Validate per-GPU roofline health
gpu-fleet validate --sim h100_sxm --count 8

# Check fleet symmetry (flag mismatched configs)
gpu-fleet symmetry --sim h100_sxm --count 8

# Find stragglers: outlier detection + automatic root-cause diagnosis
gpu-fleet straggler --sim h100_sxm --count 8
```

Straggler detection measures all GPUs, computes the fleet median, flags outliers below threshold, then runs the diagnostic engine on each straggler to identify **why** it's underperforming.

---

## Roadmap

See [ROADMAP.md](docs/ROADMAP.md) for details.

- **v0.2** ✅ vGPU Lifecycle Monitoring — trigger-point detection, contention, teardown verification, 7 alert rules
- **v0.2** ✅ CUDA Events — GPU-side hardware timestamps, automatic CPU fallback
- **v0.3** ✅ Diagnostic Engine — 6 probes for automatic GPU root-cause analysis
- **v0.3** ✅ gpu-fleet — multi-GPU cluster validation, NVLink topology, straggler detection
- **v0.3** ✅ Real MIG Detection — NVML MIG APIs for hardware vGPU enumeration + polling

## Contributing

Contributions welcome! See [CONTRIBUTING_VALIDATION.md](docs/CONTRIBUTING_VALIDATION.md) to validate your GPU hardware.

## License

MIT OR Apache-2.0
