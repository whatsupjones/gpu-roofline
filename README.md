# gpu-roofline

**GPU lifecycle monitoring and performance analysis for virtualized infrastructure.**

GPU lifecycle monitoring and performance analysis for virtualized infrastructure. The first tool that monitors vGPU instances from the moment they provision, detecting contention, verifying teardown, and catching ghost allocations. Cross-vendor roofline measurement, degradation alerting, and CI-native GPU health checks. Single Rust binary, <10 MB.

> Every existing GPU monitoring tool polls a vGPU **after** it exists. Nobody measures from the moment of creation. Spin-up overhead is invisible. Contention impact on existing tenants goes undetected until workloads fail. Ghost allocations after teardown silently leak resources. gpu-roofline fixes this.

## Study: The GPU Efficiency Gap

We conducted a [120,000-trial simulation study](docs/study-simulation-manuscript.md) identifying six categories of invisible GPU waste across the full operations stack: device-level thermal physics, virtualization partitioning, and fleet-level coordination. The simulation is calibrated against hardware-validated roofline measurements on H100 SXM and H200 systems.

**The central finding: `nvidia-smi` and DCGM detect 0% of waste events across all six categories. `gpu-roofline` detects 56.5 to 100%.**

| What Visibility Enables | Categories | Per-Event Impact |
|------------------------|-----------|-----------------|
| **Recover capacity** | Ghost allocations, Straggler tax | 512 MiB VRAM freed per teardown; 19% fleet throughput recovered per straggler GPU |
| **Inform decisions** | Contention squeeze, Burst-sustained gap | MIG vs time-slicing selection with measured data; SLAs on actual sustained performance |
| **Prevent failures** | Oversubscription | Detection before tenants experience degradation at 1.5x+ overcommit |

All six categories produce large effect sizes (Cohen's d/d_z from 0.73 to 8.55). The simulation is deterministic and reproducible (seed 42, SHA-256 verified).

**Read the full study:** [Manuscript](docs/study-simulation-manuscript.md) | [PDF](docs/print/gpu-waste-study-complete.pdf) | [Results](docs/study-results/summary.md) | [Protocol](docs/study-protocol-gpu-waste.md)

Hardware validation on bare-metal H100 is the next phase. See [Contributing Hardware Validation](#contributing-hardware-validation) below.

## The Problem

DGX Cloud, AWS, GCP, and Azure manage thousands of GPU lifecycles. MIG partitions on H100/H200, GRID time-slicing, SR-IOV, and Kubernetes device plugins all create and destroy virtual GPUs constantly. But:

- **Provisioning latency is invisible.** No tool measures spin-up overhead.
- **Contention goes undetected.** When a new vGPU appears, existing tenants get squeezed silently.
- **Teardown leaks resources.** Ghost allocations persist after vGPU destruction.
- **Performance baselines don't exist.** Nobody measures the burst-to-sustained gap under virtualization.

## Use Cases

### ML Platform Team

You manage 100+ H100s. Training jobs randomly slow down and nobody knows why. A straggler GPU turns a 3-day run into a 5-day run, or causes silent accuracy degradation.

```bash
# Find the straggler and get a root cause in one command
gpu-fleet straggler

# Pre-flight validation before launching a $50K training run
gpu-fleet validate --threshold 0.85
```

### Cloud GPU Provider / Multi-Tenant

You sell MIG slices or GRID time-sliced vGPUs. When a new tenant provisions, existing tenants get squeezed, but you don't know until they complain. Teardown leaks VRAM silently.

```bash
# Real-time lifecycle monitoring with 7 alert rules
gpu-roofline vgpu watch --daemon --log vgpu.jsonl

# Enumerate all MIG instances with metadata (JSON for Prometheus/Grafana)
gpu-roofline vgpu list --json
```

Detects: ContentionSqueeze, GhostAllocation, MemoryOvercommit, SlowProvision, SlowReclaim, OverSubscription, UnderperformingInstance.

### ML Engineer / Researcher

Your H100 training is 30% slower than the paper claims. You don't know if it's the GPU, the workload, or the cluster.

```bash
# Get an actionable diagnosis, not just "GPU slow"
gpu-roofline diagnose --device 0

# See the burst-to-sustained gap that benchmarks hide
gpu-roofline measure
```

Output: "L2 cache thrashing: working set 180MB exceeds 50MB L2. Increase batch size to amortize memory access."

### DevOps / CI Pipeline

GPU performance as a CI gate. Fail the build if the GPU regressed, before bad hardware reaches production.

```bash
# Save a baseline, then check against it in CI
gpu-roofline measure --save-baseline roofline.json
gpu-roofline check --baseline roofline.json --threshold 0.9  # exit code 1 on failure

# Quick health check (no baseline needed)
gpu-roofline validate --strict
```

JSON output + exit codes for integration with any CI system.

### Simulation-First Development

Every command supports `--sim` with profiles validated against real H100, H200, and RTX 5090 hardware. Develop your monitoring pipeline, alerting rules, and integration tests without paying for GPU time.

```bash
# Develop and test without hardware
gpu-roofline diagnose --sim degraded_h100_memory    # triggers HBM degradation finding
gpu-fleet straggler --sim h100_sxm --count 8        # tests outlier detection
gpu-roofline vgpu watch --sim grid_contention        # tests contention alerting

# Same binary deploys to production, no code changes
```

> All simulation profiles are validated against real hardware. See [docs/validation/](docs/validation/) for H100/H200/RTX 5090 test artifacts with full provenance.

---

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
| **Simulated** | Built-in scenarios for testing | N/A |

### Simulation Scenarios (No Hardware Required)

```bash
gpu-roofline vgpu scenarios
```

| Scenario | What It Tests |
|----------|--------------|
| `mig_scale_up` | 7 MIG instances on H100, hardware-partitioned, no contention |
| `grid_contention` | 4 GRID vGPUs, each new one squeezes existing tenants |
| `ghost_allocation` | Create + destroy with 512MB not reclaimed |
| `rapid_churn` | 20 create/destroy cycles, stress-tests for state leaks |

---

## Performance Measurement

Beyond lifecycle monitoring, gpu-roofline measures what no other tool does: the **sustained** performance ceiling, not just the burst peak that benchmarks report.

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

Your ML training job runs for hours at the **sustained** ceiling, 15 to 30% lower than what benchmarks report.

### Validated Hardware

| GPU | HBM BW | FP32 | FP16 Tensor | BF16 Tensor | Backend |
|-----|--------|------|-------------|-------------|---------|
| **NVIDIA H200 141GB** | **4,028 GB/s** | **59.5 TFLOPS** | **686 TFLOPS** | **686 TFLOPS** | CUDA |
| **NVIDIA H100 80GB** | **2,958 GB/s** | **59.0 TFLOPS** | **495 TFLOPS** | **495 TFLOPS** | CUDA |
| **NVIDIA RTX 5090 32GB** | **1,503 GB/s** | **95.8 TFLOPS** | **247 TFLOPS** | **247 TFLOPS** | CUDA |
| Intel UHD Graphics | 7 GB/s | 0.15 TFLOPS | N/A | N/A | Vulkan |

GPU-side CUDA Event timestamps for sub-microsecond accuracy. Automatic fallback to CPU timing on non-CUDA backends.

### Quick Start

```bash
# Install (consumer GPUs: Vulkan/DX12/Metal)
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

11 pre-built GPU profiles. Test without hardware:

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

## Library Usage

`gpu-harness` is a standalone Rust crate. Embed GPU monitoring directly in your services without shelling out to the CLI:

```toml
[dependencies]
gpu-harness = { version = "0.1", features = ["vgpu", "cuda"] }
```

```rust
use gpu_harness::vgpu::detect::{auto_detect, VgpuDetector};
use std::sync::mpsc;

// Enumerate all MIG instances on the system
let detector = auto_detect();
let instances = detector.enumerate()?;

// Watch for lifecycle events in real time
let (tx, rx) = mpsc::channel();
std::thread::spawn(move || detector.watch(tx));
for event in rx {
    println!("vGPU event: {:?}", event.event_type);
}
```

Available modules: `backend` (GpuBackend trait), `device` (discovery), `sim` (physics-based simulation), `vgpu` (lifecycle detection), `cuda_backend`, `nvml_telemetry`.

---

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

## Enterprise Integration

Build with `--features enterprise` for production monitoring stacks. See [docs/enterprise-integration.md](docs/enterprise-integration.md) for the full guide.

```bash
# Build with enterprise features
cargo install gpu-roofline --features enterprise,vgpu,cuda

# Monitor with Prometheus metrics endpoint
gpu-roofline monitor --daemon --metrics-port 9835

# Add webhook alerts (Slack, PagerDuty, OpsGenie)
gpu-roofline monitor --daemon --metrics-port 9835 \
  --webhook-url https://hooks.slack.com/services/xxx

# Deploy to Kubernetes
kubectl apply -f deploy/k8s/
```

- **Prometheus**: `/metrics` endpoint with GPU gauges, alert counters, vGPU lifecycle metrics
- **Grafana**: One-click dashboard import (`deploy/grafana/gpu-roofline-dashboard.json`)
- **Webhooks**: JSON POST on alert (configurable URLs, fire-and-forget)
- **Kubernetes**: DaemonSet with health probes, ServiceMonitor for auto-discovery
- **Health endpoint**: `/health` for liveness/readiness probes

---

## Roadmap

See [ROADMAP.md](docs/ROADMAP.md) for details.

- **v0.2** ✅ vGPU Lifecycle Monitoring: trigger-point detection, contention, teardown verification, 7 alert rules
- **v0.2** ✅ CUDA Events: GPU-side hardware timestamps, automatic CPU fallback
- **v0.3** ✅ Diagnostic Engine: 6 probes for automatic GPU root-cause analysis
- **v0.3** ✅ gpu-fleet: multi-GPU cluster validation, NVLink topology, straggler detection
- **v0.3** ✅ Real MIG Detection: NVML MIG APIs for hardware vGPU enumeration + polling
- **v0.4** ✅ Enterprise Integration: Prometheus metrics, webhook alerts, Grafana dashboard, K8s deployment

## Contributing Hardware Validation

The simulation study predicts six categories of invisible waste on H100/H200 GPUs. We need bare-metal hardware validation to confirm the simulation findings. If you have access to:

- **H100 SXM5 or H200** with MIG enabled: ghost allocations, provisioning overhead, contention
- **Multi-GPU cluster (8+ GPUs)**: straggler tax in distributed training
- **Time-sliced vGPU environment**: contention squeeze measurements
- **Any datacenter GPU**: burst-to-sustained gap thermal characterization

See the [study protocol](docs/study-protocol-gpu-waste.md) for the full experimental design (1,200 hardware trials planned). The simulation predicts specific effect sizes. We need hardware data to confirm or calibrate them.

Open an issue with the `hardware-validation` label or reach out directly. All contributed data will be credited in the follow-on publication.

## Contributing

Contributions welcome! See [CONTRIBUTING_VALIDATION.md](docs/CONTRIBUTING_VALIDATION.md) to validate your GPU hardware.

## License

MIT OR Apache-2.0
