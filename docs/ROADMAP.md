# gpu-roofline Roadmap

## v0.2.0 — vGPU Lifecycle Monitoring + RTX 5090 Validation

### vGPU Lifecycle-Aware Monitoring ✅ Implemented

Trigger-point vGPU lifecycle monitoring is implemented behind `--features vgpu`:

- **Trigger Detection** — VgpuDetector trait with platform-specific implementations (NVIDIA GRID sysfs/NVML, MIG procfs/NVML, SR-IOV sysfs, Cloud passthrough, K8s device plugin) + SimulatedDetector for testing
- **Contention Detection** — ContentionMeasurer records per-instance baselines, detects squeeze when new vGPUs appear on time-sliced GPUs. Hardware-partitioned (MIG) instances correctly skip contention checks.
- **Teardown Verification** — TeardownVerifier captures pre-teardown state and compares post-teardown to detect ghost allocations and measure reclaim latency
- **Auto Attach/Detach** — VgpuSampler event loop with 7 alert rules (SlowProvision, ContentionSqueeze, UnderperformingInstance, GhostAllocation, SlowReclaim, OverSubscription, MemoryOvercommit)
- **CLI** — `gpu-roofline vgpu watch`, `vgpu list`, `vgpu scenarios` with TUI and daemon modes
- **4 Simulation Scenarios** — mig_scale_up, grid_contention, ghost_allocation, rapid_churn
- **Validated** — End-to-end integration tests for all scenarios + scale stress tests (100/1000 lifecycle cycles)

### CUDA Events ✅ Implemented

GPU-side hardware timestamps via CUDA Events replace CPU-side `Instant::now()` timing. GpuTimer wraps CudaEvent pairs with automatic fallback to CPU timing if events are unavailable. Validated on H100: ~2% bandwidth improvement from eliminating kernel launch overhead.

### CUDA Graphs (Planned)

Batch multiple kernel launches into a single GPU graph submission. Reduces per-kernel CPU overhead from ~5µs to near zero. Enables higher-frequency monitoring without impacting workload performance.

### Consumer GPU Validation

**We validate:** RTX 5090 (flagship, proves consumer support)
**Community validates:** RTX 4090, 4080, 4070, 3090, 3080, 3060, AMD, Intel, Apple

See [CONTRIBUTING_VALIDATION.md](CONTRIBUTING_VALIDATION.md) for the community validation template.

## v0.3.0 — Diagnostic Engine + Fleet Validation ✅ Implemented

### "Why Is My GPU Slow?" Diagnostic ✅ Implemented

Six targeted probes identify root causes of GPU underperformance:

- **L2 Cache Thrashing** — working set exceeds L2 capacity
- **HBM Degradation** — partial HBM stack failure or ECC bandwidth loss
- **PCIe Bottleneck** — host-device transfers saturating PCIe link
- **Thermal Throttling** — clock reduction from cooling failure
- **Clock Stuck** — GPU locked at base clock, boost disabled
- **Compute Deficit** — achieved TFLOPS below expected for the architecture

CLI: `gpu-roofline diagnose --device 0` or `gpu-roofline diagnose --sim degraded_h100_memory`

### gpu-fleet: Multi-GPU Cluster Validation ✅ Implemented

```bash
gpu-fleet topology              # PCIe/NVLink tree + P2P bandwidth matrix
gpu-fleet validate              # Per-GPU roofline health check
gpu-fleet symmetry              # Flag mismatched configs across fleet
gpu-fleet straggler             # Identify underperforming GPUs + cause
```

Straggler detection: measures all GPUs → computes fleet median → flags outliers → runs diagnostic probes on each straggler → reports cause + fix.

### Real MIG Detection ✅ Implemented

NvidiaMigDetector upgraded from stub to real NVML-based implementation (nvml-wrapper 0.12):
- `enumerate()` via NVML MIG APIs (`mig_device_count`, `mig_device_by_index`)
- `watch()` via poll-based delta detection (enumerate periodically, diff instance sets, emit Created/Destroyed events)
- MIG hardware validation pending bare-metal H100 access (simulation-validated with 254 tests)

### TUI Dashboard Enhancements (Planned)

- Sparkline charts with historical data persistence
- Per-GPU view for multi-GPU systems
- Keyboard navigation between GPU panels
- Export session to JSON/CSV on exit

## v0.4.0 — Enterprise Integration ✅ Implemented

### Prometheus Metrics Endpoint ✅ Implemented

Feature-gated behind `--features enterprise`. Serves `/metrics` in Prometheus text exposition format:
- GPU gauges: bandwidth, gflops, temperature, clock, power, utilization, memory
- Alert counters by severity (warning/critical)
- vGPU gauges: active instance count, VRAM allocated/available, lifecycle event counters
- Health endpoint at `/health` for Kubernetes probes

### Webhook Alerts ✅ Implemented

JSON POST to configurable URLs on alert conditions. Fire-and-forget with bounded channel to avoid blocking the measurement loop. Supports multiple URLs (`--webhook-url` is repeatable).

### Grafana Dashboard ✅ Implemented

Ready-to-import JSON dashboard at `deploy/grafana/gpu-roofline-dashboard.json`:
- Health overview (utilization, temperature, alert count, power gauges)
- Performance (bandwidth + gflops time series)
- Telemetry (power, clock, memory usage)
- vGPU lifecycle (active count, VRAM allocation, event rates — collapsed by default)

### Kubernetes Deployment ✅ Implemented

DaemonSet + Service + ConfigMap + ServiceMonitor at `deploy/k8s/`:
- `nodeSelector: nvidia.com/gpu.present` for GPU-only nodes
- Liveness/readiness probes on `/health`
- Prometheus Operator ServiceMonitor for auto-discovery
- No custom operator — works with existing Prometheus + Grafana stacks

### TUI Dashboard Enhancements (Planned)

- Sparkline charts with historical data persistence
- Per-GPU view for multi-GPU systems
- Keyboard navigation between GPU panels
- Export session to JSON/CSV on exit

## v0.5.0 — Future

- Fleet-wide anomaly detection with ML baseline
- Custom Kubernetes operator with CRDs (if demand warrants)
- Historical data persistence (time-series DB integration)
- Multi-cluster federation
