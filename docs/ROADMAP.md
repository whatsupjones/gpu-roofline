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

## v0.3.0 — Diagnostic Engine + Fleet Validation

### "Why Is My GPU Slow?" Diagnostic

Instead of just measuring ceilings, tell the user **why** they're not hitting them:

```
Your H200 is running at 67% of its bandwidth ceiling.
Cause: L2 cache thrashing (working set 180MB vs 50MB L2)
Fix:   Increase batch size to amortize memory access
```

Diagnostic categories:
- Memory-bound: L2 thrashing, HBM degradation, PCIe bottleneck
- Compute-bound: low occupancy, register pressure, serial dependencies
- Thermal: throttling from paste degradation or cooling failure
- Configuration: wrong power mode, ECC overhead, MIG misconfiguration

### gpu-fleet: Multi-GPU Cluster Validation

```bash
gpu-fleet topology              # PCIe/NVLink tree view
gpu-fleet validate --roofline   # Per-GPU roofline health check
gpu-fleet symmetry              # Flag mismatched configs across fleet
gpu-fleet straggler             # Identify underperforming GPUs + cause
```

### TUI Dashboard Enhancements

- Sparkline charts with historical data persistence
- Per-GPU view for multi-GPU systems
- Keyboard navigation between GPU panels
- Export session to JSON/CSV on exit

## v0.4.0 — Enterprise Integration

- Prometheus metrics endpoint (`/metrics`)
- Grafana dashboard templates
- Kubernetes GPU health operator
- Webhook alerts (Slack, PagerDuty, OpsGenie)
- Fleet-wide anomaly detection with ML baseline
