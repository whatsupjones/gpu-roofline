# gpu-roofline Roadmap

## v0.2.0 — vGPU Lifecycle Monitoring + RTX 5090 Validation

### vGPU Lifecycle-Aware Monitoring

Current GPU monitoring tools poll **after** a vGPU exists. We measure **from the moment of creation** — capturing provisioning overhead, warm-up curves, and teardown efficiency.

**Architecture:**
- **Trigger Detection** — Hook into vGPU provisioning events (NVIDIA GRID/vGPU manager API, libvirt hooks, Kubernetes device plugin events) to detect the exact moment a vGPU is created or destroyed
- **Lifecycle Measurement** — Capture spin-up latency, initial allocation efficiency, and first-N-seconds performance curve before steady state
- **Contention Detection** — When a new vGPU spins up on a shared physical GPU, measure the impact on existing vGPU tenants in real-time
- **Teardown Verification** — When a vGPU is dropped, verify the physical GPU reclaims resources (detect ghost allocations, memory fragmentation)
- **Auto Load/Unload** — Monitor attaches when vGPU provisions, detaches when vGPU drops. Zero overhead when no vGPU is active.

**Why This Matters:**
- DGX Cloud manages thousands of vGPU lifecycles — no tool measures provisioning efficiency at the trigger point
- MIG partitions on H100/H200 share thermals but have independent compute/memory — lifecycle monitoring catches cross-partition interference
- Cloud providers (AWS, GCP, Azure) charge per-second for GPU instances — measuring spin-up waste directly impacts cost

### CUDA Events + CUDA Graphs

**CUDA Events** — Replace CPU-side `Instant::now()` timing with GPU-side hardware timestamps for sub-microsecond accuracy. Eliminates kernel launch overhead from measurements. Critical for Tensor Core kernels where dispatch time rivals execution time.

**CUDA Graphs** — Batch multiple kernel launches into a single GPU graph submission. Reduces per-kernel CPU overhead from ~5µs to near zero. Enables higher-frequency monitoring without impacting workload performance.

**Graceful Fallback Chain:**
1. CUDA Graphs (lowest overhead, batch dispatch)
2. CUDA Events (GPU-side timing, per-kernel dispatch)
3. CPU timing with stream sync (current, broadest compatibility)

System auto-selects based on GPU capability and driver version.

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
