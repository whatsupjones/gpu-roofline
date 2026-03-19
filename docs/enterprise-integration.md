# Enterprise Integration Guide

gpu-roofline ships with built-in Prometheus metrics, webhook alerts, and Kubernetes deployment support behind the `enterprise` feature flag.

## Building

```bash
# CLI + enterprise monitoring
cargo install gpu-roofline --features enterprise

# Full stack: enterprise + vGPU lifecycle + CUDA
cargo install gpu-roofline --features enterprise,vgpu,cuda
```

The `enterprise` feature adds ~2 MB to the binary (tiny-http for metrics server, ureq for webhook HTTP client).

## Prometheus Metrics

Start the monitor in daemon mode with the metrics endpoint:

```bash
gpu-roofline monitor --daemon --metrics-port 9835
```

Scrape `http://localhost:9835/metrics` from your Prometheus instance:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'gpu-roofline'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9835']
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `gpu_bandwidth_gbps` | gauge | Memory bandwidth in GB/s |
| `gpu_gflops` | gauge | Compute throughput in GFLOP/s |
| `gpu_temperature_celsius` | gauge | GPU temperature |
| `gpu_clock_mhz` | gauge | GPU clock speed |
| `gpu_power_watts` | gauge | Power draw |
| `gpu_utilization_percent` | gauge | GPU utilization |
| `gpu_memory_used_bytes` | gauge | VRAM used |
| `gpu_memory_total_bytes` | gauge | VRAM total |
| `gpu_cv` | gauge | Measurement stability (coefficient of variation) |
| `gpu_alerts_total{level}` | counter | Cumulative alerts by severity |
| `gpu_vgpu_active_count` | gauge | Active vGPU instances |
| `gpu_vgpu_vram_allocated_bytes` | gauge | Total vGPU VRAM allocated |
| `gpu_vgpu_vram_available_bytes` | gauge | Total vGPU VRAM available |
| `gpu_vgpu_events_total{type}` | counter | vGPU lifecycle events (created/destroyed) |

### Health Endpoint

`http://localhost:9835/health` returns `200 OK` — use for Kubernetes liveness/readiness probes.

## Webhook Alerts

POST JSON alerts to one or more URLs when alert conditions fire:

```bash
# Single webhook
gpu-roofline monitor --daemon --webhook-url https://hooks.slack.com/services/T.../B.../xxx

# Multiple webhooks
gpu-roofline monitor --daemon \
  --webhook-url https://hooks.slack.com/services/xxx \
  --webhook-url https://events.pagerduty.com/integration/xxx/enqueue
```

### Payload Format

```json
{
  "source": "gpu-roofline",
  "timestamp": "2026-03-18T12:00:00Z",
  "device": "NVIDIA H100 SXM",
  "alert": {
    "level": "critical",
    "rule": "BandwidthDrop",
    "message": "Bandwidth 1450 GB/s is 50% of baseline (2900 GB/s)"
  }
}
```

Webhooks fire on alert only (not every sample). Delivery is fire-and-forget — if a POST fails, it's logged and skipped. The measurement loop is never blocked by webhook latency.

## Grafana Dashboard

Import `deploy/grafana/gpu-roofline-dashboard.json` into Grafana:

1. Open Grafana → Dashboards → Import
2. Upload `gpu-roofline-dashboard.json`
3. Select your Prometheus datasource
4. Dashboard appears with 4 rows: Health Overview, Performance, Telemetry, vGPU Lifecycle

The vGPU row is collapsed by default — expand it when monitoring vGPU-enabled systems.

## Kubernetes Deployment

Deploy as a DaemonSet on GPU nodes:

```bash
# Create namespace
kubectl create namespace gpu-monitoring

# Deploy
kubectl apply -f deploy/k8s/
```

This creates:
- **DaemonSet**: Runs on every node with `nvidia.com/gpu.present=true`
- **Service**: ClusterIP exposing port 9835 for Prometheus scraping
- **ConfigMap**: Alert thresholds and webhook configuration
- **ServiceMonitor**: Auto-discovery for Prometheus Operator

### Configuration

Edit `deploy/k8s/configmap.yaml` to set webhook URLs and thresholds before deploying.

### Requirements

- NVIDIA GPU Operator or device plugin installed (provides the `nvidia.com/gpu.present` node label)
- Container needs access to NVIDIA devices (`NVIDIA_VISIBLE_DEVICES=all`)
- Prometheus Operator (optional, for ServiceMonitor auto-discovery)

## Troubleshooting

**Metrics endpoint not responding**: Check that port 9835 is not blocked by firewall. On Windows, the first run may trigger a firewall dialog.

**Webhook not delivering**: Check logs for `webhook: POST to <url> failed` messages. Common causes: wrong URL, network policy blocking egress, authentication required.

**K8s pod in CrashLoopBackOff**: The startup probe allows 50 seconds for the initial baseline measurement. If your GPU is under heavy load, the baseline may take longer. Increase `failureThreshold` in the startup probe.

**No GPU detected in K8s**: Verify `NVIDIA_VISIBLE_DEVICES=all` is set and the NVIDIA device plugin is running on the node. Check `kubectl describe pod` for device allocation issues.
