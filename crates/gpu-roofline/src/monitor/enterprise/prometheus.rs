//! Prometheus metrics endpoint — serves `/metrics` in text exposition format.
//!
//! Runs a tiny-http server in a background thread. The `MetricsSnapshot`
//! is updated from the monitoring callback; the HTTP handler reads it
//! and renders Prometheus-compatible text on each scrape.

use std::sync::{Arc, Mutex};

/// Latest metric values, updated atomically from the monitoring loop.
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    // GPU gauges
    pub bandwidth_gbps: f64,
    pub gflops: f64,
    pub temperature_c: u32,
    pub clock_mhz: u32,
    pub power_watts: f32,
    pub utilization_pct: f32,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub cv: f64,

    // Alert counters (cumulative)
    pub alerts_warning_total: u64,
    pub alerts_critical_total: u64,

    // vGPU gauges
    pub vgpu_active_count: u32,
    pub vgpu_vram_allocated_bytes: u64,
    pub vgpu_vram_available_bytes: u64,
    pub vgpu_events_created: u64,
    pub vgpu_events_destroyed: u64,
}

/// Render the current snapshot as Prometheus text exposition format.
pub fn render_metrics(snap: &MetricsSnapshot) -> String {
    let mut out = String::with_capacity(2048);

    // GPU gauges
    push_gauge(
        &mut out,
        "gpu_bandwidth_gbps",
        "Current memory bandwidth in GB/s",
        snap.bandwidth_gbps,
    );
    push_gauge(
        &mut out,
        "gpu_gflops",
        "Current compute throughput in GFLOP/s",
        snap.gflops,
    );
    push_gauge(
        &mut out,
        "gpu_temperature_celsius",
        "GPU temperature in degrees Celsius",
        snap.temperature_c as f64,
    );
    push_gauge(
        &mut out,
        "gpu_clock_mhz",
        "GPU clock speed in MHz",
        snap.clock_mhz as f64,
    );
    push_gauge(
        &mut out,
        "gpu_power_watts",
        "GPU power draw in watts",
        snap.power_watts as f64,
    );
    push_gauge(
        &mut out,
        "gpu_utilization_percent",
        "GPU utilization percentage",
        snap.utilization_pct as f64,
    );
    push_gauge(
        &mut out,
        "gpu_memory_used_bytes",
        "GPU memory used in bytes",
        snap.memory_used_bytes as f64,
    );
    push_gauge(
        &mut out,
        "gpu_memory_total_bytes",
        "GPU total memory in bytes",
        snap.memory_total_bytes as f64,
    );
    push_gauge(
        &mut out,
        "gpu_cv",
        "Measurement coefficient of variation",
        snap.cv,
    );

    // Alert counters
    out.push_str("# HELP gpu_alerts_total Cumulative count of alerts fired\n");
    out.push_str("# TYPE gpu_alerts_total counter\n");
    out.push_str(&format!(
        "gpu_alerts_total{{level=\"warning\"}} {}\n",
        snap.alerts_warning_total
    ));
    out.push_str(&format!(
        "gpu_alerts_total{{level=\"critical\"}} {}\n",
        snap.alerts_critical_total
    ));

    // vGPU gauges
    push_gauge(
        &mut out,
        "gpu_vgpu_active_count",
        "Number of active vGPU instances",
        snap.vgpu_active_count as f64,
    );
    push_gauge(
        &mut out,
        "gpu_vgpu_vram_allocated_bytes",
        "Total vGPU VRAM allocated in bytes",
        snap.vgpu_vram_allocated_bytes as f64,
    );
    push_gauge(
        &mut out,
        "gpu_vgpu_vram_available_bytes",
        "Total vGPU VRAM available in bytes",
        snap.vgpu_vram_available_bytes as f64,
    );

    // vGPU event counters
    out.push_str("# HELP gpu_vgpu_events_total Cumulative count of vGPU lifecycle events\n");
    out.push_str("# TYPE gpu_vgpu_events_total counter\n");
    out.push_str(&format!(
        "gpu_vgpu_events_total{{type=\"created\"}} {}\n",
        snap.vgpu_events_created
    ));
    out.push_str(&format!(
        "gpu_vgpu_events_total{{type=\"destroyed\"}} {}\n",
        snap.vgpu_events_destroyed
    ));

    out
}

fn push_gauge(out: &mut String, name: &str, help: &str, value: f64) {
    out.push_str(&format!("# HELP {name} {help}\n"));
    out.push_str(&format!("# TYPE {name} gauge\n"));
    out.push_str(&format!("{name} {value}\n"));
}

/// Start the HTTP metrics server in a background thread.
///
/// Returns the thread handle. The server runs until the process exits.
pub fn start_metrics_server(
    port: u16,
    registry: Arc<Mutex<MetricsSnapshot>>,
) -> Result<std::thread::JoinHandle<()>, String> {
    let addr = format!("0.0.0.0:{port}");
    let server = tiny_http::Server::http(&addr)
        .map_err(|e| format!("Failed to bind metrics server on {addr}: {e}"))?;

    tracing::info!("Prometheus metrics available at http://{addr}/metrics");
    tracing::info!("Health endpoint at http://{addr}/health");

    let handle = std::thread::spawn(move || {
        for request in server.incoming_requests() {
            let response = match request.url() {
                "/metrics" => {
                    let body = if let Ok(snap) = registry.lock() {
                        render_metrics(&snap)
                    } else {
                        "# error: metrics lock poisoned\n".to_string()
                    };
                    tiny_http::Response::from_string(body).with_header(
                        "Content-Type: text/plain; version=0.0.4; charset=utf-8"
                            .parse::<tiny_http::Header>()
                            .unwrap(),
                    )
                }
                "/health" => tiny_http::Response::from_string("OK"),
                _ => tiny_http::Response::from_string("Not Found").with_status_code(404),
            };
            let _ = request.respond(response);
        }
    });

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_metrics_contains_all_gauges() {
        let snap = MetricsSnapshot {
            bandwidth_gbps: 2847.3,
            gflops: 59000.0,
            temperature_c: 72,
            clock_mhz: 2100,
            power_watts: 350.5,
            utilization_pct: 85.2,
            memory_used_bytes: 40_000_000_000,
            memory_total_bytes: 80_000_000_000,
            cv: 0.03,
            alerts_warning_total: 5,
            alerts_critical_total: 1,
            vgpu_active_count: 3,
            vgpu_vram_allocated_bytes: 30_000_000_000,
            vgpu_vram_available_bytes: 80_000_000_000,
            vgpu_events_created: 10,
            vgpu_events_destroyed: 7,
        };

        let output = render_metrics(&snap);

        // Verify all expected metrics present
        assert!(output.contains("gpu_bandwidth_gbps 2847.3"));
        assert!(output.contains("gpu_gflops 59000"));
        assert!(output.contains("gpu_temperature_celsius 72"));
        assert!(output.contains("gpu_clock_mhz 2100"));
        assert!(output.contains("gpu_power_watts 350.5"));
        assert!(output.contains("gpu_utilization_percent 85."));
        assert!(output.contains("gpu_memory_used_bytes 40000000000"));
        assert!(output.contains("gpu_memory_total_bytes 80000000000"));
        assert!(output.contains("gpu_cv 0.03"));
        assert!(output.contains("gpu_alerts_total{level=\"warning\"} 5"));
        assert!(output.contains("gpu_alerts_total{level=\"critical\"} 1"));
        assert!(output.contains("gpu_vgpu_active_count 3"));
        assert!(output.contains("gpu_vgpu_events_total{type=\"created\"} 10"));
        assert!(output.contains("gpu_vgpu_events_total{type=\"destroyed\"} 7"));

        // Verify TYPE annotations
        assert!(output.contains("# TYPE gpu_bandwidth_gbps gauge"));
        assert!(output.contains("# TYPE gpu_alerts_total counter"));
        assert!(output.contains("# TYPE gpu_vgpu_events_total counter"));
    }

    #[test]
    fn test_render_metrics_default_snapshot() {
        let snap = MetricsSnapshot::default();
        let output = render_metrics(&snap);
        // Should render without panic even with all zeros
        assert!(output.contains("gpu_bandwidth_gbps 0"));
        assert!(output.contains("gpu_alerts_total{level=\"warning\"} 0"));
    }

    #[test]
    fn test_metrics_server_starts_and_responds() {
        let registry = Arc::new(Mutex::new(MetricsSnapshot {
            bandwidth_gbps: 1234.5,
            ..Default::default()
        }));

        // Find a free port
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let _handle = start_metrics_server(port, Arc::clone(&registry)).unwrap();

        // Give server time to start
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Test /metrics endpoint
        let resp = ureq::get(&format!("http://127.0.0.1:{port}/metrics"))
            .call()
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.into_string().unwrap();
        assert!(body.contains("gpu_bandwidth_gbps 1234.5"));

        // Test /health endpoint
        let resp = ureq::get(&format!("http://127.0.0.1:{port}/health"))
            .call()
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
}
