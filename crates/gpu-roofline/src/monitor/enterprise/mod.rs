//! Enterprise integration: Prometheus metrics endpoint and webhook alerts.
//!
//! Feature-gated behind `enterprise`. Adds an HTTP `/metrics` endpoint
//! for Prometheus scraping and webhook POST on alert conditions.

pub mod prometheus;
pub mod webhook;

use std::sync::{Arc, Mutex};

use super::alerting::Alert;
use super::MonitorSample;

/// Configuration for enterprise integrations.
#[derive(Debug, Clone)]
pub struct EnterpriseConfig {
    /// Port for Prometheus metrics HTTP server (default: 9835).
    pub metrics_port: u16,
    /// Webhook URLs to POST alerts to.
    pub webhook_urls: Vec<String>,
    /// Device name for metric labels.
    pub device_name: String,
}

impl Default for EnterpriseConfig {
    fn default() -> Self {
        Self {
            metrics_port: 9835,
            webhook_urls: Vec::new(),
            device_name: "unknown".to_string(),
        }
    }
}

/// Handle to running enterprise services (metrics server + webhook dispatcher).
pub struct EnterpriseHandle {
    registry: Arc<Mutex<prometheus::MetricsSnapshot>>,
    webhook_tx: Option<std::sync::mpsc::SyncSender<webhook::WebhookPayload>>,
    device_name: String,
    _server_handle: Option<std::thread::JoinHandle<()>>,
    _webhook_handle: Option<std::thread::JoinHandle<()>>,
}

impl EnterpriseHandle {
    /// Start enterprise services based on config.
    pub fn start(config: &EnterpriseConfig) -> Result<Self, String> {
        let registry = Arc::new(Mutex::new(prometheus::MetricsSnapshot::default()));

        // Start Prometheus HTTP server
        let server_handle =
            prometheus::start_metrics_server(config.metrics_port, Arc::clone(&registry))?;

        // Start webhook dispatcher if URLs configured
        let (webhook_tx, webhook_handle) = if !config.webhook_urls.is_empty() {
            let (tx, handle) = webhook::start_dispatcher(config.webhook_urls.clone());
            (Some(tx), Some(handle))
        } else {
            (None, None)
        };

        Ok(Self {
            registry,
            webhook_tx,
            device_name: config.device_name.clone(),
            _server_handle: Some(server_handle),
            _webhook_handle: webhook_handle,
        })
    }

    /// Update metrics from a monitoring sample.
    pub fn update_metrics(&self, sample: &MonitorSample) {
        if let Ok(mut snap) = self.registry.lock() {
            snap.bandwidth_gbps = sample.bandwidth_gbps;
            snap.gflops = sample.gflops;
            snap.temperature_c = sample.temperature_c;
            snap.clock_mhz = sample.clock_mhz;
            snap.power_watts = sample.power_watts;
            snap.utilization_pct = sample.utilization_pct;
            snap.memory_used_bytes = sample.memory_used_bytes;
            snap.memory_total_bytes = sample.memory_total_bytes;
            snap.cv = sample.cv;
            for alert in &sample.alerts {
                match alert.level {
                    super::alerting::AlertLevel::Warning => snap.alerts_warning_total += 1,
                    super::alerting::AlertLevel::Critical => snap.alerts_critical_total += 1,
                }
            }
        }
    }

    /// Update vGPU metrics.
    #[cfg(feature = "vgpu")]
    pub fn update_vgpu_metrics(&self, active_count: u32, vram_allocated: u64, vram_available: u64) {
        if let Ok(mut snap) = self.registry.lock() {
            snap.vgpu_active_count = active_count;
            snap.vgpu_vram_allocated_bytes = vram_allocated;
            snap.vgpu_vram_available_bytes = vram_available;
        }
    }

    /// Increment vGPU event counter.
    #[cfg(feature = "vgpu")]
    pub fn increment_vgpu_event(&self, event_type: &str) {
        if let Ok(mut snap) = self.registry.lock() {
            match event_type {
                "created" => snap.vgpu_events_created += 1,
                "destroyed" => snap.vgpu_events_destroyed += 1,
                _ => {}
            }
        }
    }

    /// Dispatch alerts to configured webhooks.
    pub fn dispatch_alerts(&self, alerts: &[Alert]) {
        if let Some(ref tx) = self.webhook_tx {
            for alert in alerts {
                let payload = webhook::WebhookPayload {
                    source: "gpu-roofline".to_string(),
                    timestamp: chrono::Utc::now(),
                    device: self.device_name.clone(),
                    alert: webhook::WebhookAlert {
                        level: format!("{:?}", alert.level).to_lowercase(),
                        rule: format!("{:?}", alert.rule),
                        message: alert.message.clone(),
                    },
                };
                // Non-blocking send — drop if channel full
                let _ = tx.try_send(payload);
            }
        }
    }
}
