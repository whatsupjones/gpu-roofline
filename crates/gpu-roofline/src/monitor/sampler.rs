//! Periodic performance sampling for continuous monitoring.

use std::time::{Duration, Instant};

use gpu_harness::backend::DeviceState;
use gpu_harness::GpuBackend;

use crate::ceilings::{measure_roofline, MeasureConfig};
use crate::model::RooflineModel;

use super::alerting::{Alert, AlertEngine};

/// Configuration for the monitor sampling loop.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Seconds between samples.
    pub interval_secs: u64,
    /// Total duration to monitor (0 = indefinite).
    pub duration_secs: u64,
    /// Alert if performance drops below this fraction of baseline (0.0-1.0).
    pub alert_threshold: f64,
    /// Buffer size for measurement (should exceed L2 cache).
    pub buffer_size_bytes: usize,
    /// Number of measurement iterations per sample (low = fast, noisy).
    pub iterations_per_sample: u32,
    /// Write JSON log to this path (None = stdout only).
    pub log_path: Option<String>,
    /// Run in daemon mode (no interactive output, JSON-only logging).
    pub daemon: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            interval_secs: 60,
            duration_secs: 0, // Indefinite
            alert_threshold: 0.8,
            buffer_size_bytes: 256 * 1024 * 1024, // 256 MB
            iterations_per_sample: 10,
            log_path: None,
            daemon: false,
        }
    }
}

/// Status of a single monitoring sample.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status")]
pub enum SampleStatus {
    #[serde(rename = "normal")]
    Normal,
    #[serde(rename = "warning")]
    Warning { reason: String },
    #[serde(rename = "alert")]
    Alert { reason: String },
}

/// A single monitoring sample with performance + telemetry data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MonitorSample {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub sample_index: u32,
    pub bandwidth_gbps: f64,
    pub gflops: f64,
    pub cv: f64,
    pub temperature_c: u32,
    pub clock_mhz: u32,
    pub power_watts: f32,
    pub utilization_pct: f32,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub status: SampleStatus,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub alerts: Vec<Alert>,
}

/// The monitoring sampler — runs periodic measurements and checks for degradation.
pub struct Sampler {
    config: MonitorConfig,
    alert_engine: AlertEngine,
    history: Vec<MonitorSample>,
    baseline_bw: f64,
    baseline_gflops: f64,
}

impl Sampler {
    /// Create a sampler with a baseline to compare against.
    pub fn new(config: MonitorConfig, baseline: &RooflineModel) -> Self {
        let alert_engine = AlertEngine::new(config.alert_threshold);
        Self {
            config,
            alert_engine,
            history: Vec::new(),
            baseline_bw: baseline.peak_bandwidth_gbps,
            baseline_gflops: baseline.peak_gflops,
        }
    }

    /// Run the monitoring loop. Blocks until duration expires or interrupted.
    ///
    /// Calls `on_sample` after each measurement. Return `false` from the
    /// callback to stop monitoring early.
    pub fn run<F>(
        &mut self,
        backend: &dyn GpuBackend,
        mut on_sample: F,
    ) -> Result<Vec<MonitorSample>, gpu_harness::HarnessError>
    where
        F: FnMut(&MonitorSample) -> bool,
    {
        let start = Instant::now();
        let duration = if self.config.duration_secs > 0 {
            Some(Duration::from_secs(self.config.duration_secs))
        } else {
            None
        };
        let interval = Duration::from_secs(self.config.interval_secs);

        let measure_config = MeasureConfig {
            buffer_size_bytes: self.config.buffer_size_bytes,
            measurement_iterations: self.config.iterations_per_sample,
            warmup_iterations: 2,
            kernels: vec![
                crate::kernels::BuiltinKernel::Copy,
                crate::kernels::BuiltinKernel::FmaHeavy,
            ],
            device_index: 0,
        };

        let mut sample_index = 0u32;

        loop {
            // Check duration limit
            if let Some(dur) = duration {
                if start.elapsed() >= dur {
                    break;
                }
            }

            let sample_start = Instant::now();

            // Graceful degradation: check GPU load before measuring.
            // If GPU is under heavy load (gaming, training), reduce our
            // measurement footprint to avoid impacting the user's workload.
            let pre_state = backend.device_state(0).ok();
            let gpu_busy = pre_state
                .as_ref()
                .map(|s| s.utilization_pct > 80.0)
                .unwrap_or(false);

            let active_config = if gpu_busy {
                // Light touch: fewer iterations, smaller buffer
                MeasureConfig {
                    buffer_size_bytes: self.config.buffer_size_bytes / 4,
                    measurement_iterations: 3.max(self.config.iterations_per_sample / 4),
                    warmup_iterations: 1,
                    kernels: vec![crate::kernels::BuiltinKernel::Copy],
                    device_index: 0,
                }
            } else {
                measure_config.clone()
            };

            // Run measurement (lightweight when GPU is busy)
            let roofline = measure_roofline(backend, &active_config)?;

            // Get device telemetry
            let device_state = backend.device_state(0).unwrap_or(DeviceState {
                clock_mhz: 0,
                temperature_c: 0,
                power_watts: 0.0,
                memory_used_bytes: 0,
                memory_total_bytes: 0,
                utilization_pct: 0.0,
            });

            // Find copy kernel CV for stability
            let copy_cv = roofline
                .placements
                .iter()
                .find(|p| p.name == "copy")
                .map(|p| p.cv)
                .unwrap_or(0.0);

            // Run alerting checks
            let alerts = self.alert_engine.check(
                &roofline,
                &device_state,
                self.baseline_bw,
                self.baseline_gflops,
                &self.history,
            );

            // Determine overall status from alerts
            let status = if alerts
                .iter()
                .any(|a| a.level == super::alerting::AlertLevel::Critical)
            {
                SampleStatus::Alert {
                    reason: alerts
                        .iter()
                        .filter(|a| a.level == super::alerting::AlertLevel::Critical)
                        .map(|a| a.message.clone())
                        .collect::<Vec<_>>()
                        .join("; "),
                }
            } else if !alerts.is_empty() {
                SampleStatus::Warning {
                    reason: alerts
                        .iter()
                        .map(|a| a.message.clone())
                        .collect::<Vec<_>>()
                        .join("; "),
                }
            } else {
                SampleStatus::Normal
            };

            let sample = MonitorSample {
                timestamp: chrono::Utc::now(),
                sample_index,
                bandwidth_gbps: roofline.peak_bandwidth_gbps,
                gflops: roofline.peak_gflops,
                cv: copy_cv,
                temperature_c: device_state.temperature_c,
                clock_mhz: device_state.clock_mhz,
                power_watts: device_state.power_watts,
                utilization_pct: device_state.utilization_pct,
                memory_used_bytes: device_state.memory_used_bytes,
                memory_total_bytes: device_state.memory_total_bytes,
                status,
                alerts,
            };

            self.history.push(sample.clone());

            if !on_sample(&sample) {
                break;
            }

            sample_index += 1;

            // Sleep until next interval (subtract measurement time)
            let elapsed = sample_start.elapsed();
            if elapsed < interval {
                std::thread::sleep(interval - elapsed);
            }
        }

        Ok(self.history.clone())
    }

    /// Get all collected samples.
    pub fn history(&self) -> &[MonitorSample] {
        &self.history
    }
}
