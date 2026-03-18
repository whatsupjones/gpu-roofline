use serde::{Deserialize, Serialize};

/// A measured tension between two competing GPU forces.
///
/// Tensions model how real GPU performance deviates from the static
/// roofline. Each tension has an excitatory force (wants to maximize
/// performance) and an inhibitory force (pushes performance down).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensionMeasurement {
    /// Tension name (e.g., "thermal", "power", "contention").
    pub name: String,
    /// Excitatory force description.
    pub force_a: String,
    /// Inhibitory force description.
    pub force_b: String,
    /// Percentage drop in ceiling caused by this tension.
    /// Negative means performance decreased (the common case).
    pub ceiling_delta_pct: f64,
    /// Time in seconds when this tension becomes dominant.
    pub onset_time_secs: f64,
}

/// A single time-series sample during dynamic roofline measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSample {
    /// Elapsed seconds since measurement start.
    pub elapsed_secs: f64,
    /// GPU clock speed at this sample (MHz).
    pub clock_mhz: u32,
    /// GPU temperature at this sample (Celsius).
    pub temperature_c: u32,
    /// GPU power draw at this sample (Watts).
    pub power_watts: f32,
    /// Measured GFLOP/s from the kernel running at this sample.
    pub measured_gflops: f64,
    /// Measured memory bandwidth (GB/s) at this sample.
    pub measured_bandwidth_gbps: f64,
}

/// Configuration for dynamic roofline measurement.
#[derive(Debug, Clone)]
pub struct DynamicConfig {
    /// Total measurement duration in seconds. Default: 120.
    pub duration_secs: u64,
    /// Coefficient of variation threshold for equilibrium detection.
    /// When CV drops below this over the stability window, we've reached
    /// steady state. Default: 0.02 (2%).
    pub equilibrium_cv_threshold: f64,
    /// Rolling window size for equilibrium detection (seconds). Default: 10.
    pub stability_window_secs: f64,
    /// How often to sample device state (milliseconds). Default: 500.
    pub sample_interval_ms: u64,
    /// Buffer size for kernel execution. Default: 256 MB (exceeds L2 cache).
    pub buffer_size_bytes: usize,
    /// Measurement iterations per sample. Default: 20.
    pub iterations_per_sample: u32,
    /// Whether to also measure under concurrent load. Default: false.
    pub include_contention: bool,
}

impl Default for DynamicConfig {
    fn default() -> Self {
        Self {
            duration_secs: 120,
            equilibrium_cv_threshold: 0.02,
            stability_window_secs: 10.0,
            sample_interval_ms: 500,
            buffer_size_bytes: 256 * 1024 * 1024,
            iterations_per_sample: 20,
            include_contention: false,
        }
    }
}

impl DynamicConfig {
    /// Quick measurement for testing (~10 seconds).
    pub fn quick() -> Self {
        Self {
            duration_secs: 10,
            stability_window_secs: 3.0,
            sample_interval_ms: 200,
            iterations_per_sample: 10,
            ..Default::default()
        }
    }
}
