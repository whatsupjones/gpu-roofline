use crate::device::GpuDevice;
use crate::error::HarnessError;
use serde::{Deserialize, Serialize};

/// Specification for a micro-kernel to execute on GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpec {
    pub name: String,
    pub working_set_bytes: usize,
    pub arithmetic_intensity: f64,
    pub iterations: u32,
}

/// Configuration for a kernel run.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub warmup_iterations: u32,
    pub measurement_iterations: u32,
    pub buffer_size_bytes: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            buffer_size_bytes: 16 * 1024 * 1024, // 16 MB
        }
    }
}

/// Result of a single kernel execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelResult {
    pub kernel_name: String,
    pub elapsed_us: Vec<f64>,
    pub bytes_processed: usize,
    pub flops_executed: u64,
}

impl KernelResult {
    /// Throughput in GB/s from median timing.
    pub fn bandwidth_gbps(&self) -> f64 {
        let median = self.median_us();
        if median <= 0.0 {
            return 0.0;
        }
        (self.bytes_processed as f64) / (median * 1e-6) / 1e9
    }

    /// Achieved GFLOP/s from median timing.
    pub fn gflops(&self) -> f64 {
        let median = self.median_us();
        if median <= 0.0 {
            return 0.0;
        }
        (self.flops_executed as f64) / (median * 1e-6) / 1e9
    }

    /// Median execution time in microseconds.
    pub fn median_us(&self) -> f64 {
        if self.elapsed_us.is_empty() {
            return 0.0;
        }
        let mut sorted = self.elapsed_us.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Mean execution time in microseconds.
    pub fn mean_us(&self) -> f64 {
        if self.elapsed_us.is_empty() {
            return 0.0;
        }
        self.elapsed_us.iter().sum::<f64>() / self.elapsed_us.len() as f64
    }

    /// Standard deviation of execution times in microseconds.
    pub fn stddev_us(&self) -> f64 {
        if self.elapsed_us.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_us();
        let variance = self.elapsed_us.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / (self.elapsed_us.len() - 1) as f64;
        variance.sqrt()
    }

    /// Coefficient of variation (stddev / mean). Lower = more stable.
    pub fn cv(&self) -> f64 {
        let mean = self.mean_us();
        if mean <= 0.0 {
            return 0.0;
        }
        self.stddev_us() / mean
    }
}

/// Instantaneous device state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceState {
    pub clock_mhz: u32,
    pub temperature_c: u32,
    pub power_watts: f32,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub utilization_pct: f32,
}

/// Result of a P2P bandwidth measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthResult {
    pub src_device: u32,
    pub dst_device: u32,
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
}

/// Core abstraction — all roofline/fleet code programs against this trait,
/// never against wgpu or NVML directly.
pub trait GpuBackend: Send + Sync {
    /// Run a micro-kernel and return timing + throughput.
    fn run_kernel(
        &self,
        kernel: &KernelSpec,
        config: &RunConfig,
    ) -> Result<KernelResult, HarnessError>;

    /// Query current device state (clock, temp, power).
    fn device_state(&self, device_index: u32) -> Result<DeviceState, HarnessError>;

    /// List all devices.
    fn discover_devices(&self) -> Result<Vec<GpuDevice>, HarnessError>;

    /// Measure P2P bandwidth between two devices.
    fn p2p_bandwidth(&self, src: u32, dst: u32) -> Result<BandwidthResult, HarnessError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_result_statistics() {
        let result = KernelResult {
            kernel_name: "copy".to_string(),
            elapsed_us: vec![100.0, 102.0, 98.0, 101.0, 99.0],
            bytes_processed: 16 * 1024 * 1024,
            flops_executed: 0,
        };

        assert!((result.mean_us() - 100.0).abs() < 1.0);
        assert!((result.median_us() - 100.0).abs() < 2.0);
        assert!(result.stddev_us() < 5.0);
        assert!(result.cv() < 0.05);
        assert!(result.bandwidth_gbps() > 100.0); // 16MB in ~100us
    }

    #[test]
    fn test_kernel_result_empty() {
        let result = KernelResult {
            kernel_name: "empty".to_string(),
            elapsed_us: vec![],
            bytes_processed: 0,
            flops_executed: 0,
        };

        assert_eq!(result.mean_us(), 0.0);
        assert_eq!(result.median_us(), 0.0);
        assert_eq!(result.stddev_us(), 0.0);
        assert_eq!(result.bandwidth_gbps(), 0.0);
    }
}
