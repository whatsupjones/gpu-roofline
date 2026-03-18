use gpu_harness::device::GpuDevice;

use super::baselines::{find_baseline, HardwareBaseline};
use crate::ceilings::MeasureConfig;
use crate::kernels::BuiltinKernel;

/// Generate an optimized measurement configuration for a detected GPU.
///
/// Auto-sizes buffers to exceed L2 cache, adjusts iteration count
/// based on expected noise level, and selects appropriate warmup.
pub fn adaptive_config(device: &GpuDevice) -> MeasureConfig {
    match find_baseline(device) {
        Some(baseline) => config_from_baseline(baseline),
        None => {
            // Unknown GPU — use conservative defaults
            tracing::warn!(
                "No baseline found for '{}' — using conservative defaults (256MB buffer, 100 iterations)",
                device.name
            );
            MeasureConfig::default()
        }
    }
}

/// Build a MeasureConfig from a hardware baseline.
fn config_from_baseline(baseline: &HardwareBaseline) -> MeasureConfig {
    // Buffer must exceed L2 to measure HBM/DRAM, not L2 cache
    let buffer_size = baseline.recommended_buffer_bytes;

    // Higher-bandwidth GPUs need fewer iterations (less noise per sample)
    // Lower-bandwidth GPUs need more iterations for statistical stability
    let measurement_iterations = if baseline.bandwidth_range.1 > 3000.0 {
        50 // Datacenter HBM3/3e — very stable
    } else if baseline.bandwidth_range.1 > 500.0 {
        100 // Consumer discrete — moderate noise
    } else {
        200 // Integrated — high noise
    };

    // Warmup scales with GPU class (datacenter GPUs warm up faster)
    let warmup_iterations = if baseline.flops_range.1 > 50.0 {
        5 // Datacenter
    } else if baseline.flops_range.1 > 10.0 {
        10 // Consumer discrete
    } else {
        20 // Integrated
    };

    MeasureConfig {
        buffer_size_bytes: buffer_size,
        warmup_iterations,
        measurement_iterations,
        kernels: BuiltinKernel::all().to_vec(),
        device_index: 0,
    }
}

/// Log the adaptive configuration that was selected.
pub fn log_adaptive_config(device: &GpuDevice, config: &MeasureConfig) {
    let baseline = find_baseline(device);
    let l2_info = baseline
        .map(|b| format!("L2: {}MB", b.l2_cache_mb))
        .unwrap_or_else(|| "L2: unknown".to_string());

    eprintln!(
        "  Config: {}MB buffer ({}) | {} iterations | {} warmup",
        config.buffer_size_bytes / (1024 * 1024),
        l2_info,
        config.measurement_iterations,
        config.warmup_iterations,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::device::{GpuArchitecture, GpuFeatures, GpuLimits, GpuVendor};

    fn mock_device(name: &str, cc: Option<(u32, u32)>) -> GpuDevice {
        GpuDevice {
            index: 0,
            name: name.to_string(),
            vendor: GpuVendor::Nvidia,
            architecture: GpuArchitecture::Hopper,
            memory_bytes: 80 * 1024 * 1024 * 1024,
            pci_bus_id: None,
            driver_version: None,
            features: GpuFeatures {
                compute_capability: cc,
                ..Default::default()
            },
            limits: GpuLimits::default(),
        }
    }

    #[test]
    fn test_h100_gets_256mb_buffer() {
        let device = mock_device("NVIDIA H100 80GB HBM3", Some((9, 0)));
        let config = adaptive_config(&device);
        assert_eq!(config.buffer_size_bytes, 256 * 1024 * 1024);
        assert_eq!(config.measurement_iterations, 50); // High-BW datacenter
    }

    #[test]
    fn test_rtx4090_gets_256mb_buffer() {
        let device = mock_device("NVIDIA GeForce RTX 4090", Some((8, 9)));
        let config = adaptive_config(&device);
        assert_eq!(config.buffer_size_bytes, 256 * 1024 * 1024);
        assert_eq!(config.measurement_iterations, 100); // Consumer discrete
    }

    #[test]
    fn test_integrated_gets_small_buffer() {
        let mut device = mock_device("Intel(R) UHD Graphics", None);
        device.vendor = GpuVendor::Intel;
        device.architecture = GpuArchitecture::Integrated;
        let config = adaptive_config(&device);
        assert_eq!(config.buffer_size_bytes, 16 * 1024 * 1024);
        assert_eq!(config.measurement_iterations, 200); // Integrated — noisy
    }

    #[test]
    fn test_mi300x_gets_1gb_buffer() {
        let mut device = mock_device("AMD Instinct MI300X", None);
        device.vendor = GpuVendor::Amd;
        let config = adaptive_config(&device);
        assert_eq!(config.buffer_size_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_unknown_gpu_gets_defaults() {
        let device = mock_device("Unknown Future GPU 9000", None);
        let config = adaptive_config(&device);
        assert_eq!(config.buffer_size_bytes, 256 * 1024 * 1024); // Default
    }

    #[test]
    fn test_buffer_exceeds_l2_for_all_known_gpus() {
        let test_devices = [
            ("NVIDIA H100 80GB HBM3", Some((9, 0))),
            ("NVIDIA GeForce RTX 4090", Some((8, 9))),
            ("NVIDIA GeForce RTX 3090", Some((8, 6))),
            ("AMD Instinct MI300X", None),
        ];

        for (name, cc) in test_devices {
            let device = mock_device(name, cc);
            let config = adaptive_config(&device);
            let baseline = find_baseline(&device).unwrap();
            assert!(
                config.buffer_size_bytes > baseline.l2_cache_mb as usize * 1024 * 1024,
                "{}: buffer {}MB should exceed L2 {}MB",
                name,
                config.buffer_size_bytes / (1024 * 1024),
                baseline.l2_cache_mb
            );
        }
    }
}
