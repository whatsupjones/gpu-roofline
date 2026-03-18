use gpu_harness::error::HarnessError;
use gpu_harness::{GpuBackend, KernelResult, RunConfig};

use crate::kernels::{BuiltinKernel, KernelDefinition};
use crate::model::RooflineModel;

/// Configuration for a roofline measurement session.
#[derive(Debug, Clone)]
pub struct MeasureConfig {
    /// Buffer size for kernel execution (bytes).
    /// Must exceed L2 cache to measure HBM/DRAM bandwidth.
    /// Default: 256 MB (exceeds all current GPU L2 caches).
    pub buffer_size_bytes: usize,
    /// Warm-up iterations before measurement. Default: 10.
    pub warmup_iterations: u32,
    /// Measurement iterations per kernel. Default: 100.
    pub measurement_iterations: u32,
    /// Which kernels to run. Default: all built-in.
    pub kernels: Vec<BuiltinKernel>,
    /// Device index to measure. Default: 0.
    pub device_index: u32,
}

impl Default for MeasureConfig {
    fn default() -> Self {
        Self {
            // 256 MB: exceeds L2 cache on all current GPUs
            // (H100: 50MB L2, RTX 4090: 72MB L2, MI300X: 256MB Infinity Cache)
            // This ensures we measure HBM/DRAM bandwidth, not L2.
            buffer_size_bytes: 256 * 1024 * 1024,
            warmup_iterations: 10,
            measurement_iterations: 100,
            kernels: BuiltinKernel::all().to_vec(),
            device_index: 0,
        }
    }
}

/// Run all configured kernels and build a static roofline model.
///
/// This measures burst performance (single snapshot, no thermal tracking).
/// For dynamic roofline with tension analysis, use `measure_dynamic`.
pub fn measure_roofline(
    backend: &dyn GpuBackend,
    config: &MeasureConfig,
) -> Result<RooflineModel, HarnessError> {
    let devices = backend.discover_devices()?;
    let device = devices
        .get(config.device_index as usize)
        .ok_or(HarnessError::DeviceIndexOutOfRange(config.device_index))?;

    let run_config = RunConfig {
        warmup_iterations: config.warmup_iterations,
        measurement_iterations: config.measurement_iterations,
        buffer_size_bytes: config.buffer_size_bytes,
    };

    // Run each kernel and collect measurements
    let mut measurements: Vec<(KernelDefinition, KernelResult)> = Vec::new();

    for kernel in &config.kernels {
        let spec = kernel.to_spec(config.buffer_size_bytes);
        let result = backend.run_kernel(&spec, &run_config)?;
        measurements.push((kernel.definition(), result));
    }

    // Get device state for metadata
    let state = backend.device_state(config.device_index)?;

    Ok(RooflineModel::from_measurements(
        device.name.clone(),
        &measurements,
        state.clock_mhz,
        state.temperature_c,
        state.power_watts,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::sim::{profiles, SimulatedBackend};

    #[test]
    fn test_measure_roofline_sim() {
        let backend = SimulatedBackend::new(profiles::rtx_5090());
        let config = MeasureConfig {
            measurement_iterations: 50,
            ..Default::default()
        };

        let model = measure_roofline(&backend, &config).unwrap();

        assert!(model.device_name.contains("RTX 5090"));
        assert!(model.peak_bandwidth_gbps > 0.0, "should measure bandwidth");
        assert!(model.peak_gflops > 0.0, "should measure FLOPS");
        assert!(model.ridge_point > 0.0, "should compute ridge point");
        assert_eq!(model.placements.len(), 7, "should have all 7 kernels");

        // Verify bandwidth kernels are memory-bound
        for p in &model.placements {
            if p.arithmetic_intensity < 1.0 {
                assert!(
                    matches!(p.bottleneck, crate::model::Bottleneck::MemoryBound { .. }),
                    "{} should be memory-bound",
                    p.name
                );
            }
        }
    }

    #[test]
    fn test_measure_roofline_h100() {
        let backend = SimulatedBackend::new(profiles::h100_sxm());
        let config = MeasureConfig {
            measurement_iterations: 20,
            ..Default::default()
        };

        let model = measure_roofline(&backend, &config).unwrap();

        assert!(model.device_name.contains("H100"));
        assert!(
            model.peak_bandwidth_gbps > 1000.0,
            "H100 should have >1000 GB/s bandwidth, got {:.0}",
            model.peak_bandwidth_gbps
        );
    }

    #[test]
    fn test_measure_specific_kernels() {
        let backend = SimulatedBackend::new(profiles::rtx_4090());
        let config = MeasureConfig {
            kernels: vec![BuiltinKernel::Copy, BuiltinKernel::FmaHeavy],
            measurement_iterations: 20,
            ..Default::default()
        };

        let model = measure_roofline(&backend, &config).unwrap();
        assert_eq!(model.placements.len(), 2);
        assert_eq!(model.placements[0].name, "copy");
        assert_eq!(model.placements[1].name, "fma_heavy");
    }

    #[test]
    fn test_different_gpus_different_ceilings() {
        let config = MeasureConfig {
            measurement_iterations: 20,
            ..Default::default()
        };

        let rtx = SimulatedBackend::new(profiles::rtx_5090());
        let arc = SimulatedBackend::new(profiles::arc_a770());

        let model_rtx = measure_roofline(&rtx, &config).unwrap();
        let model_arc = measure_roofline(&arc, &config).unwrap();

        assert!(
            model_rtx.peak_bandwidth_gbps > model_arc.peak_bandwidth_gbps,
            "RTX 5090 ({:.0} GB/s) should have higher BW than Arc A770 ({:.0} GB/s)",
            model_rtx.peak_bandwidth_gbps,
            model_arc.peak_bandwidth_gbps
        );
    }
}
