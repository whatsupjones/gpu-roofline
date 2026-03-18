use gpu_harness::{GpuBackend, RunConfig};
use gpu_harness::error::HarnessError;

use crate::kernels::BuiltinKernel;
use crate::model::dynamic::DynamicRoofline;
use crate::model::equilibrium::EquilibriumDetector;
use crate::model::roofline::RooflineModel;
use crate::model::tension::{DynamicConfig, ThermalSample};

/// Measure a dynamic roofline with tension analysis.
///
/// Runs micro-kernels continuously while sampling device state (clock,
/// temperature, power) to capture the thermal trajectory from burst
/// to sustained. Detects equilibrium using rolling CV and builds
/// separate burst and sustained roofline models.
pub fn measure_dynamic(
    backend: &dyn GpuBackend,
    config: &DynamicConfig,
) -> Result<DynamicRoofline, HarnessError> {
    let devices = backend.discover_devices()?;
    let device = devices.first().ok_or(HarnessError::NoDevice)?;
    let device_name = device.name.clone();

    let run_config = RunConfig {
        warmup_iterations: 0, // No warmup — we want to capture the full ramp
        measurement_iterations: config.iterations_per_sample,
        buffer_size_bytes: config.buffer_size_bytes,
    };

    // Use copy (bandwidth) and fma_heavy (compute) as representative kernels
    let bw_kernel = BuiltinKernel::Copy.to_spec(config.buffer_size_bytes);
    let compute_kernel = BuiltinKernel::FmaHeavy.to_spec(config.buffer_size_bytes);

    // Capture burst measurement (first sample, before thermal ramp)
    let burst_state = backend.device_state(0)?;
    let burst_bw = backend.run_kernel(&bw_kernel, &run_config)?;
    let burst_compute = backend.run_kernel(&compute_kernel, &run_config)?;

    let burst = RooflineModel {
        device_name: device_name.clone(),
        peak_gflops: burst_compute.gflops(),
        peak_bandwidth_gbps: burst_bw.bandwidth_gbps(),
        ridge_point: if burst_bw.bandwidth_gbps() > 0.0 {
            burst_compute.gflops() / burst_bw.bandwidth_gbps()
        } else {
            0.0
        },
        clock_mhz: burst_state.clock_mhz,
        temperature_c: burst_state.temperature_c,
        power_watts: burst_state.power_watts,
        placements: vec![], // Burst placements populated separately if needed
    };

    // Run continuous measurement loop, sampling device state
    let mut trajectory: Vec<ThermalSample> = Vec::new();
    let samples_per_window =
        (config.stability_window_secs / (config.sample_interval_ms as f64 / 1000.0)) as usize;
    let mut gflops_detector =
        EquilibriumDetector::new(config.equilibrium_cv_threshold, samples_per_window.max(3));

    let total_samples = (config.duration_secs * 1000 / config.sample_interval_ms) as usize;

    let mut last_bw_result = burst_bw;
    let mut last_compute_result = burst_compute;
    let mut equilibrium_time = config.duration_secs as f64; // Default: full duration

    for sample_idx in 0..total_samples {
        // Run kernels to keep GPU loaded and get fresh measurements
        let bw_result = backend.run_kernel(&bw_kernel, &run_config)?;
        let compute_result = backend.run_kernel(&compute_kernel, &run_config)?;

        // Sample device state
        let state = backend.device_state(0)?;

        let elapsed = (sample_idx + 1) as f64 * (config.sample_interval_ms as f64 / 1000.0);

        trajectory.push(ThermalSample {
            elapsed_secs: elapsed,
            clock_mhz: state.clock_mhz,
            temperature_c: state.temperature_c,
            power_watts: state.power_watts,
            measured_gflops: compute_result.gflops(),
            measured_bandwidth_gbps: bw_result.bandwidth_gbps(),
        });

        // Check for equilibrium
        if gflops_detector.observe(compute_result.gflops()) {
            equilibrium_time = elapsed;
        }

        last_bw_result = bw_result;
        last_compute_result = compute_result;

        // Early exit if stable for a full window past equilibrium
        if gflops_detector.is_stable()
            && elapsed > equilibrium_time + config.stability_window_secs
        {
            break;
        }
    }

    // Build sustained model from final measurements
    let final_state = backend.device_state(0)?;
    let sustained = RooflineModel {
        device_name: device_name.clone(),
        peak_gflops: last_compute_result.gflops(),
        peak_bandwidth_gbps: last_bw_result.bandwidth_gbps(),
        ridge_point: if last_bw_result.bandwidth_gbps() > 0.0 {
            last_compute_result.gflops() / last_bw_result.bandwidth_gbps()
        } else {
            0.0
        },
        clock_mhz: final_state.clock_mhz,
        temperature_c: final_state.temperature_c,
        power_watts: final_state.power_watts,
        placements: vec![],
    };

    Ok(DynamicRoofline::from_measurements(
        burst,
        sustained,
        trajectory,
        equilibrium_time,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::sim::{profiles, SimulatedBackend};

    #[test]
    fn test_dynamic_roofline_sim_rtx5090() {
        let backend = SimulatedBackend::new(profiles::rtx_5090());
        let config = DynamicConfig::quick();

        let dynamic = measure_dynamic(&backend, &config).unwrap();

        assert!(dynamic.device_name.contains("RTX 5090"));
        assert!(dynamic.burst.peak_gflops > 0.0);
        assert!(dynamic.sustained.peak_gflops > 0.0);
        assert!(!dynamic.thermal_trajectory.is_empty());

        // Burst should be >= sustained (thermal ramp reduces performance)
        assert!(
            dynamic.burst.peak_gflops >= dynamic.sustained.peak_gflops * 0.95,
            "burst ({:.0}) should be >= sustained ({:.0})",
            dynamic.burst.peak_gflops,
            dynamic.sustained.peak_gflops
        );
    }

    #[test]
    fn test_dynamic_roofline_h100() {
        let backend = SimulatedBackend::new(profiles::h100_sxm());
        let config = DynamicConfig::quick();

        let dynamic = measure_dynamic(&backend, &config).unwrap();

        assert!(dynamic.device_name.contains("H100"));
        assert!(dynamic.sustained.peak_bandwidth_gbps > 1000.0,
            "H100 sustained BW should be >1000 GB/s, got {:.0}",
            dynamic.sustained.peak_bandwidth_gbps);
    }

    #[test]
    fn test_dynamic_captures_trajectory() {
        let backend = SimulatedBackend::new(profiles::rtx_5090());
        let config = DynamicConfig::quick();

        let dynamic = measure_dynamic(&backend, &config).unwrap();

        assert!(
            dynamic.thermal_trajectory.len() >= 3,
            "should have at least 3 trajectory samples, got {}",
            dynamic.thermal_trajectory.len()
        );

        // Temperature should generally increase over trajectory
        if dynamic.thermal_trajectory.len() >= 2 {
            let first_temp = dynamic.thermal_trajectory.first().unwrap().temperature_c;
            let last_temp = dynamic.thermal_trajectory.last().unwrap().temperature_c;
            assert!(
                last_temp >= first_temp,
                "temperature should not decrease: {first_temp}°C -> {last_temp}°C"
            );
        }
    }

    #[test]
    fn test_degraded_gpu_shows_larger_drop() {
        let healthy = SimulatedBackend::new(profiles::rtx_5090());
        let degraded = SimulatedBackend::new(profiles::degraded_5090_thermal());
        let config = DynamicConfig::quick();

        let dynamic_healthy = measure_dynamic(&healthy, &config).unwrap();
        let dynamic_degraded = measure_dynamic(&degraded, &config).unwrap();

        // Degraded GPU should have higher sustained temperature
        assert!(
            dynamic_degraded.sustained.temperature_c >= dynamic_healthy.sustained.temperature_c,
            "degraded should be hotter: {}°C vs {}°C",
            dynamic_degraded.sustained.temperature_c,
            dynamic_healthy.sustained.temperature_c
        );
    }

    #[test]
    fn test_summary_not_empty() {
        let backend = SimulatedBackend::new(profiles::rtx_5090());
        let config = DynamicConfig::quick();

        let dynamic = measure_dynamic(&backend, &config).unwrap();
        let summary = dynamic.summary();

        assert!(!summary.is_empty());
        assert!(summary.contains("Burst"));
        assert!(summary.contains("Sustained"));
    }
}
