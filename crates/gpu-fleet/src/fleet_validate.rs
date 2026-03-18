//! Per-GPU roofline validation across the fleet.

use gpu_harness::error::HarnessError;
use gpu_harness::GpuBackend;
use gpu_roofline::ceilings::{measure_roofline, MeasureConfig};
use gpu_roofline::validate::{adaptive_config, find_baseline, validate_roofline};
use serde::{Deserialize, Serialize};

/// Result of fleet-wide validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetValidationResult {
    pub gpu_count: u32,
    pub pass_count: u32,
    pub gpu_results: Vec<GpuValidationEntry>,
    pub fleet_summary: FleetSummary,
    pub all_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuValidationEntry {
    pub gpu_index: u32,
    pub gpu_name: String,
    pub passed: bool,
    pub bandwidth_gbps: f64,
    pub gflops: f64,
    pub diagnosis: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetSummary {
    pub avg_bandwidth_gbps: f64,
    pub avg_gflops: f64,
    pub min_bandwidth_gbps: f64,
    pub max_bandwidth_gbps: f64,
    pub bandwidth_spread_pct: f64,
}

/// Validate each GPU in the fleet against its hardware baseline.
pub fn validate_fleet(
    backend: &dyn GpuBackend,
    threshold: f64,
) -> Result<FleetValidationResult, HarnessError> {
    let devices = backend.discover_devices()?;
    let n = devices.len();

    let mut gpu_results = Vec::new();
    let mut pass_count = 0u32;
    let mut bandwidths = Vec::new();
    let mut gflops_vec = Vec::new();

    for (i, device) in devices.iter().enumerate() {
        let baseline = find_baseline(device);
        let config = if baseline.is_some() {
            adaptive_config(device)
        } else {
            MeasureConfig::default()
        };

        // Set active device for fleet-mode backends
        backend.set_active_device(i as u32);
        let model = measure_roofline(
            backend,
            &MeasureConfig {
                device_index: i as u32,
                ..config
            },
        )?;

        let (passed, diagnosis) = if let Some(bl) = baseline {
            let result = validate_roofline(&model, bl, threshold);
            (result.passed, result.diagnosis)
        } else {
            (
                true,
                vec!["No baseline available — skipped validation".to_string()],
            )
        };

        if passed {
            pass_count += 1;
        }

        bandwidths.push(model.peak_bandwidth_gbps);
        gflops_vec.push(model.peak_gflops);

        gpu_results.push(GpuValidationEntry {
            gpu_index: i as u32,
            gpu_name: device.name.clone(),
            passed,
            bandwidth_gbps: model.peak_bandwidth_gbps,
            gflops: model.peak_gflops,
            diagnosis,
        });
    }

    let avg_bw = bandwidths.iter().sum::<f64>() / n.max(1) as f64;
    let avg_gf = gflops_vec.iter().sum::<f64>() / n.max(1) as f64;
    let min_bw = bandwidths.iter().cloned().fold(f64::MAX, f64::min);
    let max_bw = bandwidths.iter().cloned().fold(0.0_f64, f64::max);
    let spread = if avg_bw > 0.0 {
        (max_bw - min_bw) / avg_bw * 100.0
    } else {
        0.0
    };

    let all_passed = pass_count == n as u32;

    Ok(FleetValidationResult {
        gpu_count: n as u32,
        pass_count,
        gpu_results,
        fleet_summary: FleetSummary {
            avg_bandwidth_gbps: avg_bw,
            avg_gflops: avg_gf,
            min_bandwidth_gbps: min_bw,
            max_bandwidth_gbps: max_bw,
            bandwidth_spread_pct: spread,
        },
        all_passed,
    })
}

/// Print fleet validation as a table.
pub fn print_fleet_validate_table(result: &FleetValidationResult, _no_color: bool) {
    println!(
        "\ngpu-fleet validate | {} GPUs | {}/{} PASS\n",
        result.gpu_count, result.pass_count, result.gpu_count
    );

    for entry in &result.gpu_results {
        let status = if entry.passed { "PASS" } else { "FAIL" };
        println!(
            "  GPU {} | {} | {:<4} | {:.0} GB/s | {:.1} TFLOPS",
            entry.gpu_index,
            entry.gpu_name,
            status,
            entry.bandwidth_gbps,
            entry.gflops / 1000.0,
        );
        for diag in &entry.diagnosis {
            println!("         {diag}");
        }
    }

    println!(
        "\n  Fleet: avg {:.0} GB/s | spread {:.1}% | min {:.0} | max {:.0}",
        result.fleet_summary.avg_bandwidth_gbps,
        result.fleet_summary.bandwidth_spread_pct,
        result.fleet_summary.min_bandwidth_gbps,
        result.fleet_summary.max_bandwidth_gbps,
    );
}

/// Print fleet validation as JSON.
pub fn print_fleet_validate_json(result: &FleetValidationResult) {
    if let Ok(json) = serde_json::to_string_pretty(result) {
        println!("{json}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::sim::{fleet::SimulatedFleet, profiles, SimulatedBackend};

    #[test]
    fn test_fleet_validate_healthy() {
        let fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        let backend = SimulatedBackend::with_fleet(fleet);
        let result = validate_fleet(&backend, 0.8).unwrap();
        assert_eq!(result.gpu_count, 4);
        assert!(result.all_passed, "healthy fleet should all pass");
        assert!(result.fleet_summary.bandwidth_spread_pct < 10.0);
    }

    #[test]
    fn test_fleet_validate_single_gpu() {
        let backend = SimulatedBackend::new(profiles::h100_sxm());
        let result = validate_fleet(&backend, 0.8).unwrap();
        assert_eq!(result.gpu_count, 1);
        assert!(result.all_passed);
    }
}
