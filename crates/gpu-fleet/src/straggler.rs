//! Straggler detection — identify underperforming GPUs and diagnose root cause.
//!
//! Measures all GPUs in the fleet, computes the fleet median, flags GPUs
//! below a threshold, and runs the diagnostic engine on each straggler.

use gpu_harness::error::HarnessError;
use gpu_harness::GpuBackend;
use gpu_roofline::ceilings::{measure_roofline, MeasureConfig};
use gpu_roofline::diagnose::{run_diagnosis, DiagnoseConfig, DiagnosisResult, ProbeName};
use gpu_roofline::validate::baselines::find_baseline;
use serde::{Deserialize, Serialize};

/// A GPU flagged as a straggler with diagnostic findings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Straggler {
    pub gpu_index: u32,
    pub gpu_name: String,
    pub bandwidth_gbps: f64,
    pub gflops: f64,
    pub bandwidth_ratio: f64,
    pub compute_ratio: f64,
    pub diagnosis: DiagnosisResult,
}

/// Fleet-wide straggler report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StragglerReport {
    pub gpu_count: u32,
    pub fleet_median_bandwidth_gbps: f64,
    pub fleet_median_gflops: f64,
    pub threshold: f64,
    pub stragglers: Vec<Straggler>,
}

/// Detect stragglers: measure all GPUs, find outliers, diagnose cause.
pub fn detect_stragglers(
    backend: &dyn GpuBackend,
    threshold: f64,
) -> Result<StragglerReport, HarnessError> {
    let devices = backend.discover_devices()?;
    let n = devices.len();

    // Quick measurement on each GPU
    let mut measurements = Vec::new();
    for (i, device) in devices.iter().enumerate() {
        // Set active device for fleet-mode backends (SimulatedBackend uses this
        // to select the right GPU profile; real backends ignore it)
        backend.set_active_device(i as u32);

        let config = MeasureConfig {
            buffer_size_bytes: 256 * 1024 * 1024,
            warmup_iterations: 3,
            measurement_iterations: 20,
            kernels: vec![
                gpu_roofline::kernels::BuiltinKernel::Copy,
                gpu_roofline::kernels::BuiltinKernel::FmaHeavy,
            ],
            device_index: i as u32,
        };

        let model = measure_roofline(backend, &config)?;
        measurements.push((i as u32, device.name.clone(), model));
    }

    // Compute fleet medians
    let mut bw_values: Vec<f64> = measurements
        .iter()
        .map(|(_, _, m)| m.peak_bandwidth_gbps)
        .collect();
    let mut gf_values: Vec<f64> = measurements.iter().map(|(_, _, m)| m.peak_gflops).collect();
    bw_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gf_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median_bw = if n > 0 { bw_values[n / 2] } else { 0.0 };
    let median_gf = if n > 0 { gf_values[n / 2] } else { 0.0 };

    // Find stragglers (below threshold of median)
    let mut stragglers = Vec::new();
    for (idx, name, model) in &measurements {
        let bw_ratio = if median_bw > 0.0 {
            model.peak_bandwidth_gbps / median_bw
        } else {
            1.0
        };
        let gf_ratio = if median_gf > 0.0 {
            model.peak_gflops / median_gf
        } else {
            1.0
        };

        if bw_ratio < threshold || gf_ratio < threshold {
            // Run diagnostic probes on this straggler
            let device = &devices[*idx as usize];
            let diagnosis = if let Some(baseline) = find_baseline(device) {
                let diag_config = DiagnoseConfig {
                    device_index: *idx,
                    probes: ProbeName::all().to_vec(),
                    measurement_iterations: 20,
                };
                run_diagnosis(backend, baseline, &diag_config)
                    .unwrap_or_else(|_| DiagnosisResult::new(name.clone(), Vec::new(), Vec::new()))
            } else {
                DiagnosisResult::new(name.clone(), Vec::new(), vec!["no baseline".to_string()])
            };

            stragglers.push(Straggler {
                gpu_index: *idx,
                gpu_name: name.clone(),
                bandwidth_gbps: model.peak_bandwidth_gbps,
                gflops: model.peak_gflops,
                bandwidth_ratio: bw_ratio,
                compute_ratio: gf_ratio,
                diagnosis,
            });
        }
    }

    Ok(StragglerReport {
        gpu_count: n as u32,
        fleet_median_bandwidth_gbps: median_bw,
        fleet_median_gflops: median_gf,
        threshold,
        stragglers,
    })
}

/// Print straggler report as a table.
pub fn print_straggler_table(report: &StragglerReport, _no_color: bool) {
    println!(
        "\ngpu-fleet straggler | {} GPUs | threshold {:.0}% of median\n",
        report.gpu_count,
        report.threshold * 100.0
    );

    println!(
        "  Fleet median: {:.0} GB/s | {:.1} TFLOPS",
        report.fleet_median_bandwidth_gbps,
        report.fleet_median_gflops / 1000.0
    );

    if report.stragglers.is_empty() {
        println!("\n  No stragglers detected. All GPUs within threshold.");
    } else {
        println!("\n  {} straggler(s) found:\n", report.stragglers.len());
        for s in &report.stragglers {
            println!(
                "  GPU {} | {} | BW {:.0}% | Compute {:.0}%",
                s.gpu_index,
                s.gpu_name,
                s.bandwidth_ratio * 100.0,
                s.compute_ratio * 100.0,
            );
            if s.diagnosis.findings.is_empty() {
                println!("    No specific root cause identified");
            } else {
                for finding in &s.diagnosis.findings {
                    println!("    [{}] {}", finding.severity, finding.summary);
                    println!("    Cause: {}", finding.cause);
                    println!("    Fix:   {}", finding.fix);
                }
            }
            println!();
        }
    }
}

/// Print straggler report as JSON.
pub fn print_straggler_json(report: &StragglerReport) {
    if let Ok(json) = serde_json::to_string_pretty(report) {
        println!("{json}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::sim::{
        fleet::{Degradation, SimulatedFleet},
        profiles, SimulatedBackend,
    };

    #[test]
    fn test_no_stragglers_healthy_fleet() {
        let fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        let backend = SimulatedBackend::with_fleet(fleet);
        let report = detect_stragglers(&backend, 0.9).unwrap();
        assert!(
            report.stragglers.is_empty(),
            "healthy fleet should have no stragglers: {:?}",
            report
                .stragglers
                .iter()
                .map(|s| (&s.gpu_name, s.bandwidth_ratio, s.compute_ratio))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_straggler_detected_memory_degradation() {
        let mut fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        fleet.degrade_gpu(
            2,
            Degradation::MemorySubsystem {
                bandwidth_ratio: 0.6,
            },
        );
        let backend = SimulatedBackend::with_fleet(fleet);
        let report = detect_stragglers(&backend, 0.9).unwrap();

        assert!(
            !report.stragglers.is_empty(),
            "degraded GPU 2 should be flagged as straggler"
        );

        let straggler = report.stragglers.iter().find(|s| s.gpu_index == 2);
        assert!(straggler.is_some(), "GPU 2 should be the straggler");
        assert!(
            straggler.unwrap().bandwidth_ratio < 0.9,
            "straggler BW ratio should be below threshold"
        );
    }

    #[test]
    fn test_straggler_detected_clock_stuck() {
        let mut fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        fleet.degrade_gpu(1, Degradation::ClockStuck { max_mhz: 1095 });
        let backend = SimulatedBackend::with_fleet(fleet);
        let report = detect_stragglers(&backend, 0.9).unwrap();

        let straggler = report.stragglers.iter().find(|s| s.gpu_index == 1);
        assert!(
            straggler.is_some(),
            "clock-stuck GPU 1 should be straggler: {:?}",
            report
                .stragglers
                .iter()
                .map(|s| (s.gpu_index, s.compute_ratio))
                .collect::<Vec<_>>()
        );
    }
}
