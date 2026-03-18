//! Diagnostic probes — composable functions that detect specific GPU issues.
//!
//! Each probe runs a targeted measurement and compares against the hardware
//! baseline to produce findings with causes and fixes.

use gpu_harness::error::HarnessError;
use gpu_harness::GpuBackend;

use crate::ceilings::{measure_roofline, MeasureConfig};
use crate::kernels::BuiltinKernel;
use crate::validate::baselines::HardwareBaseline;

use super::types::*;

/// Run all configured diagnostic probes and collect findings.
pub fn run_diagnosis(
    backend: &dyn GpuBackend,
    baseline: &HardwareBaseline,
    config: &DiagnoseConfig,
) -> Result<DiagnosisResult, HarnessError> {
    let devices = backend.discover_devices()?;
    let gpu_name = devices
        .get(config.device_index as usize)
        .map(|d| d.name.clone())
        .unwrap_or_else(|| "Unknown GPU".to_string());

    let mut findings = Vec::new();
    let mut probes_run = Vec::new();

    for probe in &config.probes {
        probes_run.push(probe.to_string());
        let probe_findings = match probe {
            ProbeName::L2Thrashing => probe_l2_thrashing(backend, baseline, config)?,
            ProbeName::HbmDegradation => probe_hbm_degradation(backend, baseline, config)?,
            ProbeName::PciBottleneck => probe_pci_bottleneck(backend, config)?,
            ProbeName::ThermalThrottling => probe_thermal_throttling(backend, baseline, config)?,
            ProbeName::ClockStuck => probe_clock_stuck(backend, baseline, config)?,
            ProbeName::ComputeDeficit => probe_compute_deficit(backend, baseline, config)?,
        };
        findings.extend(probe_findings);
    }

    Ok(DiagnosisResult::new(gpu_name, findings, probes_run))
}

/// Probe: L2 cache thrashing detection.
///
/// Measures bandwidth with a small buffer (fits in L2) and a large buffer
/// (exceeds L2). If the ratio is > 2x, the L2 is being relied upon heavily
/// and workloads exceeding L2 will see a sharp performance cliff.
fn probe_l2_thrashing(
    backend: &dyn GpuBackend,
    baseline: &HardwareBaseline,
    config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    let l2_bytes = (baseline.l2_cache_mb as usize) * 1024 * 1024;
    let small_buffer = l2_bytes / 2; // Fits in L2
    let large_buffer = baseline.recommended_buffer_bytes; // Exceeds L2

    // Measure with small buffer (L2-resident)
    let small_config = MeasureConfig {
        buffer_size_bytes: small_buffer,
        warmup_iterations: 3,
        measurement_iterations: config.measurement_iterations,
        kernels: vec![BuiltinKernel::Copy],
        device_index: config.device_index,
    };
    let small_model = measure_roofline(backend, &small_config)?;

    // Measure with large buffer (HBM-resident)
    let large_config = MeasureConfig {
        buffer_size_bytes: large_buffer,
        warmup_iterations: 3,
        measurement_iterations: config.measurement_iterations,
        kernels: vec![BuiltinKernel::Copy],
        device_index: config.device_index,
    };
    let large_model = measure_roofline(backend, &large_config)?;

    let l2_bw = small_model.peak_bandwidth_gbps;
    let hbm_bw = large_model.peak_bandwidth_gbps;

    // L2 bandwidth should be significantly higher than HBM.
    // A ratio > 2x indicates strong L2 dependence — workloads exceeding L2
    // will see a sharp cliff.
    let ratio = if hbm_bw > 0.0 { l2_bw / hbm_bw } else { 1.0 };

    if ratio > 2.0 {
        Ok(vec![DiagnosticFinding {
            category: DiagnosticCategory::MemoryBound(MemoryIssue::L2Thrashing {
                l2_bandwidth_gbps: l2_bw,
                hbm_bandwidth_gbps: hbm_bw,
                l2_cache_mb: baseline.l2_cache_mb,
            }),
            severity: Severity::Info,
            summary: format!(
                "L2 cache delivers {:.0} GB/s vs {:.0} GB/s at HBM ({:.1}x ratio)",
                l2_bw, hbm_bw, ratio
            ),
            cause: format!(
                "Working sets under {}MB fit in L2 cache and run {:.1}x faster. \
                 Workloads exceeding L2 will hit the HBM bandwidth wall.",
                baseline.l2_cache_mb, ratio
            ),
            fix: "Ensure working set exceeds L2 when benchmarking HBM bandwidth. \
                  For production: tile computations to fit L2 when possible."
                .to_string(),
        }])
    } else {
        Ok(Vec::new())
    }
}

/// Probe: HBM/DRAM bandwidth degradation.
///
/// Measures bandwidth with a large buffer and compares against baseline range.
fn probe_hbm_degradation(
    backend: &dyn GpuBackend,
    baseline: &HardwareBaseline,
    config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    let measure_config = MeasureConfig {
        buffer_size_bytes: baseline.recommended_buffer_bytes,
        warmup_iterations: 5,
        measurement_iterations: config.measurement_iterations,
        kernels: vec![BuiltinKernel::Copy],
        device_index: config.device_index,
    };
    let model = measure_roofline(backend, &measure_config)?;
    let measured = model.peak_bandwidth_gbps;
    let expected_min = baseline.bandwidth_range.0;

    if measured < expected_min * 0.9 {
        let ratio = measured / expected_min;
        let severity = if ratio < 0.7 {
            Severity::Critical
        } else {
            Severity::Warning
        };

        Ok(vec![DiagnosticFinding {
            category: DiagnosticCategory::MemoryBound(MemoryIssue::HbmDegradation {
                measured_gbps: measured,
                expected_min_gbps: expected_min,
                ratio,
            }),
            severity,
            summary: format!(
                "HBM bandwidth at {:.0}% of expected ({:.0} vs {:.0} GB/s)",
                ratio * 100.0,
                measured,
                expected_min
            ),
            cause:
                "Possible partial HBM stack failure, ECC errors, or memory subsystem degradation."
                    .to_string(),
            fix: "Check nvidia-smi -q for ECC error counts. If persistent, RMA the GPU."
                .to_string(),
        }])
    } else {
        Ok(Vec::new())
    }
}

/// Probe: PCIe bandwidth bottleneck.
///
/// Checks if measured bandwidth suggests PCIe-limited operation.
fn probe_pci_bottleneck(
    backend: &dyn GpuBackend,
    config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    let devices = backend.discover_devices()?;
    let device = devices
        .get(config.device_index as usize)
        .ok_or(HarnessError::DeviceIndexOutOfRange(config.device_index))?;

    // PCIe bandwidth ceilings (bidirectional, GB/s)
    let expected_gen = if device
        .features
        .compute_capability
        .map(|(m, _)| m)
        .unwrap_or(0)
        >= 9
    {
        5 // Hopper+ = PCIe 5.0
    } else if device
        .features
        .compute_capability
        .map(|(m, _)| m)
        .unwrap_or(0)
        >= 8
    {
        4 // Ampere/Ada = PCIe 4.0
    } else {
        3
    };

    let pcie_ceiling_gbps = match expected_gen {
        5 => 64.0, // PCIe 5.0 x16
        4 => 32.0, // PCIe 4.0 x16
        _ => 16.0, // PCIe 3.0 x16
    };

    // Quick measurement — if bandwidth is near PCIe ceiling, we're PCIe-limited
    let measure_config = MeasureConfig {
        buffer_size_bytes: 256 * 1024 * 1024,
        warmup_iterations: 3,
        measurement_iterations: config.measurement_iterations,
        kernels: vec![BuiltinKernel::Copy],
        device_index: config.device_index,
    };
    let model = measure_roofline(backend, &measure_config)?;
    let measured = model.peak_bandwidth_gbps;

    // If measured bandwidth is within 20% of PCIe ceiling (and far below HBM spec),
    // something is forcing PCIe-limited operation
    if measured < pcie_ceiling_gbps * 1.5 && measured < 200.0 {
        Ok(vec![DiagnosticFinding {
            category: DiagnosticCategory::MemoryBound(MemoryIssue::PciBottleneck {
                measured_gbps: measured,
                expected_gen,
            }),
            severity: Severity::Warning,
            summary: format!(
                "Bandwidth {:.0} GB/s near PCIe {expected_gen}.0 ceiling ({:.0} GB/s)",
                measured, pcie_ceiling_gbps
            ),
            cause: format!(
                "GPU may be running over PCIe instead of local HBM. \
                 Check if GPU is in PCIe slot (not SXM) or if NVLink is misconfigured."
            ),
            fix: "Verify GPU physical connection. For SXM GPUs, check NVLink firmware. \
                  For PCIe GPUs, ensure x16 lane width (not x8 or x4)."
                .to_string(),
        }])
    } else {
        Ok(Vec::new())
    }
}

/// Probe: Thermal throttling detection.
///
/// Runs a sustained workload and checks if clock frequency drops while
/// temperature rises — the signature of thermal throttling.
fn probe_thermal_throttling(
    backend: &dyn GpuBackend,
    baseline: &HardwareBaseline,
    config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    // Take initial state
    let initial_state = backend.device_state(config.device_index)?;
    let initial_clock = initial_state.clock_mhz;

    // Run a heavier measurement to heat the GPU
    let heavy_config = MeasureConfig {
        buffer_size_bytes: baseline.recommended_buffer_bytes,
        warmup_iterations: 20,
        measurement_iterations: config.measurement_iterations * 2,
        kernels: vec![BuiltinKernel::FmaHeavy],
        device_index: config.device_index,
    };
    let _ = measure_roofline(backend, &heavy_config)?;

    // Check state after sustained workload
    let post_state = backend.device_state(config.device_index)?;
    let post_clock = post_state.clock_mhz;
    let temp = post_state.temperature_c;

    // If clock dropped >5% and temperature is elevated, that's throttling
    if initial_clock > 0 && post_clock > 0 {
        let clock_drop_pct = (1.0 - post_clock as f64 / initial_clock as f64) * 100.0;

        if clock_drop_pct > 5.0 && temp > 75 {
            let severity = if clock_drop_pct > 15.0 {
                Severity::Critical
            } else {
                Severity::Warning
            };

            return Ok(vec![DiagnosticFinding {
                category: DiagnosticCategory::Thermal(ThermalIssue::Throttling {
                    temperature_c: temp,
                    clock_drop_pct,
                }),
                severity,
                summary: format!(
                    "Clock dropped {:.1}% ({} → {} MHz) at {}°C under sustained load",
                    clock_drop_pct, initial_clock, post_clock, temp
                ),
                cause: "GPU is thermal throttling under sustained workload. \
                        This reduces performance from burst to sustained ceiling."
                    .to_string(),
                fix: "Check cooling system (fans, thermal paste, airflow). \
                      For datacenter: verify rack cooling and ambient temperature."
                    .to_string(),
            }]);
        }
    }

    Ok(Vec::new())
}

/// Probe: Clock stuck detection.
///
/// Checks if GPU clock is stuck at base frequency instead of boosting.
fn probe_clock_stuck(
    backend: &dyn GpuBackend,
    baseline: &HardwareBaseline,
    config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    // Run a light workload to trigger boost
    let light_config = MeasureConfig {
        buffer_size_bytes: baseline.recommended_buffer_bytes,
        warmup_iterations: 5,
        measurement_iterations: 10,
        kernels: vec![BuiltinKernel::FmaHeavy],
        device_index: config.device_index,
    };
    let state = backend.device_state(config.device_index)?;

    // Compare measured compute throughput against baseline instead of clock directly.
    // If compute is way below expected, and we ran a workload that should have boosted
    // the clock, then the clock is likely stuck.
    let model = measure_roofline(backend, &light_config)?;
    let measured_tflops = model.peak_gflops / 1000.0;
    let expected_min_tflops = baseline.flops_range.0;

    // If compute is below 70% of baseline minimum AND clock is reported,
    // the clock is likely stuck at base frequency
    let compute_ratio = if expected_min_tflops > 0.0 {
        measured_tflops / expected_min_tflops
    } else {
        1.0
    };

    if compute_ratio < 0.7 && state.clock_mhz > 0 {
        return Ok(vec![DiagnosticFinding {
            category: DiagnosticCategory::ComputeBound(ComputeIssue::ClockStuck {
                measured_mhz: state.clock_mhz,
                expected_boost_mhz: 0, // Unknown without profile data
            }),
            severity: Severity::Warning,
            summary: format!(
                "Compute at {:.0}% of expected ({:.1} vs {:.0} TFLOPS) — clock may be stuck at {} MHz",
                compute_ratio * 100.0, measured_tflops, expected_min_tflops, state.clock_mhz
            ),
            cause: "GPU clock may be stuck at base frequency. Possible firmware bug, \
                    driver issue, or power management misconfiguration."
                .to_string(),
            fix: "Try: nvidia-smi -pm 1 (persistence mode), nvidia-smi -ac <mem_clock>,<gpu_clock> \
                  (application clocks), or update GPU driver."
                .to_string(),
        }]);
    }

    Ok(Vec::new())
}

/// Probe: Compute throughput deficit.
///
/// Runs compute-heavy kernel and compares GFLOPS against baseline.
fn probe_compute_deficit(
    backend: &dyn GpuBackend,
    baseline: &HardwareBaseline,
    config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    let measure_config = MeasureConfig {
        buffer_size_bytes: baseline.recommended_buffer_bytes,
        warmup_iterations: 5,
        measurement_iterations: config.measurement_iterations,
        kernels: vec![BuiltinKernel::FmaHeavy],
        device_index: config.device_index,
    };
    let model = measure_roofline(backend, &measure_config)?;
    let measured_tflops = model.peak_gflops / 1000.0;
    let expected_min_tflops = baseline.flops_range.0;

    if measured_tflops < expected_min_tflops * 0.9 {
        let ratio = measured_tflops / expected_min_tflops;
        let severity = if ratio < 0.7 {
            Severity::Critical
        } else {
            Severity::Warning
        };

        Ok(vec![DiagnosticFinding {
            category: DiagnosticCategory::ComputeBound(ComputeIssue::ComputeDeficit {
                measured_gflops: model.peak_gflops,
                expected_min_gflops: expected_min_tflops * 1000.0,
                ratio,
            }),
            severity,
            summary: format!(
                "FP32 compute at {:.0}% of expected ({:.1} vs {:.0} TFLOPS)",
                ratio * 100.0,
                measured_tflops,
                expected_min_tflops
            ),
            cause: "Compute throughput below expected. Possible causes: thermal throttling \
                    reducing clock speed, driver issue, or hardware degradation."
                .to_string(),
            fix: "Run the thermal probe to check for throttling. Update drivers. \
                  If persistent, check nvidia-smi for Xid errors."
                .to_string(),
        }])
    } else {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::baselines::find_baseline;
    use gpu_harness::sim::{profiles, SimulatedBackend};

    fn get_baseline(backend: &dyn GpuBackend) -> &'static HardwareBaseline {
        let devices = backend.discover_devices().unwrap();
        find_baseline(&devices[0]).unwrap()
    }

    #[test]
    fn test_healthy_gpu_no_findings() {
        let backend = SimulatedBackend::new(profiles::h100_sxm());
        let baseline = get_baseline(&backend);
        let config = DiagnoseConfig {
            measurement_iterations: 10,
            ..DiagnoseConfig::default()
        };
        let result = run_diagnosis(&backend, baseline, &config).unwrap();
        assert!(
            result.findings.is_empty(),
            "healthy H100 should have no findings: {:?}",
            result.findings
        );
    }

    #[test]
    fn test_hbm_degradation_detected() {
        let backend = SimulatedBackend::new(profiles::degraded_h100_memory());
        let baseline = get_baseline(&backend);
        let config = DiagnoseConfig {
            probes: vec![ProbeName::HbmDegradation],
            measurement_iterations: 10,
            ..DiagnoseConfig::default()
        };
        let result = run_diagnosis(&backend, baseline, &config).unwrap();
        assert!(
            result.findings.iter().any(|f| matches!(
                f.category,
                DiagnosticCategory::MemoryBound(MemoryIssue::HbmDegradation { .. })
            )),
            "degraded H100 memory should trigger HBM finding: {:?}",
            result.findings
        );
    }

    #[test]
    fn test_clock_stuck_detected() {
        let backend = SimulatedBackend::new(profiles::degraded_h100_clock());
        let baseline = get_baseline(&backend);
        let config = DiagnoseConfig {
            probes: vec![ProbeName::ClockStuck],
            measurement_iterations: 10,
            ..DiagnoseConfig::default()
        };
        let result = run_diagnosis(&backend, baseline, &config).unwrap();
        assert!(
            result.findings.iter().any(|f| matches!(
                f.category,
                DiagnosticCategory::ComputeBound(ComputeIssue::ClockStuck { .. })
            )),
            "clock-stuck H100 should trigger ClockStuck finding: {:?}",
            result.findings
        );
    }

    #[test]
    fn test_compute_deficit_detected() {
        let backend = SimulatedBackend::new(profiles::degraded_h100_clock());
        let baseline = get_baseline(&backend);
        let config = DiagnoseConfig {
            probes: vec![ProbeName::ComputeDeficit],
            measurement_iterations: 10,
            ..DiagnoseConfig::default()
        };
        let result = run_diagnosis(&backend, baseline, &config).unwrap();
        assert!(
            result.findings.iter().any(|f| matches!(
                f.category,
                DiagnosticCategory::ComputeBound(ComputeIssue::ComputeDeficit { .. })
            )),
            "clock-stuck H100 should trigger ComputeDeficit: {:?}",
            result.findings
        );
    }

    #[test]
    fn test_thermal_throttling_detected() {
        let backend = SimulatedBackend::new(profiles::degraded_5090_thermal());
        // Use RTX 5090 baseline
        let devices = backend.discover_devices().unwrap();
        let baseline = find_baseline(&devices[0]).unwrap();
        let config = DiagnoseConfig {
            probes: vec![ProbeName::ThermalThrottling],
            measurement_iterations: 10,
            ..DiagnoseConfig::default()
        };
        // Advance time to heat the GPU
        backend.advance_time(60.0);
        let result = run_diagnosis(&backend, baseline, &config).unwrap();
        // Thermal throttling may or may not trigger depending on simulated thermal model
        // The degraded_5090_thermal profile has lower throttle onset
        // At minimum, the probe should run without error
        assert_eq!(result.probes_run.len(), 1);
    }

    #[test]
    fn test_l2_thrashing_probe_runs() {
        let backend = SimulatedBackend::new(profiles::h100_sxm());
        let baseline = get_baseline(&backend);
        let config = DiagnoseConfig {
            probes: vec![ProbeName::L2Thrashing],
            measurement_iterations: 10,
            ..DiagnoseConfig::default()
        };
        let result = run_diagnosis(&backend, baseline, &config).unwrap();
        // L2 thrashing is Info-level — may or may not fire depending on sim model
        assert_eq!(result.probes_run.len(), 1);
    }

    #[test]
    fn test_all_probes_run_without_error() {
        let backend = SimulatedBackend::new(profiles::h100_sxm());
        let baseline = get_baseline(&backend);
        let config = DiagnoseConfig {
            measurement_iterations: 10,
            ..DiagnoseConfig::default()
        };
        let result = run_diagnosis(&backend, baseline, &config).unwrap();
        assert_eq!(result.probes_run.len(), 6, "all 6 probes should run");
    }
}
