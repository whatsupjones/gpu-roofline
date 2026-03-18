//! Diagnostic probes — composable functions that detect specific GPU issues.

use gpu_harness::error::HarnessError;
use gpu_harness::GpuBackend;

use crate::validate::baselines::HardwareBaseline;

use super::types::{DiagnoseConfig, DiagnosisResult, DiagnosticFinding, ProbeName};

/// Run all configured diagnostic probes and collect findings.
pub fn run_diagnosis(
    backend: &dyn GpuBackend,
    baseline: &HardwareBaseline,
    config: &DiagnoseConfig,
) -> Result<DiagnosisResult, HarnessError> {
    let devices = backend.discover_devices()?;
    let gpu_name = devices
        .first()
        .map(|d| d.name.clone())
        .unwrap_or_else(|| "Unknown GPU".to_string());

    let mut findings = Vec::new();
    let mut probes_run = Vec::new();

    for probe in &config.probes {
        probes_run.push(probe.to_string());
        let probe_findings = match probe {
            ProbeName::L2Thrashing => probe_l2_thrashing(backend, baseline, config)?,
            ProbeName::HbmDegradation => probe_hbm_degradation(backend, baseline, config)?,
            ProbeName::PciBottleneck => probe_pci_bottleneck(backend, baseline, config)?,
            ProbeName::ThermalThrottling => probe_thermal_throttling(backend, config)?,
            ProbeName::ClockStuck => probe_clock_stuck(backend, baseline, config)?,
            ProbeName::ComputeDeficit => probe_compute_deficit(backend, baseline, config)?,
        };
        findings.extend(probe_findings);
    }

    Ok(DiagnosisResult::new(gpu_name, findings, probes_run))
}

fn probe_l2_thrashing(
    _backend: &dyn GpuBackend,
    _baseline: &HardwareBaseline,
    _config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    // TODO: implement in Phase 2
    Ok(Vec::new())
}

fn probe_hbm_degradation(
    _backend: &dyn GpuBackend,
    _baseline: &HardwareBaseline,
    _config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    Ok(Vec::new())
}

fn probe_pci_bottleneck(
    _backend: &dyn GpuBackend,
    _baseline: &HardwareBaseline,
    _config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    Ok(Vec::new())
}

fn probe_thermal_throttling(
    _backend: &dyn GpuBackend,
    _config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    Ok(Vec::new())
}

fn probe_clock_stuck(
    _backend: &dyn GpuBackend,
    _baseline: &HardwareBaseline,
    _config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    Ok(Vec::new())
}

fn probe_compute_deficit(
    _backend: &dyn GpuBackend,
    _baseline: &HardwareBaseline,
    _config: &DiagnoseConfig,
) -> Result<Vec<DiagnosticFinding>, HarnessError> {
    Ok(Vec::new())
}
