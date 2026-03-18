//! Fleet symmetry checking — flag mismatched GPU configurations.

use gpu_harness::error::HarnessError;
use gpu_harness::GpuBackend;
use serde::{Deserialize, Serialize};

/// Result of a symmetry check across the fleet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryReport {
    pub gpu_count: u32,
    pub mismatches: Vec<SymmetryMismatch>,
    pub is_symmetric: bool,
}

/// A single configuration mismatch between a GPU and the fleet mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryMismatch {
    pub field: String,
    pub gpu_index: u32,
    pub gpu_value: String,
    pub fleet_mode: String,
}

/// Check fleet symmetry: compare GPU configs, flag outliers.
pub fn check_symmetry(backend: &dyn GpuBackend) -> Result<SymmetryReport, HarnessError> {
    let devices = backend.discover_devices()?;
    let n = devices.len();

    if n <= 1 {
        return Ok(SymmetryReport {
            gpu_count: n as u32,
            mismatches: Vec::new(),
            is_symmetric: true,
        });
    }

    let mut mismatches = Vec::new();

    // Find mode (most common value) for each field
    let mode_name = mode_string(devices.iter().map(|d| d.name.clone()));
    let mode_memory = mode_u64(devices.iter().map(|d| d.memory_bytes));
    let mode_arch = mode_string(devices.iter().map(|d| format!("{:?}", d.architecture)));

    // Also check device state for runtime config differences
    let mut clocks = Vec::new();
    let mut temps = Vec::new();
    for (i, _) in devices.iter().enumerate() {
        if let Ok(state) = backend.device_state(i as u32) {
            clocks.push((i, state.clock_mhz));
            temps.push((i, state.temperature_c));
        }
    }

    // Check each device against the mode
    for device in &devices {
        if device.name != mode_name {
            mismatches.push(SymmetryMismatch {
                field: "name".to_string(),
                gpu_index: device.index,
                gpu_value: device.name.clone(),
                fleet_mode: mode_name.clone(),
            });
        }
        if device.memory_bytes != mode_memory {
            mismatches.push(SymmetryMismatch {
                field: "memory_bytes".to_string(),
                gpu_index: device.index,
                gpu_value: format!(
                    "{:.0} GB",
                    device.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                ),
                fleet_mode: format!("{:.0} GB", mode_memory as f64 / (1024.0 * 1024.0 * 1024.0)),
            });
        }
        let arch_str = format!("{:?}", device.architecture);
        if arch_str != mode_arch {
            mismatches.push(SymmetryMismatch {
                field: "architecture".to_string(),
                gpu_index: device.index,
                gpu_value: arch_str,
                fleet_mode: mode_arch.clone(),
            });
        }
    }

    // Clock speed symmetry (runtime — significant differences suggest degradation)
    if clocks.len() > 1 {
        let mode_clock = mode_u32(clocks.iter().map(|(_, c)| *c));
        for (idx, clock) in &clocks {
            // Flag if clock differs by >10% from fleet mode
            if mode_clock > 0 && (*clock as f64 / mode_clock as f64 - 1.0).abs() > 0.1 {
                mismatches.push(SymmetryMismatch {
                    field: "clock_mhz".to_string(),
                    gpu_index: *idx as u32,
                    gpu_value: format!("{clock} MHz"),
                    fleet_mode: format!("{mode_clock} MHz"),
                });
            }
        }
    }

    let is_symmetric = mismatches.is_empty();
    Ok(SymmetryReport {
        gpu_count: n as u32,
        mismatches,
        is_symmetric,
    })
}

/// Print symmetry report as a table.
pub fn print_symmetry_table(report: &SymmetryReport, _no_color: bool) {
    println!("\ngpu-fleet symmetry | {} GPUs\n", report.gpu_count);

    if report.is_symmetric {
        println!("  All GPUs are symmetrically configured.");
    } else {
        println!("  {} mismatch(es) found:\n", report.mismatches.len());
        for m in &report.mismatches {
            println!(
                "  GPU {} | {} = {} (fleet: {})",
                m.gpu_index, m.field, m.gpu_value, m.fleet_mode
            );
        }
    }
    println!();
}

/// Print symmetry report as JSON.
pub fn print_symmetry_json(report: &SymmetryReport) {
    if let Ok(json) = serde_json::to_string_pretty(report) {
        println!("{json}");
    }
}

// Helper: find the most common string value.
fn mode_string(values: impl Iterator<Item = String>) -> String {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(val, _)| val)
        .unwrap_or_default()
}

fn mode_u64(values: impl Iterator<Item = u64>) -> u64 {
    let mut counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(val, _)| val)
        .unwrap_or(0)
}

fn mode_u32(values: impl Iterator<Item = u32>) -> u32 {
    let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(val, _)| val)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::sim::{
        fleet::{Degradation, SimulatedFleet},
        profiles, SimulatedBackend,
    };

    #[test]
    fn test_symmetry_homogeneous_fleet() {
        let fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        let backend = SimulatedBackend::with_fleet(fleet);
        let report = check_symmetry(&backend).unwrap();
        assert!(
            report.is_symmetric,
            "homogeneous fleet should be symmetric: {:?}",
            report.mismatches
        );
    }

    #[test]
    fn test_symmetry_single_gpu() {
        let backend = SimulatedBackend::new(profiles::h100_sxm());
        let report = check_symmetry(&backend).unwrap();
        assert!(report.is_symmetric);
        assert_eq!(report.gpu_count, 1);
    }

    #[test]
    fn test_symmetry_detects_clock_mismatch() {
        let mut fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        fleet.degrade_gpu(2, Degradation::ClockStuck { max_mhz: 1095 });
        let backend = SimulatedBackend::with_fleet(fleet);
        let report = check_symmetry(&backend).unwrap();

        // Clock-stuck GPU should create a mismatch
        let clock_mismatches: Vec<_> = report
            .mismatches
            .iter()
            .filter(|m| m.field == "clock_mhz" && m.gpu_index == 2)
            .collect();

        assert!(
            !clock_mismatches.is_empty(),
            "clock-stuck GPU 2 should be detected: {:?}",
            report.mismatches
        );
    }
}
