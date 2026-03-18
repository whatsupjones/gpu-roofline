//! Data types for the GPU diagnostic engine.
//!
//! Diagnostic findings describe WHY a GPU is underperforming,
//! not just THAT it is. Each finding includes a cause and a fix.

use serde::{Deserialize, Serialize};

/// Severity of a diagnostic finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "Info"),
            Self::Warning => write!(f, "Warning"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Category of a diagnostic finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiagnosticCategory {
    MemoryBound(MemoryIssue),
    ComputeBound(ComputeIssue),
    Thermal(ThermalIssue),
    Configuration(ConfigIssue),
}

impl std::fmt::Display for DiagnosticCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryBound(issue) => write!(f, "Memory: {issue}"),
            Self::ComputeBound(issue) => write!(f, "Compute: {issue}"),
            Self::Thermal(issue) => write!(f, "Thermal: {issue}"),
            Self::Configuration(issue) => write!(f, "Config: {issue}"),
        }
    }
}

/// Memory subsystem issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryIssue {
    /// Working set fits in L2 — measuring cache, not HBM.
    L2Thrashing {
        l2_bandwidth_gbps: f64,
        hbm_bandwidth_gbps: f64,
        l2_cache_mb: u32,
    },
    /// HBM bandwidth below expected range.
    HbmDegradation {
        measured_gbps: f64,
        expected_min_gbps: f64,
        ratio: f64,
    },
    /// PCIe running at lower gen than expected.
    PciBottleneck {
        measured_gbps: f64,
        expected_gen: u32,
    },
}

impl std::fmt::Display for MemoryIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::L2Thrashing { .. } => write!(f, "L2 Cache Thrashing"),
            Self::HbmDegradation { .. } => write!(f, "HBM Degradation"),
            Self::PciBottleneck { .. } => write!(f, "PCIe Bottleneck"),
        }
    }
}

/// Compute subsystem issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeIssue {
    /// Clock frequency stuck below boost.
    ClockStuck {
        measured_mhz: u32,
        expected_boost_mhz: u32,
    },
    /// Compute throughput below expected.
    ComputeDeficit {
        measured_gflops: f64,
        expected_min_gflops: f64,
        ratio: f64,
    },
}

impl std::fmt::Display for ComputeIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClockStuck { .. } => write!(f, "Clock Stuck"),
            Self::ComputeDeficit { .. } => write!(f, "Compute Deficit"),
        }
    }
}

/// Thermal issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalIssue {
    /// GPU throttling due to temperature.
    Throttling {
        temperature_c: u32,
        clock_drop_pct: f64,
    },
}

impl std::fmt::Display for ThermalIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Throttling { .. } => write!(f, "Thermal Throttling"),
        }
    }
}

/// Configuration issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigIssue {
    /// ECC memory enabled, reducing effective bandwidth.
    EccOverhead,
}

impl std::fmt::Display for ConfigIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EccOverhead => write!(f, "ECC Overhead"),
        }
    }
}

/// A single diagnostic finding with cause and fix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticFinding {
    pub category: DiagnosticCategory,
    pub severity: Severity,
    /// Human-readable summary (e.g. "HBM bandwidth at 75% of expected").
    pub summary: String,
    /// Root cause explanation.
    pub cause: String,
    /// Suggested fix.
    pub fix: String,
}

/// Which diagnostic probe to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeName {
    L2Thrashing,
    HbmDegradation,
    PciBottleneck,
    ThermalThrottling,
    ClockStuck,
    ComputeDeficit,
}

impl ProbeName {
    pub fn all() -> &'static [ProbeName] {
        &[
            ProbeName::L2Thrashing,
            ProbeName::HbmDegradation,
            ProbeName::PciBottleneck,
            ProbeName::ThermalThrottling,
            ProbeName::ClockStuck,
            ProbeName::ComputeDeficit,
        ]
    }
}

impl std::fmt::Display for ProbeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::L2Thrashing => write!(f, "l2_thrashing"),
            Self::HbmDegradation => write!(f, "hbm_degradation"),
            Self::PciBottleneck => write!(f, "pci_bottleneck"),
            Self::ThermalThrottling => write!(f, "thermal_throttling"),
            Self::ClockStuck => write!(f, "clock_stuck"),
            Self::ComputeDeficit => write!(f, "compute_deficit"),
        }
    }
}

impl std::str::FromStr for ProbeName {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "l2_thrashing" => Ok(Self::L2Thrashing),
            "hbm_degradation" => Ok(Self::HbmDegradation),
            "pci_bottleneck" => Ok(Self::PciBottleneck),
            "thermal_throttling" | "thermal" => Ok(Self::ThermalThrottling),
            "clock_stuck" | "clock" => Ok(Self::ClockStuck),
            "compute_deficit" | "compute" => Ok(Self::ComputeDeficit),
            _ => Err(format!("unknown probe: {s}")),
        }
    }
}

/// Configuration for a diagnostic run.
pub struct DiagnoseConfig {
    pub device_index: u32,
    pub probes: Vec<ProbeName>,
    pub measurement_iterations: u32,
}

impl Default for DiagnoseConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            probes: ProbeName::all().to_vec(),
            measurement_iterations: 50,
        }
    }
}

/// Complete result of a diagnostic run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosisResult {
    pub gpu_name: String,
    pub findings: Vec<DiagnosticFinding>,
    pub probes_run: Vec<String>,
    /// Highest severity among all findings.
    pub max_severity: Option<Severity>,
}

impl DiagnosisResult {
    pub fn new(
        gpu_name: String,
        findings: Vec<DiagnosticFinding>,
        probes_run: Vec<String>,
    ) -> Self {
        let max_severity = findings.iter().map(|f| f.severity).max();
        Self {
            gpu_name,
            findings,
            probes_run,
            max_severity,
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn exit_code(&self) -> i32 {
        match self.max_severity {
            Some(Severity::Critical) => 1,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Critical);
    }

    #[test]
    fn test_diagnosis_result_healthy() {
        let result = DiagnosisResult::new(
            "Test GPU".to_string(),
            vec![],
            vec!["l2_thrashing".to_string()],
        );
        assert!(result.is_healthy());
        assert_eq!(result.exit_code(), 0);
        assert!(result.max_severity.is_none());
    }

    #[test]
    fn test_diagnosis_result_with_finding() {
        let finding = DiagnosticFinding {
            category: DiagnosticCategory::MemoryBound(MemoryIssue::HbmDegradation {
                measured_gbps: 2500.0,
                expected_min_gbps: 3350.0,
                ratio: 0.75,
            }),
            severity: Severity::Warning,
            summary: "HBM bandwidth at 75%".to_string(),
            cause: "Partial HBM stack failure".to_string(),
            fix: "RMA GPU".to_string(),
        };
        let result = DiagnosisResult::new("H100".to_string(), vec![finding], vec![]);
        assert!(!result.is_healthy());
        assert_eq!(result.exit_code(), 0); // Warning, not Critical
        assert_eq!(result.max_severity, Some(Severity::Warning));
    }

    #[test]
    fn test_diagnosis_result_critical_exit_code() {
        let finding = DiagnosticFinding {
            category: DiagnosticCategory::Thermal(ThermalIssue::Throttling {
                temperature_c: 95,
                clock_drop_pct: 25.0,
            }),
            severity: Severity::Critical,
            summary: "Severe thermal throttling".to_string(),
            cause: "Cooling failure".to_string(),
            fix: "Check fans and thermal paste".to_string(),
        };
        let result = DiagnosisResult::new("H100".to_string(), vec![finding], vec![]);
        assert_eq!(result.exit_code(), 1);
    }

    #[test]
    fn test_probe_name_parse() {
        assert_eq!(
            "l2_thrashing".parse::<ProbeName>().unwrap(),
            ProbeName::L2Thrashing
        );
        assert_eq!(
            "thermal".parse::<ProbeName>().unwrap(),
            ProbeName::ThermalThrottling
        );
        assert!("nonexistent".parse::<ProbeName>().is_err());
    }

    #[test]
    fn test_finding_serialization() {
        let finding = DiagnosticFinding {
            category: DiagnosticCategory::ComputeBound(ComputeIssue::ClockStuck {
                measured_mhz: 1095,
                expected_boost_mhz: 1830,
            }),
            severity: Severity::Warning,
            summary: "Clock stuck".to_string(),
            cause: "Firmware issue".to_string(),
            fix: "Update driver".to_string(),
        };
        let json = serde_json::to_string(&finding).unwrap();
        let _: DiagnosticFinding = serde_json::from_str(&json).unwrap();
    }
}
