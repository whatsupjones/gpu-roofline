//! GPU diagnostic engine — "Why Is My GPU Slow?"
//!
//! Runs targeted probes to identify root causes of GPU underperformance:
//! L2 cache thrashing, HBM degradation, thermal throttling, clock issues,
//! PCIe bottlenecks, and compute deficits. Each finding includes a cause
//! and a suggested fix.

pub mod probes;
pub mod report;
pub mod types;

pub use probes::run_diagnosis;
pub use report::{print_diagnosis_json, print_diagnosis_table};
pub use types::{
    DiagnoseConfig, DiagnosisResult, DiagnosticCategory, DiagnosticFinding, ProbeName, Severity,
};
