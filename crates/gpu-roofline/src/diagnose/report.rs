//! Output formatting for diagnostic results.

use super::types::{DiagnosisResult, Severity};

/// Print diagnosis results as a colored terminal table.
pub fn print_diagnosis_table(result: &DiagnosisResult, _no_color: bool) {
    println!("\ngpu-roofline diagnose | {}\n", result.gpu_name);

    if result.is_healthy() {
        println!("  No issues detected. GPU is healthy.\n");
        println!("  {} probes run | 0 findings", result.probes_run.len());
        return;
    }

    for finding in &result.findings {
        let severity_marker = match finding.severity {
            Severity::Critical => "!!",
            Severity::Warning => "!",
            Severity::Info => "i",
        };
        println!("  [{severity_marker}] {}", finding.summary);
        println!("  Cause: {}", finding.cause);
        println!("  Fix:   {}", finding.fix);
        println!();
    }

    let max_sev = result
        .max_severity
        .map(|s| s.to_string())
        .unwrap_or_else(|| "None".to_string());
    println!(
        "  {} probes run | {} finding(s) | Max severity: {}",
        result.probes_run.len(),
        result.findings.len(),
        max_sev,
    );
}

/// Print diagnosis results as JSON.
pub fn print_diagnosis_json(result: &DiagnosisResult) {
    if let Ok(json) = serde_json::to_string_pretty(result) {
        println!("{json}");
    }
}
