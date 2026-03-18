use colored::Colorize;
use comfy_table::{presets::UTF8_FULL_CONDENSED, Cell, ContentArrangement, Table};

use super::checks::ValidationResult;

/// Print validation result as a colored terminal table.
pub fn print_validation_table(result: &ValidationResult, no_color: bool) {
    if no_color {
        colored::control::set_override(false);
    }

    println!(
        "\n{} {} | {}",
        "gpu-roofline validate".bold().cyan(),
        env!("CARGO_PKG_VERSION"),
        result.gpu_name.bold()
    );

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec![
        Cell::new("Check"),
        Cell::new("Measured"),
        Cell::new("Expected"),
        Cell::new("Status"),
    ]);

    for check in &result.checks {
        let status = if check.passed {
            "PASS".green().bold().to_string()
        } else {
            "FAIL".red().bold().to_string()
        };

        table.add_row(vec![
            Cell::new(&check.name),
            Cell::new(&check.measured),
            Cell::new(&check.expected),
            Cell::new(&status),
        ]);
    }

    println!("\n{table}");

    // Diagnosis
    if !result.diagnosis.is_empty() {
        println!("\n  {}:", "Diagnosis".bold().yellow());
        for (i, diag) in result.diagnosis.iter().enumerate() {
            let prefix = if i == result.diagnosis.len() - 1 {
                "└─"
            } else {
                "├─"
            };
            println!("  {} {}", prefix, diag);
        }
    }

    // Final result
    let result_str = if result.passed {
        format!("PASS ({}/{} checks)", result.pass_count, result.total_count)
            .green()
            .bold()
            .to_string()
    } else {
        format!(
            "FAIL ({}/{} checks passed)",
            result.pass_count, result.total_count
        )
        .red()
        .bold()
        .to_string()
    };

    println!("\n  Result: {}", result_str);
    println!("  Exit code: {}\n", result.exit_code());
}

/// Print validation result as JSON.
pub fn print_validation_json(result: &ValidationResult) {
    match serde_json::to_string_pretty(result) {
        Ok(json) => println!("{json}"),
        Err(e) => eprintln!("Error serializing validation result: {e}"),
    }
}
