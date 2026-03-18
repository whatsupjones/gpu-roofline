use colored::Colorize;
use comfy_table::{presets::UTF8_FULL_CONDENSED, Cell, ContentArrangement, Table};

use crate::model::dynamic::DynamicRoofline;
use crate::model::roofline::RooflineModel;
use crate::model::Bottleneck;

/// Print a static roofline model as a colored terminal table.
pub fn print_static(model: &RooflineModel, no_color: bool) {
    if no_color {
        colored::control::set_override(false);
    }

    println!(
        "\n{} {} | {}",
        "gpu-roofline".bold().cyan(),
        env!("CARGO_PKG_VERSION"),
        model.device_name.bold()
    );
    println!(
        "  Peak FLOPS:     {} (FP32)",
        format_gflops(model.peak_gflops).green()
    );
    println!(
        "  Peak Bandwidth: {}",
        format!("{:.0} GB/s", model.peak_bandwidth_gbps).green()
    );
    println!(
        "  Ridge Point:    {} FLOP/byte",
        format!("{:.1}", model.ridge_point).yellow()
    );
    println!(
        "  State:          {} MHz | {}°C | {:.0}W\n",
        model.clock_mhz, model.temperature_c, model.power_watts
    );

    print_placements_table(&model.placements, no_color);
}

/// Print a dynamic roofline with tension analysis.
pub fn print_dynamic(dynamic: &DynamicRoofline, no_color: bool) {
    if no_color {
        colored::control::set_override(false);
    }

    println!(
        "\n{} {} | {} | {}",
        "gpu-roofline".bold().cyan(),
        env!("CARGO_PKG_VERSION"),
        dynamic.device_name.bold(),
        "Dynamic Roofline".bold().magenta()
    );

    // Burst line
    println!(
        "\n  {} {:.1} GFLOP/s | {:.0} GB/s | {} MHz | {}°C | {:.0}W",
        "Burst:    ".bold(),
        dynamic.burst.peak_gflops,
        dynamic.burst.peak_bandwidth_gbps,
        dynamic.burst.clock_mhz,
        dynamic.burst.temperature_c,
        dynamic.burst.power_watts,
    );

    // Sustained line
    println!(
        "  {} {:.1} GFLOP/s | {:.0} GB/s | {} MHz | {}°C | {:.0}W",
        "Sustained:".bold(),
        dynamic.sustained.peak_gflops,
        dynamic.sustained.peak_bandwidth_gbps,
        dynamic.sustained.clock_mhz,
        dynamic.sustained.temperature_c,
        dynamic.sustained.power_watts,
    );

    // Equilibrium and net drop
    println!(
        "  {} {:.1}s | Net drop: {}",
        "Equilibrium:".bold(),
        dynamic.equilibrium_time_secs,
        format!("{:.1}%", dynamic.net_ceiling_drop_pct).red()
    );

    // Tensions
    if !dynamic.tensions.is_empty() {
        println!("\n  {}:", "Tension Analysis".bold().yellow());
        for t in &dynamic.tensions {
            let arrow = if t.ceiling_delta_pct < 0.0 {
                "↓".red().to_string()
            } else {
                "↑".green().to_string()
            };
            println!(
                "    {} {:12} {:.1}% after {:.1}s",
                arrow,
                t.name,
                t.ceiling_delta_pct.abs(),
                t.onset_time_secs
            );
            println!("      {} {}", "excites:".dimmed(), t.force_a.dimmed());
            println!("      {} {}", "inhibits:".dimmed(), t.force_b.dimmed());
        }
    }

    // Kernel placements (if available)
    if !dynamic.sustained.placements.is_empty() {
        println!();
        print_placements_table(&dynamic.sustained.placements, no_color);
    }

    println!();
}

fn print_placements_table(placements: &[crate::model::KernelPlacement], _no_color: bool) {
    if placements.is_empty() {
        return;
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec![
        Cell::new("Kernel"),
        Cell::new("AI (FLOP/B)"),
        Cell::new("GFLOP/s"),
        Cell::new("BW (GB/s)"),
        Cell::new("Efficiency"),
        Cell::new("Bottleneck"),
        Cell::new("Median"),
        Cell::new("CV"),
    ]);

    for p in placements {
        let efficiency_str = format!("{:.0}%", p.efficiency * 100.0);
        let efficiency_cell = if p.efficiency >= 0.7 {
            Cell::new(efficiency_str.to_string())
        } else {
            Cell::new(format!("{efficiency_str} ⚠"))
        };

        let bottleneck_str = match p.bottleneck {
            Bottleneck::ComputeBound => "Compute".to_string(),
            Bottleneck::MemoryBound { level } => format!("Memory ({level})"),
        };

        table.add_row(vec![
            Cell::new(&p.name),
            Cell::new(format!("{:.2}", p.arithmetic_intensity)),
            Cell::new(format!("{:.1}", p.achieved_gflops)),
            Cell::new(format!("{:.0}", p.achieved_bandwidth_gbps)),
            efficiency_cell,
            Cell::new(bottleneck_str),
            Cell::new(format!("{:.1} µs", p.median_us)),
            Cell::new(format!("{:.1}%", p.cv * 100.0)),
        ]);
    }

    println!("{table}");
}

/// Format GFLOP/s with appropriate unit (GFLOP/s or TFLOP/s).
fn format_gflops(gflops: f64) -> String {
    if gflops >= 1000.0 {
        format!("{:.1} TFLOP/s", gflops / 1000.0)
    } else {
        format!("{:.1} GFLOP/s", gflops)
    }
}
