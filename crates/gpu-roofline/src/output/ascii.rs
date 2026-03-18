use crate::model::dynamic::DynamicRoofline;
use crate::model::roofline::RooflineModel;

const CHART_WIDTH: usize = 70;
const CHART_HEIGHT: usize = 18;

/// Print an ASCII roofline chart for a static model.
pub fn print_static_ascii(model: &RooflineModel) {
    println!("\n  {} | {}", model.device_name, "Static Roofline");
    println!(
        "  Peak: {} | {:.0} GB/s | Ridge: {:.1} FLOP/byte\n",
        format_gflops(model.peak_gflops),
        model.peak_bandwidth_gbps,
        model.ridge_point,
    );
    print_roofline_chart(
        model.peak_gflops,
        model.peak_bandwidth_gbps,
        model.ridge_point,
        &model.placements,
        None,
        None,
    );
}

/// Print an ASCII roofline chart for a dynamic model (burst + sustained overlay).
pub fn print_dynamic_ascii(dynamic: &DynamicRoofline) {
    println!(
        "\n  {} | Dynamic Roofline",
        dynamic.device_name
    );
    println!(
        "  Burst:     {} | {:.0} GB/s",
        format_gflops(dynamic.burst.peak_gflops),
        dynamic.burst.peak_bandwidth_gbps,
    );
    println!(
        "  Sustained: {} | {:.0} GB/s | Drop: {:.1}%\n",
        format_gflops(dynamic.sustained.peak_gflops),
        dynamic.sustained.peak_bandwidth_gbps,
        dynamic.net_ceiling_drop_pct,
    );

    print_roofline_chart(
        dynamic.burst.peak_gflops,
        dynamic.burst.peak_bandwidth_gbps,
        dynamic.burst.ridge_point,
        &dynamic.sustained.placements,
        Some(dynamic.sustained.peak_gflops),
        Some(dynamic.sustained.peak_bandwidth_gbps),
    );

    // Tension summary below chart
    if !dynamic.tensions.is_empty() {
        println!("  Tensions:");
        for t in &dynamic.tensions {
            println!(
                "    {} {:12} {:>5.1}% after {:.1}s",
                if t.ceiling_delta_pct < 0.0 { "|-" } else { "|+" },
                t.name,
                t.ceiling_delta_pct.abs(),
                t.onset_time_secs
            );
        }
    }
    println!();
}

fn print_roofline_chart(
    peak_gflops: f64,
    peak_bw_gbps: f64,
    ridge_point: f64,
    placements: &[crate::model::KernelPlacement],
    sustained_gflops: Option<f64>,
    _sustained_bw: Option<f64>,
) {
    // Log-scale X axis: arithmetic intensity from 0.01 to 1000
    let x_min: f64 = 0.01;
    let x_max: f64 = 1000.0;
    let y_max = peak_gflops * 1.1; // 10% headroom
    let y_min = peak_gflops * 0.001; // 3 orders of magnitude

    let log_x_min = x_min.log10();
    let log_x_max = x_max.log10();
    let log_y_min = y_min.log10();
    let log_y_max = y_max.log10();

    // Build character grid
    let mut grid = vec![vec![' '; CHART_WIDTH]; CHART_HEIGHT];

    // Draw roofline (bandwidth slope + compute ceiling)
    for col in 0..CHART_WIDTH {
        let log_ai = log_x_min + (col as f64 / CHART_WIDTH as f64) * (log_x_max - log_x_min);
        let ai = 10.0_f64.powf(log_ai);

        // Burst ceiling
        let ceiling = (peak_bw_gbps * ai).min(peak_gflops);
        let log_ceiling = ceiling.log10();
        let row = ((log_y_max - log_ceiling) / (log_y_max - log_y_min) * (CHART_HEIGHT - 1) as f64)
            as i32;
        if row >= 0 && (row as usize) < CHART_HEIGHT {
            grid[row as usize][col] = if ai >= ridge_point { '=' } else { '/' };
        }

        // Sustained ceiling (if dynamic)
        if let Some(s_gflops) = sustained_gflops {
            let s_ceiling = (peak_bw_gbps * 0.93 * ai).min(s_gflops); // Approx sustained BW
            let log_s = s_ceiling.log10();
            let s_row =
                ((log_y_max - log_s) / (log_y_max - log_y_min) * (CHART_HEIGHT - 1) as f64) as i32;
            if s_row >= 0 && (s_row as usize) < CHART_HEIGHT && s_row != row {
                grid[s_row as usize][col] = '-';
            }
        }
    }

    // Place kernel markers
    for p in placements {
        if p.arithmetic_intensity <= 0.0 {
            continue;
        }
        let log_ai = p.arithmetic_intensity.log10();
        let log_perf = if p.achieved_gflops > 0.0 {
            p.achieved_gflops.log10()
        } else {
            continue;
        };

        let col = ((log_ai - log_x_min) / (log_x_max - log_x_min) * CHART_WIDTH as f64) as i32;
        let row =
            ((log_y_max - log_perf) / (log_y_max - log_y_min) * (CHART_HEIGHT - 1) as f64) as i32;

        if col >= 0 && (col as usize) < CHART_WIDTH && row >= 0 && (row as usize) < CHART_HEIGHT {
            grid[row as usize][col as usize] = '*';
        }
    }

    // Y-axis labels
    let y_labels = [
        (0, format!("{}", format_gflops(peak_gflops))),
        (CHART_HEIGHT / 2, format!("{}", format_gflops(10.0_f64.powf((log_y_max + log_y_min) / 2.0)))),
        (CHART_HEIGHT - 1, format!("{}", format_gflops(y_min))),
    ];

    // Print chart
    println!("  Performance (GFLOP/s)");
    for row in 0..CHART_HEIGHT {
        let label = y_labels
            .iter()
            .find(|(r, _)| *r == row)
            .map(|(_, l)| format!("{:>12}", l))
            .unwrap_or_else(|| "            ".to_string());

        let line: String = grid[row].iter().collect();
        println!("  {} |{}", label, line);
    }

    // X-axis
    let x_axis: String = (0..CHART_WIDTH).map(|_| '-').collect();
    println!("  {:>12} +{}", "", x_axis);
    println!(
        "  {:>12}  {:<10} {:<10} {:<10} {:<10} {:<10}",
        "", "0.01", "0.1", "1", "10", "100"
    );
    println!("  {:>12}  Arithmetic Intensity (FLOP/byte)", "");

    // Legend
    println!();
    if sustained_gflops.is_some() {
        println!("  Legend: / = burst BW slope | = = burst compute | - = sustained ceiling | * = kernel");
    } else {
        println!("  Legend: / = BW slope | = = compute ceiling | * = kernel");
    }
}

fn format_gflops(gflops: f64) -> String {
    if gflops >= 1000.0 {
        format!("{:.1}T", gflops / 1000.0)
    } else if gflops >= 1.0 {
        format!("{:.0}G", gflops)
    } else {
        format!("{:.2}G", gflops)
    }
}
