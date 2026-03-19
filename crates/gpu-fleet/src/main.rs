#![allow(dead_code)]

use std::process;

use clap::Parser;
use gpu_harness::sim::{fleet::SimulatedFleet, profiles, SimulatedBackend};
use gpu_harness::GpuBackend;

mod cli;
mod fleet_validate;
mod straggler;
mod symmetry;
mod topology;

use cli::{Cli, Commands, OutputFormat};

fn main() {
    let cli = Cli::parse();

    let filter = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    let exit_code = match cli.command {
        Commands::Topology { sim, count } => cmd_topology(sim, count, &cli.format, cli.no_color),
        Commands::Validate {
            threshold,
            sim,
            count,
        } => cmd_validate(threshold, sim, count, &cli.format, cli.no_color),
        Commands::Symmetry { sim, count } => cmd_symmetry(sim, count, &cli.format, cli.no_color),
        Commands::Straggler {
            threshold,
            sim,
            count,
        } => cmd_straggler(threshold, sim, count, &cli.format, cli.no_color),
    };

    process::exit(exit_code);
}

fn get_fleet_backend(sim: &Option<String>, count: u32) -> Result<Box<dyn GpuBackend>, String> {
    match sim {
        Some(profile_name) => {
            let profile = profiles::profile_by_name(profile_name).ok_or_else(|| {
                format!(
                    "Unknown profile '{}'. Run 'gpu-roofline profiles' for available profiles.",
                    profile_name
                )
            })?;
            let fleet = SimulatedFleet::homogeneous(profile, count);
            Ok(Box::new(SimulatedBackend::with_fleet(fleet)))
        }
        None => {
            // Real hardware: try CUDA first, then wgpu
            #[cfg(feature = "cuda")]
            {
                if let Ok(backend) = gpu_harness::CudaBackend::new() {
                    let devices = backend.discover_devices().unwrap_or_default();
                    eprintln!("Discovered {} GPU(s) via CUDA", devices.len());
                    for d in &devices {
                        eprintln!(
                            "  GPU {}: {} | {:.1} GB VRAM",
                            d.index,
                            d.name,
                            d.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                        );
                    }
                    return Ok(Box::new(backend));
                }
            }
            #[cfg(feature = "wgpu-backend")]
            {
                if let Ok(backend) = gpu_harness::WgpuBackend::new() {
                    let devices = backend.discover_devices().unwrap_or_default();
                    eprintln!("Discovered {} GPU(s) via wgpu", devices.len());
                    return Ok(Box::new(backend));
                }
            }
            Err(
                "No GPU backend available. Use --sim <profile> --count <n> for simulation."
                    .to_string(),
            )
        }
    }
}

fn cmd_topology(sim: Option<String>, count: u32, format: &OutputFormat, no_color: bool) -> i32 {
    let backend = match get_fleet_backend(&sim, count) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };

    let view = match topology::discover_topology(backend.as_ref()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Topology discovery failed: {e}");
            return 1;
        }
    };

    match format {
        OutputFormat::Json => topology::print_topology_json(&view),
        OutputFormat::Table => topology::print_topology_tree(&view, no_color),
    }
    0
}

fn cmd_symmetry(sim: Option<String>, count: u32, format: &OutputFormat, no_color: bool) -> i32 {
    let backend = match get_fleet_backend(&sim, count) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };

    let report = match symmetry::check_symmetry(backend.as_ref()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Symmetry check failed: {e}");
            return 1;
        }
    };

    match format {
        OutputFormat::Json => symmetry::print_symmetry_json(&report),
        OutputFormat::Table => symmetry::print_symmetry_table(&report, no_color),
    }

    if report.is_symmetric {
        0
    } else {
        1
    }
}

fn cmd_validate(
    threshold: f64,
    sim: Option<String>,
    count: u32,
    format: &OutputFormat,
    no_color: bool,
) -> i32 {
    let backend = match get_fleet_backend(&sim, count) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };

    eprintln!("Validating {} GPUs...", count);
    let result = match fleet_validate::validate_fleet(backend.as_ref(), threshold) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Fleet validation failed: {e}");
            return 1;
        }
    };

    match format {
        OutputFormat::Json => fleet_validate::print_fleet_validate_json(&result),
        OutputFormat::Table => fleet_validate::print_fleet_validate_table(&result, no_color),
    }

    if result.all_passed {
        0
    } else {
        1
    }
}

fn cmd_straggler(
    threshold: f64,
    sim: Option<String>,
    count: u32,
    format: &OutputFormat,
    no_color: bool,
) -> i32 {
    let backend = match get_fleet_backend(&sim, count) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };

    eprintln!("Measuring {} GPUs for straggler detection...", count);
    let report = match straggler::detect_stragglers(backend.as_ref(), threshold) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Straggler detection failed: {e}");
            return 1;
        }
    };

    match format {
        OutputFormat::Json => straggler::print_straggler_json(&report),
        OutputFormat::Table => straggler::print_straggler_table(&report, no_color),
    }

    if report.stragglers.is_empty() {
        0
    } else {
        1
    }
}
