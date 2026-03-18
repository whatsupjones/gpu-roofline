#![allow(dead_code)]

use std::process;

use clap::Parser;
use gpu_harness::sim::{fleet::SimulatedFleet, profiles, SimulatedBackend};
use gpu_harness::GpuBackend;

mod cli;
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
            // Real hardware: try to create a backend
            eprintln!("Live hardware fleet discovery not yet implemented.");
            eprintln!("Use --sim <profile> --count <n> for simulation mode.");
            Err("No --sim provided".to_string())
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
    _threshold: f64,
    _sim: Option<String>,
    _count: u32,
    _format: &OutputFormat,
    _no_color: bool,
) -> i32 {
    eprintln!("Fleet validate: coming in Phase 6");
    0
}

fn cmd_straggler(
    _threshold: f64,
    _sim: Option<String>,
    _count: u32,
    _format: &OutputFormat,
    _no_color: bool,
) -> i32 {
    eprintln!("Fleet straggler: coming in Phase 6");
    0
}
