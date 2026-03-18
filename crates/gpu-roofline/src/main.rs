// The binary uses the lib's modules directly (not via the crate re-exports),
// so some pub items appear unused from the binary's perspective. They are
// part of the public library API for downstream consumers.
#![allow(dead_code)]

use std::process;

use clap::Parser;
use gpu_harness::sim::{profiles, SimulatedBackend};
use gpu_harness::GpuBackend;
#[cfg(feature = "wgpu-backend")]
use gpu_harness::WgpuBackend;

mod ceilings;
mod cli;
mod kernels;
mod model;
mod output;
mod validate;

use cli::{BackendChoice, Cli, Commands, OutputFormat};

fn main() {
    let cli = Cli::parse();

    // Set up tracing
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
        Commands::Measure {
            device: _,
            burst,
            duration,
            sim,
            save_baseline,
        } => cmd_measure(
            burst,
            duration,
            sim,
            save_baseline,
            &cli.format,
            cli.no_color,
            &cli.backend,
        ),
        Commands::Check {
            baseline,
            threshold,
            sim,
        } => cmd_check(&baseline, threshold, sim, cli.no_color, &cli.backend),
        Commands::Validate {
            threshold,
            strict,
            sim,
            baseline: _,
        } => cmd_validate(
            if strict { 0.9 } else { threshold },
            sim,
            &cli.format,
            cli.no_color,
            &cli.backend,
        ),
        Commands::Profiles => cmd_profiles(),
    };

    process::exit(exit_code);
}

fn get_backend(
    sim: &Option<String>,
    backend_choice: &BackendChoice,
) -> Result<Box<dyn GpuBackend>, String> {
    match sim {
        Some(profile_name) => {
            let profile = profiles::profile_by_name(profile_name).ok_or_else(|| {
                format!(
                    "Unknown profile '{}'. Run 'gpu-roofline profiles' to list available profiles.",
                    profile_name
                )
            })?;
            Ok(Box::new(SimulatedBackend::new(profile)))
        }
        None => {
            // CUDA backend (datacenter GPUs)
            if matches!(backend_choice, BackendChoice::Cuda) {
                #[cfg(feature = "cuda")]
                {
                    match gpu_harness::CudaBackend::new() {
                        Ok(backend) => {
                            let devices = backend.discover_devices().unwrap_or_default();
                            for d in &devices {
                                eprintln!(
                                    "  GPU: {} | CUDA sm_{}{} | {:.1} GB VRAM",
                                    d.name,
                                    d.features.compute_capability.map(|(m, _)| m).unwrap_or(0),
                                    d.features.compute_capability.map(|(_, m)| m).unwrap_or(0),
                                    d.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                                );
                            }
                            return Ok(Box::new(backend));
                        }
                        Err(e) => {
                            return Err(format!(
                                "CUDA backend failed: {e}\n\
                                 Try --backend auto for Vulkan/DX12 fallback."
                            ));
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err("CUDA backend not available. Build with --features cuda.\n\
                         Try --backend auto for Vulkan/DX12."
                        .to_string());
                }
            }

            // Auto mode: try CUDA first (for datacenter), then wgpu
            if matches!(backend_choice, BackendChoice::Auto) {
                #[cfg(feature = "cuda")]
                {
                    if let Ok(backend) = gpu_harness::CudaBackend::new() {
                        let devices = backend.discover_devices().unwrap_or_default();
                        for d in &devices {
                            eprintln!(
                                "  GPU: {} | CUDA sm_{}{} | {:.1} GB VRAM",
                                d.name,
                                d.features.compute_capability.map(|(m, _)| m).unwrap_or(0),
                                d.features.compute_capability.map(|(_, m)| m).unwrap_or(0),
                                d.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                            );
                        }
                        return Ok(Box::new(backend));
                    }
                    // CUDA failed, fall through to wgpu
                }
            }

            // wgpu backend (Vulkan/DX12/Metal/GL)
            #[cfg(feature = "wgpu-backend")]
            {
                use gpu_harness::wgpu_backend::GpuApiBackend;
                let api = match backend_choice {
                    BackendChoice::Auto => GpuApiBackend::Auto,
                    BackendChoice::Vulkan => GpuApiBackend::Vulkan,
                    BackendChoice::Dx12 => GpuApiBackend::Dx12,
                    BackendChoice::Metal => GpuApiBackend::Metal,
                    BackendChoice::Gl => GpuApiBackend::Gl,
                    BackendChoice::Cuda => unreachable!(), // Handled above
                };
                match WgpuBackend::with_backend(api) {
                    Ok(backend) => {
                        for meta in &backend.metadata {
                            eprintln!(
                                "  GPU: {} | {} | {} | Driver: {}{}",
                                meta.name,
                                meta.backend,
                                meta.device_type,
                                meta.driver,
                                if meta.is_virtual {
                                    " | ⚠ Virtual GPU (cloud passthrough)"
                                } else {
                                    ""
                                }
                            );
                        }
                        Ok(Box::new(backend))
                    }
                    Err(e) => Err(format!(
                        "{e}\n\nUse --sim <profile> for simulation mode.\n\
                         Run 'gpu-roofline profiles' to list available profiles."
                    )),
                }
            }
            #[cfg(not(feature = "wgpu-backend"))]
            {
                let _ = backend_choice;
                Err(
                    "No GPU backend available. Build with --features wgpu-backend or --features cuda.\n\
                     Use --sim <profile> for simulation mode."
                        .to_string(),
                )
            }
        }
    }
}

fn cmd_measure(
    burst: bool,
    duration: u64,
    sim: Option<String>,
    save_baseline: Option<String>,
    format: &OutputFormat,
    no_color: bool,
    backend_choice: &BackendChoice,
) -> i32 {
    let backend = match get_backend(&sim, backend_choice) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };

    if burst {
        // Static burst-only roofline
        let config = ceilings::MeasureConfig::default();
        match ceilings::measure_roofline(backend.as_ref(), &config) {
            Ok(model) => {
                match format {
                    OutputFormat::Table => output::table::print_static(&model, no_color),
                    OutputFormat::Json => output::json::print_static_json(&model),
                    OutputFormat::Ascii => output::ascii::print_static_ascii(&model),
                }
                if let Some(path) = save_baseline {
                    save_json(&model, &path);
                }
                0
            }
            Err(e) => {
                eprintln!("Measurement failed: {e}");
                1
            }
        }
    } else {
        // Full dynamic roofline with tension analysis
        let config = model::DynamicConfig {
            duration_secs: duration,
            ..if sim.is_some() {
                model::DynamicConfig::quick()
            } else {
                model::DynamicConfig::default()
            }
        };
        match ceilings::measure_dynamic(backend.as_ref(), &config) {
            Ok(dynamic) => {
                match format {
                    OutputFormat::Table => output::table::print_dynamic(&dynamic, no_color),
                    OutputFormat::Json => output::json::print_dynamic_json(&dynamic),
                    OutputFormat::Ascii => output::ascii::print_dynamic_ascii(&dynamic),
                }
                if let Some(path) = save_baseline {
                    save_json(&dynamic, &path);
                }
                0
            }
            Err(e) => {
                eprintln!("Measurement failed: {e}");
                1
            }
        }
    }
}

fn cmd_check(
    baseline_path: &str,
    threshold: f64,
    sim: Option<String>,
    _no_color: bool,
    backend_choice: &BackendChoice,
) -> i32 {
    // Load baseline
    let baseline_json = match std::fs::read_to_string(baseline_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read baseline '{}': {}", baseline_path, e);
            return 2;
        }
    };

    let baseline: model::DynamicRoofline = match serde_json::from_str(&baseline_json) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to parse baseline JSON: {}", e);
            return 2;
        }
    };

    // Measure current
    let backend = match get_backend(&sim, backend_choice) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };

    let config = if sim.is_some() {
        model::DynamicConfig::quick()
    } else {
        model::DynamicConfig::default()
    };

    let current = match ceilings::measure_dynamic(backend.as_ref(), &config) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Measurement failed: {e}");
            return 1;
        }
    };

    // Compare sustained performance against baseline
    let ratio = if baseline.sustained.peak_gflops > 0.0 {
        current.sustained.peak_gflops / baseline.sustained.peak_gflops
    } else {
        1.0
    };

    let bw_ratio = if baseline.sustained.peak_bandwidth_gbps > 0.0 {
        current.sustained.peak_bandwidth_gbps / baseline.sustained.peak_bandwidth_gbps
    } else {
        1.0
    };

    println!(
        "Baseline: {:.1} GFLOP/s | {:.0} GB/s ({})",
        baseline.sustained.peak_gflops,
        baseline.sustained.peak_bandwidth_gbps,
        baseline.device_name,
    );
    println!(
        "Current:  {:.1} GFLOP/s | {:.0} GB/s ({})",
        current.sustained.peak_gflops, current.sustained.peak_bandwidth_gbps, current.device_name,
    );
    println!("Compute:  {:.1}% of baseline", ratio * 100.0);
    println!("Bandwidth: {:.1}% of baseline", bw_ratio * 100.0);

    if ratio >= threshold && bw_ratio >= threshold {
        println!("\nResult: PASS (threshold: {:.0}%)", threshold * 100.0);
        0
    } else {
        println!("\nResult: FAIL (threshold: {:.0}%)", threshold * 100.0);
        if ratio < threshold {
            println!(
                "  Compute regression: {:.1}% < {:.0}%",
                ratio * 100.0,
                threshold * 100.0
            );
        }
        if bw_ratio < threshold {
            println!(
                "  Bandwidth regression: {:.1}% < {:.0}%",
                bw_ratio * 100.0,
                threshold * 100.0
            );
        }
        1
    }
}

fn cmd_validate(
    threshold: f64,
    sim: Option<String>,
    format: &OutputFormat,
    no_color: bool,
    backend_choice: &BackendChoice,
) -> i32 {
    let backend = match get_backend(&sim, backend_choice) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };

    // Discover GPU and find baseline
    let devices = match backend.discover_devices() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to discover devices: {e}");
            return 2;
        }
    };

    let device = match devices.first() {
        Some(d) => d,
        None => {
            eprintln!("No GPU found");
            return 2;
        }
    };

    let hw_baseline = match validate::find_baseline(device) {
        Some(b) => b,
        None => {
            eprintln!(
                "No baseline found for '{}'. Cannot validate unknown GPU.\n\
                 Use 'gpu-roofline profiles' to see supported GPUs.",
                device.name
            );
            return 2;
        }
    };

    // Use adaptive config for this GPU
    let config = validate::adaptive_config(device);
    validate::log_adaptive_config(device, &config);

    // Run measurement
    let roofline = match ceilings::measure_roofline(backend.as_ref(), &config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Measurement failed: {e}");
            return 1;
        }
    };

    // Validate against baseline
    let result = validate::validate_roofline(&roofline, hw_baseline, threshold);

    // Output
    match format {
        OutputFormat::Json => validate::print_validation_json(&result),
        _ => validate::print_validation_table(&result, no_color),
    }

    result.exit_code()
}

fn cmd_profiles() -> i32 {
    println!("Available simulation profiles:\n");
    for name in profiles::available_profiles() {
        if let Some(p) = profiles::profile_by_name(name) {
            println!(
                "  {:<25} {} | {} | {:.0} GB VRAM | {:.0}W TDP",
                name,
                p.vendor,
                p.architecture,
                p.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                p.power.tdp_watts,
            );
        }
    }
    println!("\nUsage: gpu-roofline measure --sim <profile>");
    0
}

fn save_json<T: serde::Serialize>(data: &T, path: &str) {
    match serde_json::to_string_pretty(data) {
        Ok(json) => match std::fs::write(path, json) {
            Ok(()) => eprintln!("Baseline saved to {path}"),
            Err(e) => eprintln!("Failed to save baseline: {e}"),
        },
        Err(e) => eprintln!("Failed to serialize baseline: {e}"),
    }
}
