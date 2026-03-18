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
mod diagnose;
mod kernels;
mod model;
mod monitor;
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
        Commands::Monitor {
            interval,
            duration,
            alert_threshold,
            daemon,
            log,
            sim,
        } => cmd_monitor(
            interval,
            duration,
            alert_threshold,
            daemon,
            log,
            sim,
            &cli.format,
            cli.no_color,
            &cli.backend,
        ),
        Commands::Diagnose {
            device: _,
            probes,
            sim,
        } => cmd_diagnose(probes, sim, &cli.format, cli.no_color, &cli.backend),
        Commands::Profiles => cmd_profiles(),
        #[cfg(feature = "vgpu")]
        Commands::Vgpu { action } => cmd_vgpu(action, &cli.format),
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

#[allow(clippy::too_many_arguments)]
fn cmd_monitor(
    interval: u64,
    duration: u64,
    alert_threshold: f64,
    daemon: bool,
    log_path: Option<String>,
    sim: Option<String>,
    format: &OutputFormat,
    _no_color: bool,
    backend_choice: &BackendChoice,
) -> i32 {
    let backend = match get_backend(&sim, backend_choice) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };

    // Get device info for display
    let devices = backend.discover_devices().unwrap_or_default();
    let device = devices.first();
    let device_name = device
        .map(|d| d.name.clone())
        .unwrap_or_else(|| "Unknown GPU".to_string());
    let driver_version = device
        .and_then(|d| d.driver_version.clone())
        .unwrap_or_default();
    let compute_cap = device
        .and_then(|d| d.features.compute_capability)
        .map(|(maj, min)| format!("sm_{}{}", maj, min))
        .unwrap_or_default();
    let vram_total = device.map(|d| d.memory_bytes).unwrap_or(0);

    // Run initial baseline measurement
    eprintln!("Taking baseline measurement on {device_name}...");
    let baseline_config = ceilings::MeasureConfig {
        buffer_size_bytes: 256 * 1024 * 1024,
        measurement_iterations: 20,
        warmup_iterations: 5,
        kernels: vec![
            crate::kernels::BuiltinKernel::Copy,
            crate::kernels::BuiltinKernel::FmaHeavy,
        ],
        device_index: 0,
    };

    let baseline = match ceilings::measure_roofline(backend.as_ref(), &baseline_config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Baseline measurement failed: {e}");
            return 1;
        }
    };

    eprintln!(
        "Baseline: {:.0} GB/s | {:.1} GFLOP/s",
        baseline.peak_bandwidth_gbps, baseline.peak_gflops
    );

    let monitor_config = monitor::MonitorConfig {
        interval_secs: interval,
        duration_secs: duration,
        alert_threshold,
        buffer_size_bytes: 256 * 1024 * 1024,
        iterations_per_sample: 10,
        log_path: log_path.clone(),
        daemon,
    };

    let mut sampler = monitor::Sampler::new(monitor_config, &baseline);

    // Open log file if requested
    let mut log_file = log_path.as_ref().and_then(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
    });

    // Daemon mode: JSON lines, no TUI
    if daemon {
        let result = sampler.run(backend.as_ref(), |sample| {
            if let Some(ref mut file) = log_file {
                use std::io::Write;
                if let Ok(json) = serde_json::to_string(sample) {
                    let _ = writeln!(file, "{json}");
                }
            }
            if let Ok(json) = serde_json::to_string(sample) {
                println!("{json}");
            }
            true
        });
        return match result {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("Monitoring error: {e}");
                1
            }
        };
    }

    // TUI mode (default for interactive)
    let mut tui_state = monitor::tui::TuiState::new(monitor::tui::TuiConfig {
        baseline_bw: baseline.peak_bandwidth_gbps,
        baseline_gflops: baseline.peak_gflops,
        device_name,
        driver_version,
        compute_capability: compute_cap,
        max_clock_mhz: baseline.clock_mhz,
        tdp_watts: if baseline.power_watts > 0.0 {
            baseline.power_watts
        } else {
            700.0
        },
        vram_total_bytes: vram_total,
    });

    let mut terminal = match monitor::tui::init_terminal() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to initialize TUI: {e}");
            eprintln!("Try --daemon for headless mode.");
            return 2;
        }
    };

    let result = sampler.run(backend.as_ref(), |sample| {
        // Write JSON log line
        if let Some(ref mut file) = log_file {
            use std::io::Write;
            if let Ok(json) = serde_json::to_string(sample) {
                let _ = writeln!(file, "{json}");
            }
        }

        tui_state.push_sample(sample.clone());

        // Render TUI
        let _ = terminal.draw(|frame| {
            monitor::tui::draw(frame, &tui_state);
        });

        // Check for quit key
        !monitor::tui::poll_input()
    });

    // Restore terminal
    let _ = monitor::tui::restore_terminal();

    match result {
        Ok(samples) => {
            eprintln!("Monitoring complete: {} samples collected", samples.len());

            if !samples.is_empty() {
                let avg_bw: f64 =
                    samples.iter().map(|s| s.bandwidth_gbps).sum::<f64>() / samples.len() as f64;
                let avg_gflops: f64 =
                    samples.iter().map(|s| s.gflops).sum::<f64>() / samples.len() as f64;
                let alert_count: usize = samples.iter().map(|s| s.alerts.len()).sum();

                eprintln!(
                    "Average: {:.0} GB/s | {:.0} GFLOP/s | {} alerts",
                    avg_bw, avg_gflops, alert_count
                );
            }

            // JSON summary output
            if matches!(format, OutputFormat::Json) {
                if let Ok(json) = serde_json::to_string_pretty(&samples) {
                    println!("{json}");
                }
            }

            0
        }
        Err(e) => {
            eprintln!("Monitoring error: {e}");
            1
        }
    }
}

fn cmd_diagnose(
    probes: Option<Vec<String>>,
    sim: Option<String>,
    format: &OutputFormat,
    no_color: bool,
    backend_choice: &BackendChoice,
) -> i32 {
    use diagnose::{DiagnoseConfig, ProbeName};

    let backend = match get_backend(&sim, backend_choice) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };

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

    let baseline = match validate::find_baseline(device) {
        Some(b) => b,
        None => {
            eprintln!(
                "No baseline found for '{}'. Cannot diagnose unknown GPU.",
                device.name
            );
            return 2;
        }
    };

    // Parse probe names if specified
    let probe_list = match probes {
        Some(names) => {
            let mut parsed = Vec::new();
            for name in &names {
                match name.parse::<ProbeName>() {
                    Ok(p) => parsed.push(p),
                    Err(e) => {
                        eprintln!("Invalid probe name '{name}': {e}");
                        eprintln!(
                            "Available: l2_thrashing, hbm_degradation, pci_bottleneck, \
                             thermal, clock, compute"
                        );
                        return 2;
                    }
                }
            }
            parsed
        }
        None => ProbeName::all().to_vec(),
    };

    let config = DiagnoseConfig {
        device_index: 0,
        probes: probe_list,
        measurement_iterations: 50,
    };

    eprintln!(
        "Running {} diagnostic probes on {}...",
        config.probes.len(),
        device.name
    );

    let result = match diagnose::run_diagnosis(backend.as_ref(), baseline, &config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Diagnosis failed: {e}");
            return 1;
        }
    };

    match format {
        OutputFormat::Json => diagnose::print_diagnosis_json(&result),
        _ => diagnose::print_diagnosis_table(&result, no_color),
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

#[cfg(feature = "vgpu")]
fn cmd_vgpu(action: cli::VgpuAction, format: &OutputFormat) -> i32 {
    match action {
        cli::VgpuAction::Watch {
            device: _,
            daemon,
            log,
            sim,
            contention_threshold,
        } => cmd_vgpu_watch(sim, daemon, log, contention_threshold, format),
        cli::VgpuAction::List { sim, json } => cmd_vgpu_list(sim, json),
        cli::VgpuAction::Scenarios => cmd_vgpu_scenarios(),
    }
}

#[cfg(feature = "vgpu")]
fn cmd_vgpu_watch(
    sim: Option<String>,
    daemon: bool,
    log_path: Option<String>,
    contention_threshold: f64,
    _format: &OutputFormat,
) -> i32 {
    use gpu_harness::vgpu::{self, SimulatedDetector, VgpuDetector};
    use monitor::vgpu_sampler::{VgpuMonitorConfig, VgpuSampler};
    use std::sync::mpsc;

    let (detector, physical_vram, technology, partitioning_mode, scenario_name): (
        Box<dyn VgpuDetector>,
        u64,
        _,
        _,
        String,
    ) = match &sim {
        Some(name) => {
            let scenario = match vgpu::scenario_by_name(name) {
                Some(s) => s,
                None => {
                    eprintln!(
                        "Unknown scenario '{}'. Run 'gpu-roofline vgpu scenarios' for a list.",
                        name
                    );
                    return 2;
                }
            };
            let tech = scenario.technology;
            let mode = scenario.partitioning_mode;
            let vram = scenario.physical_vram_bytes;
            let sname = scenario.name.clone();
            (
                Box::new(SimulatedDetector::new(scenario)) as Box<dyn VgpuDetector>,
                vram,
                tech,
                mode,
                sname,
            )
        }
        None => {
            // Real hardware detection
            let composite = vgpu::detect::auto_detect();
            if !composite.is_available() {
                eprintln!("No vGPU technology detected on this system.");
                eprintln!("Use --sim <scenario> for simulation mode.");
                eprintln!("Run 'gpu-roofline vgpu scenarios' for available scenarios.");
                return 2;
            }
            let tech = composite.technology();
            (
                Box::new(composite) as Box<dyn VgpuDetector>,
                0, // Will be filled from NVML if available
                tech,
                gpu_harness::vgpu::state::PartitioningMode::HardwarePartitioned,
                "live".to_string(),
            )
        }
    };

    let config = VgpuMonitorConfig {
        sample_interval_secs: 1,
        contention_threshold,
        daemon,
        log_path: log_path.clone(),
    };

    let mut sampler = VgpuSampler::new(config, physical_vram);
    sampler.set_technology(technology, partitioning_mode);

    let (tx, rx) = mpsc::channel();

    // Run detector in background thread
    let detector_handle = std::thread::spawn(move || {
        if let Err(e) = detector.watch(tx) {
            tracing::error!("detector error: {e}");
        }
    });

    // Open log file if requested
    let mut log_file = log_path.as_ref().and_then(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
    });

    if daemon {
        // Daemon mode: JSON lines only
        let events = sampler.run(rx, |event, alerts| {
            if let Some(ref mut file) = log_file {
                use std::io::Write;
                if let Ok(json) = serde_json::to_string(event) {
                    let _ = writeln!(file, "{json}");
                }
            }
            if let Ok(json) = serde_json::to_string(event) {
                println!("{json}");
            }
            for alert in alerts {
                if let Ok(json) = serde_json::to_string(alert) {
                    eprintln!("{json}");
                }
            }
            true
        });

        let _ = detector_handle.join();
        eprintln!("{} events processed", events.len());
        0
    } else {
        // TUI mode
        let mut tui_state = monitor::vgpu_tui::VgpuTuiState::new(Some(scenario_name));

        let mut terminal = match monitor::tui::init_terminal() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to initialize TUI: {e}");
                eprintln!("Try --daemon for headless mode.");
                return 2;
            }
        };

        let events = sampler.run(rx, |event, alerts| {
            // Write JSON log line
            if let Some(ref mut file) = log_file {
                use std::io::Write;
                if let Ok(json) = serde_json::to_string(event) {
                    let _ = writeln!(file, "{json}");
                }
            }

            tui_state.push_event(event, alerts);

            // Render TUI
            let _ = terminal.draw(|frame| {
                monitor::vgpu_tui::draw(frame, &tui_state);
            });

            // Check for quit key
            !monitor::vgpu_tui::poll_quit()
        });

        // Restore terminal
        let _ = monitor::tui::restore_terminal();

        let _ = detector_handle.join();

        eprintln!(
            "vGPU monitoring complete: {} events, {} alerts",
            events.len(),
            sampler.alerts().len()
        );
        0
    }
}

#[cfg(feature = "vgpu")]
fn cmd_vgpu_list(sim: Option<String>, json: bool) -> i32 {
    use gpu_harness::vgpu::{self, SimulatedDetector, VgpuDetector};

    let instances = match &sim {
        Some(name) => {
            let scenario = match vgpu::scenario_by_name(name) {
                Some(s) => s,
                None => {
                    eprintln!("Unknown scenario '{name}'.");
                    return 2;
                }
            };

            // For list, we run the scenario quickly and collect created instances
            let detector = SimulatedDetector::new(vgpu::VgpuSimScenario {
                name: scenario.name.clone(),
                description: scenario.description.clone(),
                technology: scenario.technology,
                partitioning_mode: scenario.partitioning_mode,
                physical_vram_bytes: scenario.physical_vram_bytes,
                events: scenario
                    .events
                    .into_iter()
                    .map(|mut e| {
                        e.delay_secs = 0.0;
                        e
                    })
                    .collect(),
            });

            let (tx, rx) = std::sync::mpsc::channel();
            detector.watch(tx).ok();

            let mut instances = Vec::new();
            for event in rx.try_iter() {
                if let vgpu::VgpuEventType::Created { instance, .. } = event.event_type {
                    instances.push(instance);
                }
            }
            instances
        }
        None => {
            // Real hardware enumeration
            let composite = vgpu::detect::auto_detect();
            if !composite.is_available() {
                eprintln!("No vGPU technology detected on this system.");
                eprintln!("Use --sim <scenario> for simulation.");
                return 2;
            }
            match composite.enumerate() {
                Ok(insts) => insts,
                Err(e) => {
                    eprintln!("Enumeration failed: {e}");
                    return 1;
                }
            }
        }
    };

    if json {
        match serde_json::to_string_pretty(&instances) {
            Ok(j) => println!("{j}"),
            Err(e) => {
                eprintln!("JSON error: {e}");
                return 1;
            }
        }
    } else {
        println!(
            "{:<15} {:<25} {:<15} {:<10} {:<10}",
            "ID", "Name", "Technology", "VRAM", "Phase"
        );
        println!("{}", "-".repeat(75));
        for inst in &instances {
            println!(
                "{:<15} {:<25} {:<15} {:<10} {:<10}",
                inst.id,
                inst.name,
                inst.technology,
                format!(
                    "{:.0}G",
                    inst.vram_allocated_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                ),
                inst.phase,
            );
        }
        println!("\n{} instance(s)", instances.len());
    }
    0
}

#[cfg(feature = "vgpu")]
fn cmd_vgpu_scenarios() -> i32 {
    use gpu_harness::vgpu;

    println!("Available vGPU simulation scenarios:\n");
    for name in vgpu::available_scenarios() {
        if let Some(s) = vgpu::scenario_by_name(name) {
            println!(
                "  {:<20} {} | {} | {} events",
                name,
                s.technology,
                s.partitioning_mode_label(),
                s.events.len(),
            );
            println!("  {:<20} {}", "", s.description);
            println!();
        }
    }
    println!("Usage: gpu-roofline vgpu watch --sim <scenario>");
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
