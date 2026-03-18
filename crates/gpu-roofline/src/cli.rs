use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "gpu-roofline",
    about = "Cross-vendor GPU roofline model with dynamic tension analysis",
    version,
    after_help = "Examples:\n  \
        gpu-roofline measure                     Full dynamic roofline (~120s)\n  \
        gpu-roofline measure --burst             Quick burst-only (~10s)\n  \
        gpu-roofline measure --sim rtx_5090      Simulated RTX 5090 (no GPU needed)\n  \
        gpu-roofline measure --format json        Machine-readable output\n  \
        gpu-roofline validate                    Preflight GPU health check\n  \
        gpu-roofline check --baseline r.json     CI regression check"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Output format
    #[arg(long, global = true, default_value = "table")]
    pub format: OutputFormat,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Graphics API backend (auto-detects by default)
    #[arg(long, global = true, default_value = "auto")]
    pub backend: BackendChoice,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Measure GPU roofline (burst + sustained with tension analysis)
    Measure {
        /// Device index to measure (default: 0)
        #[arg(short, long, default_value = "0")]
        device: u32,

        /// Burst-only mode (quick static roofline, skip thermal ramp)
        #[arg(long)]
        burst: bool,

        /// Measurement duration in seconds (dynamic mode only)
        #[arg(long, default_value = "120")]
        duration: u64,

        /// Use simulated GPU profile instead of real hardware
        #[arg(long, value_name = "PROFILE")]
        sim: Option<String>,

        /// Save results as baseline JSON file
        #[arg(long, value_name = "PATH")]
        save_baseline: Option<String>,
    },

    /// Check performance against a baseline (CI mode)
    Check {
        /// Path to baseline JSON file
        #[arg(long, required = true)]
        baseline: String,

        /// Minimum efficiency threshold (0.0-1.0). Exit 1 if below.
        #[arg(long, default_value = "0.9")]
        threshold: f64,

        /// Use simulated GPU profile
        #[arg(long, value_name = "PROFILE")]
        sim: Option<String>,
    },

    /// Validate GPU against expected performance baselines
    Validate {
        /// Minimum performance threshold (0.0-1.0). Default: 0.8 (80%).
        #[arg(long, default_value = "0.8")]
        threshold: f64,

        /// Strict mode: threshold 0.9 (90%)
        #[arg(long)]
        strict: bool,

        /// Use simulated GPU profile
        #[arg(long, value_name = "PROFILE")]
        sim: Option<String>,

        /// Compare against a previous measurement baseline JSON
        #[arg(long, value_name = "PATH")]
        baseline: Option<String>,
    },

    /// Continuously monitor GPU performance and alert on degradation
    Monitor {
        /// Seconds between performance samples (default: 60)
        #[arg(long, default_value = "60")]
        interval: u64,

        /// Total monitoring duration in seconds (0 = indefinite)
        #[arg(long, default_value = "0")]
        duration: u64,

        /// Alert if performance drops below this fraction of baseline (0.0-1.0)
        #[arg(long, default_value = "0.8")]
        alert_threshold: f64,

        /// Run in daemon mode (JSON logging only, no interactive output)
        #[arg(long)]
        daemon: bool,

        /// Write JSON log to this file path
        #[arg(long, value_name = "PATH")]
        log: Option<String>,

        /// Use simulated GPU profile
        #[arg(long, value_name = "PROFILE")]
        sim: Option<String>,
    },

    /// List available simulation profiles
    Profiles,

    /// vGPU lifecycle monitoring: detect provisioning, measure contention, verify teardown
    #[cfg(feature = "vgpu")]
    Vgpu {
        #[command(subcommand)]
        action: VgpuAction,
    },
}

/// vGPU subcommands.
#[cfg(feature = "vgpu")]
#[derive(Subcommand)]
pub enum VgpuAction {
    /// Watch vGPU lifecycle events in real time
    Watch {
        /// Device index to watch (default: all)
        #[arg(short, long)]
        device: Option<u32>,

        /// Run in daemon mode (JSON logging only)
        #[arg(long)]
        daemon: bool,

        /// Write JSON log to this file path
        #[arg(long, value_name = "PATH")]
        log: Option<String>,

        /// Use simulated vGPU scenario instead of real hardware
        #[arg(long, value_name = "SCENARIO")]
        sim: Option<String>,

        /// Contention detection threshold (% drop, default: 5)
        #[arg(long, default_value = "5")]
        contention_threshold: f64,
    },

    /// List current vGPU instances
    List {
        /// Use simulated vGPU scenario
        #[arg(long, value_name = "SCENARIO")]
        sim: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List available simulation scenarios
    Scenarios,
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    /// Colored terminal table
    Table,
    /// ASCII roofline chart
    Ascii,
    /// Machine-readable JSON
    Json,
}

#[derive(Clone, ValueEnum)]
pub enum BackendChoice {
    /// Auto-detect best available backend
    Auto,
    /// Force Vulkan (Linux default)
    Vulkan,
    /// Force DirectX 12 (Windows)
    Dx12,
    /// Force Metal (macOS)
    Metal,
    /// Force OpenGL (broad compatibility)
    Gl,
    /// CUDA native compute (datacenter GPUs: H100, H200, A100)
    Cuda,
}
