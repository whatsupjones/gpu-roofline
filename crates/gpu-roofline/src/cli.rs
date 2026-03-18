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
        gpu-roofline measure --json              Machine-readable output\n  \
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

    /// List available simulation profiles
    Profiles,
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
}
