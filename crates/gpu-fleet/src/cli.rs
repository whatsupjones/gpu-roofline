use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "gpu-fleet",
    about = "Multi-GPU fleet validation, topology discovery, and straggler detection",
    version,
    after_help = "Examples:\n  \
        gpu-fleet topology --sim h100_sxm --count 8    Show NVLink topology\n  \
        gpu-fleet validate --sim h100_sxm --count 8    Per-GPU health check\n  \
        gpu-fleet symmetry --sim h100_sxm --count 8    Flag config mismatches\n  \
        gpu-fleet straggler --sim h100_sxm --count 8   Find underperformers"
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
}

#[derive(Subcommand)]
pub enum Commands {
    /// Show PCIe/NVLink topology tree and P2P bandwidth matrix
    Topology {
        /// Use simulated GPU fleet
        #[arg(long, value_name = "PROFILE")]
        sim: Option<String>,

        /// Number of GPUs in simulated fleet
        #[arg(long, default_value = "8")]
        count: u32,
    },

    /// Per-GPU roofline health check across the fleet
    Validate {
        /// Performance threshold (0.0-1.0, default: 0.8)
        #[arg(long, default_value = "0.8")]
        threshold: f64,

        /// Use simulated GPU fleet
        #[arg(long, value_name = "PROFILE")]
        sim: Option<String>,

        /// Number of GPUs in simulated fleet
        #[arg(long, default_value = "8")]
        count: u32,
    },

    /// Flag mismatched GPU configurations across the fleet
    Symmetry {
        /// Use simulated GPU fleet
        #[arg(long, value_name = "PROFILE")]
        sim: Option<String>,

        /// Number of GPUs in simulated fleet
        #[arg(long, default_value = "8")]
        count: u32,
    },

    /// Identify underperforming GPUs and diagnose root cause
    Straggler {
        /// Flag GPUs below this fraction of fleet median (default: 0.9)
        #[arg(long, default_value = "0.9")]
        threshold: f64,

        /// Use simulated GPU fleet
        #[arg(long, value_name = "PROFILE")]
        sim: Option<String>,

        /// Number of GPUs in simulated fleet
        #[arg(long, default_value = "8")]
        count: u32,
    },
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Table,
    Json,
}
