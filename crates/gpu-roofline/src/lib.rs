//! gpu-roofline: Cross-vendor dynamic roofline model generator.
//!
//! Measures GPU performance ceilings (peak FLOPS + peak bandwidth),
//! models the dynamic performance envelope using tension parameters
//! (thermal, power, contention), and detects the true sustained ceiling.
//!
//! # Quick Start (Simulation)
//!
//! ```rust
//! use gpu_roofline::ceilings::measure_roofline;
//! use gpu_roofline::ceilings::MeasureConfig;
//! use gpu_harness::sim::{SimulatedBackend, profiles};
//!
//! let backend = SimulatedBackend::new(profiles::h100_sxm());
//! let model = measure_roofline(&backend, &MeasureConfig::default()).unwrap();
//!
//! println!("Peak: {:.1} GFLOP/s, {:.0} GB/s", model.peak_gflops, model.peak_bandwidth_gbps);
//! println!("Ridge point: {:.1} FLOP/byte", model.ridge_point);
//! ```

pub mod ceilings;
pub mod kernels;
pub mod model;
pub mod output;

// Re-export key types for convenience
pub use ceilings::{measure_dynamic, measure_roofline, MeasureConfig};
pub use kernels::{BuiltinKernel, KernelDefinition};
pub use model::{Bottleneck, DynamicConfig, DynamicRoofline, KernelPlacement, RooflineModel};
