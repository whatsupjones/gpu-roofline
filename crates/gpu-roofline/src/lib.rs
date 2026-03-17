//! gpu-roofline: Cross-vendor dynamic roofline model generator.
//!
//! Measures GPU performance ceilings (peak FLOPS + peak bandwidth),
//! models the dynamic performance envelope using tension parameters
//! (thermal, power, contention), and detects the true sustained ceiling.

pub use gpu_harness;
