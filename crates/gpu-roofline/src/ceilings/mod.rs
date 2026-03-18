//! Ceiling measurement: runs micro-kernels to determine peak bandwidth and compute.

pub mod dynamic;
pub mod measure;

pub use dynamic::measure_dynamic;
pub use measure::{measure_roofline, MeasureConfig};
