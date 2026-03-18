//! Ceiling measurement: runs micro-kernels to determine peak bandwidth and compute.

pub mod measure;

pub use measure::{measure_roofline, MeasureConfig};
