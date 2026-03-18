//! Built-in micro-kernels for roofline measurement.
//!
//! These kernels span the roofline from pure memory-bound (copy)
//! to pure compute-bound (fma_heavy), allowing the tool to
//! characterize the full performance envelope.

pub mod definitions;

pub use definitions::{BuiltinKernel, KernelDefinition};
