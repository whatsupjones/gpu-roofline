//! gpu-fleet: Multi-GPU fleet validation, topology discovery, and straggler detection.
//!
//! Validates multi-GPU clusters for performance correctness using roofline-based
//! analysis. Discovers topology, measures inter-GPU bandwidth, detects stragglers,
//! and validates each GPU against its expected performance model.

pub mod symmetry;
pub mod topology;

pub use gpu_harness;
