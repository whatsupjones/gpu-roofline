//! gpu-harness: GPU device discovery, simulation, and backend abstraction.
//!
//! Provides the `GpuBackend` trait that all gpu-tools crates program against,
//! plus a physics-based simulation engine for testing without hardware.

pub mod backend;
pub mod device;
pub mod error;
pub mod sim;

pub use backend::{
    BandwidthResult, DeviceState, GpuBackend, KernelResult, KernelSpec, RunConfig,
};
pub use device::{GpuArchitecture, GpuDevice, GpuFeatures, GpuLimits, GpuVendor};
pub use error::HarnessError;
