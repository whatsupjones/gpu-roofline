#![allow(clippy::needless_range_loop)]
//! gpu-harness: GPU device discovery, simulation, and backend abstraction.
//!
//! Provides the `GpuBackend` trait that all gpu-tools crates program against,
//! plus a physics-based simulation engine for testing without hardware.

pub mod backend;
pub mod cuda_backend;
pub mod device;
pub mod error;
pub mod nvml_telemetry;
pub mod sim;
pub mod study;
pub mod wgpu_backend;

#[cfg(feature = "vgpu")]
pub mod vgpu;

pub use backend::{BandwidthResult, DeviceState, GpuBackend, KernelResult, KernelSpec, RunConfig};
pub use device::{GpuArchitecture, GpuDevice, GpuFeatures, GpuLimits, GpuVendor};
pub use error::HarnessError;
pub use nvml_telemetry::NvmlTelemetry;

#[cfg(feature = "wgpu-backend")]
pub use wgpu_backend::WgpuBackend;

#[cfg(feature = "cuda")]
pub use cuda_backend::CudaBackend;
