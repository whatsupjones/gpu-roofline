//! GPU Simulation Engine
//!
//! Physics-based GPU models for testing roofline and fleet logic without hardware.
//! The simulation provides expected baselines that real measurements validate against.

pub mod bandwidth;
pub mod fleet;
pub mod gpu_model;
pub mod power;
pub mod profiles;
pub mod thermal;

pub use bandwidth::{BandwidthModel, MemoryLevel};
pub use fleet::{Degradation, SimGpuInstance, SimTopology, SimulatedFleet};
pub use gpu_model::SimGpuProfile;
pub use power::PowerModel;
pub use profiles::*;
pub use thermal::ThermalModel;
