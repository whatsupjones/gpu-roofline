//! Roofline model construction and analysis.

pub mod dynamic;
pub mod equilibrium;
pub mod roofline;
pub mod tension;

pub use dynamic::DynamicRoofline;
pub use roofline::{Bottleneck, KernelPlacement, RooflineModel};
pub use tension::DynamicConfig;
