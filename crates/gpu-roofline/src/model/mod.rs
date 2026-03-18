//! Roofline model construction and analysis.

pub mod roofline;

pub use roofline::{
    Bottleneck, KernelPlacement, RooflineModel,
};
