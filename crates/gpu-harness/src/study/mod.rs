//! GPU Waste Study — Simulation Phase
//!
//! Implements the simulation phase from the study protocol:
//! ~120,000 trials across 6 waste categories, with noise injection,
//! statistical analysis, and cost model projections.

pub mod cost_model;
pub mod noise;
pub mod runner;
pub mod scenarios;
pub mod stats;

pub use cost_model::{CostModelParams, CostProjection, FleetScale, MeasuredEffects};
pub use noise::NoiseModel;
pub use runner::{CategoryResult, SimulationConfig, SimulationResults, TrialRecord};
pub use scenarios::WasteCategory;
pub use stats::CategoryStats;
