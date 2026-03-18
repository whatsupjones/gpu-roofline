//! Continuous GPU performance monitoring with degradation alerting.
//!
//! Periodically samples GPU performance (lightweight burst measurement)
//! and compares against baseline to detect degradation, thermal ramp,
//! and instability in real-time.

pub mod alerting;
pub mod sampler;
pub mod tui;

// Re-export for library consumers
#[allow(unused_imports)]
pub use alerting::{Alert, AlertLevel, AlertRule};
pub use sampler::{MonitorConfig, MonitorSample, SampleStatus, Sampler};
#[allow(unused_imports)]
pub use tui::TuiState;

#[cfg(feature = "vgpu")]
pub mod vgpu_alerting;
#[cfg(feature = "vgpu")]
pub mod vgpu_sampler;
#[cfg(feature = "vgpu")]
pub mod vgpu_tui;
