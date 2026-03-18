//! Continuous GPU performance monitoring with degradation alerting.
//!
//! Periodically samples GPU performance (lightweight burst measurement)
//! and compares against baseline to detect degradation, thermal ramp,
//! and instability in real-time.

pub mod alerting;
pub mod sampler;

pub use alerting::AlertLevel;
// Re-export for library consumers
#[allow(unused_imports)]
pub use alerting::{Alert, AlertRule};
pub use sampler::{MonitorConfig, MonitorSample, SampleStatus, Sampler};
