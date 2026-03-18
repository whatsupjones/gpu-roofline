//! GPU validation engine.
//!
//! Runs roofline measurement and compares against built-in per-GPU
//! hardware baselines. Produces pass/fail report with smart diagnosis.

pub mod adaptive;
pub mod baselines;
pub mod checks;
pub mod report;

pub use adaptive::adaptive_config;
pub use adaptive::log_adaptive_config;
pub use baselines::find_baseline;
pub use checks::validate_roofline;
pub use report::print_validation_json;
pub use report::print_validation_table;
