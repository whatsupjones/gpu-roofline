//! Degradation detection and alert generation for GPU monitoring.
//!
//! Detects:
//! - Sudden degradation: current sample below threshold of baseline
//! - Gradual degradation: rolling average declining over N samples
//! - Thermal ramp: temperature exceeding expected operating range
//! - Instability: CV increasing over time (noisy measurements)

use gpu_harness::backend::DeviceState;

use crate::model::RooflineModel;

use super::MonitorSample;

/// Severity level for monitoring alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AlertLevel {
    /// Performance slightly below expectations.
    Warning,
    /// Significant performance degradation detected.
    Critical,
}

/// A specific alert generated during monitoring.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Alert {
    pub level: AlertLevel,
    pub rule: AlertRule,
    pub message: String,
}

/// Which rule triggered the alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AlertRule {
    /// Current bandwidth below threshold of baseline.
    BandwidthDrop,
    /// Current FLOPS below threshold of baseline.
    ComputeDrop,
    /// Temperature above expected range.
    ThermalExceedance,
    /// Measurement instability (high CV).
    Instability,
    /// Rolling average declining over recent samples.
    GradualDegradation,
}

impl std::fmt::Display for AlertRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BandwidthDrop => write!(f, "Bandwidth Drop"),
            Self::ComputeDrop => write!(f, "Compute Drop"),
            Self::ThermalExceedance => write!(f, "Thermal"),
            Self::Instability => write!(f, "Instability"),
            Self::GradualDegradation => write!(f, "Gradual Degradation"),
        }
    }
}

/// Engine that evaluates all alert rules against current state.
pub struct AlertEngine {
    /// Fraction of baseline below which we alert (e.g., 0.8 = 80%).
    threshold: f64,
    /// Temperature ceiling (°C) before warning.
    thermal_warning_c: u32,
    /// Temperature ceiling (°C) before critical alert.
    thermal_critical_c: u32,
    /// CV above which we flag instability.
    cv_warning: f64,
    /// Number of samples for rolling average trend detection.
    trend_window: usize,
}

impl AlertEngine {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            thermal_warning_c: 80,
            thermal_critical_c: 90,
            cv_warning: 0.05, // 5% CV is concerning
            trend_window: 5,
        }
    }

    /// Check all alert rules and return any triggered alerts.
    pub fn check(
        &self,
        roofline: &RooflineModel,
        device_state: &DeviceState,
        baseline_bw: f64,
        baseline_gflops: f64,
        history: &[MonitorSample],
    ) -> Vec<Alert> {
        let mut alerts = Vec::new();

        // 1. Sudden bandwidth drop
        if baseline_bw > 0.0 {
            let bw_ratio = roofline.peak_bandwidth_gbps / baseline_bw;
            if bw_ratio < self.threshold * 0.9 {
                alerts.push(Alert {
                    level: AlertLevel::Critical,
                    rule: AlertRule::BandwidthDrop,
                    message: format!(
                        "Bandwidth {:.0} GB/s is {:.0}% of baseline ({:.0} GB/s)",
                        roofline.peak_bandwidth_gbps,
                        bw_ratio * 100.0,
                        baseline_bw
                    ),
                });
            } else if bw_ratio < self.threshold {
                alerts.push(Alert {
                    level: AlertLevel::Warning,
                    rule: AlertRule::BandwidthDrop,
                    message: format!(
                        "Bandwidth {:.0} GB/s is {:.0}% of baseline ({:.0} GB/s)",
                        roofline.peak_bandwidth_gbps,
                        bw_ratio * 100.0,
                        baseline_bw
                    ),
                });
            }
        }

        // 2. Sudden compute drop
        if baseline_gflops > 0.0 {
            let flops_ratio = roofline.peak_gflops / baseline_gflops;
            if flops_ratio < self.threshold * 0.9 {
                alerts.push(Alert {
                    level: AlertLevel::Critical,
                    rule: AlertRule::ComputeDrop,
                    message: format!(
                        "Compute {:.0} GFLOP/s is {:.0}% of baseline ({:.0} GFLOP/s)",
                        roofline.peak_gflops,
                        flops_ratio * 100.0,
                        baseline_gflops
                    ),
                });
            } else if flops_ratio < self.threshold {
                alerts.push(Alert {
                    level: AlertLevel::Warning,
                    rule: AlertRule::ComputeDrop,
                    message: format!(
                        "Compute {:.0} GFLOP/s is {:.0}% of baseline ({:.0} GFLOP/s)",
                        roofline.peak_gflops,
                        flops_ratio * 100.0,
                        baseline_gflops
                    ),
                });
            }
        }

        // 3. Thermal exceedance
        if device_state.temperature_c > 0 {
            if device_state.temperature_c >= self.thermal_critical_c {
                alerts.push(Alert {
                    level: AlertLevel::Critical,
                    rule: AlertRule::ThermalExceedance,
                    message: format!(
                        "GPU temperature {}°C exceeds critical threshold ({}°C)",
                        device_state.temperature_c, self.thermal_critical_c
                    ),
                });
            } else if device_state.temperature_c >= self.thermal_warning_c {
                alerts.push(Alert {
                    level: AlertLevel::Warning,
                    rule: AlertRule::ThermalExceedance,
                    message: format!(
                        "GPU temperature {}°C exceeds warning threshold ({}°C)",
                        device_state.temperature_c, self.thermal_warning_c
                    ),
                });
            }
        }

        // 4. Measurement instability
        let copy_cv = roofline
            .placements
            .iter()
            .find(|p| p.name == "copy")
            .map(|p| p.cv)
            .unwrap_or(0.0);

        if copy_cv > self.cv_warning {
            alerts.push(Alert {
                level: AlertLevel::Warning,
                rule: AlertRule::Instability,
                message: format!(
                    "Copy kernel CV {:.1}% indicates measurement noise (expected <{:.0}%)",
                    copy_cv * 100.0,
                    self.cv_warning * 100.0
                ),
            });
        }

        // 5. Gradual degradation (rolling BW average declining)
        if history.len() >= self.trend_window {
            let recent = &history[history.len() - self.trend_window..];
            let first_half_avg: f64 = recent[..self.trend_window / 2]
                .iter()
                .map(|s| s.bandwidth_gbps)
                .sum::<f64>()
                / (self.trend_window / 2) as f64;
            let second_half_avg: f64 = recent[self.trend_window / 2..]
                .iter()
                .map(|s| s.bandwidth_gbps)
                .sum::<f64>()
                / (self.trend_window - self.trend_window / 2) as f64;

            if first_half_avg > 0.0 {
                let trend = second_half_avg / first_half_avg;
                if trend < 0.95 {
                    alerts.push(Alert {
                        level: AlertLevel::Warning,
                        rule: AlertRule::GradualDegradation,
                        message: format!(
                            "Bandwidth trending down: {:.0} → {:.0} GB/s over last {} samples ({:.1}% decline)",
                            first_half_avg,
                            second_half_avg,
                            self.trend_window,
                            (1.0 - trend) * 100.0,
                        ),
                    });
                }
            }
        }

        alerts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::roofline::{Bottleneck, KernelPlacement};
    use gpu_harness::sim::MemoryLevel;

    fn mock_roofline(bw: f64, gflops: f64, cv: f64) -> RooflineModel {
        RooflineModel {
            device_name: "Test GPU".to_string(),
            peak_gflops: gflops,
            peak_bandwidth_gbps: bw,
            ridge_point: gflops / bw,
            clock_mhz: 2100,
            temperature_c: 65,
            power_watts: 300.0,
            placements: vec![KernelPlacement {
                name: "copy".to_string(),
                arithmetic_intensity: 0.0,
                achieved_gflops: 0.0,
                achieved_bandwidth_gbps: bw,
                efficiency: 1.0,
                bottleneck: Bottleneck::MemoryBound { level: MemoryLevel::Hbm },
                median_us: 100.0,
                stddev_us: 1.0,
                cv,
            }],
        }
    }

    fn mock_device_state(temp: u32) -> DeviceState {
        DeviceState {
            clock_mhz: 2100,
            temperature_c: temp,
            power_watts: 300.0,
            memory_used_bytes: 0,
            memory_total_bytes: 80 * 1024 * 1024 * 1024,
            utilization_pct: 95.0,
        }
    }

    #[test]
    fn test_no_alerts_when_healthy() {
        let engine = AlertEngine::new(0.8);
        let roofline = mock_roofline(2900.0, 59000.0, 0.003);
        let state = mock_device_state(65);
        let alerts = engine.check(&roofline, &state, 2900.0, 59000.0, &[]);
        assert!(alerts.is_empty(), "expected no alerts: {:?}", alerts);
    }

    #[test]
    fn test_bandwidth_drop_warning() {
        let engine = AlertEngine::new(0.8);
        let roofline = mock_roofline(2200.0, 59000.0, 0.003);
        let state = mock_device_state(65);
        let alerts = engine.check(&roofline, &state, 2900.0, 59000.0, &[]);
        assert!(alerts.iter().any(|a| a.rule == AlertRule::BandwidthDrop));
    }

    #[test]
    fn test_bandwidth_drop_critical() {
        let engine = AlertEngine::new(0.8);
        // 50% of baseline = way below 0.8 * 0.9 = 72%
        let roofline = mock_roofline(1450.0, 59000.0, 0.003);
        let state = mock_device_state(65);
        let alerts = engine.check(&roofline, &state, 2900.0, 59000.0, &[]);
        assert!(alerts
            .iter()
            .any(|a| a.rule == AlertRule::BandwidthDrop && a.level == AlertLevel::Critical));
    }

    #[test]
    fn test_thermal_warning() {
        let engine = AlertEngine::new(0.8);
        let roofline = mock_roofline(2900.0, 59000.0, 0.003);
        let state = mock_device_state(85);
        let alerts = engine.check(&roofline, &state, 2900.0, 59000.0, &[]);
        assert!(alerts
            .iter()
            .any(|a| a.rule == AlertRule::ThermalExceedance));
    }

    #[test]
    fn test_thermal_critical() {
        let engine = AlertEngine::new(0.8);
        let roofline = mock_roofline(2900.0, 59000.0, 0.003);
        let state = mock_device_state(95);
        let alerts = engine.check(&roofline, &state, 2900.0, 59000.0, &[]);
        assert!(alerts
            .iter()
            .any(|a| a.rule == AlertRule::ThermalExceedance && a.level == AlertLevel::Critical));
    }

    #[test]
    fn test_instability_warning() {
        let engine = AlertEngine::new(0.8);
        let roofline = mock_roofline(2900.0, 59000.0, 0.08); // 8% CV
        let state = mock_device_state(65);
        let alerts = engine.check(&roofline, &state, 2900.0, 59000.0, &[]);
        assert!(alerts.iter().any(|a| a.rule == AlertRule::Instability));
    }

    #[test]
    fn test_gradual_degradation() {
        let engine = AlertEngine::new(0.8);
        let roofline = mock_roofline(2900.0, 59000.0, 0.003);
        let state = mock_device_state(65);

        // Create history with declining bandwidth
        let mut history: Vec<MonitorSample> = Vec::new();
        for i in 0..5 {
            let bw = if i < 2 { 2900.0 } else { 2600.0 }; // Drop in second half
            history.push(MonitorSample {
                timestamp: chrono::Utc::now(),
                sample_index: i,
                bandwidth_gbps: bw,
                gflops: 59000.0,
                cv: 0.003,
                temperature_c: 65,
                clock_mhz: 2100,
                power_watts: 300.0,
                utilization_pct: 95.0,
                status: super::super::SampleStatus::Normal,
                alerts: vec![],
            });
        }

        let alerts = engine.check(&roofline, &state, 2900.0, 59000.0, &history);
        assert!(alerts
            .iter()
            .any(|a| a.rule == AlertRule::GradualDegradation));
    }
}
