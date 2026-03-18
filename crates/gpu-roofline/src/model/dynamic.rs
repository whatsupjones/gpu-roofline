use serde::{Deserialize, Serialize};

use super::roofline::RooflineModel;
use super::tension::{TensionMeasurement, ThermalSample};

/// A dynamic roofline model capturing both burst and sustained performance
/// with tension analysis showing how competing forces shape the envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRoofline {
    /// Device name.
    pub device_name: String,
    /// Burst roofline (t=0, peak boost clock, before thermal ramp).
    pub burst: RooflineModel,
    /// Sustained roofline (at thermal equilibrium — the REAL ceiling).
    pub sustained: RooflineModel,
    /// Measured tensions between competing forces.
    pub tensions: Vec<TensionMeasurement>,
    /// Time in seconds to reach thermal equilibrium.
    pub equilibrium_time_secs: f64,
    /// Full thermal trajectory (clock, temp, power, performance over time).
    pub thermal_trajectory: Vec<ThermalSample>,
    /// Net ceiling drop from burst to sustained (percentage).
    pub net_ceiling_drop_pct: f64,
}

impl DynamicRoofline {
    /// Build a dynamic roofline from burst + sustained measurements
    /// plus thermal trajectory data.
    pub fn from_measurements(
        burst: RooflineModel,
        sustained: RooflineModel,
        trajectory: Vec<ThermalSample>,
        equilibrium_time_secs: f64,
    ) -> Self {
        let device_name = burst.device_name.clone();

        // Calculate net ceiling drop
        let net_ceiling_drop_pct = if burst.peak_gflops > 0.0 {
            ((burst.peak_gflops - sustained.peak_gflops) / burst.peak_gflops) * 100.0
        } else {
            0.0
        };

        // Extract tensions from the trajectory
        let tensions =
            Self::extract_tensions(&burst, &sustained, &trajectory, equilibrium_time_secs);

        Self {
            device_name,
            burst,
            sustained,
            tensions,
            equilibrium_time_secs,
            thermal_trajectory: trajectory,
            net_ceiling_drop_pct,
        }
    }

    /// Extract tension measurements from burst/sustained delta and trajectory.
    fn extract_tensions(
        burst: &RooflineModel,
        sustained: &RooflineModel,
        trajectory: &[ThermalSample],
        equilibrium_time_secs: f64,
    ) -> Vec<TensionMeasurement> {
        let mut tensions = Vec::new();

        // Thermal tension: clock reduction due to temperature
        let clock_drop_pct = if burst.clock_mhz > 0 {
            ((burst.clock_mhz as f64 - sustained.clock_mhz as f64) / burst.clock_mhz as f64) * 100.0
        } else {
            0.0
        };

        if clock_drop_pct.abs() > 0.5 {
            // Find when thermal throttling started (clock first dropped by >1%)
            let onset = trajectory
                .iter()
                .find(|s| {
                    let drop =
                        (burst.clock_mhz as f64 - s.clock_mhz as f64) / burst.clock_mhz as f64;
                    drop > 0.01
                })
                .map(|s| s.elapsed_secs)
                .unwrap_or(0.0);

            tensions.push(TensionMeasurement {
                name: "thermal".to_string(),
                force_a: format!("Boost clock targets {} MHz", burst.clock_mhz),
                force_b: format!(
                    "Thermal limit forces throttle to {} MHz ({}°C)",
                    sustained.clock_mhz, sustained.temperature_c
                ),
                ceiling_delta_pct: -clock_drop_pct,
                onset_time_secs: onset,
            });
        }

        // Power tension: additional drop from power limit
        let gflops_drop_pct = if burst.peak_gflops > 0.0 {
            ((burst.peak_gflops - sustained.peak_gflops) / burst.peak_gflops) * 100.0
        } else {
            0.0
        };

        let power_tension_pct = gflops_drop_pct - clock_drop_pct;
        if power_tension_pct > 1.0 {
            tensions.push(TensionMeasurement {
                name: "power".to_string(),
                force_a: "Workload demands peak compute".to_string(),
                force_b: format!(
                    "TDP cap at {:.0}W forces clock scaling",
                    sustained.power_watts
                ),
                ceiling_delta_pct: -power_tension_pct,
                onset_time_secs: equilibrium_time_secs * 0.6, // Power tension typically starts mid-ramp
            });
        }

        // Bandwidth tension
        let bw_drop_pct = if burst.peak_bandwidth_gbps > 0.0 {
            ((burst.peak_bandwidth_gbps - sustained.peak_bandwidth_gbps)
                / burst.peak_bandwidth_gbps)
                * 100.0
        } else {
            0.0
        };

        if bw_drop_pct > 1.0 {
            tensions.push(TensionMeasurement {
                name: "bandwidth".to_string(),
                force_a: format!("Peak bandwidth {:.0} GB/s", burst.peak_bandwidth_gbps),
                force_b: format!(
                    "Sustained drops to {:.0} GB/s under thermal load",
                    sustained.peak_bandwidth_gbps
                ),
                ceiling_delta_pct: -bw_drop_pct,
                onset_time_secs: equilibrium_time_secs * 0.3,
            });
        }

        tensions
    }

    /// Summary string for terminal output.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Dynamic Roofline: {} | Burst → Sustained",
            self.device_name
        ));
        lines.push(format!(
            "  Burst:     {:.1} GFLOP/s | {:.0} GB/s | {} MHz | {}°C",
            self.burst.peak_gflops,
            self.burst.peak_bandwidth_gbps,
            self.burst.clock_mhz,
            self.burst.temperature_c
        ));
        lines.push(format!(
            "  Sustained: {:.1} GFLOP/s | {:.0} GB/s | {} MHz | {}°C",
            self.sustained.peak_gflops,
            self.sustained.peak_bandwidth_gbps,
            self.sustained.clock_mhz,
            self.sustained.temperature_c
        ));
        lines.push(format!(
            "  Equilibrium: {:.1}s | Net drop: {:.1}%",
            self.equilibrium_time_secs, self.net_ceiling_drop_pct
        ));

        if !self.tensions.is_empty() {
            lines.push("  Tensions:".to_string());
            for t in &self.tensions {
                lines.push(format!(
                    "    {} {}: {:.1}% after {:.1}s",
                    if t.ceiling_delta_pct < 0.0 {
                        "↓"
                    } else {
                        "↑"
                    },
                    t.name,
                    t.ceiling_delta_pct.abs(),
                    t.onset_time_secs
                ));
            }
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::roofline::RooflineModel;

    fn mock_burst() -> RooflineModel {
        RooflineModel {
            device_name: "Test GPU".to_string(),
            peak_gflops: 80000.0,
            peak_bandwidth_gbps: 1000.0,
            ridge_point: 80.0,
            clock_mhz: 2520,
            temperature_c: 45,
            power_watts: 280.0,
            placements: vec![],
        }
    }

    fn mock_sustained() -> RooflineModel {
        RooflineModel {
            device_name: "Test GPU".to_string(),
            peak_gflops: 65000.0, // 18.75% drop
            peak_bandwidth_gbps: 940.0,
            ridge_point: 69.1,
            clock_mhz: 2100, // Throttled
            temperature_c: 83,
            power_watts: 420.0,
            placements: vec![],
        }
    }

    fn mock_trajectory() -> Vec<ThermalSample> {
        (0..60)
            .map(|i| {
                let t = i as f64 * 2.0; // 2-second intervals
                let temp = 45.0 + 38.0 * (1.0 - (-t / 30.0).exp());
                let clock = 2520.0 - 420.0 * (1.0 - (-t / 30.0).exp());
                ThermalSample {
                    elapsed_secs: t,
                    clock_mhz: clock as u32,
                    temperature_c: temp as u32,
                    power_watts: 280.0 + 140.0 * (1.0 - (-t / 20.0).exp()) as f32,
                    measured_gflops: 80000.0 * (clock / 2520.0),
                    measured_bandwidth_gbps: 1000.0 * 0.94_f64.powf(t / 60.0),
                }
            })
            .collect()
    }

    #[test]
    fn test_dynamic_roofline_construction() {
        let dynamic = DynamicRoofline::from_measurements(
            mock_burst(),
            mock_sustained(),
            mock_trajectory(),
            34.0,
        );

        assert_eq!(dynamic.device_name, "Test GPU");
        assert!(
            dynamic.net_ceiling_drop_pct > 15.0,
            "should show significant drop"
        );
        assert!(
            dynamic.net_ceiling_drop_pct < 25.0,
            "drop should be reasonable"
        );
        assert!(!dynamic.tensions.is_empty(), "should detect tensions");
    }

    #[test]
    fn test_thermal_tension_detected() {
        let dynamic = DynamicRoofline::from_measurements(
            mock_burst(),
            mock_sustained(),
            mock_trajectory(),
            34.0,
        );

        let thermal = dynamic.tensions.iter().find(|t| t.name == "thermal");
        assert!(thermal.is_some(), "should detect thermal tension");
        let thermal = thermal.unwrap();
        assert!(
            thermal.ceiling_delta_pct < 0.0,
            "thermal should reduce performance"
        );
    }

    #[test]
    fn test_no_tension_when_identical() {
        let burst = mock_burst();
        let sustained = burst.clone(); // Same as burst = no tension

        let dynamic = DynamicRoofline::from_measurements(burst, sustained, vec![], 0.0);

        assert!(
            dynamic.tensions.is_empty(),
            "no tension when burst == sustained"
        );
        assert!((dynamic.net_ceiling_drop_pct).abs() < 0.1);
    }

    #[test]
    fn test_summary_output() {
        let dynamic = DynamicRoofline::from_measurements(
            mock_burst(),
            mock_sustained(),
            mock_trajectory(),
            34.0,
        );

        let summary = dynamic.summary();
        assert!(summary.contains("Burst"));
        assert!(summary.contains("Sustained"));
        assert!(summary.contains("Equilibrium"));
        assert!(summary.contains("thermal"));
    }
}
