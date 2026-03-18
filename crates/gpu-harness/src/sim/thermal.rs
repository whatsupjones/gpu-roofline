use serde::{Deserialize, Serialize};

/// Physics-based thermal model for GPU simulation.
///
/// Models heat generation (power draw) vs heat dissipation (cooler capacity)
/// as a first-order system approaching thermal equilibrium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalModel {
    /// Ambient / idle temperature in Celsius.
    pub ambient_c: f32,
    /// Temperature at which throttling begins.
    pub throttle_onset_c: f32,
    /// Temperature at which max throttle is applied.
    pub throttle_max_c: f32,
    /// Emergency shutdown temperature.
    pub shutdown_c: f32,
    /// Thermal mass: joules required to raise GPU temp by 1 degree.
    /// Higher = slower to heat, slower to cool.
    pub thermal_mass_j_per_c: f32,
    /// Cooler dissipation capacity in watts at max fan speed.
    pub cooling_capacity_watts: f32,
}

impl ThermalModel {
    /// Calculate temperature at a given time under constant power draw.
    ///
    /// Uses Newton's law of cooling: dT/dt = (P - C*(T - T_ambient)) / M
    /// where P = power, C = cooling coefficient, M = thermal mass, T_ambient = ambient temp.
    ///
    /// Analytical solution: T(t) = T_ambient + (P/C) * (1 - e^(-C*t/M))
    /// where C = cooling_capacity / (throttle_max - ambient) is the cooling coefficient per degree.
    pub fn temperature_at(&self, power_watts: f32, elapsed_secs: f64) -> f32 {
        let temp_range = self.throttle_max_c - self.ambient_c;
        if temp_range <= 0.0 {
            return self.ambient_c;
        }

        // Cooling coefficient: watts per degree Celsius
        let cooling_coeff = self.cooling_capacity_watts / temp_range;

        // Equilibrium temperature delta above ambient
        let equilibrium_delta = power_watts / cooling_coeff;

        // Time constant (seconds to reach ~63% of equilibrium)
        let tau = self.thermal_mass_j_per_c / cooling_coeff;

        // Exponential approach to equilibrium
        let delta = equilibrium_delta * (1.0 - (-elapsed_secs as f32 / tau).exp());

        self.ambient_c + delta
    }

    /// Clock throttle factor based on current temperature.
    ///
    /// Returns a multiplier in [0.0, 1.0]:
    /// - 1.0 = no throttle (below onset)
    /// - 0.0 = maximum throttle (at or above throttle_max)
    /// - Linear interpolation between onset and max
    pub fn throttle_factor(&self, temp_c: f32) -> f32 {
        if temp_c <= self.throttle_onset_c {
            1.0
        } else if temp_c >= self.throttle_max_c {
            // Don't go below 60% — GPU doesn't fully stop, it clocks way down
            0.6
        } else {
            let range = self.throttle_max_c - self.throttle_onset_c;
            let above_onset = temp_c - self.throttle_onset_c;
            let throttle_amount = above_onset / range;
            1.0 - (throttle_amount * 0.4) // Linear from 1.0 to 0.6
        }
    }

    /// Time in seconds to reach thermal equilibrium (CV < threshold).
    pub fn time_to_equilibrium(&self, _power_watts: f32) -> f64 {
        let temp_range = self.throttle_max_c - self.ambient_c;
        if temp_range <= 0.0 {
            return 0.0;
        }
        let cooling_coeff = self.cooling_capacity_watts / temp_range;
        let tau = self.thermal_mass_j_per_c / cooling_coeff;
        // 3 * tau ≈ 95% of equilibrium (good enough for CV < 2%)
        (tau * 3.0) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rtx_4090_thermal() -> ThermalModel {
        ThermalModel {
            ambient_c: 35.0,
            throttle_onset_c: 78.0,
            throttle_max_c: 90.0,
            shutdown_c: 95.0,
            thermal_mass_j_per_c: 15.0,
            cooling_capacity_watts: 350.0,
        }
    }

    #[test]
    fn test_temperature_starts_at_ambient() {
        let model = rtx_4090_thermal();
        let temp = model.temperature_at(450.0, 0.0);
        assert!(
            (temp - 35.0).abs() < 0.1,
            "temp at t=0 should be ambient, got {temp}"
        );
    }

    #[test]
    fn test_temperature_rises_over_time() {
        let model = rtx_4090_thermal();
        let t1 = model.temperature_at(450.0, 5.0);
        let t2 = model.temperature_at(450.0, 30.0);
        let t3 = model.temperature_at(450.0, 120.0);

        assert!(t1 > 35.0, "should be above ambient at 5s");
        assert!(t2 > t1, "should be hotter at 30s than 5s");
        assert!(t3 > t2, "should be hotter at 120s than 30s");
    }

    #[test]
    fn test_temperature_approaches_equilibrium() {
        let model = rtx_4090_thermal();
        let t_long = model.temperature_at(350.0, 300.0);

        // With 350W power and 350W cooling capacity, equilibrium should be
        // at ambient + (power/cooling_coeff) = 35 + (350 / (350/55)) ≈ 35 + 55 = 90
        // But this is the max case. The model should approach but not exceed throttle_max.
        assert!(t_long > 70.0, "should be well above ambient at equilibrium");
        assert!(t_long < 120.0, "should not diverge");
    }

    #[test]
    fn test_throttle_factor_below_onset() {
        let model = rtx_4090_thermal();
        assert_eq!(model.throttle_factor(35.0), 1.0);
        assert_eq!(model.throttle_factor(77.0), 1.0);
    }

    #[test]
    fn test_throttle_factor_above_onset() {
        let model = rtx_4090_thermal();
        let factor = model.throttle_factor(84.0);
        assert!(factor < 1.0, "should throttle above onset");
        assert!(factor > 0.6, "should not be at max throttle");
    }

    #[test]
    fn test_throttle_factor_at_max() {
        let model = rtx_4090_thermal();
        assert_eq!(model.throttle_factor(90.0), 0.6);
        assert_eq!(model.throttle_factor(95.0), 0.6);
    }

    #[test]
    fn test_time_to_equilibrium() {
        let model = rtx_4090_thermal();
        let t_eq = model.time_to_equilibrium(450.0);
        assert!(t_eq > 1.0, "equilibrium should take more than 1s");
        assert!(
            t_eq < 60.0,
            "equilibrium should be reached within 60s for consumer GPU"
        );
    }
}
