use serde::{Deserialize, Serialize};

/// Power delivery model for GPU simulation.
///
/// Models the relationship between workload intensity, clock speed,
/// and power draw, including TDP-limited clock scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerModel {
    /// Thermal Design Power limit in watts.
    pub tdp_watts: f32,
    /// Base (guaranteed minimum) clock in MHz.
    pub base_clock_mhz: u32,
    /// Maximum boost clock in MHz.
    pub boost_clock_mhz: u32,
    /// Power scaling exponent (clock^exponent ∝ power).
    /// Typically ~1.5 for modern GPUs (voltage * frequency * capacitance).
    pub power_exponent: f32,
    /// Idle power draw in watts.
    pub idle_watts: f32,
}

impl PowerModel {
    /// Compute the achievable clock speed and power draw for a given workload.
    ///
    /// `intensity` is 0.0 (idle) to 1.0 (full load).
    /// `throttle_factor` is from the thermal model (1.0 = no thermal throttle).
    ///
    /// Returns (power_watts, clock_mhz).
    pub fn compute_state(&self, intensity: f32, throttle_factor: f32) -> (f32, u32) {
        let intensity = intensity.clamp(0.0, 1.0);
        let throttle_factor = throttle_factor.clamp(0.0, 1.0);

        if intensity < 0.01 {
            return (self.idle_watts, self.base_clock_mhz);
        }

        // Target clock = base + intensity * (boost - base), scaled by thermal throttle
        let clock_range = (self.boost_clock_mhz - self.base_clock_mhz) as f32;
        let target_clock = self.base_clock_mhz as f32 + intensity * clock_range * throttle_factor;

        // Power at this clock: P = idle + (target/boost)^exponent * (tdp - idle)
        let clock_ratio = (target_clock - self.base_clock_mhz as f32) / clock_range;
        let dynamic_power =
            clock_ratio.powf(self.power_exponent) * (self.tdp_watts - self.idle_watts);
        let total_power = self.idle_watts + dynamic_power * intensity;

        // TDP cap: if power exceeds TDP, reduce clock
        if total_power > self.tdp_watts {
            let capped_clock = self.tdp_limited_clock(intensity);
            let capped_power = self.tdp_watts; // At TDP limit
            (capped_power, capped_clock as u32)
        } else {
            (total_power, target_clock as u32)
        }
    }

    /// Calculate the maximum clock achievable within TDP budget.
    fn tdp_limited_clock(&self, intensity: f32) -> f32 {
        let available_dynamic = (self.tdp_watts - self.idle_watts) / intensity.max(0.01);
        let clock_range = (self.boost_clock_mhz - self.base_clock_mhz) as f32;
        let max_ratio = (available_dynamic / (self.tdp_watts - self.idle_watts))
            .powf(1.0 / self.power_exponent)
            .min(1.0);
        self.base_clock_mhz as f32 + max_ratio * clock_range
    }

    /// Peak theoretical FLOPS at a given clock speed.
    ///
    /// `cuda_cores` is the number of FP32 CUDA cores (or equivalent).
    /// FP32 FLOPS = cores * clock_mhz * 1e6 * 2 (FMA = 2 ops)
    pub fn peak_flops_at_clock(cuda_cores: u32, clock_mhz: u32) -> f64 {
        cuda_cores as f64 * clock_mhz as f64 * 1e6 * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rtx_4090_power() -> PowerModel {
        PowerModel {
            tdp_watts: 450.0,
            base_clock_mhz: 2235,
            boost_clock_mhz: 2520,
            power_exponent: 1.5,
            idle_watts: 25.0,
        }
    }

    #[test]
    fn test_idle_power() {
        let model = rtx_4090_power();
        let (power, clock) = model.compute_state(0.0, 1.0);
        assert!(
            (power - 25.0).abs() < 1.0,
            "idle power should be ~25W, got {power}"
        );
        assert_eq!(clock, 2235, "idle clock should be base");
    }

    #[test]
    fn test_full_load_power() {
        let model = rtx_4090_power();
        let (power, clock) = model.compute_state(1.0, 1.0);
        assert!(power <= 450.0, "power should not exceed TDP, got {power}");
        assert!(clock >= 2235, "clock should be at least base, got {clock}");
        assert!(clock <= 2520, "clock should not exceed boost, got {clock}");
    }

    #[test]
    fn test_thermal_throttle_reduces_clock() {
        let model = rtx_4090_power();
        let (_, clock_full) = model.compute_state(1.0, 1.0);
        let (_, clock_throttled) = model.compute_state(1.0, 0.7);
        assert!(
            clock_throttled < clock_full,
            "throttled clock {clock_throttled} should be less than full {clock_full}"
        );
    }

    #[test]
    fn test_partial_load() {
        let model = rtx_4090_power();
        let (power_50, _) = model.compute_state(0.5, 1.0);
        let (power_100, _) = model.compute_state(1.0, 1.0);
        assert!(
            power_50 < power_100,
            "50% load power {power_50} should be less than 100% {power_100}"
        );
    }

    #[test]
    fn test_peak_flops_calculation() {
        // RTX 4090: 16384 cores, 2520 MHz boost
        let flops = PowerModel::peak_flops_at_clock(16384, 2520);
        let tflops = flops / 1e12;
        assert!(
            (tflops - 82.6).abs() < 1.0,
            "RTX 4090 peak should be ~82.6 TFLOPS, got {tflops:.1}"
        );
    }
}
