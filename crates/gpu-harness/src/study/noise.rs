//! Noise injection layer for simulation realism.
//!
//! Adds measurement uncertainty matching real-world instrumentation:
//! - NVML memory reading jitter: N(0, 512 KiB)
//! - Spin-up latency jitter: LogNormal(mu, 0.3)
//! - Bandwidth measurement variance: N(0, 0.02 * baseline)
//! - Background memory spikes: Poisson(0.01) * Uniform(1 MiB, 64 MiB)
//! - Thermal measurement noise: N(0, 1.0 C)
//! - Fleet performance jitter: N(0, 0.02 * peak)

use rand::Rng;

/// Noise model parameters matching the study protocol Section 14.3.
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Sigma for NVML memory reading jitter in bytes.
    pub memory_sigma_bytes: f64,
    /// Sigma for log-normal latency jitter.
    pub latency_log_sigma: f64,
    /// Relative sigma for bandwidth measurement variance.
    pub bandwidth_relative_sigma: f64,
    /// Poisson lambda for background memory spikes per sample.
    pub spike_lambda: f64,
    /// Min spike size in bytes.
    pub spike_min_bytes: u64,
    /// Max spike size in bytes.
    pub spike_max_bytes: u64,
    /// Sigma for thermal measurement noise in Celsius.
    pub thermal_sigma_c: f64,
    /// Relative sigma for fleet performance jitter.
    pub fleet_relative_sigma: f64,
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            memory_sigma_bytes: 512.0 * 1024.0,       // 512 KiB
            latency_log_sigma: 0.3,
            bandwidth_relative_sigma: 0.02,
            spike_lambda: 0.01,
            spike_min_bytes: 1024 * 1024,              // 1 MiB
            spike_max_bytes: 64 * 1024 * 1024,         // 64 MiB
            thermal_sigma_c: 1.0,
            fleet_relative_sigma: 0.02,
        }
    }
}

impl NoiseModel {
    /// Add Gaussian noise to a memory reading (bytes).
    pub fn jitter_memory(&self, value: u64, rng: &mut impl Rng) -> u64 {
        let noise = self.gaussian(rng, 0.0, self.memory_sigma_bytes);
        (value as f64 + noise).max(0.0) as u64
    }

    /// Add log-normal jitter to a latency value (ms).
    pub fn jitter_latency(&self, base_ms: f64, rng: &mut impl Rng) -> f64 {
        let mu = base_ms.ln();
        let log_val = self.gaussian(rng, mu, self.latency_log_sigma);
        log_val.exp().max(0.1)
    }

    /// Add Gaussian noise to a bandwidth measurement (GB/s).
    pub fn jitter_bandwidth(&self, baseline_gbps: f64, rng: &mut impl Rng) -> f64 {
        let sigma = self.bandwidth_relative_sigma * baseline_gbps;
        let noise = self.gaussian(rng, 0.0, sigma);
        (baseline_gbps + noise).max(0.0)
    }

    /// Generate a background memory spike (returns spike size in bytes, or 0).
    pub fn memory_spike(&self, rng: &mut impl Rng) -> u64 {
        let p: f64 = rng.gen();
        if p < self.spike_lambda {
            rng.gen_range(self.spike_min_bytes..=self.spike_max_bytes)
        } else {
            0
        }
    }

    /// Add Gaussian noise to a temperature reading.
    pub fn jitter_thermal(&self, temp_c: f32, rng: &mut impl Rng) -> f32 {
        let noise = self.gaussian(rng, 0.0, self.thermal_sigma_c as f64);
        (temp_c + noise as f32).max(0.0)
    }

    /// Add fleet performance jitter to a measurement.
    pub fn jitter_fleet(&self, value: f64, rng: &mut impl Rng) -> f64 {
        let sigma = self.fleet_relative_sigma * value;
        let noise = self.gaussian(rng, 0.0, sigma);
        (value + noise).max(0.0)
    }

    /// Box-Muller Gaussian random number.
    fn gaussian(&self, rng: &mut impl Rng, mean: f64, sigma: f64) -> f64 {
        let u1: f64 = rng.gen::<f64>().max(1e-10);
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + sigma * z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_memory_jitter_near_original() {
        let noise = NoiseModel::default();
        let mut rng = seeded_rng();
        let original = 1024 * 1024 * 1024u64; // 1 GiB
        let jittered = noise.jitter_memory(original, &mut rng);
        let diff = (jittered as f64 - original as f64).abs();
        // Should be within a few MB of original (sigma = 512 KiB)
        assert!(diff < 10.0 * 1024.0 * 1024.0, "diff too large: {diff}");
    }

    #[test]
    fn test_latency_jitter_positive() {
        let noise = NoiseModel::default();
        let mut rng = seeded_rng();
        for _ in 0..100 {
            let jittered = noise.jitter_latency(200.0, &mut rng);
            assert!(jittered > 0.0);
        }
    }

    #[test]
    fn test_bandwidth_jitter_distribution() {
        let noise = NoiseModel::default();
        let mut rng = seeded_rng();
        let baseline = 3350.0;
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            sum += noise.jitter_bandwidth(baseline, &mut rng);
        }
        let mean = sum / n as f64;
        // Mean should be close to baseline
        assert!((mean - baseline).abs() < 10.0, "mean {mean} far from baseline {baseline}");
    }
}
