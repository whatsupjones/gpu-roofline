use gpu_harness::sim::MemoryLevel;
use gpu_harness::KernelResult;
use serde::{Deserialize, Serialize};

use crate::kernels::KernelDefinition;

/// A measured roofline model for a single GPU at a specific thermal state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineModel {
    /// Device name.
    pub device_name: String,
    /// Peak measured FP32 FLOPS (from compute-heavy kernel).
    pub peak_gflops: f64,
    /// Peak measured memory bandwidth in GB/s (from copy kernel).
    pub peak_bandwidth_gbps: f64,
    /// Ridge point: arithmetic intensity where compute meets bandwidth.
    /// Kernels below this are memory-bound; above are compute-bound.
    pub ridge_point: f64,
    /// GPU clock at time of measurement (MHz).
    pub clock_mhz: u32,
    /// GPU temperature at time of measurement (Celsius).
    pub temperature_c: u32,
    /// GPU power draw at time of measurement (Watts).
    pub power_watts: f32,
    /// Individual kernel placements on the roofline.
    pub placements: Vec<KernelPlacement>,
}

impl RooflineModel {
    /// Build a roofline model from a set of kernel measurements.
    ///
    /// Extracts peak bandwidth (from lowest-AI kernel) and peak FLOPS
    /// (from highest-AI kernel), then places all kernels on the model.
    pub fn from_measurements(
        device_name: String,
        measurements: &[(KernelDefinition, KernelResult)],
        clock_mhz: u32,
        temperature_c: u32,
        power_watts: f32,
    ) -> Self {
        // Find peak bandwidth from bandwidth kernels (lowest AI)
        let peak_bandwidth_gbps = measurements
            .iter()
            .filter(|(def, _)| def.arithmetic_intensity < 1.0)
            .map(|(_, result)| result.bandwidth_gbps())
            .fold(0.0_f64, f64::max);

        // Find peak GFLOPS from compute kernels (highest AI)
        let peak_gflops = measurements
            .iter()
            .filter(|(def, _)| def.arithmetic_intensity >= 1.0)
            .map(|(_, result)| result.gflops())
            .fold(0.0_f64, f64::max);

        // Ridge point: where peak_gflops = peak_bandwidth * AI
        // => AI = peak_gflops / peak_bandwidth
        let ridge_point = if peak_bandwidth_gbps > 0.0 {
            peak_gflops / peak_bandwidth_gbps
        } else {
            0.0
        };

        // Place each kernel on the roofline
        let placements: Vec<KernelPlacement> = measurements
            .iter()
            .map(|(def, result)| {
                let ai = def.arithmetic_intensity;
                let achieved_gflops = result.gflops();
                let achieved_bw = result.bandwidth_gbps();

                // Roofline ceiling at this AI
                let ceiling_gflops = (peak_bandwidth_gbps * ai).min(peak_gflops);

                let efficiency = if ceiling_gflops > 0.0 {
                    (achieved_gflops / ceiling_gflops).min(1.0)
                } else if peak_bandwidth_gbps > 0.0 {
                    // For AI=0, measure bandwidth efficiency instead
                    (achieved_bw / peak_bandwidth_gbps).min(1.0)
                } else {
                    0.0
                };

                let bottleneck = if ai < ridge_point {
                    Bottleneck::MemoryBound {
                        level: MemoryLevel::Hbm, // Default; hierarchical model refines this
                    }
                } else {
                    Bottleneck::ComputeBound
                };

                KernelPlacement {
                    name: def.name.to_string(),
                    arithmetic_intensity: ai,
                    achieved_gflops,
                    achieved_bandwidth_gbps: achieved_bw,
                    efficiency,
                    bottleneck,
                    median_us: result.median_us(),
                    stddev_us: result.stddev_us(),
                    cv: result.cv(),
                }
            })
            .collect();

        Self {
            device_name,
            peak_gflops,
            peak_bandwidth_gbps,
            ridge_point,
            clock_mhz,
            temperature_c,
            power_watts,
            placements,
        }
    }

    /// Get the roofline ceiling (GFLOP/s) at a given arithmetic intensity.
    pub fn ceiling_at(&self, arithmetic_intensity: f64) -> f64 {
        (self.peak_bandwidth_gbps * arithmetic_intensity).min(self.peak_gflops)
    }

    /// Find underperforming kernels (efficiency below threshold).
    pub fn underperformers(&self, threshold: f64) -> Vec<&KernelPlacement> {
        self.placements
            .iter()
            .filter(|p| p.efficiency < threshold)
            .collect()
    }
}

/// A kernel's position on the roofline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPlacement {
    /// Kernel name.
    pub name: String,
    /// Arithmetic intensity (FLOP/byte).
    pub arithmetic_intensity: f64,
    /// Measured GFLOP/s.
    pub achieved_gflops: f64,
    /// Measured memory bandwidth (GB/s).
    pub achieved_bandwidth_gbps: f64,
    /// Fraction of roofline ceiling achieved (0.0–1.0).
    pub efficiency: f64,
    /// Whether the kernel is compute-bound or memory-bound.
    pub bottleneck: Bottleneck,
    /// Median execution time (microseconds).
    pub median_us: f64,
    /// Standard deviation of execution time (microseconds).
    pub stddev_us: f64,
    /// Coefficient of variation (lower = more stable).
    pub cv: f64,
}

/// Performance bottleneck classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bottleneck {
    ComputeBound,
    MemoryBound { level: MemoryLevel },
}

impl std::fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComputeBound => write!(f, "Compute-bound"),
            Self::MemoryBound { level } => write!(f, "Memory-bound ({level})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::BuiltinKernel;

    fn mock_measurement(
        kernel: BuiltinKernel,
        elapsed_us: f64,
        buffer_bytes: usize,
    ) -> (KernelDefinition, KernelResult) {
        let def = kernel.definition();
        let elements = buffer_bytes / 16; // vec4<f32> = 16 bytes
        let total_flops = elements as u64 * def.flops_per_element;
        (
            def.clone(),
            KernelResult {
                kernel_name: def.name.to_string(),
                elapsed_us: vec![elapsed_us; 100],
                bytes_processed: buffer_bytes,
                flops_executed: total_flops,
            },
        )
    }

    #[test]
    fn test_roofline_from_measurements() {
        let buf = 16 * 1024 * 1024; // 16 MB
        let measurements = vec![
            mock_measurement(BuiltinKernel::Copy, 16.0, buf), // ~1000 GB/s
            mock_measurement(BuiltinKernel::Triad, 24.0, buf), // ~667 GB/s
            mock_measurement(BuiltinKernel::FmaLight, 20.0, buf),
            mock_measurement(BuiltinKernel::FmaHeavy, 30.0, buf),
        ];

        let model = RooflineModel::from_measurements(
            "Test GPU".to_string(),
            &measurements,
            2520,
            65,
            350.0,
        );

        assert!(
            model.peak_bandwidth_gbps > 0.0,
            "should have measured bandwidth"
        );
        assert!(model.peak_gflops > 0.0, "should have measured FLOPS");
        assert!(model.ridge_point > 0.0, "should have a ridge point");
        assert_eq!(model.placements.len(), 4);
    }

    #[test]
    fn test_ceiling_at() {
        let model = RooflineModel {
            device_name: "Test".to_string(),
            peak_gflops: 80000.0,        // 80 TFLOPS
            peak_bandwidth_gbps: 1000.0, // 1000 GB/s
            ridge_point: 80.0,           // 80 FLOP/byte
            clock_mhz: 2520,
            temperature_c: 65,
            power_watts: 350.0,
            placements: vec![],
        };

        // Memory-bound region (AI < ridge point)
        let ceiling_1 = model.ceiling_at(1.0);
        assert!((ceiling_1 - 1000.0).abs() < 1.0, "at AI=1: {ceiling_1}");

        // At ridge point
        let ceiling_ridge = model.ceiling_at(80.0);
        assert!(
            (ceiling_ridge - 80000.0).abs() < 1.0,
            "at ridge: {ceiling_ridge}"
        );

        // Compute-bound region (AI > ridge point)
        let ceiling_high = model.ceiling_at(200.0);
        assert!(
            (ceiling_high - 80000.0).abs() < 1.0,
            "above ridge: {ceiling_high}"
        );
    }

    #[test]
    fn test_underperformers() {
        let model = RooflineModel {
            device_name: "Test".to_string(),
            peak_gflops: 80000.0,
            peak_bandwidth_gbps: 1000.0,
            ridge_point: 80.0,
            clock_mhz: 2520,
            temperature_c: 65,
            power_watts: 350.0,
            placements: vec![
                KernelPlacement {
                    name: "good".to_string(),
                    arithmetic_intensity: 1.0,
                    achieved_gflops: 950.0,
                    achieved_bandwidth_gbps: 950.0,
                    efficiency: 0.95,
                    bottleneck: Bottleneck::MemoryBound {
                        level: MemoryLevel::Hbm,
                    },
                    median_us: 10.0,
                    stddev_us: 0.5,
                    cv: 0.05,
                },
                KernelPlacement {
                    name: "bad".to_string(),
                    arithmetic_intensity: 8.0,
                    achieved_gflops: 4000.0,
                    achieved_bandwidth_gbps: 500.0,
                    efficiency: 0.50,
                    bottleneck: Bottleneck::MemoryBound {
                        level: MemoryLevel::Hbm,
                    },
                    median_us: 25.0,
                    stddev_us: 2.0,
                    cv: 0.08,
                },
            ],
        };

        let under = model.underperformers(0.70);
        assert_eq!(under.len(), 1);
        assert_eq!(under[0].name, "bad");
    }
}
