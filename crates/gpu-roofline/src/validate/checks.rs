use serde::{Deserialize, Serialize};

use super::baselines::HardwareBaseline;
use crate::model::{Bottleneck, RooflineModel};

/// Result of a single validation check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCheck {
    pub name: String,
    pub measured: String,
    pub expected: String,
    pub passed: bool,
    pub detail: Option<String>,
}

/// Aggregate validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub gpu_name: String,
    pub checks: Vec<ValidationCheck>,
    pub diagnosis: Vec<String>,
    pub passed: bool,
    pub pass_count: usize,
    pub total_count: usize,
}

impl ValidationResult {
    pub fn exit_code(&self) -> i32 {
        if self.passed {
            0
        } else {
            1
        }
    }
}

/// Run all validation checks against a roofline measurement.
pub fn validate_roofline(
    model: &RooflineModel,
    baseline: &HardwareBaseline,
    threshold: f64,
) -> ValidationResult {
    let mut checks = Vec::new();
    let mut diagnosis = Vec::new();

    // Scale expected ranges by threshold
    let bw_min = baseline.bandwidth_range.0 * threshold;
    let bw_max = baseline.bandwidth_range.1;
    let flops_min = baseline.flops_range.0 * threshold;
    let flops_max = baseline.flops_range.1;

    // Check 1: Bandwidth
    let bw_passed =
        model.peak_bandwidth_gbps >= bw_min && model.peak_bandwidth_gbps <= bw_max * 1.15;
    checks.push(ValidationCheck {
        name: "Bandwidth".to_string(),
        measured: format!("{:.0} GB/s", model.peak_bandwidth_gbps),
        expected: format!("{:.0}-{:.0} GB/s", bw_min, bw_max),
        passed: bw_passed,
        detail: if !bw_passed {
            Some(format!(
                "{:.1}% of expected minimum",
                (model.peak_bandwidth_gbps / baseline.bandwidth_range.0) * 100.0
            ))
        } else {
            None
        },
    });

    // Check 2: FP32 Compute
    let flops_tflops = model.peak_gflops / 1000.0;
    let flops_passed = flops_tflops >= flops_min && flops_tflops <= flops_max * 1.15;
    checks.push(ValidationCheck {
        name: "FP32 Compute".to_string(),
        measured: format!("{:.1} TFLOPS", flops_tflops),
        expected: format!("{:.0}-{:.0} TFLOPS", flops_min, flops_max),
        passed: flops_passed,
        detail: if !flops_passed {
            Some(format!(
                "{:.1}% of expected minimum",
                (flops_tflops / baseline.flops_range.0) * 100.0
            ))
        } else {
            None
        },
    });

    // Check 3: Measurement stability (CV)
    let max_cv = model
        .placements
        .iter()
        .filter(|p| p.median_us > 0.0)
        .map(|p| p.cv * 100.0)
        .fold(0.0_f64, f64::max);

    let stability_passed = max_cv <= baseline.max_cv_percent;
    checks.push(ValidationCheck {
        name: "Stability".to_string(),
        measured: format!("CV {:.1}%", max_cv),
        expected: format!("< {:.0}%", baseline.max_cv_percent),
        passed: stability_passed,
        detail: if !stability_passed {
            Some(
                "High variance indicates thermal throttling, concurrent load, or unstable driver"
                    .to_string(),
            )
        } else {
            None
        },
    });

    // Check 4: Roofline shape
    let working_placements: Vec<_> = model
        .placements
        .iter()
        .filter(|p| p.median_us > 0.0 && p.achieved_gflops > 0.0)
        .collect();

    let correct_shape = working_placements
        .iter()
        .filter(|p| {
            let expected_memory_bound = p.arithmetic_intensity < model.ridge_point;
            let is_memory_bound = matches!(p.bottleneck, Bottleneck::MemoryBound { .. });
            expected_memory_bound == is_memory_bound
        })
        .count();

    let total_working = working_placements.len();
    let shape_passed = total_working == 0 || correct_shape == total_working;
    checks.push(ValidationCheck {
        name: "Roofline Shape".to_string(),
        measured: format!("{}/{}", correct_shape, total_working),
        expected: "All correct".to_string(),
        passed: shape_passed,
        detail: if !shape_passed {
            Some("Some kernels classified in wrong roofline region — possible driver or compiler issue".to_string())
        } else {
            None
        },
    });

    // Generate diagnosis for failures
    if !bw_passed && flops_passed {
        diagnosis.push(
            "Bandwidth below expected but compute normal — possible HBM/DRAM degradation, \
             or buffer sized within L2 cache"
                .to_string(),
        );
    }
    if bw_passed && !flops_passed {
        diagnosis.push(
            "Compute below expected but bandwidth normal — possible thermal throttling, \
             clock stuck at base, or driver issue"
                .to_string(),
        );
    }
    if !bw_passed && !flops_passed {
        diagnosis.push(
            "Both bandwidth and compute below expected — possible power delivery issue, \
             PCIe running at lower gen, or GPU in reduced mode"
                .to_string(),
        );
    }
    if !stability_passed {
        diagnosis.push(
            "High measurement variance — possible concurrent GPU load, thermal throttling \
             during measurement, or unstable driver state"
                .to_string(),
        );
    }

    let pass_count = checks.iter().filter(|c| c.passed).count();
    let total_count = checks.len();
    let passed = pass_count == total_count;

    ValidationResult {
        gpu_name: model.device_name.clone(),
        checks,
        diagnosis,
        passed,
        pass_count,
        total_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{KernelPlacement, RooflineModel};
    use gpu_harness::sim::MemoryLevel;

    fn mock_h100_model() -> RooflineModel {
        RooflineModel {
            device_name: "NVIDIA H100 80GB HBM3".to_string(),
            peak_gflops: 59500.0,
            peak_bandwidth_gbps: 2893.0,
            ridge_point: 20.6,
            clock_mhz: 1830,
            temperature_c: 35,
            power_watts: 300.0,
            placements: vec![
                KernelPlacement {
                    name: "copy".to_string(),
                    arithmetic_intensity: 0.0,
                    achieved_gflops: 0.0,
                    achieved_bandwidth_gbps: 2893.0,
                    efficiency: 1.0,
                    bottleneck: Bottleneck::MemoryBound {
                        level: MemoryLevel::Hbm,
                    },
                    median_us: 185.6,
                    stddev_us: 0.7,
                    cv: 0.004,
                },
                KernelPlacement {
                    name: "fma_heavy".to_string(),
                    arithmetic_intensity: 64.0,
                    achieved_gflops: 59500.0,
                    achieved_bandwidth_gbps: 930.0,
                    efficiency: 1.0,
                    bottleneck: Bottleneck::ComputeBound,
                    median_us: 577.0,
                    stddev_us: 0.6,
                    cv: 0.001,
                },
            ],
        }
    }

    fn h100_baseline() -> &'static HardwareBaseline {
        super::super::baselines::find_baseline(&gpu_harness::device::GpuDevice {
            index: 0,
            name: "NVIDIA H100 80GB HBM3".to_string(),
            vendor: gpu_harness::device::GpuVendor::Nvidia,
            architecture: gpu_harness::device::GpuArchitecture::Hopper,
            memory_bytes: 80 * 1024 * 1024 * 1024,
            pci_bus_id: None,
            driver_version: None,
            features: gpu_harness::device::GpuFeatures {
                compute_capability: Some((9, 0)),
                ..Default::default()
            },
            limits: gpu_harness::device::GpuLimits::default(),
        })
        .unwrap()
    }

    #[test]
    fn test_h100_passes_validation() {
        let model = mock_h100_model();
        let baseline = h100_baseline();
        let result = validate_roofline(&model, baseline, 0.8);

        assert!(
            result.passed,
            "H100 validated results should pass: {:?}",
            result.checks
        );
        assert_eq!(result.pass_count, result.total_count);
        assert!(result.diagnosis.is_empty());
    }

    #[test]
    fn test_degraded_bandwidth_fails() {
        let mut model = mock_h100_model();
        model.peak_bandwidth_gbps = 1500.0; // Way below 2700-3100 range
        model.placements[0].achieved_bandwidth_gbps = 1500.0;

        let baseline = h100_baseline();
        let result = validate_roofline(&model, baseline, 0.8);

        assert!(!result.passed, "Degraded bandwidth should fail");
        assert!(!result.diagnosis.is_empty(), "Should have diagnosis");
        assert!(
            result.diagnosis[0].contains("bandwidth") || result.diagnosis[0].contains("Bandwidth"),
            "Diagnosis should mention bandwidth"
        );
    }

    #[test]
    fn test_degraded_compute_fails() {
        let mut model = mock_h100_model();
        model.peak_gflops = 30000.0; // 30 TFLOPS, below 55-65 range

        let baseline = h100_baseline();
        let result = validate_roofline(&model, baseline, 0.8);

        assert!(!result.passed, "Degraded compute should fail");
    }

    #[test]
    fn test_high_cv_fails() {
        let mut model = mock_h100_model();
        model.placements[0].cv = 0.15; // 15% CV

        let baseline = h100_baseline();
        let result = validate_roofline(&model, baseline, 0.8);

        assert!(!result.passed, "High CV should fail stability check");
    }

    #[test]
    fn test_strict_threshold() {
        let model = mock_h100_model();
        let baseline = h100_baseline();

        // At 80% threshold, should pass (2893 > 2700*0.8 = 2160)
        let result_80 = validate_roofline(&model, baseline, 0.8);
        assert!(result_80.passed);

        // At 99% threshold, might fail depending on exact numbers
        let result_99 = validate_roofline(&model, baseline, 0.99);
        // 2893 vs 2700*0.99 = 2673 — still passes
        assert!(result_99.passed);
    }
}
