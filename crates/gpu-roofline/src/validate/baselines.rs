use gpu_harness::device::GpuDevice;
/// Expected performance range for a specific GPU model.
///
/// Baselines are ranges (not single values) because real performance varies
/// across SKUs (SXM vs PCIe), cloud vs bare metal, driver versions, and
/// thermal conditions at measurement time.
#[derive(Debug, Clone)]
pub struct HardwareBaseline {
    /// Human-readable GPU model name.
    pub model: &'static str,
    /// Name substrings to match against detected GPU name.
    pub name_patterns: &'static [&'static str],
    /// Compute capability to match (if CUDA).
    pub compute_capability: Option<(u32, u32)>,
    /// Expected HBM/DRAM bandwidth range (GB/s).
    pub bandwidth_range: (f64, f64),
    /// Expected FP32 FLOPS range (TFLOPS).
    pub flops_range: (f64, f64),
    /// L2 cache size (MB) — determines minimum buffer size.
    pub l2_cache_mb: u32,
    /// Recommended buffer size (bytes) — must exceed L2 for HBM measurement.
    pub recommended_buffer_bytes: usize,
    /// Expected ridge point range (FLOP/byte).
    pub ridge_point_range: (f64, f64),
    /// Expected maximum CV (coefficient of variation) for stable measurement.
    pub max_cv_percent: f64,
}

/// All known hardware baselines.
///
/// Ordered from most specific to least specific — matching stops at first hit.
pub fn all_baselines() -> &'static [HardwareBaseline] {
    &[
        // NVIDIA Datacenter — Hopper
        HardwareBaseline {
            model: "H100 SXM5",
            name_patterns: &["H100", "h100"],
            compute_capability: Some((9, 0)),
            bandwidth_range: (2700.0, 3100.0),
            flops_range: (55.0, 65.0),
            l2_cache_mb: 50,
            recommended_buffer_bytes: 256 * 1024 * 1024,
            ridge_point_range: (18.0, 24.0),
            max_cv_percent: 5.0,
        },
        HardwareBaseline {
            model: "H200 SXM",
            name_patterns: &["H200", "h200"],
            compute_capability: Some((9, 0)),
            bandwidth_range: (3800.0, 4500.0),
            flops_range: (55.0, 65.0),
            l2_cache_mb: 50,
            recommended_buffer_bytes: 256 * 1024 * 1024,
            ridge_point_range: (12.0, 18.0),
            max_cv_percent: 5.0,
        },
        // NVIDIA Datacenter — Ampere
        HardwareBaseline {
            model: "A100 SXM4",
            name_patterns: &["A100", "a100"],
            compute_capability: Some((8, 0)),
            bandwidth_range: (1800.0, 2100.0),
            flops_range: (16.0, 20.0),
            l2_cache_mb: 40,
            recommended_buffer_bytes: 256 * 1024 * 1024,
            ridge_point_range: (8.0, 12.0),
            max_cv_percent: 5.0,
        },
        // NVIDIA Datacenter — Blackwell
        HardwareBaseline {
            model: "B200",
            name_patterns: &["B200", "b200"],
            compute_capability: Some((10, 0)),
            bandwidth_range: (6000.0, 8000.0),
            flops_range: (60.0, 80.0),
            l2_cache_mb: 64,
            recommended_buffer_bytes: 512 * 1024 * 1024,
            ridge_point_range: (8.0, 14.0),
            max_cv_percent: 5.0,
        },
        // NVIDIA Consumer — Blackwell
        HardwareBaseline {
            model: "RTX 5090",
            name_patterns: &["5090", "RTX 509"],
            compute_capability: Some((10, 0)),
            bandwidth_range: (1500.0, 1800.0),
            flops_range: (90.0, 110.0),
            l2_cache_mb: 96,
            recommended_buffer_bytes: 512 * 1024 * 1024,
            ridge_point_range: (50.0, 75.0),
            max_cv_percent: 5.0,
        },
        // NVIDIA Consumer — Ada Lovelace
        HardwareBaseline {
            model: "RTX 4090",
            name_patterns: &["4090", "RTX 409"],
            compute_capability: Some((8, 9)),
            bandwidth_range: (800.0, 1000.0),
            flops_range: (75.0, 85.0),
            l2_cache_mb: 72,
            recommended_buffer_bytes: 256 * 1024 * 1024,
            ridge_point_range: (75.0, 110.0),
            max_cv_percent: 5.0,
        },
        HardwareBaseline {
            model: "RTX 4080",
            name_patterns: &["4080", "RTX 408"],
            compute_capability: Some((8, 9)),
            bandwidth_range: (600.0, 750.0),
            flops_range: (45.0, 55.0),
            l2_cache_mb: 64,
            recommended_buffer_bytes: 256 * 1024 * 1024,
            ridge_point_range: (60.0, 95.0),
            max_cv_percent: 5.0,
        },
        // NVIDIA Consumer — Ampere
        HardwareBaseline {
            model: "RTX 3090",
            name_patterns: &["3090", "RTX 309"],
            compute_capability: Some((8, 6)),
            bandwidth_range: (700.0, 900.0),
            flops_range: (30.0, 38.0),
            l2_cache_mb: 6,
            recommended_buffer_bytes: 64 * 1024 * 1024,
            ridge_point_range: (33.0, 55.0),
            max_cv_percent: 5.0,
        },
        HardwareBaseline {
            model: "RTX 3080",
            name_patterns: &["3080", "RTX 308"],
            compute_capability: Some((8, 6)),
            bandwidth_range: (600.0, 760.0),
            flops_range: (25.0, 32.0),
            l2_cache_mb: 5,
            recommended_buffer_bytes: 64 * 1024 * 1024,
            ridge_point_range: (33.0, 55.0),
            max_cv_percent: 5.0,
        },
        // AMD Datacenter
        HardwareBaseline {
            model: "MI300X",
            name_patterns: &["MI300", "mi300"],
            compute_capability: None,
            bandwidth_range: (4500.0, 5300.0),
            flops_range: (150.0, 170.0),
            l2_cache_mb: 256, // Infinity Cache
            recommended_buffer_bytes: 1024 * 1024 * 1024,
            ridge_point_range: (28.0, 38.0),
            max_cv_percent: 5.0,
        },
        // Intel Discrete
        HardwareBaseline {
            model: "Arc A770",
            name_patterns: &["A770", "Arc A7"],
            compute_capability: None,
            bandwidth_range: (450.0, 560.0),
            flops_range: (15.0, 20.0),
            l2_cache_mb: 16,
            recommended_buffer_bytes: 128 * 1024 * 1024,
            ridge_point_range: (27.0, 45.0),
            max_cv_percent: 10.0,
        },
        // Intel Integrated (broad match — last resort)
        HardwareBaseline {
            model: "Intel Integrated",
            name_patterns: &["UHD", "Iris", "Intel"],
            compute_capability: None,
            bandwidth_range: (5.0, 50.0),
            flops_range: (0.05, 0.5),
            l2_cache_mb: 2,
            recommended_buffer_bytes: 16 * 1024 * 1024,
            ridge_point_range: (10.0, 50.0),
            max_cv_percent: 15.0,
        },
    ]
}

/// Find the best-matching baseline for a detected GPU.
///
/// Matches by name patterns first (most reliable), then by compute
/// capability if no name match is found.
pub fn find_baseline(device: &GpuDevice) -> Option<&'static HardwareBaseline> {
    let name_lower = device.name.to_lowercase();

    // First pass: match by name patterns
    for baseline in all_baselines() {
        for pattern in baseline.name_patterns {
            if name_lower.contains(&pattern.to_lowercase()) {
                return Some(baseline);
            }
        }
    }

    // Second pass: match by compute capability (CUDA only)
    if let Some(cc) = device.features.compute_capability {
        for baseline in all_baselines() {
            if let Some(expected_cc) = baseline.compute_capability {
                if cc == expected_cc {
                    return Some(baseline);
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::device::{GpuArchitecture, GpuFeatures, GpuLimits, GpuVendor};

    fn mock_device(name: &str, cc: Option<(u32, u32)>) -> GpuDevice {
        GpuDevice {
            index: 0,
            name: name.to_string(),
            vendor: GpuVendor::Nvidia,
            architecture: GpuArchitecture::Hopper,
            memory_bytes: 80 * 1024 * 1024 * 1024,
            pci_bus_id: None,
            driver_version: None,
            features: GpuFeatures {
                compute_capability: cc,
                ..Default::default()
            },
            limits: GpuLimits::default(),
        }
    }

    #[test]
    fn test_find_h100_by_name() {
        let device = mock_device("NVIDIA H100 80GB HBM3", Some((9, 0)));
        let baseline = find_baseline(&device);
        assert!(baseline.is_some());
        assert_eq!(baseline.unwrap().model, "H100 SXM5");
    }

    #[test]
    fn test_find_rtx4090_by_name() {
        let device = mock_device("NVIDIA GeForce RTX 4090", Some((8, 9)));
        let baseline = find_baseline(&device);
        assert!(baseline.is_some());
        assert_eq!(baseline.unwrap().model, "RTX 4090");
    }

    #[test]
    fn test_find_intel_uhd() {
        let device = mock_device("Intel(R) UHD Graphics", None);
        let baseline = find_baseline(&device);
        assert!(baseline.is_some());
        assert_eq!(baseline.unwrap().model, "Intel Integrated");
    }

    #[test]
    fn test_find_mi300x() {
        let mut device = mock_device("AMD Instinct MI300X", None);
        device.vendor = GpuVendor::Amd;
        let baseline = find_baseline(&device);
        assert!(baseline.is_some());
        assert_eq!(baseline.unwrap().model, "MI300X");
    }

    #[test]
    fn test_unknown_gpu_returns_none() {
        let device = mock_device("Unknown Future GPU", None);
        let baseline = find_baseline(&device);
        assert!(baseline.is_none());
    }

    #[test]
    fn test_all_baselines_have_valid_ranges() {
        for baseline in all_baselines() {
            assert!(
                baseline.bandwidth_range.1 > baseline.bandwidth_range.0,
                "{}: bandwidth range invalid",
                baseline.model
            );
            assert!(
                baseline.flops_range.1 > baseline.flops_range.0,
                "{}: FLOPS range invalid",
                baseline.model
            );
            assert!(
                baseline.recommended_buffer_bytes > baseline.l2_cache_mb as usize * 1024 * 1024,
                "{}: buffer should exceed L2 cache",
                baseline.model
            );
            assert!(
                baseline.max_cv_percent > 0.0,
                "{}: max CV should be positive",
                baseline.model
            );
        }
    }

    #[test]
    fn test_h100_baseline_matches_validated_results() {
        let device = mock_device("NVIDIA H100 80GB HBM3", Some((9, 0)));
        let baseline = find_baseline(&device).unwrap();

        // Our validated H100 result: 2893 GB/s, 59.5 TFLOPS
        assert!(
            2893.0 >= baseline.bandwidth_range.0 && 2893.0 <= baseline.bandwidth_range.1,
            "H100 validated BW (2893) should be in range {:?}",
            baseline.bandwidth_range
        );
        assert!(
            59.5 >= baseline.flops_range.0 && 59.5 <= baseline.flops_range.1,
            "H100 validated FLOPS (59.5) should be in range {:?}",
            baseline.flops_range
        );
    }
}
