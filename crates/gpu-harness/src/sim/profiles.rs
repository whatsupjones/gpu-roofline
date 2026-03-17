use crate::device::{GpuArchitecture, GpuVendor};

use super::bandwidth::BandwidthModel;
use super::gpu_model::SimGpuProfile;
use super::power::PowerModel;
use super::thermal::ThermalModel;

// =============================================================================
// NVIDIA CONSUMER — Current Generation
// =============================================================================

/// NVIDIA GeForce RTX 5090 (Blackwell, GB202) — current consumer flagship.
/// 512-bit bus, 32GB GDDR7, 575W TDP.
pub fn rtx_5090() -> SimGpuProfile {
    SimGpuProfile {
        name: "NVIDIA GeForce RTX 5090".to_string(),
        vendor: GpuVendor::Nvidia,
        architecture: GpuArchitecture::Blackwell,
        vram_bytes: 32 * 1024 * 1024 * 1024,
        cuda_cores: 21760,
        sm_count: 170,
        thermal: ThermalModel {
            ambient_c: 35.0,
            throttle_onset_c: 80.0,
            throttle_max_c: 90.0,
            shutdown_c: 95.0,
            thermal_mass_j_per_c: 18.0,
            cooling_capacity_watts: 400.0,
        },
        power: PowerModel {
            tdp_watts: 575.0,
            base_clock_mhz: 2010,
            boost_clock_mhz: 2407,
            power_exponent: 1.5,
            idle_watts: 30.0,
        },
        bandwidth: BandwidthModel {
            l1_bandwidth_gbps: 7200.0,
            l2_bandwidth_gbps: 3600.0,
            hbm_bandwidth_gbps: 1792.0, // GDDR7, 512-bit bus
            pcie_bandwidth_gbps: 63.0,  // PCIe 5.0 x16
            l1_size_kb: 128,
            l2_size_mb: 96,
            vram_gb: 32,
            sustained_ratio: 0.93,
        },
        nvlink_bandwidth_gbps: None, // Consumer card, no NVLink
        pcie_gen: 5,
        pcie_lanes: 16,
    }
}

/// NVIDIA GeForce RTX 4090 (Ada Lovelace) — widely deployed reference.
/// 384-bit bus, 24GB GDDR6X, 450W TDP.
pub fn rtx_4090() -> SimGpuProfile {
    SimGpuProfile {
        name: "NVIDIA GeForce RTX 4090".to_string(),
        vendor: GpuVendor::Nvidia,
        architecture: GpuArchitecture::Ada,
        vram_bytes: 24 * 1024 * 1024 * 1024,
        cuda_cores: 16384,
        sm_count: 128,
        thermal: ThermalModel {
            ambient_c: 35.0,
            throttle_onset_c: 78.0,
            throttle_max_c: 90.0,
            shutdown_c: 95.0,
            thermal_mass_j_per_c: 15.0,
            cooling_capacity_watts: 350.0,
        },
        power: PowerModel {
            tdp_watts: 450.0,
            base_clock_mhz: 2235,
            boost_clock_mhz: 2520,
            power_exponent: 1.5,
            idle_watts: 25.0,
        },
        bandwidth: BandwidthModel {
            l1_bandwidth_gbps: 5800.0,
            l2_bandwidth_gbps: 2900.0,
            hbm_bandwidth_gbps: 1008.0,
            pcie_bandwidth_gbps: 63.0,
            l1_size_kb: 128,
            l2_size_mb: 72,
            vram_gb: 24,
            sustained_ratio: 0.93,
        },
        nvlink_bandwidth_gbps: None,
        pcie_gen: 4,
        pcie_lanes: 16,
    }
}

// =============================================================================
// NVIDIA DATACENTER — Current Generation
// =============================================================================

/// NVIDIA H100 SXM5 (Hopper) — current datacenter flagship.
/// HBM3, 80GB, 700W TDP, NVLink 4.0.
pub fn h100_sxm() -> SimGpuProfile {
    SimGpuProfile {
        name: "NVIDIA H100 SXM5 80GB".to_string(),
        vendor: GpuVendor::Nvidia,
        architecture: GpuArchitecture::Hopper,
        vram_bytes: 80 * 1024 * 1024 * 1024,
        cuda_cores: 16896,
        sm_count: 132,
        thermal: ThermalModel {
            ambient_c: 25.0, // Datacenter cooling
            throttle_onset_c: 75.0,
            throttle_max_c: 83.0,
            shutdown_c: 90.0,
            thermal_mass_j_per_c: 25.0, // Larger die = more thermal mass
            cooling_capacity_watts: 600.0,
        },
        power: PowerModel {
            tdp_watts: 700.0,
            base_clock_mhz: 1095,
            boost_clock_mhz: 1830,
            power_exponent: 1.5,
            idle_watts: 50.0,
        },
        bandwidth: BandwidthModel {
            l1_bandwidth_gbps: 8400.0,
            l2_bandwidth_gbps: 4200.0,
            hbm_bandwidth_gbps: 3350.0, // HBM3
            pcie_bandwidth_gbps: 63.0,
            l1_size_kb: 256,
            l2_size_mb: 50,
            vram_gb: 80,
            sustained_ratio: 0.95, // HBM sustains better than GDDR
        },
        nvlink_bandwidth_gbps: Some(900.0), // 18 links, 50 GB/s each
        pcie_gen: 5,
        pcie_lanes: 16,
    }
}

/// NVIDIA H200 SXM (Hopper refresh) — HBM3e, 141GB.
pub fn h200_sxm() -> SimGpuProfile {
    let mut profile = h100_sxm();
    profile.name = "NVIDIA H200 SXM 141GB".to_string();
    profile.vram_bytes = 141 * 1024 * 1024 * 1024;
    profile.bandwidth.hbm_bandwidth_gbps = 4800.0; // HBM3e
    profile.bandwidth.l2_bandwidth_gbps = 5500.0; // Faster L2 to maintain hierarchy
    profile.bandwidth.vram_gb = 141;
    profile
}

/// NVIDIA B200 (Blackwell datacenter) — next-gen datacenter.
/// HBM3e, 192GB, 1000W TDP, NVLink 5.0.
pub fn b200() -> SimGpuProfile {
    SimGpuProfile {
        name: "NVIDIA B200 192GB".to_string(),
        vendor: GpuVendor::Nvidia,
        architecture: GpuArchitecture::Blackwell,
        vram_bytes: 192 * 1024 * 1024 * 1024,
        cuda_cores: 18432,
        sm_count: 144,
        thermal: ThermalModel {
            ambient_c: 22.0,
            throttle_onset_c: 72.0,
            throttle_max_c: 80.0,
            shutdown_c: 87.0,
            thermal_mass_j_per_c: 30.0,
            cooling_capacity_watts: 800.0,
        },
        power: PowerModel {
            tdp_watts: 1000.0,
            base_clock_mhz: 1000,
            boost_clock_mhz: 1800,
            power_exponent: 1.5,
            idle_watts: 60.0,
        },
        bandwidth: BandwidthModel {
            l1_bandwidth_gbps: 14000.0,
            l2_bandwidth_gbps: 9000.0,
            hbm_bandwidth_gbps: 8000.0, // HBM3e, wider bus
            pcie_bandwidth_gbps: 128.0, // PCIe 6.0
            l1_size_kb: 256,
            l2_size_mb: 64,
            vram_gb: 192,
            sustained_ratio: 0.96,
        },
        nvlink_bandwidth_gbps: Some(1800.0), // NVLink 5.0
        pcie_gen: 6,
        pcie_lanes: 16,
    }
}

// =============================================================================
// AMD — Current Generation
// =============================================================================

/// AMD Instinct MI300X — AMD datacenter flagship.
/// HBM3, 192GB, 750W TDP.
pub fn mi300x() -> SimGpuProfile {
    SimGpuProfile {
        name: "AMD Instinct MI300X".to_string(),
        vendor: GpuVendor::Amd,
        architecture: GpuArchitecture::Cdna3,
        vram_bytes: 192 * 1024 * 1024 * 1024,
        cuda_cores: 19456, // Stream processors
        sm_count: 304,     // Compute units
        thermal: ThermalModel {
            ambient_c: 25.0,
            throttle_onset_c: 80.0,
            throttle_max_c: 90.0,
            shutdown_c: 100.0,
            thermal_mass_j_per_c: 28.0,
            cooling_capacity_watts: 600.0,
        },
        power: PowerModel {
            tdp_watts: 750.0,
            base_clock_mhz: 1000,
            boost_clock_mhz: 2100,
            power_exponent: 1.5,
            idle_watts: 55.0,
        },
        bandwidth: BandwidthModel {
            l1_bandwidth_gbps: 6400.0,
            l2_bandwidth_gbps: 5500.0, // 256MB Infinity Cache, near-HBM bandwidth
            hbm_bandwidth_gbps: 5300.0, // 8-stack HBM3
            pcie_bandwidth_gbps: 63.0,
            l1_size_kb: 32,
            l2_size_mb: 256, // Infinity Cache
            vram_gb: 192,
            sustained_ratio: 0.94,
        },
        nvlink_bandwidth_gbps: None, // Uses Infinity Fabric
        pcie_gen: 5,
        pcie_lanes: 16,
    }
}

// =============================================================================
// INTEL — Current Generation
// =============================================================================

/// Intel Arc A770 (Alchemist) — Intel discrete flagship.
/// 16GB GDDR6, 225W TDP.
pub fn arc_a770() -> SimGpuProfile {
    SimGpuProfile {
        name: "Intel Arc A770 16GB".to_string(),
        vendor: GpuVendor::Intel,
        architecture: GpuArchitecture::ArcAlchemist,
        vram_bytes: 16 * 1024 * 1024 * 1024,
        cuda_cores: 4096, // Xe cores equivalent
        sm_count: 32,     // Xe-HPG render slices
        thermal: ThermalModel {
            ambient_c: 35.0,
            throttle_onset_c: 85.0,
            throttle_max_c: 95.0,
            shutdown_c: 100.0,
            thermal_mass_j_per_c: 10.0,
            cooling_capacity_watts: 200.0,
        },
        power: PowerModel {
            tdp_watts: 225.0,
            base_clock_mhz: 2100,
            boost_clock_mhz: 2400,
            power_exponent: 1.5,
            idle_watts: 15.0,
        },
        bandwidth: BandwidthModel {
            l1_bandwidth_gbps: 2400.0,
            l2_bandwidth_gbps: 1200.0,
            hbm_bandwidth_gbps: 560.0,
            pcie_bandwidth_gbps: 32.0, // PCIe 4.0 x16
            l1_size_kb: 64,
            l2_size_mb: 16,
            vram_gb: 16,
            sustained_ratio: 0.91,
        },
        nvlink_bandwidth_gbps: None,
        pcie_gen: 4,
        pcie_lanes: 16,
    }
}

// =============================================================================
// DEGRADED PROFILES — For straggler/parity detection testing
// =============================================================================

/// RTX 5090 with degraded thermal paste — throttles earlier.
pub fn degraded_5090_thermal() -> SimGpuProfile {
    let mut profile = rtx_5090();
    profile.name = "NVIDIA GeForce RTX 5090 [DEGRADED: thermal]".to_string();
    // Dried thermal paste: throttle onset 12C earlier
    profile.thermal.throttle_onset_c = 68.0;
    profile.thermal.throttle_max_c = 80.0;
    // Reduced cooling efficiency
    profile.thermal.cooling_capacity_watts = 280.0;
    profile
}

/// H100 with degraded NVLink — fewer active links.
pub fn degraded_h100_nvlink() -> SimGpuProfile {
    let mut profile = h100_sxm();
    profile.name = "NVIDIA H100 SXM5 [DEGRADED: NVLink]".to_string();
    // Only 12 of 18 NVLink lanes active
    profile.nvlink_bandwidth_gbps = Some(600.0);
    profile
}

/// H100 with clock stuck at base — driver/firmware bug.
pub fn degraded_h100_clock() -> SimGpuProfile {
    let mut profile = h100_sxm();
    profile.name = "NVIDIA H100 SXM5 [DEGRADED: clock stuck]".to_string();
    // Boost clock stuck at base
    profile.power.boost_clock_mhz = profile.power.base_clock_mhz;
    profile
}

/// H100 with memory subsystem degradation — reduced HBM bandwidth.
pub fn degraded_h100_memory() -> SimGpuProfile {
    let mut profile = h100_sxm();
    profile.name = "NVIDIA H100 SXM5 [DEGRADED: memory]".to_string();
    // One HBM stack underperforming: ~75% of expected bandwidth
    profile.bandwidth.hbm_bandwidth_gbps *= 0.75;
    profile
}

/// Get a profile by name (for CLI --sim flag).
pub fn profile_by_name(name: &str) -> Option<SimGpuProfile> {
    match name.to_lowercase().as_str() {
        "rtx_5090" | "5090" => Some(rtx_5090()),
        "rtx_4090" | "4090" => Some(rtx_4090()),
        "h100" | "h100_sxm" => Some(h100_sxm()),
        "h200" | "h200_sxm" => Some(h200_sxm()),
        "b200" => Some(b200()),
        "mi300x" | "mi300" => Some(mi300x()),
        "arc_a770" | "a770" => Some(arc_a770()),
        "degraded_5090" => Some(degraded_5090_thermal()),
        "degraded_h100_nvlink" => Some(degraded_h100_nvlink()),
        "degraded_h100_clock" => Some(degraded_h100_clock()),
        "degraded_h100_memory" => Some(degraded_h100_memory()),
        _ => None,
    }
}

/// List all available profile names.
pub fn available_profiles() -> Vec<&'static str> {
    vec![
        "rtx_5090",
        "rtx_4090",
        "h100_sxm",
        "h200_sxm",
        "b200",
        "mi300x",
        "arc_a770",
        "degraded_5090",
        "degraded_h100_nvlink",
        "degraded_h100_clock",
        "degraded_h100_memory",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_profiles_load() {
        for name in available_profiles() {
            let profile = profile_by_name(name);
            assert!(profile.is_some(), "profile '{name}' should exist");
            let p = profile.unwrap();
            assert!(!p.name.is_empty());
            assert!(p.cuda_cores > 0);
            assert!(p.vram_bytes > 0);
        }
    }

    #[test]
    fn test_rtx_5090_peak_tflops() {
        let profile = rtx_5090();
        let tflops = profile.peak_burst_tflops();
        // RTX 5090: 21760 cores * 2407 MHz * 2 / 1e12 ≈ 104.8 TFLOPS
        assert!(tflops > 90.0, "RTX 5090 should be >90 TFLOPS, got {tflops:.1}");
        assert!(tflops < 120.0, "RTX 5090 should be <120 TFLOPS, got {tflops:.1}");
    }

    #[test]
    fn test_h100_peak_tflops() {
        let profile = h100_sxm();
        let tflops = profile.peak_burst_tflops();
        // H100: 16896 cores * 1830 MHz * 2 / 1e12 ≈ 61.8 TFLOPS FP32
        assert!(tflops > 50.0, "H100 should be >50 TFLOPS FP32, got {tflops:.1}");
        assert!(tflops < 75.0, "H100 should be <75 TFLOPS FP32, got {tflops:.1}");
    }

    #[test]
    fn test_degraded_profiles_are_worse() {
        let healthy = rtx_5090();
        let degraded = degraded_5090_thermal();

        // Degraded should throttle earlier
        assert!(
            degraded.thermal.throttle_onset_c < healthy.thermal.throttle_onset_c,
            "degraded should throttle earlier"
        );

        // Degraded sustained should be lower
        assert!(
            degraded.peak_sustained_tflops() < healthy.peak_sustained_tflops()
                || degraded.thermal.cooling_capacity_watts < healthy.thermal.cooling_capacity_watts,
            "degraded profile should perform worse"
        );
    }

    #[test]
    fn test_h100_has_nvlink() {
        let profile = h100_sxm();
        assert!(profile.nvlink_bandwidth_gbps.is_some());
        assert!(profile.nvlink_bandwidth_gbps.unwrap() > 800.0);
    }

    #[test]
    fn test_consumer_cards_no_nvlink() {
        assert!(rtx_5090().nvlink_bandwidth_gbps.is_none());
        assert!(rtx_4090().nvlink_bandwidth_gbps.is_none());
        assert!(arc_a770().nvlink_bandwidth_gbps.is_none());
    }

    #[test]
    fn test_bandwidth_hierarchy() {
        for name in available_profiles() {
            let p = profile_by_name(name).unwrap();
            assert!(
                p.bandwidth.l1_bandwidth_gbps > p.bandwidth.l2_bandwidth_gbps,
                "{}: L1 should be faster than L2",
                name
            );
            assert!(
                p.bandwidth.l2_bandwidth_gbps > p.bandwidth.hbm_bandwidth_gbps,
                "{}: L2 should be faster than HBM",
                name
            );
            assert!(
                p.bandwidth.hbm_bandwidth_gbps > p.bandwidth.pcie_bandwidth_gbps,
                "{}: HBM should be faster than PCIe",
                name
            );
        }
    }

    #[test]
    fn test_ridge_point_reasonable() {
        let profile = h100_sxm();
        let ridge = profile.ridge_point_burst();
        // H100: ~61.8 TFLOPS / 3350 GB/s ≈ 18.4 FLOP/byte
        assert!(ridge > 10.0, "H100 ridge point should be >10, got {ridge:.1}");
        assert!(ridge < 30.0, "H100 ridge point should be <30, got {ridge:.1}");
    }
}
