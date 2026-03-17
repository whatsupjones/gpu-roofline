use serde::{Deserialize, Serialize};

use super::gpu_model::SimGpuProfile;

/// A simulated multi-GPU fleet for testing straggler detection and parity validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedFleet {
    pub gpus: Vec<SimGpuInstance>,
    pub topology: SimTopology,
}

/// A single GPU instance in a simulated fleet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimGpuInstance {
    pub index: u32,
    pub profile: SimGpuProfile,
    /// Random performance jitter (0.0 = deterministic, 0.05 = 5% variance).
    pub jitter: f64,
    /// Injected degradation for testing detection logic.
    pub degradation: Option<Degradation>,
}

/// Types of degradation that can be injected into a simulated GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Degradation {
    /// Dried thermal paste: GPU heats faster, throttles earlier.
    ThermalPasteDried { extra_degrees_c: f32 },
    /// Fewer active NVLink lanes (degraded interconnect).
    NvlinkDegraded { active_links: u32, expected_links: u32 },
    /// PCIe running at lower gen than expected.
    PcieFallback { actual_gen: u32, expected_gen: u32 },
    /// Memory subsystem degradation (partial HBM stack failure).
    MemorySubsystem { bandwidth_ratio: f64 },
    /// Clock stuck at a lower frequency (driver/firmware bug).
    ClockStuck { max_mhz: u32 },
}

/// Simulated GPU interconnect topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimTopology {
    pub gpu_count: u32,
    pub nvswitch_count: u32,
    pub numa_nodes: Vec<u32>,
    pub p2p_bandwidth_gbps: Vec<Vec<f64>>,
}

impl SimulatedFleet {
    /// Create a homogeneous fleet of N identical GPUs with NVSwitch full mesh.
    pub fn homogeneous(profile: SimGpuProfile, count: u32) -> Self {
        let nvlink_bw = profile.nvlink_bandwidth_gbps.unwrap_or(0.0);
        let gpus = (0..count)
            .map(|i| SimGpuInstance {
                index: i,
                profile: profile.clone(),
                jitter: 0.02, // 2% natural variance
                degradation: None,
            })
            .collect();

        let p2p = Self::full_mesh_bandwidth(count, nvlink_bw);

        SimulatedFleet {
            gpus,
            topology: SimTopology {
                gpu_count: count,
                nvswitch_count: if nvlink_bw > 0.0 { 4 } else { 0 },
                numa_nodes: (0..count).map(|i| i / (count / 2).max(1)).collect(),
                p2p_bandwidth_gbps: p2p,
            },
        }
    }

    /// Inject a degradation into a specific GPU.
    pub fn degrade_gpu(&mut self, index: u32, degradation: Degradation) {
        if let Some(gpu) = self.gpus.get_mut(index as usize) {
            // Apply degradation to the topology as well
            match &degradation {
                Degradation::NvlinkDegraded { active_links, expected_links } => {
                    let ratio = *active_links as f64 / *expected_links as f64;
                    for j in 0..self.topology.gpu_count as usize {
                        if j != index as usize {
                            self.topology.p2p_bandwidth_gbps[index as usize][j] *= ratio;
                            self.topology.p2p_bandwidth_gbps[j][index as usize] *= ratio;
                        }
                    }
                }
                Degradation::MemorySubsystem { bandwidth_ratio } => {
                    gpu.profile.bandwidth.hbm_bandwidth_gbps *= bandwidth_ratio;
                }
                Degradation::ClockStuck { max_mhz } => {
                    gpu.profile.power.boost_clock_mhz = *max_mhz;
                }
                Degradation::ThermalPasteDried { extra_degrees_c } => {
                    gpu.profile.thermal.throttle_onset_c -= extra_degrees_c;
                    gpu.profile.thermal.throttle_max_c -= extra_degrees_c;
                }
                Degradation::PcieFallback { actual_gen, .. } => {
                    gpu.profile.pcie_gen = *actual_gen;
                    // PCIe bandwidth roughly halves per generation step down
                    gpu.profile.bandwidth.pcie_bandwidth_gbps *=
                        0.5_f64.powi((gpu.profile.pcie_gen as i32 - *actual_gen as i32).max(0));
                }
            }
            gpu.degradation = Some(degradation);
        }
    }

    /// Generate a full-mesh P2P bandwidth matrix.
    fn full_mesh_bandwidth(count: u32, nvlink_bw: f64) -> Vec<Vec<f64>> {
        let n = count as usize;
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    matrix[i][j] = if nvlink_bw > 0.0 {
                        nvlink_bw
                    } else {
                        32.0 // PCIe P2P fallback
                    };
                }
            }
        }
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::profiles;

    #[test]
    fn test_homogeneous_fleet_creation() {
        let fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 8);
        assert_eq!(fleet.gpus.len(), 8);
        assert_eq!(fleet.topology.gpu_count, 8);
        assert_eq!(fleet.topology.nvswitch_count, 4);
    }

    #[test]
    fn test_p2p_bandwidth_matrix_symmetric() {
        let fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        let p2p = &fleet.topology.p2p_bandwidth_gbps;
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p2p[i][j], p2p[j][i], "P2P matrix should be symmetric");
            }
            assert_eq!(p2p[i][i], 0.0, "self-bandwidth should be 0");
        }
    }

    #[test]
    fn test_nvlink_degradation_injection() {
        let mut fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 8);
        let original_bw = fleet.topology.p2p_bandwidth_gbps[2][0];

        fleet.degrade_gpu(2, Degradation::NvlinkDegraded {
            active_links: 12,
            expected_links: 18,
        });

        let degraded_bw = fleet.topology.p2p_bandwidth_gbps[2][0];
        assert!(
            degraded_bw < original_bw,
            "degraded BW {degraded_bw} should be less than original {original_bw}"
        );
        // 12/18 = 66.7%
        let ratio = degraded_bw / original_bw;
        assert!((ratio - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_clock_stuck_degradation() {
        let mut fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        let original_boost = fleet.gpus[1].profile.power.boost_clock_mhz;

        fleet.degrade_gpu(1, Degradation::ClockStuck { max_mhz: 1095 });

        let stuck_boost = fleet.gpus[1].profile.power.boost_clock_mhz;
        assert_eq!(stuck_boost, 1095);
        assert!(stuck_boost < original_boost);
    }

    #[test]
    fn test_memory_subsystem_degradation() {
        let mut fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        let original_bw = fleet.gpus[3].profile.bandwidth.hbm_bandwidth_gbps;

        fleet.degrade_gpu(3, Degradation::MemorySubsystem { bandwidth_ratio: 0.75 });

        let degraded_bw = fleet.gpus[3].profile.bandwidth.hbm_bandwidth_gbps;
        assert!((degraded_bw - original_bw * 0.75).abs() < 1.0);
    }

    #[test]
    fn test_consumer_fleet_no_nvswitch() {
        let fleet = SimulatedFleet::homogeneous(profiles::rtx_5090(), 2);
        assert_eq!(fleet.topology.nvswitch_count, 0);
        // P2P should fall back to PCIe
        assert!(fleet.topology.p2p_bandwidth_gbps[0][1] < 100.0);
    }
}
