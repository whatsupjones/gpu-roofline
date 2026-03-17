use serde::{Deserialize, Serialize};

/// Memory bandwidth model for GPU simulation.
///
/// Models hierarchical memory bandwidth (L1 > L2 > HBM/DRAM > PCIe)
/// with working-set-dependent ceiling selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthModel {
    /// L1 cache bandwidth in GB/s (typically 3-6 TB/s).
    pub l1_bandwidth_gbps: f64,
    /// L2 cache bandwidth in GB/s (typically 1.5-3 TB/s).
    pub l2_bandwidth_gbps: f64,
    /// HBM or DRAM bandwidth in GB/s.
    pub hbm_bandwidth_gbps: f64,
    /// PCIe host-device bandwidth in GB/s.
    pub pcie_bandwidth_gbps: f64,
    /// L1 cache size in KB per SM (typically 128-256 KB).
    pub l1_size_kb: u32,
    /// L2 cache size in MB (typically 48-96 MB).
    pub l2_size_mb: u32,
    /// Total VRAM in GB.
    pub vram_gb: u32,
    /// Sustained bandwidth ratio (sustained / burst). Typically 0.90-0.97.
    pub sustained_ratio: f64,
}

/// Which memory level is the bandwidth bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLevel {
    L1,
    L2,
    Hbm,
    Pcie,
}

impl std::fmt::Display for MemoryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::L1 => write!(f, "L1 Cache"),
            Self::L2 => write!(f, "L2 Cache"),
            Self::Hbm => write!(f, "HBM/DRAM"),
            Self::Pcie => write!(f, "PCIe"),
        }
    }
}

impl BandwidthModel {
    /// Determine effective bandwidth for a given working set size.
    ///
    /// Returns (bandwidth_gbps, memory_level) based on which cache level
    /// the working set fits in.
    pub fn effective_bandwidth(&self, working_set_bytes: usize) -> (f64, MemoryLevel) {
        let working_set_kb = working_set_bytes / 1024;
        let working_set_mb = working_set_kb / 1024;

        if working_set_kb <= self.l1_size_kb as usize {
            (self.l1_bandwidth_gbps, MemoryLevel::L1)
        } else if working_set_mb <= self.l2_size_mb as usize {
            (self.l2_bandwidth_gbps, MemoryLevel::L2)
        } else if working_set_mb <= (self.vram_gb as usize * 1024) {
            (self.hbm_bandwidth_gbps, MemoryLevel::Hbm)
        } else {
            (self.pcie_bandwidth_gbps, MemoryLevel::Pcie)
        }
    }

    /// Sustained bandwidth at a given memory level.
    pub fn sustained_bandwidth(&self, level: MemoryLevel) -> f64 {
        let burst = match level {
            MemoryLevel::L1 => self.l1_bandwidth_gbps,
            MemoryLevel::L2 => self.l2_bandwidth_gbps,
            MemoryLevel::Hbm => self.hbm_bandwidth_gbps,
            MemoryLevel::Pcie => self.pcie_bandwidth_gbps,
        };
        burst * self.sustained_ratio
    }

    /// Apply contention degradation (0.0 = no contention, 1.0 = max contention).
    pub fn degraded_bandwidth(
        &self,
        working_set_bytes: usize,
        contention: f64,
    ) -> (f64, MemoryLevel) {
        let (bw, level) = self.effective_bandwidth(working_set_bytes);
        let contention = contention.clamp(0.0, 1.0);
        // Contention reduces bandwidth: at max contention, drops to ~50% of burst
        let degraded = bw * (1.0 - contention * 0.5);
        (degraded, level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rtx_4090_bandwidth() -> BandwidthModel {
        BandwidthModel {
            l1_bandwidth_gbps: 5800.0,
            l2_bandwidth_gbps: 2900.0,
            hbm_bandwidth_gbps: 1008.0,
            pcie_bandwidth_gbps: 63.0,
            l1_size_kb: 128,
            l2_size_mb: 72,
            vram_gb: 24,
            sustained_ratio: 0.93,
        }
    }

    #[test]
    fn test_l1_bandwidth_small_working_set() {
        let model = rtx_4090_bandwidth();
        let (bw, level) = model.effective_bandwidth(64 * 1024); // 64 KB
        assert_eq!(level, MemoryLevel::L1);
        assert!((bw - 5800.0).abs() < 0.1);
    }

    #[test]
    fn test_l2_bandwidth_medium_working_set() {
        let model = rtx_4090_bandwidth();
        let (bw, level) = model.effective_bandwidth(16 * 1024 * 1024); // 16 MB
        assert_eq!(level, MemoryLevel::L2);
        assert!((bw - 2900.0).abs() < 0.1);
    }

    #[test]
    fn test_hbm_bandwidth_large_working_set() {
        let model = rtx_4090_bandwidth();
        let (bw, level) = model.effective_bandwidth(256 * 1024 * 1024); // 256 MB
        assert_eq!(level, MemoryLevel::Hbm);
        assert!((bw - 1008.0).abs() < 0.1);
    }

    #[test]
    fn test_sustained_less_than_burst() {
        let model = rtx_4090_bandwidth();
        let sustained = model.sustained_bandwidth(MemoryLevel::Hbm);
        assert!(sustained < model.hbm_bandwidth_gbps);
        assert!(sustained > model.hbm_bandwidth_gbps * 0.9);
    }

    #[test]
    fn test_contention_degrades_bandwidth() {
        let model = rtx_4090_bandwidth();
        let (bw_clean, _) = model.degraded_bandwidth(256 * 1024 * 1024, 0.0);
        let (bw_contested, _) = model.degraded_bandwidth(256 * 1024 * 1024, 0.8);
        assert!(bw_contested < bw_clean);
        assert!(bw_contested > 0.0);
    }

    #[test]
    fn test_memory_hierarchy_ordering() {
        let model = rtx_4090_bandwidth();
        assert!(model.l1_bandwidth_gbps > model.l2_bandwidth_gbps);
        assert!(model.l2_bandwidth_gbps > model.hbm_bandwidth_gbps);
        assert!(model.hbm_bandwidth_gbps > model.pcie_bandwidth_gbps);
    }
}
