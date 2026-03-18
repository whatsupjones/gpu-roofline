use serde::{Deserialize, Serialize};

use super::bandwidth::BandwidthModel;
use super::power::PowerModel;
use super::thermal::ThermalModel;
use crate::device::{GpuArchitecture, GpuVendor};

/// Complete simulated GPU profile combining thermal, power, and bandwidth models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimGpuProfile {
    pub name: String,
    pub vendor: GpuVendor,
    pub architecture: GpuArchitecture,
    pub vram_bytes: u64,
    pub cuda_cores: u32,
    pub sm_count: u32,
    pub thermal: ThermalModel,
    pub power: PowerModel,
    pub bandwidth: BandwidthModel,
    pub nvlink_bandwidth_gbps: Option<f64>,
    pub pcie_gen: u32,
    pub pcie_lanes: u32,
}

impl SimGpuProfile {
    /// Peak burst FLOPS (FP32) at maximum boost clock.
    pub fn peak_burst_tflops(&self) -> f64 {
        PowerModel::peak_flops_at_clock(self.cuda_cores, self.power.boost_clock_mhz) / 1e12
    }

    /// Peak burst memory bandwidth in GB/s (HBM level).
    pub fn peak_burst_bandwidth_gbps(&self) -> f64 {
        self.bandwidth.hbm_bandwidth_gbps
    }

    /// Estimated sustained FLOPS accounting for thermal equilibrium.
    pub fn peak_sustained_tflops(&self) -> f64 {
        // At thermal equilibrium, clock is typically base + 60-80% of (boost - base)
        let sustained_clock = self.power.base_clock_mhz
            + ((self.power.boost_clock_mhz - self.power.base_clock_mhz) as f32 * 0.7) as u32;
        PowerModel::peak_flops_at_clock(self.cuda_cores, sustained_clock) / 1e12
    }

    /// Ridge point: arithmetic intensity where compute-bound meets memory-bound.
    pub fn ridge_point_burst(&self) -> f64 {
        let peak_flops_gbyte = self.peak_burst_tflops() * 1e3; // GFLOP/s
        peak_flops_gbyte / self.bandwidth.hbm_bandwidth_gbps
    }
}
