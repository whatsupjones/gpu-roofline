//! Parameterized scenario builders for all 6 waste categories.
//!
//! Each builder generates simulation parameters for one trial.
//! The runner calls these repeatedly to produce N trials per condition.

use serde::{Deserialize, Serialize};

use crate::sim::fleet::Degradation;
use crate::sim::gpu_model::SimGpuProfile;
use crate::sim::profiles;
use crate::sim::power::PowerModel;

use super::noise::NoiseModel;
use rand::Rng;

/// The six waste categories from the study protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WasteCategory {
    GhostAllocation,
    ContentionSqueeze,
    ProvisioningOverhead,
    BurstSustainedGap,
    StragglerTax,
    Oversubscription,
}

impl WasteCategory {
    pub fn all() -> &'static [WasteCategory] {
        &[
            WasteCategory::GhostAllocation,
            WasteCategory::ContentionSqueeze,
            WasteCategory::ProvisioningOverhead,
            WasteCategory::BurstSustainedGap,
            WasteCategory::StragglerTax,
            WasteCategory::Oversubscription,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            WasteCategory::GhostAllocation => "ghost_allocation",
            WasteCategory::ContentionSqueeze => "contention_squeeze",
            WasteCategory::ProvisioningOverhead => "provisioning_overhead",
            WasteCategory::BurstSustainedGap => "burst_sustained_gap",
            WasteCategory::StragglerTax => "straggler_tax",
            WasteCategory::Oversubscription => "oversubscription",
        }
    }

    pub fn index(&self) -> u8 {
        match self {
            WasteCategory::GhostAllocation => 1,
            WasteCategory::ContentionSqueeze => 2,
            WasteCategory::ProvisioningOverhead => 3,
            WasteCategory::BurstSustainedGap => 4,
            WasteCategory::StragglerTax => 5,
            WasteCategory::Oversubscription => 6,
        }
    }
}

impl std::fmt::Display for WasteCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// =============================================================================
// Category 1: Ghost Allocations
// =============================================================================

/// Teardown methods for ghost allocation trials.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TeardownMethod {
    Clean,
    UnderLoad,
    RapidChurn,
}

impl TeardownMethod {
    pub fn all() -> &'static [TeardownMethod] {
        &[TeardownMethod::Clean, TeardownMethod::UnderLoad, TeardownMethod::RapidChurn]
    }

    pub fn name(&self) -> &'static str {
        match self {
            TeardownMethod::Clean => "clean",
            TeardownMethod::UnderLoad => "under_load",
            TeardownMethod::RapidChurn => "rapid_churn",
        }
    }
}

/// MIG profiles for ghost allocation trials.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MigProfile {
    Mig1g10gb,
    Mig2g20gb,
    Mig3g40gb,
}

impl MigProfile {
    pub fn all() -> &'static [MigProfile] {
        &[MigProfile::Mig1g10gb, MigProfile::Mig2g20gb, MigProfile::Mig3g40gb]
    }

    pub fn name(&self) -> &'static str {
        match self {
            MigProfile::Mig1g10gb => "1g.10gb",
            MigProfile::Mig2g20gb => "2g.20gb",
            MigProfile::Mig3g40gb => "3g.40gb",
        }
    }

    pub fn vram_bytes(&self) -> u64 {
        match self {
            MigProfile::Mig1g10gb => 10 * 1024 * 1024 * 1024,
            MigProfile::Mig2g20gb => 20 * 1024 * 1024 * 1024,
            MigProfile::Mig3g40gb => 40 * 1024 * 1024 * 1024,
        }
    }
}

/// Parameters for one ghost allocation trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GhostTrialParams {
    pub teardown_method: String,
    pub mig_profile: String,
    pub vram_allocated_bytes: u64,
    pub ghost_bytes_injected: u64,
    pub is_control: bool,
}

/// Result of one ghost allocation trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GhostTrialResult {
    pub ghost_bytes_injected: u64,
    pub ghost_bytes_measured: i64,
    pub reclaim_latency_ms: f64,
    pub detected: bool,
    pub detection_threshold_bytes: u64,
}

/// Run one ghost allocation trial.
pub fn run_ghost_trial(
    params: &GhostTrialParams,
    noise: &NoiseModel,
    detection_threshold: u64,
    rng: &mut impl Rng,
) -> GhostTrialResult {
    let vram = params.vram_allocated_bytes;

    // Simulate pre-teardown memory: vram + some base usage
    let base_usage: u64 = 2 * 1024 * 1024 * 1024; // 2 GiB base
    let pre_teardown_used = base_usage + vram;

    // After teardown, ghost_bytes remain unreleased
    let actual_freed = vram - params.ghost_bytes_injected;
    let post_teardown_used = pre_teardown_used - actual_freed;

    // Add NVML reading noise + possible memory spike
    let pre_measured = noise.jitter_memory(pre_teardown_used, rng);
    let spike = noise.memory_spike(rng);
    let post_measured = noise.jitter_memory(post_teardown_used + spike, rng);

    // Compute measured ghost bytes (can be negative due to noise)
    let measured_freed = pre_measured as i64 - post_measured as i64;
    let ghost_measured = vram as i64 - measured_freed;

    // Base reclaim latency depends on teardown method
    let base_latency_ms = match params.teardown_method.as_str() {
        "clean" => 50.0,
        "under_load" => 150.0,
        "rapid_churn" => 30.0,
        _ => 50.0,
    };
    let reclaim_latency_ms = noise.jitter_latency(base_latency_ms, rng);

    let detected = ghost_measured > detection_threshold as i64;

    GhostTrialResult {
        ghost_bytes_injected: params.ghost_bytes_injected,
        ghost_bytes_measured: ghost_measured,
        reclaim_latency_ms,
        detected,
        detection_threshold_bytes: detection_threshold,
    }
}

/// Generate ghost trial parameters for one condition.
pub fn ghost_trial_params(
    method: TeardownMethod,
    profile: MigProfile,
    is_control: bool,
    rng: &mut impl Rng,
) -> GhostTrialParams {
    let ghost_bytes = if is_control {
        0
    } else {
        // Uniform(0, 1024 MiB) for treatment
        rng.gen_range(0..=(1024 * 1024 * 1024u64))
    };

    GhostTrialParams {
        teardown_method: method.name().to_string(),
        mig_profile: profile.name().to_string(),
        vram_allocated_bytes: profile.vram_bytes(),
        ghost_bytes_injected: ghost_bytes,
        is_control,
    }
}

// =============================================================================
// Category 2: Contention Squeeze
// =============================================================================

/// Parameters for one contention squeeze trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionTrialParams {
    pub tenant_count: u32,
    pub partitioning_mode: String,
    pub is_hardware_partitioned: bool,
}

/// Result of one contention squeeze trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionTrialResult {
    pub bandwidth_ratios: Vec<f64>,
    pub compute_ratios: Vec<f64>,
    pub mean_bandwidth_ratio: f64,
    pub mean_compute_ratio: f64,
    pub detected: bool,
    pub contention_threshold: f64,
}

/// Run one contention squeeze trial.
pub fn run_contention_trial(
    params: &ContentionTrialParams,
    noise: &NoiseModel,
    threshold: f64,
    rng: &mut impl Rng,
) -> ContentionTrialResult {
    let n = params.tenant_count as usize;

    let (bw_ratios, compute_ratios) = if params.is_hardware_partitioned {
        // MIG: no contention, each tenant gets full share
        let bw: Vec<f64> = (0..n).map(|_| noise.jitter_bandwidth(1.0, rng)).collect();
        let comp: Vec<f64> = (0..n).map(|_| noise.jitter_bandwidth(1.0, rng)).collect();
        (bw, comp)
    } else {
        // Time-sliced: each tenant gets 1/N of bandwidth
        let fair_share = 1.0 / n as f64;
        let bw: Vec<f64> = (0..n).map(|_| noise.jitter_bandwidth(fair_share, rng)).collect();
        let comp: Vec<f64> = (0..n).map(|_| noise.jitter_bandwidth(fair_share, rng)).collect();
        (bw, comp)
    };

    let mean_bw = bw_ratios.iter().sum::<f64>() / n as f64;
    let mean_comp = compute_ratios.iter().sum::<f64>() / n as f64;

    // Detection: any tenant dropped below (1 - threshold) of baseline
    let detected = if !params.is_hardware_partitioned && n > 1 {
        bw_ratios.iter().any(|&r| r < 1.0 - threshold)
    } else {
        false
    };

    ContentionTrialResult {
        bandwidth_ratios: bw_ratios,
        compute_ratios,
        mean_bandwidth_ratio: mean_bw,
        mean_compute_ratio: mean_comp,
        detected,
        contention_threshold: threshold,
    }
}

// =============================================================================
// Category 3: Provisioning Overhead
// =============================================================================

/// GPU load states for provisioning trials.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadState {
    Cold,
    Warm,
    Hot,
}

impl LoadState {
    pub fn all() -> &'static [LoadState] {
        &[LoadState::Cold, LoadState::Warm, LoadState::Hot]
    }

    pub fn name(&self) -> &'static str {
        match self {
            LoadState::Cold => "cold",
            LoadState::Warm => "warm",
            LoadState::Hot => "hot",
        }
    }

    /// Extra latency penalty in ms due to GPU thermal/load state.
    pub fn latency_penalty_ms(&self) -> f64 {
        match self {
            LoadState::Cold => 0.0,
            LoadState::Warm => 30.0,
            LoadState::Hot => 80.0,
        }
    }
}

/// MIG profiles for provisioning with base latencies.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProvisionProfile {
    Mig1g10gb,
    Mig2g20gb,
    Mig3g40gb,
    Mig4g40gb,
    Mig7g80gb,
}

impl ProvisionProfile {
    pub fn all() -> &'static [ProvisionProfile] {
        &[
            ProvisionProfile::Mig1g10gb,
            ProvisionProfile::Mig2g20gb,
            ProvisionProfile::Mig3g40gb,
            ProvisionProfile::Mig4g40gb,
            ProvisionProfile::Mig7g80gb,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ProvisionProfile::Mig1g10gb => "1g.10gb",
            ProvisionProfile::Mig2g20gb => "2g.20gb",
            ProvisionProfile::Mig3g40gb => "3g.40gb",
            ProvisionProfile::Mig4g40gb => "4g.40gb",
            ProvisionProfile::Mig7g80gb => "7g.80gb",
        }
    }

    /// Base spin-up latency in ms (larger profiles take longer).
    pub fn base_latency_ms(&self) -> f64 {
        match self {
            ProvisionProfile::Mig1g10gb => 120.0,
            ProvisionProfile::Mig2g20gb => 180.0,
            ProvisionProfile::Mig3g40gb => 250.0,
            ProvisionProfile::Mig4g40gb => 320.0,
            ProvisionProfile::Mig7g80gb => 450.0,
        }
    }

    /// Max concurrent partitions for this profile (larger = fewer fit).
    pub fn max_concurrent(&self) -> u32 {
        match self {
            ProvisionProfile::Mig1g10gb => 7,
            ProvisionProfile::Mig2g20gb => 3,
            ProvisionProfile::Mig3g40gb => 2,
            ProvisionProfile::Mig4g40gb => 1,
            ProvisionProfile::Mig7g80gb => 1,
        }
    }
}

/// Parameters for one provisioning overhead trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionTrialParams {
    pub profile: String,
    pub load_state: String,
    pub concurrent_partitions: u32,
    pub base_latency_ms: f64,
}

/// Result of one provisioning overhead trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionTrialResult {
    pub spin_up_latency_ms: f64,
    pub nvidia_smi_reported_ms: f64, // Always ~0
    pub dead_time_fraction: f64,
    pub detected: bool,
}

/// Run one provisioning overhead trial.
pub fn run_provision_trial(
    params: &ProvisionTrialParams,
    noise: &NoiseModel,
    rng: &mut impl Rng,
) -> ProvisionTrialResult {
    // Concurrent partitions add latency (contention on MIG reconfiguration)
    let concurrent_penalty = params.concurrent_partitions as f64 * 15.0;
    let total_base = params.base_latency_ms + concurrent_penalty;

    // Apply log-normal jitter
    let spin_up_latency_ms = noise.jitter_latency(total_base, rng);

    // nvidia-smi always reports ~0ms (instantaneous command return)
    let nvidia_smi_reported = noise.jitter_latency(0.5, rng).min(2.0);

    // Dead time fraction: spin-up / typical inter-provision interval (e.g. 300s)
    let inter_provision_secs = 300.0;
    let dead_time_fraction = (spin_up_latency_ms / 1000.0) / inter_provision_secs;

    ProvisionTrialResult {
        spin_up_latency_ms,
        nvidia_smi_reported_ms: nvidia_smi_reported,
        dead_time_fraction,
        detected: spin_up_latency_ms > 10.0, // gpu-roofline always detects >10ms
    }
}

/// Generate valid provisioning trial cells (profile × load × concurrent).
pub fn provision_valid_cells() -> Vec<(ProvisionProfile, LoadState, u32)> {
    let mut cells = Vec::new();
    for profile in ProvisionProfile::all() {
        let max_c = profile.max_concurrent();
        let concurrent_levels: Vec<u32> = [0, 1, 3, 6]
            .iter()
            .filter(|&&c| c < max_c)
            .copied()
            .collect();
        for load in LoadState::all() {
            for &concurrent in &concurrent_levels {
                cells.push((*profile, *load, concurrent));
            }
        }
    }
    cells
}

// =============================================================================
// Category 4: Burst-to-Sustained Gap
// =============================================================================

/// Workload types for burst-to-sustained trials.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WorkloadType {
    ComputeBound,
    MemoryBound,
    Mixed,
}

impl WorkloadType {
    pub fn all() -> &'static [WorkloadType] {
        &[WorkloadType::ComputeBound, WorkloadType::MemoryBound, WorkloadType::Mixed]
    }

    pub fn name(&self) -> &'static str {
        match self {
            WorkloadType::ComputeBound => "compute_bound",
            WorkloadType::MemoryBound => "memory_bound",
            WorkloadType::Mixed => "mixed",
        }
    }

    /// Workload intensity factor for power model (0.0-1.0).
    pub fn intensity(&self) -> f32 {
        match self {
            WorkloadType::ComputeBound => 1.0,
            WorkloadType::MemoryBound => 0.6,
            WorkloadType::Mixed => 0.8,
        }
    }
}

/// GPU profile names for burst-sustained trials.
pub fn burst_sustained_profiles() -> Vec<(&'static str, SimGpuProfile)> {
    vec![
        ("h100_sxm", profiles::h100_sxm()),
        ("h200_sxm", profiles::h200_sxm()),
        ("rtx_5090", profiles::rtx_5090()),
        ("rtx_4090", profiles::rtx_4090()),
        ("mi300x", profiles::mi300x()),
    ]
}

/// Parameters for one burst-to-sustained trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstSustainedTrialParams {
    pub gpu_profile: String,
    pub workload_type: String,
    pub intensity: f32,
}

/// Result of one burst-to-sustained trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstSustainedTrialResult {
    pub burst_gflops: f64,
    pub sustained_gflops: f64,
    pub burst_bandwidth_gbps: f64,
    pub sustained_bandwidth_gbps: f64,
    pub burst_clock_mhz: u32,
    pub sustained_clock_mhz: u32,
    pub burst_temp_c: f32,
    pub sustained_temp_c: f32,
    pub gap_pct: f64,
    pub equilibrium_time_secs: f64,
    pub detected: bool,
}

/// Run one burst-to-sustained gap trial.
pub fn run_burst_sustained_trial(
    params: &BurstSustainedTrialParams,
    profile: &SimGpuProfile,
    noise: &NoiseModel,
    rng: &mut impl Rng,
) -> BurstSustainedTrialResult {
    let intensity = params.intensity;

    // === Burst phase (t=0, cold GPU) ===
    let burst_temp = noise.jitter_thermal(profile.thermal.ambient_c, rng);
    let burst_throttle = profile.thermal.throttle_factor(burst_temp);
    let (burst_power, burst_clock) = profile.power.compute_state(intensity, burst_throttle);
    let burst_gflops_raw = PowerModel::peak_flops_at_clock(profile.cuda_cores, burst_clock) / 1e9;
    let burst_gflops = noise.jitter_fleet(burst_gflops_raw, rng);
    let burst_bw = noise.jitter_bandwidth(profile.bandwidth.hbm_bandwidth_gbps, rng);

    // === Sustained phase (thermal equilibrium) ===
    let equilibrium_time = profile.thermal.time_to_equilibrium(burst_power);
    let sustained_temp_raw = profile.thermal.temperature_at(burst_power, equilibrium_time);
    let sustained_temp = noise.jitter_thermal(sustained_temp_raw, rng);
    let sustained_throttle = profile.thermal.throttle_factor(sustained_temp);
    let (_sustained_power, sustained_clock) = profile.power.compute_state(intensity, sustained_throttle);
    let sustained_gflops_raw = PowerModel::peak_flops_at_clock(profile.cuda_cores, sustained_clock) / 1e9;
    let sustained_gflops = noise.jitter_fleet(sustained_gflops_raw, rng);
    let sustained_bw = noise.jitter_bandwidth(
        profile.bandwidth.hbm_bandwidth_gbps * profile.bandwidth.sustained_ratio,
        rng,
    );

    let gap_pct = if burst_gflops > 0.0 {
        ((burst_gflops - sustained_gflops) / burst_gflops * 100.0).max(0.0)
    } else {
        0.0
    };

    BurstSustainedTrialResult {
        burst_gflops,
        sustained_gflops,
        burst_bandwidth_gbps: burst_bw,
        sustained_bandwidth_gbps: sustained_bw,
        burst_clock_mhz: burst_clock,
        sustained_clock_mhz: sustained_clock,
        burst_temp_c: burst_temp,
        sustained_temp_c: sustained_temp,
        gap_pct,
        equilibrium_time_secs: equilibrium_time,
        detected: gap_pct > 1.0, // gpu-roofline detects any >1% gap
    }
}

// =============================================================================
// Category 5: Straggler Tax
// =============================================================================

/// Degradation types for straggler trials.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StragglerDegradationType {
    ThermalPaste,
    NvLink,
    PcieFallback,
    MemorySubsystem,
    ClockStuck,
}

impl StragglerDegradationType {
    pub fn all() -> &'static [StragglerDegradationType] {
        &[
            StragglerDegradationType::ThermalPaste,
            StragglerDegradationType::NvLink,
            StragglerDegradationType::PcieFallback,
            StragglerDegradationType::MemorySubsystem,
            StragglerDegradationType::ClockStuck,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            StragglerDegradationType::ThermalPaste => "thermal_paste",
            StragglerDegradationType::NvLink => "nvlink",
            StragglerDegradationType::PcieFallback => "pcie_fallback",
            StragglerDegradationType::MemorySubsystem => "memory_subsystem",
            StragglerDegradationType::ClockStuck => "clock_stuck",
        }
    }

    /// Create a Degradation enum at a given severity level (0=mild, 1=moderate, 2=severe).
    pub fn to_degradation(&self, severity: u8) -> Degradation {
        match self {
            StragglerDegradationType::ThermalPaste => {
                let extra = match severity {
                    0 => 5.0,
                    1 => 10.0,
                    _ => 20.0,
                };
                Degradation::ThermalPasteDried { extra_degrees_c: extra }
            }
            StragglerDegradationType::NvLink => {
                let (active, expected) = match severity {
                    0 => (15, 18),
                    1 => (12, 18),
                    _ => (6, 18),
                };
                Degradation::NvlinkDegraded { active_links: active, expected_links: expected }
            }
            StragglerDegradationType::PcieFallback => {
                let actual = match severity {
                    0 => 4,
                    1 => 3,
                    _ => 2,
                };
                Degradation::PcieFallback { actual_gen: actual, expected_gen: 5 }
            }
            StragglerDegradationType::MemorySubsystem => {
                let ratio = match severity {
                    0 => 0.85,
                    1 => 0.70,
                    _ => 0.50,
                };
                Degradation::MemorySubsystem { bandwidth_ratio: ratio }
            }
            StragglerDegradationType::ClockStuck => {
                let max_mhz = match severity {
                    0 => 1500,
                    1 => 1200,
                    _ => 1095,
                };
                Degradation::ClockStuck { max_mhz }
            }
        }
    }
}

/// Parameters for one straggler tax trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StragglerTrialParams {
    pub fleet_size: u32,
    pub degradation_type: String,
    pub severity: u8,
    pub is_control: bool,
}

/// Result of one straggler tax trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StragglerTrialResult {
    pub fleet_median_gflops: f64,
    pub straggler_gflops: f64,
    pub effective_fleet_throughput: f64,
    pub ideal_fleet_throughput: f64,
    pub straggler_tax_pct: f64,
    pub straggler_detected: bool,
    pub detection_threshold: f64,
}

/// Run one straggler tax trial using fleet simulation.
pub fn run_straggler_trial(
    params: &StragglerTrialParams,
    noise: &NoiseModel,
    detection_threshold: f64,
    rng: &mut impl Rng,
) -> StragglerTrialResult {
    let profile = profiles::h100_sxm();
    let n = params.fleet_size;

    // Compute peak GFLOPS for healthy GPU
    let healthy_gflops = PowerModel::peak_flops_at_clock(
        profile.cuda_cores,
        profile.power.boost_clock_mhz,
    ) / 1e9;

    // Generate per-GPU measurements with fleet jitter
    let mut gpu_gflops: Vec<f64> = (0..n)
        .map(|_| noise.jitter_fleet(healthy_gflops, rng))
        .collect();

    // Apply degradation to GPU 0 if not control
    if !params.is_control {
        let degradation_type = match params.degradation_type.as_str() {
            "thermal_paste" => StragglerDegradationType::ThermalPaste,
            "nvlink" => StragglerDegradationType::NvLink,
            "pcie_fallback" => StragglerDegradationType::PcieFallback,
            "memory_subsystem" => StragglerDegradationType::MemorySubsystem,
            "clock_stuck" => StragglerDegradationType::ClockStuck,
            _ => StragglerDegradationType::MemorySubsystem,
        };

        // Compute degraded GPU performance
        let mut degraded_profile = profile.clone();
        let deg = degradation_type.to_degradation(params.severity);
        match &deg {
            Degradation::ThermalPasteDried { extra_degrees_c } => {
                degraded_profile.thermal.throttle_onset_c -= extra_degrees_c;
                degraded_profile.thermal.throttle_max_c -= extra_degrees_c;
                // Compute sustained performance (throttled)
                let eq_time = degraded_profile.thermal.time_to_equilibrium(
                    degraded_profile.power.tdp_watts,
                );
                let temp = degraded_profile.thermal.temperature_at(
                    degraded_profile.power.tdp_watts,
                    eq_time,
                );
                let throttle = degraded_profile.thermal.throttle_factor(temp);
                let (_, clock) = degraded_profile.power.compute_state(1.0, throttle);
                let degraded_gflops = PowerModel::peak_flops_at_clock(
                    degraded_profile.cuda_cores,
                    clock,
                ) / 1e9;
                gpu_gflops[0] = noise.jitter_fleet(degraded_gflops, rng);
            }
            Degradation::MemorySubsystem { bandwidth_ratio } => {
                // Memory degradation reduces bandwidth-bound performance
                let degraded_gflops = healthy_gflops * bandwidth_ratio;
                gpu_gflops[0] = noise.jitter_fleet(degraded_gflops, rng);
            }
            Degradation::ClockStuck { max_mhz } => {
                let degraded_gflops = PowerModel::peak_flops_at_clock(
                    profile.cuda_cores,
                    *max_mhz,
                ) / 1e9;
                gpu_gflops[0] = noise.jitter_fleet(degraded_gflops, rng);
            }
            Degradation::NvlinkDegraded { active_links, expected_links } => {
                let ratio = *active_links as f64 / *expected_links as f64;
                // NVLink degradation mainly affects distributed training sync
                // Reduce effective throughput by communication overhead
                let degraded = healthy_gflops * (0.5 + 0.5 * ratio);
                gpu_gflops[0] = noise.jitter_fleet(degraded, rng);
            }
            Degradation::PcieFallback { actual_gen, expected_gen } => {
                let gen_diff = (*expected_gen as i32 - *actual_gen as i32).max(0);
                let ratio = 0.5_f64.powi(gen_diff);
                // PCIe fallback reduces data transfer performance
                let degraded = healthy_gflops * (0.7 + 0.3 * ratio);
                gpu_gflops[0] = noise.jitter_fleet(degraded, rng);
            }
        }
    }

    // Fleet metrics
    let mut sorted = gpu_gflops.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_gflops = sorted[n as usize / 2];
    let min_gflops = sorted[0];

    let effective = min_gflops * n as f64;
    let ideal = median_gflops * n as f64;
    let tax_pct = if ideal > 0.0 {
        ((1.0 - effective / ideal) * 100.0).max(0.0)
    } else {
        0.0
    };

    // Detection: straggler if any GPU < threshold * median
    let straggler_detected = gpu_gflops.iter().any(|&g| g < detection_threshold * median_gflops);

    StragglerTrialResult {
        fleet_median_gflops: median_gflops,
        straggler_gflops: min_gflops,
        effective_fleet_throughput: effective,
        ideal_fleet_throughput: ideal,
        straggler_tax_pct: tax_pct,
        straggler_detected,
        detection_threshold,
    }
}

// =============================================================================
// Category 6: Oversubscription
// =============================================================================

/// Parameters for one oversubscription trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OversubTrialParams {
    pub overcommit_ratio: f64,
    pub instance_count: u32,
    pub physical_vram_bytes: u64,
}

/// Result of one oversubscription trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OversubTrialResult {
    pub total_allocated_bytes: u64,
    pub physical_vram_bytes: u64,
    pub actual_overcommit_ratio: f64,
    pub allocation_failure_rate: f64,
    pub performance_degradation_pct: f64,
    pub gpu_roofline_detected: bool,
    pub nvidia_smi_detected: bool,
    pub dcgm_detected: bool,
}

/// Run one oversubscription trial.
pub fn run_oversub_trial(
    params: &OversubTrialParams,
    noise: &NoiseModel,
    rng: &mut impl Rng,
) -> OversubTrialResult {
    let per_instance_alloc = (params.physical_vram_bytes as f64 * params.overcommit_ratio
        / params.instance_count as f64) as u64;
    let total_allocated = per_instance_alloc * params.instance_count as u64;

    let actual_ratio = total_allocated as f64 / params.physical_vram_bytes as f64;

    // Allocation failure probability increases with overcommit
    let failure_rate = if actual_ratio > 1.0 {
        (1.0 - 1.0 / actual_ratio).max(0.0)
    } else {
        0.0
    };

    // Add noise to failure rate
    let noisy_failure_rate = (failure_rate + noise.jitter_bandwidth(0.0, rng).abs() * 0.01)
        .clamp(0.0, 1.0);

    // Performance degradation when overcommitted
    let perf_degradation = if actual_ratio > 1.0 {
        ((1.0 - 1.0 / actual_ratio) * 100.0).max(0.0)
    } else {
        0.0
    };
    let noisy_perf_deg = (perf_degradation + noise.jitter_fleet(0.0, rng).abs()).max(0.0);

    // Detection: gpu-roofline checks sum > physical
    let gpu_roofline_detected = total_allocated > params.physical_vram_bytes;
    // nvidia-smi and DCGM don't aggregate across instances
    let nvidia_smi_detected = false;
    let dcgm_detected = false;

    OversubTrialResult {
        total_allocated_bytes: total_allocated,
        physical_vram_bytes: params.physical_vram_bytes,
        actual_overcommit_ratio: actual_ratio,
        allocation_failure_rate: noisy_failure_rate,
        performance_degradation_pct: noisy_perf_deg,
        gpu_roofline_detected,
        nvidia_smi_detected,
        dcgm_detected,
    }
}
