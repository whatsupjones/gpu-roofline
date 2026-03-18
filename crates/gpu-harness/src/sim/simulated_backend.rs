use rand::Rng;

use crate::backend::{
    BandwidthResult, DeviceState, GpuBackend, KernelResult, KernelSpec, RunConfig,
};
use crate::device::{GpuDevice, GpuFeatures, GpuLimits};
use crate::error::HarnessError;

use super::fleet::SimulatedFleet;
use super::gpu_model::SimGpuProfile;
use super::power::PowerModel;

/// Simulated GPU backend — physics-based model, no hardware needed.
///
/// Implements `GpuBackend` using thermal, power, and bandwidth models
/// to produce realistic timing data. Supports single-GPU and fleet modes.
pub struct SimulatedBackend {
    /// Single-GPU mode: one profile.
    profile: SimGpuProfile,
    /// Fleet mode: multiple GPUs with topology.
    fleet: Option<SimulatedFleet>,
    /// Simulated elapsed time (advances with each kernel run).
    elapsed_secs: std::sync::atomic::AtomicU64,
    /// Workload intensity for power/thermal model (0.0-1.0).
    workload_intensity: f32,
    /// Active device index for kernel execution (fleet mode).
    active_device: std::sync::atomic::AtomicU32,
}

impl SimulatedBackend {
    /// Create a single-GPU simulated backend.
    pub fn new(profile: SimGpuProfile) -> Self {
        Self {
            profile,
            fleet: None,
            elapsed_secs: std::sync::atomic::AtomicU64::new(0),
            workload_intensity: 1.0,
            active_device: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Create a fleet-mode simulated backend.
    pub fn with_fleet(fleet: SimulatedFleet) -> Self {
        let profile = fleet.gpus[0].profile.clone();
        Self {
            profile,
            fleet: Some(fleet),
            elapsed_secs: std::sync::atomic::AtomicU64::new(0),
            workload_intensity: 1.0,
            active_device: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Set the active device for kernel execution (fleet mode).
    /// Subsequent `run_kernel()` calls will use this device's profile.
    pub fn set_active_device(&self, index: u32) {
        self.active_device
            .store(index, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the current active device index.
    pub fn active_device_index(&self) -> u32 {
        self.active_device
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Set workload intensity (0.0 = idle, 1.0 = full load).
    pub fn set_intensity(&mut self, intensity: f32) {
        self.workload_intensity = intensity.clamp(0.0, 1.0);
    }

    /// Get simulated elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        f64::from_bits(self.elapsed_secs.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Advance simulated time.
    pub fn advance_time(&self, secs: f64) {
        let current = self.elapsed_secs();
        let new = current + secs;
        self.elapsed_secs
            .store(new.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Reset simulated time to zero.
    pub fn reset_time(&self) {
        self.elapsed_secs
            .store(0.0_f64.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Get profile for a specific device index (fleet mode).
    fn profile_for(&self, device_index: u32) -> Result<&SimGpuProfile, HarnessError> {
        if let Some(fleet) = &self.fleet {
            fleet
                .gpus
                .get(device_index as usize)
                .map(|g| &g.profile)
                .ok_or(HarnessError::DeviceIndexOutOfRange(device_index))
        } else if device_index == 0 {
            Ok(&self.profile)
        } else {
            Err(HarnessError::DeviceIndexOutOfRange(device_index))
        }
    }

    /// Simulate kernel execution timing for a given profile.
    fn simulate_kernel_timing(
        &self,
        profile: &SimGpuProfile,
        kernel: &KernelSpec,
        config: &RunConfig,
        jitter: f64,
    ) -> KernelResult {
        let mut rng = rand::thread_rng();
        let elapsed = self.elapsed_secs();

        // Determine thermal state
        let temp = profile
            .thermal
            .temperature_at(profile.power.tdp_watts, elapsed);
        let throttle = profile.thermal.throttle_factor(temp);

        // Determine power state (clock speed)
        let (_, clock_mhz) = profile
            .power
            .compute_state(self.workload_intensity, throttle);

        // Determine effective bandwidth for this working set
        let (theoretical_bw_gbps, _level) = profile
            .bandwidth
            .effective_bandwidth(kernel.working_set_bytes);

        // Apply kernel efficiency: real kernels achieve less than theoretical max
        let achievable_bw_gbps = theoretical_bw_gbps * profile.bandwidth.kernel_efficiency;

        // Calculate expected timing
        let bytes = config.buffer_size_bytes as f64;
        let elements = bytes / 16.0; // float4/vec4 = 16 bytes per element
                                     // AI = FLOP / bytes_transferred. bytes_transferred = read + write = 32 per element
        let flops_per_element = kernel.arithmetic_intensity * 32.0;
        let total_flops = (elements * flops_per_element) as u64;

        // Time is max(compute_time, memory_time) — roofline model
        let peak_flops = PowerModel::peak_flops_at_clock(profile.cuda_cores, clock_mhz);
        let compute_time_us = if peak_flops > 0.0 {
            (total_flops as f64 / peak_flops) * 1e6
        } else {
            f64::MAX
        };

        // Memory time: total traffic (read + write) / achievable bandwidth
        let total_traffic_bytes = bytes * 2.0; // read + write for copy-like kernels
        let memory_time_us = if achievable_bw_gbps > 0.0 {
            (total_traffic_bytes / (achievable_bw_gbps * 1e9)) * 1e6
        } else {
            f64::MAX
        };

        let base_time_us = compute_time_us.max(memory_time_us);

        // Generate timing samples with jitter
        let elapsed_us: Vec<f64> = (0..config.measurement_iterations)
            .map(|_| {
                let jitter_factor = 1.0 + rng.gen_range(-jitter..jitter);
                (base_time_us * jitter_factor).max(0.1) // Minimum 0.1 us
            })
            .collect();

        // Advance simulated time by total measurement duration
        let total_measurement_secs = (base_time_us * config.measurement_iterations as f64) / 1e6;
        self.advance_time(total_measurement_secs);

        KernelResult {
            kernel_name: kernel.name.clone(),
            elapsed_us,
            bytes_processed: config.buffer_size_bytes,
            flops_executed: total_flops,
        }
    }
}

impl GpuBackend for SimulatedBackend {
    fn run_kernel(
        &self,
        kernel: &KernelSpec,
        config: &RunConfig,
    ) -> Result<KernelResult, HarnessError> {
        let device_idx = self.active_device_index();
        let (profile, jitter) = if let Some(fleet) = &self.fleet {
            let gpu = fleet
                .gpus
                .get(device_idx as usize)
                .ok_or(HarnessError::DeviceIndexOutOfRange(device_idx))?;
            (&gpu.profile, gpu.jitter)
        } else {
            (&self.profile, 0.02)
        };

        Ok(self.simulate_kernel_timing(profile, kernel, config, jitter))
    }

    fn device_state(&self, device_index: u32) -> Result<DeviceState, HarnessError> {
        let profile = self.profile_for(device_index)?;
        let elapsed = self.elapsed_secs();

        let temp = profile
            .thermal
            .temperature_at(profile.power.tdp_watts * self.workload_intensity, elapsed);
        let throttle = profile.thermal.throttle_factor(temp);
        let (power, clock) = profile
            .power
            .compute_state(self.workload_intensity, throttle);

        Ok(DeviceState {
            clock_mhz: clock,
            temperature_c: temp as u32,
            power_watts: power,
            memory_used_bytes: 0,
            memory_total_bytes: profile.vram_bytes,
            utilization_pct: self.workload_intensity * 100.0,
        })
    }

    fn discover_devices(&self) -> Result<Vec<GpuDevice>, HarnessError> {
        let profiles: Vec<&SimGpuProfile> = if let Some(fleet) = &self.fleet {
            fleet.gpus.iter().map(|g| &g.profile).collect()
        } else {
            vec![&self.profile]
        };

        Ok(profiles
            .iter()
            .enumerate()
            .map(|(i, p)| GpuDevice {
                index: i as u32,
                name: p.name.clone(),
                vendor: p.vendor,
                architecture: p.architecture,
                memory_bytes: p.vram_bytes,
                pci_bus_id: Some(format!("0000:{:02x}:00.0", i)),
                driver_version: Some("sim-1.0".to_string()),
                features: GpuFeatures {
                    timestamp_queries: true,
                    shader_f16: true,
                    shader_int64: true,
                    compute_capability: match p.architecture {
                        crate::device::GpuArchitecture::Blackwell => Some((10, 0)),
                        crate::device::GpuArchitecture::Hopper => Some((9, 0)),
                        crate::device::GpuArchitecture::Ada => Some((8, 9)),
                        crate::device::GpuArchitecture::Ampere => Some((8, 0)),
                        _ => None,
                    },
                    tensor_cores: matches!(
                        p.architecture,
                        crate::device::GpuArchitecture::Blackwell
                            | crate::device::GpuArchitecture::Hopper
                            | crate::device::GpuArchitecture::Ada
                            | crate::device::GpuArchitecture::Ampere
                    ),
                    rt_cores: matches!(
                        p.architecture,
                        crate::device::GpuArchitecture::Blackwell
                            | crate::device::GpuArchitecture::Ada
                            | crate::device::GpuArchitecture::Ampere
                    ),
                },
                limits: GpuLimits::default(),
            })
            .collect())
    }

    fn p2p_bandwidth(&self, src: u32, dst: u32) -> Result<BandwidthResult, HarnessError> {
        if let Some(fleet) = &self.fleet {
            let n = fleet.topology.gpu_count;
            if src >= n || dst >= n {
                return Err(HarnessError::DeviceIndexOutOfRange(src.max(dst)));
            }

            let bw = fleet.topology.p2p_bandwidth_gbps[src as usize][dst as usize];

            // Add ~2% jitter to P2P measurement
            let mut rng = rand::thread_rng();
            let jitter = 1.0 + rng.gen_range(-0.02..0.02);

            Ok(BandwidthResult {
                src_device: src,
                dst_device: dst,
                bandwidth_gbps: bw * jitter,
                latency_us: if bw > 100.0 { 1.5 } else { 8.0 }, // NVLink vs PCIe latency
            })
        } else {
            Err(HarnessError::FeatureNotSupported(
                "P2P requires fleet mode".to_string(),
            ))
        }
    }

    fn set_active_device(&self, index: u32) {
        self.active_device
            .store(index, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::profiles;

    fn copy_kernel() -> KernelSpec {
        KernelSpec {
            name: "copy".to_string(),
            working_set_bytes: 16 * 1024 * 1024,
            arithmetic_intensity: 0.0, // Pure memory bandwidth
            iterations: 1,
        }
    }

    fn fma_heavy_kernel() -> KernelSpec {
        KernelSpec {
            name: "fma_heavy".to_string(),
            working_set_bytes: 16 * 1024 * 1024,
            arithmetic_intensity: 256.0, // Compute-bound
            iterations: 1,
        }
    }

    #[test]
    fn test_single_gpu_discover() {
        let backend = SimulatedBackend::new(profiles::rtx_5090());
        let devices = backend.discover_devices().unwrap();
        assert_eq!(devices.len(), 1);
        assert!(devices[0].name.contains("RTX 5090"));
        assert_eq!(devices[0].features.compute_capability, Some((10, 0)));
        assert!(devices[0].features.tensor_cores);
    }

    #[test]
    fn test_fleet_discover() {
        let fleet = crate::sim::SimulatedFleet::homogeneous(profiles::h100_sxm(), 8);
        let backend = SimulatedBackend::with_fleet(fleet);
        let devices = backend.discover_devices().unwrap();
        assert_eq!(devices.len(), 8);
        for (i, d) in devices.iter().enumerate() {
            assert!(d.name.contains("H100"));
            assert_eq!(d.index, i as u32);
        }
    }

    #[test]
    fn test_kernel_produces_results() {
        let backend = SimulatedBackend::new(profiles::rtx_5090());
        let config = RunConfig {
            warmup_iterations: 0,
            measurement_iterations: 50,
            buffer_size_bytes: 16 * 1024 * 1024,
        };
        let result = backend.run_kernel(&copy_kernel(), &config).unwrap();

        assert_eq!(result.kernel_name, "copy");
        assert_eq!(result.elapsed_us.len(), 50);
        assert!(result.bandwidth_gbps() > 0.0);
        assert!(result.cv() < 0.1, "CV should be small: {}", result.cv());
    }

    #[test]
    fn test_memory_bound_vs_compute_bound() {
        let backend = SimulatedBackend::new(profiles::h100_sxm());
        let config = RunConfig::default();

        let copy_result = backend.run_kernel(&copy_kernel(), &config).unwrap();
        let fma_result = backend.run_kernel(&fma_heavy_kernel(), &config).unwrap();

        // Copy should be memory-bound (high bandwidth, low FLOPS)
        assert!(
            copy_result.bandwidth_gbps() > 100.0,
            "copy bandwidth should be significant: {} GB/s",
            copy_result.bandwidth_gbps()
        );

        // FMA heavy should report FLOPS
        assert!(fma_result.flops_executed > 0);
    }

    #[test]
    fn test_device_state_thermal_ramp() {
        let backend = SimulatedBackend::new(profiles::rtx_5090());

        // At t=0, should be near ambient
        let state_0 = backend.device_state(0).unwrap();
        assert!(
            state_0.temperature_c < 50,
            "should be cool at start: {}C",
            state_0.temperature_c
        );

        // Advance time significantly
        backend.advance_time(60.0);

        let state_60 = backend.device_state(0).unwrap();
        assert!(
            state_60.temperature_c > state_0.temperature_c,
            "should be hotter after 60s: {}C vs {}C",
            state_60.temperature_c,
            state_0.temperature_c
        );
    }

    #[test]
    fn test_device_state_clock_throttle() {
        let backend = SimulatedBackend::new(profiles::rtx_5090());

        let state_cool = backend.device_state(0).unwrap();
        backend.advance_time(120.0);
        let state_hot = backend.device_state(0).unwrap();

        // Hot GPU should have lower clock (throttled)
        assert!(
            state_hot.clock_mhz <= state_cool.clock_mhz,
            "hot clock {} should be <= cool clock {}",
            state_hot.clock_mhz,
            state_cool.clock_mhz
        );
    }

    #[test]
    fn test_fleet_p2p_bandwidth() {
        let fleet = crate::sim::SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        let backend = SimulatedBackend::with_fleet(fleet);

        let bw = backend.p2p_bandwidth(0, 1).unwrap();
        assert!(
            bw.bandwidth_gbps > 800.0,
            "H100 NVLink should be >800 GB/s: {}",
            bw.bandwidth_gbps
        );
        assert!(bw.latency_us < 5.0, "NVLink latency should be low");
    }

    #[test]
    fn test_fleet_p2p_out_of_range() {
        let fleet = crate::sim::SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        let backend = SimulatedBackend::with_fleet(fleet);

        let result = backend.p2p_bandwidth(0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_gpu_p2p_not_supported() {
        let backend = SimulatedBackend::new(profiles::rtx_5090());
        let result = backend.p2p_bandwidth(0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_degraded_gpu_lower_bandwidth() {
        let mut fleet = crate::sim::SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);

        // Measure healthy P2P
        let backend_healthy = SimulatedBackend::with_fleet(fleet.clone());
        let bw_healthy = backend_healthy.p2p_bandwidth(2, 0).unwrap().bandwidth_gbps;

        // Degrade GPU 2's NVLink
        fleet.degrade_gpu(
            2,
            crate::sim::Degradation::NvlinkDegraded {
                active_links: 12,
                expected_links: 18,
            },
        );

        let backend_degraded = SimulatedBackend::with_fleet(fleet);
        let bw_degraded = backend_degraded.p2p_bandwidth(2, 0).unwrap().bandwidth_gbps;

        assert!(
            bw_degraded < bw_healthy * 0.75,
            "degraded BW {bw_degraded:.0} should be <75% of healthy {bw_healthy:.0}"
        );
    }
}
