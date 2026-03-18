//! NVML telemetry for real GPU temperature, clock, power, and utilization.
//!
//! Feature-gated behind `nvml`. Provides real-time device state on NVIDIA GPUs
//! via the NVIDIA Management Library (NVML), which ships with the driver.

#[cfg(feature = "nvml")]
mod inner {
    use crate::backend::DeviceState;
    use crate::error::HarnessError;

    use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
    use nvml_wrapper::Nvml;

    /// NVML-backed telemetry provider for NVIDIA GPUs.
    ///
    /// Wraps the NVML library to query real-time temperature, clock speed,
    /// power draw, memory usage, and GPU utilization. One instance covers
    /// all devices — use `query_state(device_index)` per GPU.
    pub struct NvmlTelemetry {
        nvml: Nvml,
    }

    impl NvmlTelemetry {
        /// Initialize NVML. Call once; reuse for all device queries.
        pub fn new() -> Result<Self, HarnessError> {
            let nvml = Nvml::init()
                .map_err(|e| HarnessError::BackendUnavailable(format!("NVML init failed: {e}")))?;
            Ok(Self { nvml })
        }

        /// Query real-time device state for a specific GPU.
        pub fn query_state(&self, device_index: u32) -> Result<DeviceState, HarnessError> {
            let device = self
                .nvml
                .device_by_index(device_index)
                .map_err(|_| HarnessError::DeviceIndexOutOfRange(device_index))?;

            let temperature_c = device.temperature(TemperatureSensor::Gpu).unwrap_or(0);

            let clock_mhz = device.clock_info(Clock::Graphics).unwrap_or(0);

            let power_milliwatts = device.power_usage().unwrap_or(0);
            let power_watts = power_milliwatts as f32 / 1000.0;

            let mem_info = device.memory_info().ok();
            let (memory_used_bytes, memory_total_bytes) = match mem_info {
                Some(info) => (info.used, info.total),
                None => (0, 0),
            };

            let utilization = device.utilization_rates().ok();
            let utilization_pct = utilization.map(|u| u.gpu as f32).unwrap_or(0.0);

            Ok(DeviceState {
                clock_mhz,
                temperature_c,
                power_watts,
                memory_used_bytes,
                memory_total_bytes,
                utilization_pct,
            })
        }

        /// Get the NVML driver version string.
        pub fn driver_version(&self) -> Option<String> {
            self.nvml.sys_driver_version().ok()
        }

        /// Get the number of NVML-visible devices.
        pub fn device_count(&self) -> u32 {
            self.nvml.device_count().unwrap_or(0)
        }

        /// Get PCI bus ID for a device (e.g., "0000:41:00.0").
        pub fn pci_bus_id(&self, device_index: u32) -> Option<String> {
            self.nvml
                .device_by_index(device_index)
                .ok()
                .and_then(|d| d.pci_info().ok())
                .map(|pci| pci.bus_id)
        }
    }
}

#[cfg(feature = "nvml")]
pub use inner::NvmlTelemetry;

#[cfg(not(feature = "nvml"))]
pub struct NvmlTelemetry;

#[cfg(not(feature = "nvml"))]
impl NvmlTelemetry {
    pub fn new() -> Result<Self, crate::error::HarnessError> {
        Err(crate::error::HarnessError::FeatureNotSupported(
            "NVML support not compiled. Build with --features nvml".to_string(),
        ))
    }
}
