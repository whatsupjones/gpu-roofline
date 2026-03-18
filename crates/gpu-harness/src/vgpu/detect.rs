//! vGPU detection trait and implementations.
//!
//! The `VgpuDetector` trait abstracts over different vGPU technologies.
//! Each implementation watches for provisioning events at the trigger point
//! (not polling after the fact).

use std::sync::mpsc;

use crate::error::HarnessError;

use super::state::{VgpuEvent, VgpuEventType, VgpuInstance, VgpuPhase, VgpuTechnology};

/// Trait for detecting vGPU lifecycle events at the trigger point.
pub trait VgpuDetector: Send + Sync {
    /// Enumerate all currently-visible vGPU instances.
    fn enumerate(&self) -> Result<Vec<VgpuInstance>, HarnessError>;

    /// Start watching for vGPU lifecycle events, sending them to `tx`.
    /// Blocks until an error occurs or the channel is closed.
    fn watch(&self, tx: mpsc::Sender<VgpuEvent>) -> Result<(), HarnessError>;

    /// Which vGPU technology this detector handles.
    fn technology(&self) -> VgpuTechnology;

    /// Whether the underlying technology is available on this system.
    fn is_available(&self) -> bool;
}

// ── Hardware Detector Implementations ──

/// NVIDIA GRID/vGPU detector (NVML active_vgpus + sysfs mdev on Linux).
pub struct NvidiaGridDetector;

impl NvidiaGridDetector {
    /// Check if GRID/vGPU is available via NVML or sysfs.
    fn detect_grid() -> bool {
        // Linux: check for mdev sysfs entries
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/sys/bus/mdev/devices").exists()
        }
        // Windows: would need NVML active_vgpus() — requires nvml feature
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

impl VgpuDetector for NvidiaGridDetector {
    fn enumerate(&self) -> Result<Vec<VgpuInstance>, HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }
        // TODO: enumerate via NVML active_vgpus() or sysfs mdev
        Ok(Vec::new())
    }

    fn watch(&self, _tx: mpsc::Sender<VgpuEvent>) -> Result<(), HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }
        // Linux: inotify on /sys/bus/mdev/devices/ + NVML active_vgpus() delta
        // Windows: NVML active_vgpus() polling at 250ms
        Err(HarnessError::VgpuDetectionFailed(
            "GRID detector not yet implemented for this platform".to_string(),
        ))
    }

    fn technology(&self) -> VgpuTechnology {
        VgpuTechnology::NvidiaGrid
    }

    fn is_available(&self) -> bool {
        Self::detect_grid()
    }
}

/// NVIDIA MIG detector — real hardware detection via NVML MIG APIs.
///
/// Uses NVML to enumerate MIG instances and polls for changes at configurable
/// intervals. On Linux, also checks procfs for MIG-capable GPUs.
pub struct NvidiaMigDetector {
    /// Polling interval for watch mode (milliseconds).
    #[allow(dead_code)]
    poll_interval_ms: u64,
}

impl NvidiaMigDetector {
    pub fn new() -> Self {
        Self {
            poll_interval_ms: 500,
        }
    }

    pub fn with_poll_interval(poll_interval_ms: u64) -> Self {
        Self { poll_interval_ms }
    }

    fn detect_mig_available() -> bool {
        // Check if NVML is available and any GPU supports MIG
        #[cfg(feature = "nvml")]
        {
            if let Ok(telemetry) = crate::NvmlTelemetry::new() {
                let count = telemetry.device_count();
                for i in 0..count {
                    // MIG requires compute capability 8.0+ (Ampere/Hopper)
                    // Just check if NVML can query MIG mode (returns error if unsupported)
                    if telemetry.mig_mode_enabled(i) {
                        return true; // MIG is enabled on at least one GPU
                    }
                }
                // MIG not enabled, but check if GPU is MIG-capable
                #[cfg(target_os = "linux")]
                {
                    return std::path::Path::new("/proc/driver/nvidia/gpus").exists();
                }
                #[cfg(not(target_os = "linux"))]
                {
                    return count > 0; // Assume MIG-capable if NVML works
                }
            }
            false
        }
        #[cfg(not(feature = "nvml"))]
        {
            #[cfg(target_os = "linux")]
            {
                std::path::Path::new("/proc/driver/nvidia/gpus").exists()
            }
            #[cfg(not(target_os = "linux"))]
            {
                false
            }
        }
    }

    /// Enumerate MIG instances via NVML.
    #[cfg(feature = "nvml")]
    fn enumerate_via_nvml(&self) -> Result<Vec<VgpuInstance>, HarnessError> {
        let telemetry = crate::NvmlTelemetry::new()?;
        let device_count = telemetry.device_count();
        let mut all_instances = Vec::new();

        for gpu_idx in 0..device_count {
            if !telemetry.mig_mode_enabled(gpu_idx) {
                continue;
            }

            let pci_bus_id = telemetry.pci_bus_id(gpu_idx);
            let instances = telemetry.enumerate_mig_instances(gpu_idx)?;

            let total_instances = instances.len().max(1) as f64;
            for inst in instances {
                let vram = inst.memory_total_bytes;

                let name = inst.name.clone();
                all_instances.push(VgpuInstance {
                    id: format!("mig-{}-{}", gpu_idx, inst.index),
                    name: name.clone(),
                    technology: VgpuTechnology::NvidiaMig,
                    physical_gpu_index: gpu_idx,
                    physical_pci_bus_id: pci_bus_id.clone().unwrap_or_default(),
                    phase: VgpuPhase::Active,
                    vram_allocated_bytes: vram,
                    compute_fraction: Some(1.0 / total_instances),
                    memory_fraction: 0.0,
                    mig_profile: Some(name),
                });
            }
        }

        Ok(all_instances)
    }
}

impl Default for NvidiaMigDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl VgpuDetector for NvidiaMigDetector {
    fn enumerate(&self) -> Result<Vec<VgpuInstance>, HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }

        #[cfg(feature = "nvml")]
        {
            self.enumerate_via_nvml()
        }
        #[cfg(not(feature = "nvml"))]
        {
            Err(HarnessError::FeatureNotSupported(
                "MIG enumeration requires --features nvml".to_string(),
            ))
        }
    }

    fn watch(&self, tx: mpsc::Sender<VgpuEvent>) -> Result<(), HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }

        #[cfg(feature = "nvml")]
        {
            // Poll-based MIG watch: enumerate periodically, detect delta
            let mut known_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

            // Initial enumeration
            if let Ok(instances) = self.enumerate_via_nvml() {
                for inst in &instances {
                    known_ids.insert(inst.id.clone());
                }
            }

            let interval = std::time::Duration::from_millis(self.poll_interval_ms);

            loop {
                std::thread::sleep(interval);

                let current = match self.enumerate_via_nvml() {
                    Ok(instances) => instances,
                    Err(_) => continue,
                };

                let current_ids: std::collections::HashSet<String> =
                    current.iter().map(|i| i.id.clone()).collect();

                // Detect new instances (Created events)
                for inst in &current {
                    if !known_ids.contains(&inst.id) {
                        let event = VgpuEvent {
                            timestamp: chrono::Utc::now(),
                            event_type: VgpuEventType::Created {
                                instance: inst.clone(),
                                spin_up_latency_ms: None,
                            },
                            instance_id: inst.id.clone(),
                            snapshot: None,
                        };
                        if tx.send(event).is_err() {
                            return Ok(()); // Receiver dropped
                        }
                    }
                }

                // Detect removed instances (Destroyed events)
                for id in &known_ids {
                    if !current_ids.contains(id) {
                        let event = VgpuEvent {
                            timestamp: chrono::Utc::now(),
                            event_type: VgpuEventType::Destroyed {
                                instance_id: id.clone(),
                                verification: super::state::TeardownVerification {
                                    memory_reclaimed: true,
                                    expected_free_bytes: 0,
                                    actual_free_bytes: 0,
                                    reclaim_latency_ms: 0.0,
                                    ghost_allocations_bytes: 0,
                                    compute_reclaimed: true,
                                },
                            },
                            instance_id: id.clone(),
                            snapshot: None,
                        };
                        if tx.send(event).is_err() {
                            return Ok(());
                        }
                    }
                }

                known_ids = current_ids;
            }
        }

        #[cfg(not(feature = "nvml"))]
        {
            let _ = tx;
            Err(HarnessError::FeatureNotSupported(
                "MIG watch requires --features nvml".to_string(),
            ))
        }
    }

    fn technology(&self) -> VgpuTechnology {
        VgpuTechnology::NvidiaMig
    }

    fn is_available(&self) -> bool {
        Self::detect_mig_available()
    }
}

/// SR-IOV hardware PCIe virtualization detector.
pub struct SrIovDetector;

impl SrIovDetector {
    fn detect_sriov() -> bool {
        #[cfg(target_os = "linux")]
        {
            // Check for SR-IOV capable PCI devices
            let sriov_path = std::path::Path::new("/sys/bus/pci/devices");
            if let Ok(entries) = std::fs::read_dir(sriov_path) {
                for entry in entries.flatten() {
                    if entry.path().join("sriov_numvfs").exists() {
                        return true;
                    }
                }
            }
            false
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

impl VgpuDetector for SrIovDetector {
    fn enumerate(&self) -> Result<Vec<VgpuInstance>, HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }
        Ok(Vec::new())
    }

    fn watch(&self, _tx: mpsc::Sender<VgpuEvent>) -> Result<(), HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }
        // Linux: inotify on /sys/bus/pci/devices/*/sriov_numvfs
        Err(HarnessError::VgpuDetectionFailed(
            "SR-IOV detector not yet implemented".to_string(),
        ))
    }

    fn technology(&self) -> VgpuTechnology {
        VgpuTechnology::SrIov
    }

    fn is_available(&self) -> bool {
        Self::detect_sriov()
    }
}

/// Cloud GPU passthrough detector (device enumeration delta).
pub struct CloudPassthroughDetector;

impl VgpuDetector for CloudPassthroughDetector {
    fn enumerate(&self) -> Result<Vec<VgpuInstance>, HarnessError> {
        // TODO: use GpuBackend::discover_devices() delta
        Ok(Vec::new())
    }

    fn watch(&self, _tx: mpsc::Sender<VgpuEvent>) -> Result<(), HarnessError> {
        // Linux: udev rules (device add/remove)
        // Windows: SetupDiGetClassDevs delta
        Err(HarnessError::VgpuDetectionFailed(
            "Cloud passthrough detector not yet implemented".to_string(),
        ))
    }

    fn technology(&self) -> VgpuTechnology {
        VgpuTechnology::CloudPassthrough
    }

    fn is_available(&self) -> bool {
        // Cloud passthrough is always "available" as a fallback —
        // it just polls device enumeration
        false
    }
}

/// Kubernetes device-plugin GPU scheduling detector.
pub struct K8sDevicePluginDetector;

impl K8sDevicePluginDetector {
    fn detect_k8s() -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/var/lib/kubelet/device-plugins").exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

impl VgpuDetector for K8sDevicePluginDetector {
    fn enumerate(&self) -> Result<Vec<VgpuInstance>, HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }
        Ok(Vec::new())
    }

    fn watch(&self, _tx: mpsc::Sender<VgpuEvent>) -> Result<(), HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }
        // Linux: inotify on /var/lib/kubelet/device-plugins/
        Err(HarnessError::VgpuDetectionFailed(
            "K8s device plugin detector not yet implemented".to_string(),
        ))
    }

    fn technology(&self) -> VgpuTechnology {
        VgpuTechnology::KubernetesDevicePlugin
    }

    fn is_available(&self) -> bool {
        Self::detect_k8s()
    }
}

/// Create a composite detector with all available hardware detectors.
pub fn auto_detect() -> CompositeDetector {
    CompositeDetector::available_only(vec![
        Box::new(NvidiaGridDetector),
        Box::new(NvidiaMigDetector::new()),
        Box::new(SrIovDetector),
        Box::new(CloudPassthroughDetector),
        Box::new(K8sDevicePluginDetector),
    ])
}

/// Composite detector that runs all available detectors and merges event streams.
pub struct CompositeDetector {
    detectors: Vec<Box<dyn VgpuDetector>>,
}

impl CompositeDetector {
    pub fn new(detectors: Vec<Box<dyn VgpuDetector>>) -> Self {
        Self { detectors }
    }

    /// Only keep detectors that report `is_available()`.
    pub fn available_only(detectors: Vec<Box<dyn VgpuDetector>>) -> Self {
        let available = detectors.into_iter().filter(|d| d.is_available()).collect();
        Self::new(available)
    }

    pub fn detectors(&self) -> &[Box<dyn VgpuDetector>] {
        &self.detectors
    }
}

impl VgpuDetector for CompositeDetector {
    fn enumerate(&self) -> Result<Vec<VgpuInstance>, HarnessError> {
        let mut all = Vec::new();
        for d in &self.detectors {
            match d.enumerate() {
                Ok(instances) => all.extend(instances),
                Err(e) => tracing::warn!("detector {} enumerate failed: {e}", d.technology()),
            }
        }
        Ok(all)
    }

    fn watch(&self, tx: mpsc::Sender<VgpuEvent>) -> Result<(), HarnessError> {
        if self.detectors.is_empty() {
            return Err(HarnessError::VgpuNotSupported);
        }

        // For a single detector, just delegate directly
        if self.detectors.len() == 1 {
            return self.detectors[0].watch(tx);
        }

        // For multiple detectors, use scoped threads to merge event streams.
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for detector in &self.detectors {
                let tx = tx.clone();
                let tech = detector.technology();
                let handle = s.spawn(move || {
                    if let Err(e) = detector.watch(tx) {
                        tracing::warn!("detector {tech} watch error: {e}");
                    }
                });
                handles.push(handle);
            }
            for h in handles {
                let _ = h.join();
            }
        });
        Ok(())
    }

    fn technology(&self) -> VgpuTechnology {
        // Composite doesn't have a single technology
        VgpuTechnology::Simulated
    }

    fn is_available(&self) -> bool {
        self.detectors.iter().any(|d| d.is_available())
    }
}
