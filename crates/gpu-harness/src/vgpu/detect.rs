//! vGPU detection trait and implementations.
//!
//! The `VgpuDetector` trait abstracts over different vGPU technologies.
//! Each implementation watches for provisioning events at the trigger point
//! (not polling after the fact).

use std::sync::mpsc;

use crate::error::HarnessError;

use super::state::{VgpuEvent, VgpuInstance, VgpuTechnology};

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

/// NVIDIA MIG detector (NVML MIG APIs + procfs on Linux).
pub struct NvidiaMigDetector;

impl NvidiaMigDetector {
    fn detect_mig() -> bool {
        #[cfg(target_os = "linux")]
        {
            // Check for MIG-capable GPU via procfs
            std::path::Path::new("/proc/driver/nvidia/gpus").exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

impl VgpuDetector for NvidiaMigDetector {
    fn enumerate(&self) -> Result<Vec<VgpuInstance>, HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }
        // TODO: enumerate via NVML MIG APIs
        Ok(Vec::new())
    }

    fn watch(&self, _tx: mpsc::Sender<VgpuEvent>) -> Result<(), HarnessError> {
        if !self.is_available() {
            return Err(HarnessError::VgpuNotSupported);
        }
        // Linux: inotify on /proc/driver/nvidia/gpus/*/mig/ + NVML MIG APIs
        // Windows: NVML MIG APIs delta at 250ms
        Err(HarnessError::VgpuDetectionFailed(
            "MIG detector not yet implemented for this platform".to_string(),
        ))
    }

    fn technology(&self) -> VgpuTechnology {
        VgpuTechnology::NvidiaMig
    }

    fn is_available(&self) -> bool {
        Self::detect_mig()
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
        Box::new(NvidiaMigDetector),
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
