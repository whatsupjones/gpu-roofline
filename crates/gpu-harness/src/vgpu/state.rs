//! Core data types for vGPU lifecycle monitoring.
//!
//! Pure structs and enums — no business logic, no external dependencies
//! beyond serde/chrono for serialization.

use serde::{Deserialize, Serialize};

/// Virtualization technology backing a vGPU instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VgpuTechnology {
    /// NVIDIA GRID/vGPU (VMware, Citrix, KVM).
    NvidiaGrid,
    /// Multi-Instance GPU (H100/H200 hardware partitioning).
    NvidiaMig,
    /// SR-IOV hardware PCIe virtualization.
    SrIov,
    /// Cloud GPU passthrough (AWS p4d/p5, GCP a2/a3, Azure ND).
    CloudPassthrough,
    /// Kubernetes device-plugin GPU scheduling.
    KubernetesDevicePlugin,
    /// Simulated vGPU for testing.
    Simulated,
}

impl std::fmt::Display for VgpuTechnology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NvidiaGrid => write!(f, "NVIDIA GRID"),
            Self::NvidiaMig => write!(f, "NVIDIA MIG"),
            Self::SrIov => write!(f, "SR-IOV"),
            Self::CloudPassthrough => write!(f, "Cloud Passthrough"),
            Self::KubernetesDevicePlugin => write!(f, "K8s Device Plugin"),
            Self::Simulated => write!(f, "Simulated"),
        }
    }
}

/// Lifecycle phase of a vGPU instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VgpuPhase {
    Provisioning,
    Active,
    Teardown,
    Destroyed,
}

impl std::fmt::Display for VgpuPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provisioning => write!(f, "Provisioning"),
            Self::Active => write!(f, "Active"),
            Self::Teardown => write!(f, "Teardown"),
            Self::Destroyed => write!(f, "Destroyed"),
        }
    }
}

/// How the physical GPU is partitioned among vGPU instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitioningMode {
    /// MIG, SR-IOV — isolated resources, no contention.
    HardwarePartitioned,
    /// GRID time-slicing — shared resources, contention possible.
    TimeSliced,
    /// Entire GPU dedicated to one VM.
    Passthrough,
    /// Unknown partitioning mode.
    Unknown,
}

/// A single vGPU instance on a physical GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VgpuInstance {
    /// UUID from NVML or sysfs path.
    pub id: String,
    /// Human-readable name (e.g. "GRID V100D-8Q", "MIG 1g.10gb").
    pub name: String,
    /// Virtualization technology backing this instance.
    pub technology: VgpuTechnology,
    /// Index of the physical GPU hosting this instance.
    pub physical_gpu_index: u32,
    /// PCI bus ID of the physical GPU (if known).
    pub physical_pci_bus_id: Option<String>,
    /// Current lifecycle phase.
    pub phase: VgpuPhase,
    /// VRAM allocated to this instance in bytes.
    pub vram_allocated_bytes: u64,
    /// Fraction of physical GPU compute assigned (0.0–1.0), if known.
    pub compute_fraction: Option<f64>,
    /// Fraction of physical GPU memory assigned (0.0–1.0).
    pub memory_fraction: f64,
    /// MIG profile string (e.g. "1g.10gb", "3g.40gb").
    pub mig_profile: Option<String>,
}

/// Performance snapshot of a vGPU at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VgpuSnapshot {
    pub bandwidth_gbps: f64,
    pub gflops: f64,
    pub memory_used_bytes: u64,
    pub memory_allocated_bytes: u64,
    pub utilization_pct: f32,
    pub temperature_c: u32,
    pub power_watts: f32,
    pub encoder_utilization_pct: Option<f32>,
    pub decoder_utilization_pct: Option<f32>,
}

/// Result of verifying resource reclamation after vGPU teardown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeardownVerification {
    /// Whether memory was fully reclaimed.
    pub memory_reclaimed: bool,
    /// Expected bytes freed (= vGPU's vram_allocated_bytes).
    pub expected_free_bytes: u64,
    /// Actual bytes freed (delta in physical GPU memory_used).
    pub actual_free_bytes: u64,
    /// Time from teardown signal to resources freed.
    pub reclaim_latency_ms: f64,
    /// Memory not freed despite vGPU being gone.
    pub ghost_allocations_bytes: u64,
    /// Whether compute resources were reclaimed.
    pub compute_reclaimed: bool,
}

/// A lifecycle event for a vGPU instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VgpuEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: VgpuEventType,
    pub instance_id: String,
    pub snapshot: Option<VgpuSnapshot>,
}

/// The type of vGPU lifecycle event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VgpuEventType {
    /// A new vGPU was created.
    Created {
        instance: VgpuInstance,
        spin_up_latency_ms: Option<f64>,
    },
    /// vGPU transitioned to Active phase.
    Active { instance_id: String },
    /// Periodic performance sample.
    Sampled { instance_id: String },
    /// Existing tenants experienced performance degradation.
    ContentionDetected {
        affected_instances: Vec<String>,
        bandwidth_impact: Vec<f64>,
        compute_impact: Vec<f64>,
        caused_by: String,
    },
    /// Teardown initiated for a vGPU.
    TeardownStarted { instance_id: String },
    /// vGPU has been destroyed and resources verified.
    Destroyed {
        instance_id: String,
        verification: TeardownVerification,
    },
}

/// Aggregate state of all vGPU instances on a single physical GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VgpuState {
    pub physical_gpu_index: u32,
    pub instances: Vec<VgpuInstance>,
    pub total_vram_allocated_bytes: u64,
    pub total_vram_available_bytes: u64,
    pub active_count: u32,
    pub technology: VgpuTechnology,
    pub partitioning_mode: PartitioningMode,
    pub recent_events: Vec<VgpuEvent>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vgpu_technology_display() {
        assert_eq!(VgpuTechnology::NvidiaGrid.to_string(), "NVIDIA GRID");
        assert_eq!(VgpuTechnology::NvidiaMig.to_string(), "NVIDIA MIG");
        assert_eq!(VgpuTechnology::Simulated.to_string(), "Simulated");
    }

    #[test]
    fn test_vgpu_phase_display() {
        assert_eq!(VgpuPhase::Provisioning.to_string(), "Provisioning");
        assert_eq!(VgpuPhase::Active.to_string(), "Active");
        assert_eq!(VgpuPhase::Destroyed.to_string(), "Destroyed");
    }

    #[test]
    fn test_vgpu_instance_serialization() {
        let instance = VgpuInstance {
            id: "test-001".to_string(),
            name: "MIG 1g.10gb".to_string(),
            technology: VgpuTechnology::NvidiaMig,
            physical_gpu_index: 0,
            physical_pci_bus_id: Some("0000:3b:00.0".to_string()),
            phase: VgpuPhase::Active,
            vram_allocated_bytes: 10 * 1024 * 1024 * 1024,
            compute_fraction: Some(1.0 / 7.0),
            memory_fraction: 0.125,
            mig_profile: Some("1g.10gb".to_string()),
        };

        let json = serde_json::to_string(&instance).unwrap();
        let deserialized: VgpuInstance = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test-001");
        assert_eq!(deserialized.technology, VgpuTechnology::NvidiaMig);
    }

    #[test]
    fn test_teardown_verification_ghost_detection() {
        let verification = TeardownVerification {
            memory_reclaimed: false,
            expected_free_bytes: 10 * 1024 * 1024 * 1024,
            actual_free_bytes: 9_500_000_000,
            reclaim_latency_ms: 150.0,
            ghost_allocations_bytes: 500_000_000,
            compute_reclaimed: true,
        };
        assert!(!verification.memory_reclaimed);
        assert!(verification.ghost_allocations_bytes > 0);
    }

    #[test]
    fn test_vgpu_event_serialization() {
        let event = VgpuEvent {
            timestamp: chrono::Utc::now(),
            event_type: VgpuEventType::TeardownStarted {
                instance_id: "test-001".to_string(),
            },
            instance_id: "test-001".to_string(),
            snapshot: None,
        };

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: VgpuEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.instance_id, "test-001");
    }
}
