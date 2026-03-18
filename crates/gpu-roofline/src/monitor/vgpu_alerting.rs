//! vGPU-specific alert rules for lifecycle monitoring.
//!
//! Seven rules covering the full lifecycle: provision → active → teardown.

use super::alerting::{Alert, AlertLevel};

use gpu_harness::vgpu::{
    PartitioningMode, TeardownVerification, VgpuEvent, VgpuEventType, VgpuState,
};

/// Which vGPU-specific rule triggered the alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VgpuAlertRule {
    /// vGPU provisioning took longer than expected.
    SlowProvision,
    /// Existing tenant performance dropped after new vGPU appeared.
    ContentionSqueeze,
    /// vGPU performing below expected fraction of physical GPU.
    UnderperformingInstance,
    /// Teardown left unreleased memory (ghost allocation).
    GhostAllocation,
    /// Resource reclamation took too long.
    SlowReclaim,
    /// More vGPUs than recommended for the GPU.
    OverSubscription,
    /// Sum of vGPU memory allocations exceeds physical VRAM.
    MemoryOvercommit,
}

impl std::fmt::Display for VgpuAlertRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SlowProvision => write!(f, "Slow Provision"),
            Self::ContentionSqueeze => write!(f, "Contention Squeeze"),
            Self::UnderperformingInstance => write!(f, "Underperforming Instance"),
            Self::GhostAllocation => write!(f, "Ghost Allocation"),
            Self::SlowReclaim => write!(f, "Slow Reclaim"),
            Self::OverSubscription => write!(f, "Over-Subscription"),
            Self::MemoryOvercommit => write!(f, "Memory Overcommit"),
        }
    }
}

/// A vGPU-specific alert with both the generic alert fields and the vGPU rule.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VgpuAlert {
    pub level: AlertLevel,
    pub rule: VgpuAlertRule,
    pub message: String,
}

/// Configuration for the vGPU alert engine.
pub struct VgpuAlertConfig {
    /// Provision latency above which we alert (ms).
    pub slow_provision_ms: f64,
    /// Contention threshold (fraction, e.g. 0.05 = 5% drop).
    pub contention_threshold: f64,
    /// Reclaim latency above which we alert (ms).
    pub slow_reclaim_ms: f64,
    /// Maximum safe vGPU count per physical GPU.
    pub max_safe_density: u32,
}

impl Default for VgpuAlertConfig {
    fn default() -> Self {
        Self {
            slow_provision_ms: 500.0,
            contention_threshold: 0.05,
            slow_reclaim_ms: 1000.0,
            max_safe_density: 8,
        }
    }
}

/// Engine that evaluates vGPU-specific alert rules.
pub struct VgpuAlertEngine {
    config: VgpuAlertConfig,
}

impl VgpuAlertEngine {
    pub fn new(config: VgpuAlertConfig) -> Self {
        Self { config }
    }

    /// Check an event for alerts.
    pub fn check_event(&self, event: &VgpuEvent) -> Vec<VgpuAlert> {
        let mut alerts = Vec::new();

        match &event.event_type {
            VgpuEventType::Created {
                spin_up_latency_ms: Some(latency),
                ..
            } => {
                // Rule 1: SlowProvision
                if *latency > self.config.slow_provision_ms {
                    alerts.push(VgpuAlert {
                        level: AlertLevel::Warning,
                        rule: VgpuAlertRule::SlowProvision,
                        message: format!(
                            "vGPU {} provisioned in {:.0}ms (threshold: {:.0}ms)",
                            event.instance_id, latency, self.config.slow_provision_ms
                        ),
                    });
                }
            }
            VgpuEventType::Created {
                spin_up_latency_ms: None,
                ..
            } => {}
            VgpuEventType::ContentionDetected {
                affected_instances,
                bandwidth_impact,
                caused_by,
                ..
            } => {
                // Rule 2: ContentionSqueeze
                for (i, id) in affected_instances.iter().enumerate() {
                    let bw_impact = bandwidth_impact.get(i).copied().unwrap_or(1.0);
                    let drop_pct = (1.0 - bw_impact) * 100.0;
                    if drop_pct > self.config.contention_threshold * 100.0 {
                        alerts.push(VgpuAlert {
                            level: AlertLevel::Critical,
                            rule: VgpuAlertRule::ContentionSqueeze,
                            message: format!(
                                "vGPU {id} lost {drop_pct:.1}% bandwidth when {caused_by} provisioned"
                            ),
                        });
                    }
                }
            }
            VgpuEventType::Destroyed { verification, .. } => {
                self.check_teardown(&event.instance_id, verification, &mut alerts);
            }
            _ => {}
        }

        alerts
    }

    /// Check aggregate vGPU state for alerts.
    pub fn check_state(&self, state: &VgpuState) -> Vec<VgpuAlert> {
        let mut alerts = Vec::new();

        // Rule 6: OverSubscription
        if state.active_count > self.config.max_safe_density {
            alerts.push(VgpuAlert {
                level: AlertLevel::Warning,
                rule: VgpuAlertRule::OverSubscription,
                message: format!(
                    "GPU {} has {} active vGPUs (safe density: {})",
                    state.physical_gpu_index, state.active_count, self.config.max_safe_density
                ),
            });
        }

        // Rule 7: MemoryOvercommit
        if state.total_vram_allocated_bytes > state.total_vram_available_bytes {
            let overcommit_gb = (state.total_vram_allocated_bytes
                - state.total_vram_available_bytes) as f64
                / (1024.0 * 1024.0 * 1024.0);
            alerts.push(VgpuAlert {
                level: AlertLevel::Critical,
                rule: VgpuAlertRule::MemoryOvercommit,
                message: format!(
                    "GPU {} memory overcommitted by {:.1} GB ({:.1}/{:.1} GB)",
                    state.physical_gpu_index,
                    overcommit_gb,
                    state.total_vram_allocated_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    state.total_vram_available_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                ),
            });
        }

        // Rule 3: UnderperformingInstance
        // Only meaningful for time-sliced mode where we can compare expected vs actual
        if state.partitioning_mode == PartitioningMode::TimeSliced {
            for instance in &state.instances {
                if let Some(compute_fraction) = instance.compute_fraction {
                    // If we have a snapshot in recent events for this instance, check it
                    for event in &state.recent_events {
                        if event.instance_id == instance.id {
                            if let Some(ref snapshot) = event.snapshot {
                                // If utilization is far below expected fraction, flag it
                                let expected_util = compute_fraction * 100.0;
                                if (snapshot.utilization_pct as f64) < expected_util * 0.5 {
                                    alerts.push(VgpuAlert {
                                        level: AlertLevel::Warning,
                                        rule: VgpuAlertRule::UnderperformingInstance,
                                        message: format!(
                                            "vGPU {} at {:.0}% utilization (expected ~{:.0}%)",
                                            instance.id, snapshot.utilization_pct, expected_util
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        alerts
    }

    fn check_teardown(
        &self,
        instance_id: &str,
        verification: &TeardownVerification,
        alerts: &mut Vec<VgpuAlert>,
    ) {
        // Rule 4: GhostAllocation
        if !verification.memory_reclaimed && verification.ghost_allocations_bytes > 0 {
            let ghost_mb = verification.ghost_allocations_bytes as f64 / (1024.0 * 1024.0);
            alerts.push(VgpuAlert {
                level: AlertLevel::Critical,
                rule: VgpuAlertRule::GhostAllocation,
                message: format!("vGPU {instance_id} teardown left {ghost_mb:.0} MB unreleased"),
            });
        }

        // Rule 5: SlowReclaim
        if verification.reclaim_latency_ms > self.config.slow_reclaim_ms {
            alerts.push(VgpuAlert {
                level: AlertLevel::Warning,
                rule: VgpuAlertRule::SlowReclaim,
                message: format!(
                    "vGPU {instance_id} reclaim took {:.0}ms (threshold: {:.0}ms)",
                    verification.reclaim_latency_ms, self.config.slow_reclaim_ms
                ),
            });
        }
    }
}

/// Convert a VgpuAlert into the generic Alert type for unified logging.
impl From<VgpuAlert> for Alert {
    fn from(va: VgpuAlert) -> Self {
        Alert {
            level: va.level,
            rule: super::alerting::AlertRule::BandwidthDrop, // Best-fit generic rule
            message: format!("[vGPU:{}] {}", va.rule, va.message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::vgpu::*;

    fn default_engine() -> VgpuAlertEngine {
        VgpuAlertEngine::new(VgpuAlertConfig::default())
    }

    fn make_instance(id: &str) -> VgpuInstance {
        VgpuInstance {
            id: id.to_string(),
            name: format!("Test vGPU {id}"),
            technology: VgpuTechnology::NvidiaGrid,
            physical_gpu_index: 0,
            physical_pci_bus_id: None,
            phase: VgpuPhase::Active,
            vram_allocated_bytes: 4 * 1024 * 1024 * 1024,
            compute_fraction: None,
            memory_fraction: 0.25,
            mig_profile: None,
        }
    }

    #[test]
    fn test_slow_provision_alert() {
        let engine = default_engine();
        let event = VgpuEvent {
            timestamp: chrono::Utc::now(),
            event_type: VgpuEventType::Created {
                instance: make_instance("test-1"),
                spin_up_latency_ms: Some(800.0), // > 500ms threshold
            },
            instance_id: "test-1".to_string(),
            snapshot: None,
        };

        let alerts = engine.check_event(&event);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].rule, VgpuAlertRule::SlowProvision);
        assert_eq!(alerts[0].level, AlertLevel::Warning);
    }

    #[test]
    fn test_fast_provision_no_alert() {
        let engine = default_engine();
        let event = VgpuEvent {
            timestamp: chrono::Utc::now(),
            event_type: VgpuEventType::Created {
                instance: make_instance("test-1"),
                spin_up_latency_ms: Some(100.0), // < 500ms threshold
            },
            instance_id: "test-1".to_string(),
            snapshot: None,
        };

        let alerts = engine.check_event(&event);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_contention_squeeze_alert() {
        let engine = default_engine();
        let event = VgpuEvent {
            timestamp: chrono::Utc::now(),
            event_type: VgpuEventType::ContentionDetected {
                affected_instances: vec!["a".to_string(), "b".to_string()],
                bandwidth_impact: vec![0.75, 0.98], // a dropped 25%, b dropped 2%
                compute_impact: vec![0.8, 0.99],
                caused_by: "c".to_string(),
            },
            instance_id: "c".to_string(),
            snapshot: None,
        };

        let alerts = engine.check_event(&event);
        assert_eq!(alerts.len(), 1); // Only 'a' exceeds 5% threshold
        assert_eq!(alerts[0].rule, VgpuAlertRule::ContentionSqueeze);
        assert_eq!(alerts[0].level, AlertLevel::Critical);
    }

    #[test]
    fn test_ghost_allocation_alert() {
        let engine = default_engine();
        let event = VgpuEvent {
            timestamp: chrono::Utc::now(),
            event_type: VgpuEventType::Destroyed {
                instance_id: "test-1".to_string(),
                verification: TeardownVerification {
                    memory_reclaimed: false,
                    expected_free_bytes: 4_000_000_000,
                    actual_free_bytes: 3_500_000_000,
                    reclaim_latency_ms: 50.0,
                    ghost_allocations_bytes: 500_000_000,
                    compute_reclaimed: true,
                },
            },
            instance_id: "test-1".to_string(),
            snapshot: None,
        };

        let alerts = engine.check_event(&event);
        assert!(alerts
            .iter()
            .any(|a| a.rule == VgpuAlertRule::GhostAllocation));
    }

    #[test]
    fn test_slow_reclaim_alert() {
        let engine = default_engine();
        let event = VgpuEvent {
            timestamp: chrono::Utc::now(),
            event_type: VgpuEventType::Destroyed {
                instance_id: "test-1".to_string(),
                verification: TeardownVerification {
                    memory_reclaimed: true,
                    expected_free_bytes: 4_000_000_000,
                    actual_free_bytes: 4_000_000_000,
                    reclaim_latency_ms: 2000.0, // > 1000ms threshold
                    ghost_allocations_bytes: 0,
                    compute_reclaimed: true,
                },
            },
            instance_id: "test-1".to_string(),
            snapshot: None,
        };

        let alerts = engine.check_event(&event);
        assert!(alerts.iter().any(|a| a.rule == VgpuAlertRule::SlowReclaim));
    }

    #[test]
    fn test_oversubscription_alert() {
        let engine = default_engine();
        let state = VgpuState {
            physical_gpu_index: 0,
            instances: Vec::new(),
            total_vram_allocated_bytes: 0,
            total_vram_available_bytes: 80 * 1024 * 1024 * 1024,
            active_count: 12, // > 8 safe density
            technology: VgpuTechnology::NvidiaGrid,
            partitioning_mode: PartitioningMode::TimeSliced,
            recent_events: Vec::new(),
        };

        let alerts = engine.check_state(&state);
        assert!(alerts
            .iter()
            .any(|a| a.rule == VgpuAlertRule::OverSubscription));
    }

    #[test]
    fn test_memory_overcommit_alert() {
        let engine = default_engine();
        let state = VgpuState {
            physical_gpu_index: 0,
            instances: Vec::new(),
            total_vram_allocated_bytes: 20 * 1024 * 1024 * 1024,
            total_vram_available_bytes: 16 * 1024 * 1024 * 1024,
            active_count: 4,
            technology: VgpuTechnology::NvidiaGrid,
            partitioning_mode: PartitioningMode::TimeSliced,
            recent_events: Vec::new(),
        };

        let alerts = engine.check_state(&state);
        assert!(alerts
            .iter()
            .any(|a| a.rule == VgpuAlertRule::MemoryOvercommit));
    }
}
