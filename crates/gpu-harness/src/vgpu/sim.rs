//! Simulated vGPU scenarios for testing without hardware.
//!
//! Provides built-in scenarios that replay vGPU lifecycle events
//! (creation, contention, teardown, ghost allocations) on a timer.

use std::sync::mpsc;

use crate::error::HarnessError;

use super::detect::VgpuDetector;
use super::state::*;

/// A scheduled action in a simulation scenario.
#[derive(Debug, Clone)]
pub enum SimAction {
    CreateVgpu {
        instance: VgpuInstance,
        spin_up_latency_ms: f64,
    },
    DestroyVgpu {
        instance_id: String,
        ghost_bytes: u64,
    },
}

/// A timed event in a simulation scenario.
#[derive(Debug, Clone)]
pub struct ScheduledEvent {
    pub delay_secs: f64,
    pub action: SimAction,
}

/// A complete simulation scenario with a name and sequence of events.
#[derive(Debug, Clone)]
pub struct VgpuSimScenario {
    pub name: String,
    pub description: String,
    pub events: Vec<ScheduledEvent>,
    pub technology: VgpuTechnology,
    pub partitioning_mode: PartitioningMode,
    pub physical_vram_bytes: u64,
}

impl VgpuSimScenario {
    /// Human-readable label for the partitioning mode.
    pub fn partitioning_mode_label(&self) -> &'static str {
        match self.partitioning_mode {
            PartitioningMode::HardwarePartitioned => "Hardware Partitioned",
            PartitioningMode::TimeSliced => "Time-Sliced",
            PartitioningMode::Passthrough => "Passthrough",
            PartitioningMode::Unknown => "Unknown",
        }
    }
}

/// List all available scenario names.
pub fn available_scenarios() -> Vec<&'static str> {
    vec![
        "mig_scale_up",
        "grid_contention",
        "ghost_allocation",
        "rapid_churn",
    ]
}

/// Look up a built-in scenario by name.
pub fn scenario_by_name(name: &str) -> Option<VgpuSimScenario> {
    match name {
        "mig_scale_up" => Some(mig_scale_up()),
        "grid_contention" => Some(grid_contention()),
        "ghost_allocation" => Some(ghost_allocation()),
        "rapid_churn" => Some(rapid_churn()),
        _ => None,
    }
}

/// 7 MIG instances on H100, sequential creation, no contention (hardware-partitioned).
fn mig_scale_up() -> VgpuSimScenario {
    let mut events = Vec::new();
    for i in 0..7 {
        events.push(ScheduledEvent {
            delay_secs: 1.0 + i as f64 * 2.0,
            action: SimAction::CreateVgpu {
                instance: VgpuInstance {
                    id: format!("mig-{i}"),
                    name: format!("MIG 1g.10gb #{i}"),
                    technology: VgpuTechnology::NvidiaMig,
                    physical_gpu_index: 0,
                    physical_pci_bus_id: Some("0000:3b:00.0".to_string()),
                    phase: VgpuPhase::Provisioning,
                    vram_allocated_bytes: 10 * 1024 * 1024 * 1024,
                    compute_fraction: Some(1.0 / 7.0),
                    memory_fraction: 1.0 / 8.0,
                    mig_profile: Some("1g.10gb".to_string()),
                },
                spin_up_latency_ms: 120.0 + i as f64 * 15.0,
            },
        });
    }

    VgpuSimScenario {
        name: "mig_scale_up".to_string(),
        description: "7 MIG instances on H100, sequential creation, no contention".to_string(),
        events,
        technology: VgpuTechnology::NvidiaMig,
        partitioning_mode: PartitioningMode::HardwarePartitioned,
        physical_vram_bytes: 80 * 1024 * 1024 * 1024,
    }
}

/// 4 GRID vGPUs, each new one squeezes existing tenants.
fn grid_contention() -> VgpuSimScenario {
    let mut events = Vec::new();
    for i in 0..4 {
        events.push(ScheduledEvent {
            delay_secs: 1.0 + i as f64 * 3.0,
            action: SimAction::CreateVgpu {
                instance: VgpuInstance {
                    id: format!("grid-{i}"),
                    name: format!("GRID V100D-4Q #{i}"),
                    technology: VgpuTechnology::NvidiaGrid,
                    physical_gpu_index: 0,
                    physical_pci_bus_id: Some("0000:3b:00.0".to_string()),
                    phase: VgpuPhase::Provisioning,
                    vram_allocated_bytes: 4 * 1024 * 1024 * 1024,
                    compute_fraction: None, // Time-sliced, no guaranteed fraction
                    memory_fraction: 0.25,
                    mig_profile: None,
                },
                spin_up_latency_ms: 250.0 + i as f64 * 50.0,
            },
        });
    }

    VgpuSimScenario {
        name: "grid_contention".to_string(),
        description: "4 GRID vGPUs with time-sliced contention".to_string(),
        events,
        technology: VgpuTechnology::NvidiaGrid,
        partitioning_mode: PartitioningMode::TimeSliced,
        physical_vram_bytes: 16 * 1024 * 1024 * 1024,
    }
}

/// Create + destroy with 512MB not reclaimed.
fn ghost_allocation() -> VgpuSimScenario {
    VgpuSimScenario {
        name: "ghost_allocation".to_string(),
        description: "Create and destroy vGPU with 512MB ghost allocation".to_string(),
        events: vec![
            ScheduledEvent {
                delay_secs: 1.0,
                action: SimAction::CreateVgpu {
                    instance: VgpuInstance {
                        id: "ghost-0".to_string(),
                        name: "GRID V100D-8Q".to_string(),
                        technology: VgpuTechnology::NvidiaGrid,
                        physical_gpu_index: 0,
                        physical_pci_bus_id: Some("0000:3b:00.0".to_string()),
                        phase: VgpuPhase::Provisioning,
                        vram_allocated_bytes: 8 * 1024 * 1024 * 1024,
                        compute_fraction: None,
                        memory_fraction: 0.5,
                        mig_profile: None,
                    },
                    spin_up_latency_ms: 200.0,
                },
            },
            ScheduledEvent {
                delay_secs: 5.0,
                action: SimAction::DestroyVgpu {
                    instance_id: "ghost-0".to_string(),
                    ghost_bytes: 512 * 1024 * 1024,
                },
            },
        ],
        technology: VgpuTechnology::NvidiaGrid,
        partitioning_mode: PartitioningMode::TimeSliced,
        physical_vram_bytes: 16 * 1024 * 1024 * 1024,
    }
}

/// 20 create/destroy cycles in 60s, test for state leaks.
fn rapid_churn() -> VgpuSimScenario {
    let mut events = Vec::new();
    for i in 0..20 {
        let base = i as f64 * 3.0;
        events.push(ScheduledEvent {
            delay_secs: base + 0.5,
            action: SimAction::CreateVgpu {
                instance: VgpuInstance {
                    id: format!("churn-{i}"),
                    name: format!("MIG 1g.5gb #{i}"),
                    technology: VgpuTechnology::NvidiaMig,
                    physical_gpu_index: 0,
                    physical_pci_bus_id: Some("0000:3b:00.0".to_string()),
                    phase: VgpuPhase::Provisioning,
                    vram_allocated_bytes: 5 * 1024 * 1024 * 1024,
                    compute_fraction: Some(1.0 / 7.0),
                    memory_fraction: 1.0 / 8.0,
                    mig_profile: Some("1g.5gb".to_string()),
                },
                spin_up_latency_ms: 100.0,
            },
        });
        events.push(ScheduledEvent {
            delay_secs: base + 2.0,
            action: SimAction::DestroyVgpu {
                instance_id: format!("churn-{i}"),
                ghost_bytes: 0,
            },
        });
    }

    VgpuSimScenario {
        name: "rapid_churn".to_string(),
        description: "20 create/destroy cycles in 60s to test state leaks".to_string(),
        events,
        technology: VgpuTechnology::NvidiaMig,
        partitioning_mode: PartitioningMode::HardwarePartitioned,
        physical_vram_bytes: 80 * 1024 * 1024 * 1024,
    }
}

/// Simulated vGPU detector that replays scenario events.
pub struct SimulatedDetector {
    scenario: VgpuSimScenario,
}

impl SimulatedDetector {
    pub fn new(scenario: VgpuSimScenario) -> Self {
        Self { scenario }
    }

    pub fn scenario(&self) -> &VgpuSimScenario {
        &self.scenario
    }
}

impl VgpuDetector for SimulatedDetector {
    fn enumerate(&self) -> Result<Vec<VgpuInstance>, HarnessError> {
        // At start, no instances exist yet
        Ok(Vec::new())
    }

    fn watch(&self, tx: mpsc::Sender<VgpuEvent>) -> Result<(), HarnessError> {
        let start = std::time::Instant::now();

        for scheduled in &self.scenario.events {
            let target = std::time::Duration::from_secs_f64(scheduled.delay_secs);
            let elapsed = start.elapsed();
            if elapsed < target {
                std::thread::sleep(target - elapsed);
            }

            let event = match &scheduled.action {
                SimAction::CreateVgpu {
                    instance,
                    spin_up_latency_ms,
                } => VgpuEvent {
                    timestamp: chrono::Utc::now(),
                    event_type: VgpuEventType::Created {
                        instance: instance.clone(),
                        spin_up_latency_ms: Some(*spin_up_latency_ms),
                    },
                    instance_id: instance.id.clone(),
                    snapshot: None,
                },
                SimAction::DestroyVgpu {
                    instance_id,
                    ghost_bytes,
                } => VgpuEvent {
                    timestamp: chrono::Utc::now(),
                    event_type: VgpuEventType::Destroyed {
                        instance_id: instance_id.clone(),
                        verification: TeardownVerification {
                            memory_reclaimed: *ghost_bytes == 0,
                            expected_free_bytes: 0, // Filled by caller
                            actual_free_bytes: 0,
                            reclaim_latency_ms: 50.0,
                            ghost_allocations_bytes: *ghost_bytes,
                            compute_reclaimed: *ghost_bytes == 0,
                        },
                    },
                    instance_id: instance_id.clone(),
                    snapshot: None,
                },
            };

            if tx.send(event).is_err() {
                break; // Receiver dropped
            }
        }

        Ok(())
    }

    fn technology(&self) -> VgpuTechnology {
        VgpuTechnology::Simulated
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_scenarios() {
        let names = available_scenarios();
        assert_eq!(names.len(), 4);
        assert!(names.contains(&"mig_scale_up"));
        assert!(names.contains(&"grid_contention"));
        assert!(names.contains(&"ghost_allocation"));
        assert!(names.contains(&"rapid_churn"));
    }

    #[test]
    fn test_scenario_by_name() {
        for name in available_scenarios() {
            assert!(
                scenario_by_name(name).is_some(),
                "scenario '{name}' should exist"
            );
        }
        assert!(scenario_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_mig_scale_up_scenario() {
        let s = scenario_by_name("mig_scale_up").unwrap();
        assert_eq!(s.events.len(), 7);
        assert_eq!(s.partitioning_mode, PartitioningMode::HardwarePartitioned);
        assert_eq!(s.technology, VgpuTechnology::NvidiaMig);
    }

    #[test]
    fn test_grid_contention_scenario() {
        let s = scenario_by_name("grid_contention").unwrap();
        assert_eq!(s.events.len(), 4);
        assert_eq!(s.partitioning_mode, PartitioningMode::TimeSliced);
    }

    #[test]
    fn test_ghost_allocation_scenario() {
        let s = scenario_by_name("ghost_allocation").unwrap();
        assert_eq!(s.events.len(), 2);
        // Second event should be a destroy with ghost bytes
        match &s.events[1].action {
            SimAction::DestroyVgpu { ghost_bytes, .. } => {
                assert_eq!(*ghost_bytes, 512 * 1024 * 1024);
            }
            _ => panic!("expected DestroyVgpu"),
        }
    }

    #[test]
    fn test_rapid_churn_scenario() {
        let s = scenario_by_name("rapid_churn").unwrap();
        assert_eq!(s.events.len(), 40); // 20 creates + 20 destroys
    }

    #[test]
    fn test_simulated_detector_enumerate() {
        let s = scenario_by_name("mig_scale_up").unwrap();
        let detector = SimulatedDetector::new(s);
        assert!(detector.is_available());
        assert_eq!(detector.technology(), VgpuTechnology::Simulated);
        let instances = detector.enumerate().unwrap();
        assert!(instances.is_empty()); // No instances at start
    }

    #[test]
    fn test_simulated_detector_watch() {
        // Use ghost_allocation (only 2 events, fastest scenario)
        let s = scenario_by_name("ghost_allocation").unwrap();
        // Override delays to be instant for testing
        let fast_scenario = VgpuSimScenario {
            name: s.name.clone(),
            description: s.description.clone(),
            technology: s.technology,
            partitioning_mode: s.partitioning_mode,
            physical_vram_bytes: s.physical_vram_bytes,
            events: s
                .events
                .into_iter()
                .map(|mut e| {
                    e.delay_secs = 0.0;
                    e
                })
                .collect(),
        };
        let detector = SimulatedDetector::new(fast_scenario);

        let (tx, rx) = mpsc::channel();
        detector.watch(tx).unwrap();

        let events: Vec<_> = rx.try_iter().collect();
        assert_eq!(events.len(), 2);

        // First event should be Created
        assert!(matches!(
            &events[0].event_type,
            VgpuEventType::Created { .. }
        ));
        // Second should be Destroyed
        assert!(matches!(
            &events[1].event_type,
            VgpuEventType::Destroyed { .. }
        ));
    }
}
