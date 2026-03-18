#![cfg(feature = "vgpu")]
//! Scale stress tests for vGPU lifecycle monitoring.
//!
//! Validates state management under high-volume lifecycle churn
//! and extreme instance density.

use std::sync::mpsc;

use gpu_harness::vgpu::*;
use gpu_roofline::monitor::vgpu_alerting::VgpuAlertRule;
use gpu_roofline::monitor::vgpu_sampler::{VgpuMonitorConfig, VgpuSampler};

/// Build a churn scenario with N create/destroy cycles.
fn build_churn_scenario(cycles: u32) -> VgpuSimScenario {
    let mut events = Vec::new();
    for i in 0..cycles {
        events.push(ScheduledEvent {
            delay_secs: 0.0,
            action: SimAction::CreateVgpu {
                instance: VgpuInstance {
                    id: format!("stress-{i}"),
                    name: format!("Stress vGPU #{i}"),
                    technology: VgpuTechnology::NvidiaMig,
                    physical_gpu_index: 0,
                    physical_pci_bus_id: None,
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
            delay_secs: 0.0,
            action: SimAction::DestroyVgpu {
                instance_id: format!("stress-{i}"),
                ghost_bytes: 0,
            },
        });
    }

    VgpuSimScenario {
        name: format!("stress_churn_{cycles}"),
        description: format!("{cycles} create/destroy cycles"),
        events,
        technology: VgpuTechnology::NvidiaMig,
        partitioning_mode: PartitioningMode::HardwarePartitioned,
        physical_vram_bytes: 80 * 1024 * 1024 * 1024,
    }
}

/// Run a scenario through the full pipeline and return the sampler.
fn run_scenario_sampler(scenario: VgpuSimScenario) -> (VgpuSampler, Vec<VgpuEvent>) {
    let physical_vram = scenario.physical_vram_bytes;
    let tech = scenario.technology;
    let mode = scenario.partitioning_mode;

    let detector = SimulatedDetector::new(scenario);
    let config = VgpuMonitorConfig {
        sample_interval_secs: 1,
        contention_threshold: 5.0,
        ..VgpuMonitorConfig::default()
    };
    let mut sampler = VgpuSampler::new(config, physical_vram);
    sampler.set_technology(tech, mode);

    let (tx, rx) = mpsc::channel();
    detector.watch(tx).unwrap();
    let events = sampler.run(rx, |_event, _alerts| true);

    (sampler, events)
}

#[test]
fn test_churn_100_cycles() {
    let scenario = build_churn_scenario(100);
    let (sampler, events) = run_scenario_sampler(scenario);

    assert_eq!(events.len(), 200, "100 creates + 100 destroys");
    assert_eq!(sampler.active_count(), 0, "all instances destroyed");
    assert_eq!(sampler.total_instance_count(), 0, "instance map empty");
    assert_eq!(
        sampler.contention_baseline_count(),
        0,
        "no leaked baselines"
    );
}

#[test]
#[ignore] // Slow — run with `cargo test -- --ignored`
fn test_churn_1000_cycles() {
    let scenario = build_churn_scenario(1000);
    let (sampler, events) = run_scenario_sampler(scenario);

    assert_eq!(events.len(), 2000);
    assert_eq!(sampler.active_count(), 0);
    assert_eq!(sampler.total_instance_count(), 0);
    assert_eq!(sampler.contention_baseline_count(), 0);
}

/// Build a scenario that creates N instances without destroying them.
fn build_density_scenario(count: u32, vram_per_instance_gb: u64) -> VgpuSimScenario {
    let mut events = Vec::new();
    for i in 0..count {
        events.push(ScheduledEvent {
            delay_secs: 0.0,
            action: SimAction::CreateVgpu {
                instance: VgpuInstance {
                    id: format!("dense-{i}"),
                    name: format!("Dense vGPU #{i}"),
                    technology: VgpuTechnology::NvidiaGrid,
                    physical_gpu_index: 0,
                    physical_pci_bus_id: None,
                    phase: VgpuPhase::Provisioning,
                    vram_allocated_bytes: vram_per_instance_gb * 1024 * 1024 * 1024,
                    compute_fraction: None,
                    memory_fraction: 1.0 / count as f64,
                    mig_profile: None,
                },
                spin_up_latency_ms: 100.0,
            },
        });
    }

    VgpuSimScenario {
        name: "density_test".to_string(),
        description: format!("{count} concurrent instances"),
        events,
        technology: VgpuTechnology::NvidiaGrid,
        partitioning_mode: PartitioningMode::TimeSliced,
        physical_vram_bytes: 16 * 1024 * 1024 * 1024,
    }
}

#[test]
fn test_peak_density_oversubscription() {
    // Create 12 instances — above default max_safe_density of 8
    let scenario = build_density_scenario(12, 2);
    let (sampler, events) = run_scenario_sampler(scenario);

    assert_eq!(events.len(), 12);
    assert_eq!(sampler.active_count(), 12);

    let alerts = sampler.alerts();
    let oversub: Vec<_> = alerts
        .iter()
        .filter(|a| a.rule == VgpuAlertRule::OverSubscription)
        .collect();
    assert!(
        !oversub.is_empty(),
        "OverSubscription should fire at 12 instances (safe density = 8): alerts = {:?}",
        alerts
    );
}

#[test]
fn test_peak_density_memory_overcommit() {
    // Create 5 instances at 4GB each = 20GB on a 16GB GPU
    let scenario = build_density_scenario(5, 4);
    let (sampler, events) = run_scenario_sampler(scenario);

    assert_eq!(events.len(), 5);

    let alerts = sampler.alerts();
    let overcommit: Vec<_> = alerts
        .iter()
        .filter(|a| a.rule == VgpuAlertRule::MemoryOvercommit)
        .collect();
    assert!(
        !overcommit.is_empty(),
        "MemoryOvercommit should fire at 20GB on 16GB GPU: alerts = {:?}",
        alerts
    );
}

#[test]
fn test_daemon_json_output_structure() {
    // Validate that events serialize to valid JSON
    let scenario = build_churn_scenario(5);
    let detector = SimulatedDetector::new(scenario);

    let (tx, rx) = mpsc::channel();
    detector.watch(tx).unwrap();

    let events: Vec<VgpuEvent> = rx.try_iter().collect();
    assert_eq!(events.len(), 10);

    // Every event should serialize to valid JSON
    for event in &events {
        let json = serde_json::to_string(event).expect("event should serialize to JSON");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("JSON should parse back");

        // Must have required fields
        assert!(parsed.get("timestamp").is_some());
        assert!(parsed.get("event_type").is_some());
        assert!(parsed.get("instance_id").is_some());
    }

    // Verify Created events have instance data
    let created_json = serde_json::to_string(&events[0]).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&created_json).unwrap();
    let event_type = parsed.get("event_type").unwrap();
    assert!(
        event_type.get("Created").is_some(),
        "first event should be Created"
    );
}
