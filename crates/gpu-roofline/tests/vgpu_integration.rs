#![cfg(feature = "vgpu")]
//! End-to-end integration tests for vGPU lifecycle monitoring.
//!
//! These tests run complete simulation scenarios through the full pipeline:
//! SimulatedDetector → VgpuSampler → VgpuAlertEngine → event/alert output.

use std::sync::mpsc;

use gpu_harness::vgpu::*;
use gpu_roofline::monitor::vgpu_alerting::VgpuAlertRule;
use gpu_roofline::monitor::vgpu_sampler::{VgpuMonitorConfig, VgpuSampler};

/// Helper: create a fast (zero-delay) version of a scenario and run it through the sampler.
fn run_scenario(name: &str) -> (VgpuSampler, Vec<VgpuEvent>) {
    let scenario = scenario_by_name(name).expect("scenario should exist");
    let physical_vram = scenario.physical_vram_bytes;
    let technology = scenario.technology;
    let partitioning = scenario.partitioning_mode;

    // Zero out delays for fast testing
    let fast = VgpuSimScenario {
        name: scenario.name.clone(),
        description: scenario.description.clone(),
        technology: scenario.technology,
        partitioning_mode: scenario.partitioning_mode,
        physical_vram_bytes: scenario.physical_vram_bytes,
        events: scenario
            .events
            .into_iter()
            .map(|mut e| {
                e.delay_secs = 0.0;
                e
            })
            .collect(),
    };

    let detector = SimulatedDetector::new(fast);

    let config = VgpuMonitorConfig {
        sample_interval_secs: 1,
        contention_threshold: 5.0, // 5%
        ..VgpuMonitorConfig::default()
    };
    let mut sampler = VgpuSampler::new(config, physical_vram);
    sampler.set_technology(technology, partitioning);

    let (tx, rx) = mpsc::channel();

    // Run detector synchronously (fast scenario = no sleeps)
    detector.watch(tx).unwrap();

    // Process all events through the sampler
    let events = sampler.run(rx, |_event, _alerts| true);

    (sampler, events)
}

#[test]
fn test_mig_scale_up_e2e() {
    let (sampler, events) = run_scenario("mig_scale_up");

    // 7 MIG instances should have been created
    let created_count = events
        .iter()
        .filter(|e| matches!(e.event_type, VgpuEventType::Created { .. }))
        .count();
    assert_eq!(created_count, 7, "should have 7 Created events");

    // MIG is hardware-partitioned: no contention events
    let contention_count = events
        .iter()
        .filter(|e| matches!(e.event_type, VgpuEventType::ContentionDetected { .. }))
        .count();
    assert_eq!(contention_count, 0, "MIG should have 0 contention events");

    // All 7 should still be active (no destroys)
    assert_eq!(sampler.active_count(), 7);

    // No alerts should fire (latencies are all < 500ms threshold)
    let alerts = sampler.alerts();
    let slow_provisions: Vec<_> = alerts
        .iter()
        .filter(|a| a.rule == VgpuAlertRule::SlowProvision)
        .collect();
    assert!(
        slow_provisions.is_empty(),
        "MIG spin-up latencies are under threshold"
    );
}

#[test]
fn test_grid_contention_e2e() {
    let (sampler, events) = run_scenario("grid_contention");

    // 4 creates + 3 contention events = 7 total
    assert_eq!(
        events.len(),
        7,
        "should have 7 events (4 creates + 3 contention)"
    );

    let created_count = events
        .iter()
        .filter(|e| matches!(e.event_type, VgpuEventType::Created { .. }))
        .count();
    assert_eq!(created_count, 4);

    let contention_count = events
        .iter()
        .filter(|e| matches!(e.event_type, VgpuEventType::ContentionDetected { .. }))
        .count();
    assert_eq!(contention_count, 3, "should have 3 contention events");

    // ContentionSqueeze alert should fire (bandwidth drops > 5%)
    let alerts = sampler.alerts();
    let squeeze_alerts: Vec<_> = alerts
        .iter()
        .filter(|a| a.rule == VgpuAlertRule::ContentionSqueeze)
        .collect();
    assert!(
        !squeeze_alerts.is_empty(),
        "ContentionSqueeze should fire for time-sliced vGPUs: alerts = {:?}",
        alerts
    );

    // All 4 instances should still be active
    assert_eq!(sampler.active_count(), 4);
}

#[test]
fn test_ghost_allocation_e2e() {
    let (sampler, events) = run_scenario("ghost_allocation");

    assert_eq!(events.len(), 2, "should have 2 events (create + destroy)");

    // GhostAllocation alert should fire
    let alerts = sampler.alerts();
    let ghost_alerts: Vec<_> = alerts
        .iter()
        .filter(|a| a.rule == VgpuAlertRule::GhostAllocation)
        .collect();
    assert!(
        !ghost_alerts.is_empty(),
        "GhostAllocation alert should fire: alerts = {:?}",
        alerts
    );

    // Instance should be cleaned up
    assert_eq!(sampler.active_count(), 0, "no instances should remain");
    assert_eq!(
        sampler.total_instance_count(),
        0,
        "instance map should be empty"
    );
    assert_eq!(
        sampler.contention_baseline_count(),
        0,
        "contention baselines should be cleaned up"
    );
}

#[test]
fn test_rapid_churn_no_state_leak() {
    let (sampler, events) = run_scenario("rapid_churn");

    // 20 creates + 20 destroys = 40 events
    assert_eq!(events.len(), 40, "should have 40 events");

    let created = events
        .iter()
        .filter(|e| matches!(e.event_type, VgpuEventType::Created { .. }))
        .count();
    let destroyed = events
        .iter()
        .filter(|e| matches!(e.event_type, VgpuEventType::Destroyed { .. }))
        .count();
    assert_eq!(created, 20);
    assert_eq!(destroyed, 20);

    // All instances should be cleaned up — zero leaks
    assert_eq!(
        sampler.active_count(),
        0,
        "no active instances should remain"
    );
    assert_eq!(
        sampler.total_instance_count(),
        0,
        "instance map should be empty after all destroys"
    );
    assert_eq!(
        sampler.contention_baseline_count(),
        0,
        "contention baselines should be fully cleaned up"
    );
}
