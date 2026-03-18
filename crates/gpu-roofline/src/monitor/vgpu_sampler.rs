//! Auto-attach/detach sampler for vGPU lifecycle monitoring.
//!
//! Receives events from a VgpuDetector and manages per-instance monitoring:
//! - Created → auto-attach: add monitor, take baseline, check contention
//! - TeardownStarted → capture pre-teardown state
//! - Destroyed → auto-detach: verify teardown, remove monitor

use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

use gpu_harness::vgpu::{
    ContentionMeasurer, TeardownVerifier, VgpuEvent, VgpuEventType, VgpuInstance, VgpuPhase,
    VgpuSnapshot, VgpuState, VgpuTechnology,
};

use super::vgpu_alerting::{VgpuAlert, VgpuAlertConfig, VgpuAlertEngine};

/// Per-instance monitoring state.
struct InstanceMonitor {
    instance: VgpuInstance,
}

/// Configuration for vGPU monitoring.
#[derive(Debug, Clone)]
pub struct VgpuMonitorConfig {
    /// Seconds between periodic samples of active instances.
    pub sample_interval_secs: u64,
    /// Contention threshold (fraction, e.g. 0.05 = 5%).
    pub contention_threshold: f64,
    /// Run in daemon mode (JSON lines only).
    pub daemon: bool,
    /// Write JSON log to this path.
    pub log_path: Option<String>,
}

impl Default for VgpuMonitorConfig {
    fn default() -> Self {
        Self {
            sample_interval_secs: 5,
            contention_threshold: 0.05,
            daemon: false,
            log_path: None,
        }
    }
}

/// The vGPU sampler: auto-attach/detach event loop.
pub struct VgpuSampler {
    instances: HashMap<String, InstanceMonitor>,
    contention: ContentionMeasurer,
    teardown: TeardownVerifier,
    alert_engine: VgpuAlertEngine,
    config: VgpuMonitorConfig,
    events: Vec<VgpuEvent>,
    alerts: Vec<VgpuAlert>,
    /// Simulated physical VRAM for state tracking.
    physical_vram_bytes: u64,
    /// Simulated current memory used.
    simulated_memory_used: u64,
    /// Technology being monitored.
    technology: VgpuTechnology,
    /// Partitioning mode.
    partitioning_mode: gpu_harness::vgpu::PartitioningMode,
}

impl VgpuSampler {
    pub fn new(config: VgpuMonitorConfig, physical_vram_bytes: u64) -> Self {
        let alert_config = VgpuAlertConfig {
            contention_threshold: config.contention_threshold / 100.0, // Convert from % to fraction
            ..VgpuAlertConfig::default()
        };

        Self {
            instances: HashMap::new(),
            contention: ContentionMeasurer::new(config.contention_threshold / 100.0),
            teardown: TeardownVerifier::new(),
            alert_engine: VgpuAlertEngine::new(alert_config),
            config,
            events: Vec::new(),
            alerts: Vec::new(),
            physical_vram_bytes,
            simulated_memory_used: 0,
            technology: VgpuTechnology::Simulated,
            partitioning_mode: gpu_harness::vgpu::PartitioningMode::Unknown,
        }
    }

    /// Set the technology and partitioning mode for this sampler.
    pub fn set_technology(
        &mut self,
        tech: VgpuTechnology,
        mode: gpu_harness::vgpu::PartitioningMode,
    ) {
        self.technology = tech;
        self.partitioning_mode = mode;
    }

    /// Run the event loop, consuming events from the detector channel.
    ///
    /// Calls `on_event` for each lifecycle event. Return `false` to stop.
    pub fn run<F>(&mut self, rx: mpsc::Receiver<VgpuEvent>, mut on_event: F) -> Vec<VgpuEvent>
    where
        F: FnMut(&VgpuEvent, &[VgpuAlert]) -> bool,
    {
        let timeout = Duration::from_secs(self.config.sample_interval_secs);

        loop {
            match rx.recv_timeout(timeout) {
                Ok(event) => {
                    let alerts = self.process_event(&event);
                    self.events.push(event.clone());

                    if !on_event(&event, &alerts) {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // No events — periodic check (could sample active instances here)
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Detector finished (simulation complete or error)
                    break;
                }
            }
        }

        self.events.clone()
    }

    /// Process a single event and return any triggered alerts.
    pub fn process_event(&mut self, event: &VgpuEvent) -> Vec<VgpuAlert> {
        let mut alerts = Vec::new();

        match &event.event_type {
            VgpuEventType::Created {
                instance,
                spin_up_latency_ms: _,
            } => {
                self.handle_created(instance);
            }
            VgpuEventType::Active { instance_id } => {
                if let Some(monitor) = self.instances.get_mut(instance_id) {
                    monitor.instance.phase = VgpuPhase::Active;
                }
            }
            VgpuEventType::Sampled { instance_id: _ } => {
                // Could update snapshots here
            }
            VgpuEventType::ContentionDetected { .. } => {
                // Contention events are generated by the measurer
            }
            VgpuEventType::TeardownStarted { instance_id } => {
                self.handle_teardown_started(instance_id);
            }
            VgpuEventType::Destroyed {
                instance_id,
                verification: _,
            } => {
                self.handle_destroyed(instance_id);
            }
        }

        // Check event-level alerts
        let event_alerts = self.alert_engine.check_event(event);
        alerts.extend(event_alerts.clone());

        // Check aggregate state alerts
        let state = self.build_state();
        let state_alerts = self.alert_engine.check_state(&state);
        alerts.extend(state_alerts);

        self.alerts.extend(alerts.clone());
        alerts
    }

    fn handle_created(&mut self, instance: &VgpuInstance) {
        // Auto-attach: create monitor for this instance
        let monitor = InstanceMonitor {
            instance: VgpuInstance {
                phase: VgpuPhase::Active,
                ..instance.clone()
            },
        };

        // Record baseline for contention detection
        let baseline = VgpuSnapshot {
            bandwidth_gbps: 0.0, // Will be populated on first sample
            gflops: 0.0,
            memory_used_bytes: 0,
            memory_allocated_bytes: instance.vram_allocated_bytes,
            utilization_pct: 0.0,
            temperature_c: 0,
            power_watts: 0.0,
            encoder_utilization_pct: None,
            decoder_utilization_pct: None,
        };
        self.contention.record_baseline(&instance.id, baseline);

        self.simulated_memory_used += instance.vram_allocated_bytes;
        self.instances.insert(instance.id.clone(), monitor);
    }

    fn handle_teardown_started(&mut self, instance_id: &str) {
        if let Some(monitor) = self.instances.get_mut(instance_id) {
            monitor.instance.phase = VgpuPhase::Teardown;
        }

        // Capture pre-teardown state
        let vram = self
            .instances
            .get(instance_id)
            .map(|m| m.instance.vram_allocated_bytes)
            .unwrap_or(0);

        self.teardown
            .capture_pre_teardown(instance_id, self.simulated_memory_used, vram);
    }

    fn handle_destroyed(&mut self, instance_id: &str) {
        // Auto-detach: remove monitor
        if let Some(monitor) = self.instances.remove(instance_id) {
            self.simulated_memory_used = self
                .simulated_memory_used
                .saturating_sub(monitor.instance.vram_allocated_bytes);
        }
        self.contention.remove_baseline(instance_id);
    }

    fn build_state(&self) -> VgpuState {
        let instances: Vec<VgpuInstance> = self
            .instances
            .values()
            .map(|m| m.instance.clone())
            .collect();

        let total_allocated: u64 = instances.iter().map(|i| i.vram_allocated_bytes).sum();
        let active_count = instances
            .iter()
            .filter(|i| i.phase == VgpuPhase::Active)
            .count() as u32;

        // Include last 10 events for state-level alert checks
        let recent_events: Vec<VgpuEvent> =
            self.events.iter().rev().take(10).rev().cloned().collect();

        VgpuState {
            physical_gpu_index: 0,
            instances,
            total_vram_allocated_bytes: total_allocated,
            total_vram_available_bytes: self.physical_vram_bytes,
            active_count,
            technology: self.technology,
            partitioning_mode: self.partitioning_mode,
            recent_events,
        }
    }

    /// Get all collected events.
    pub fn events(&self) -> &[VgpuEvent] {
        &self.events
    }

    /// Get all triggered alerts.
    pub fn alerts(&self) -> &[VgpuAlert] {
        &self.alerts
    }

    /// Get current instance count.
    pub fn active_count(&self) -> usize {
        self.instances
            .values()
            .filter(|m| m.instance.phase == VgpuPhase::Active)
            .count()
    }

    /// Get number of tracked contention baselines (for leak testing).
    pub fn contention_baseline_count(&self) -> usize {
        self.contention.tracked_count()
    }

    /// Get total instance count (including non-active).
    pub fn total_instance_count(&self) -> usize {
        self.instances.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::vgpu::*;

    fn make_create_event(id: &str, latency_ms: f64) -> VgpuEvent {
        VgpuEvent {
            timestamp: chrono::Utc::now(),
            event_type: VgpuEventType::Created {
                instance: VgpuInstance {
                    id: id.to_string(),
                    name: format!("Test {id}"),
                    technology: VgpuTechnology::NvidiaGrid,
                    physical_gpu_index: 0,
                    physical_pci_bus_id: None,
                    phase: VgpuPhase::Provisioning,
                    vram_allocated_bytes: 4 * 1024 * 1024 * 1024,
                    compute_fraction: None,
                    memory_fraction: 0.25,
                    mig_profile: None,
                },
                spin_up_latency_ms: Some(latency_ms),
            },
            instance_id: id.to_string(),
            snapshot: None,
        }
    }

    fn make_destroy_event(id: &str, ghost_bytes: u64) -> VgpuEvent {
        VgpuEvent {
            timestamp: chrono::Utc::now(),
            event_type: VgpuEventType::Destroyed {
                instance_id: id.to_string(),
                verification: TeardownVerification {
                    memory_reclaimed: ghost_bytes == 0,
                    expected_free_bytes: 4 * 1024 * 1024 * 1024,
                    actual_free_bytes: 4 * 1024 * 1024 * 1024 - ghost_bytes,
                    reclaim_latency_ms: 50.0,
                    ghost_allocations_bytes: ghost_bytes,
                    compute_reclaimed: ghost_bytes == 0,
                },
            },
            instance_id: id.to_string(),
            snapshot: None,
        }
    }

    #[test]
    fn test_auto_attach_on_create() {
        let mut sampler = VgpuSampler::new(VgpuMonitorConfig::default(), 16 * 1024 * 1024 * 1024);

        let event = make_create_event("v1", 100.0);
        sampler.process_event(&event);

        assert_eq!(sampler.active_count(), 1);
    }

    #[test]
    fn test_auto_detach_on_destroy() {
        let mut sampler = VgpuSampler::new(VgpuMonitorConfig::default(), 16 * 1024 * 1024 * 1024);

        sampler.process_event(&make_create_event("v1", 100.0));
        assert_eq!(sampler.active_count(), 1);

        sampler.process_event(&make_destroy_event("v1", 0));
        assert_eq!(sampler.active_count(), 0);
    }

    #[test]
    fn test_ghost_allocation_triggers_alert() {
        let mut sampler = VgpuSampler::new(VgpuMonitorConfig::default(), 16 * 1024 * 1024 * 1024);

        sampler.process_event(&make_create_event("v1", 100.0));
        let alerts = sampler.process_event(&make_destroy_event("v1", 512 * 1024 * 1024));

        assert!(alerts
            .iter()
            .any(|a| a.rule == super::super::vgpu_alerting::VgpuAlertRule::GhostAllocation));
    }

    #[test]
    fn test_run_with_simulated_events() {
        let config = VgpuMonitorConfig {
            sample_interval_secs: 1,
            ..VgpuMonitorConfig::default()
        };
        let mut sampler = VgpuSampler::new(config, 80 * 1024 * 1024 * 1024);

        let (tx, rx) = mpsc::channel();

        // Send events from another thread
        std::thread::spawn(move || {
            tx.send(make_create_event("sim-0", 120.0)).unwrap();
            tx.send(make_create_event("sim-1", 150.0)).unwrap();
            tx.send(make_destroy_event("sim-0", 0)).unwrap();
            // Drop tx to signal completion
        });

        let events = sampler.run(rx, |_event, _alerts| true);
        assert_eq!(events.len(), 3);
        assert_eq!(sampler.active_count(), 1); // sim-1 still active
    }
}
