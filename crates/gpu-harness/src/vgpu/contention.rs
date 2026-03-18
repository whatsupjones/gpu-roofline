//! Contention measurement for time-sliced vGPU environments.
//!
//! Records performance baselines for active vGPU instances and detects
//! squeeze when a new vGPU is provisioned on the same physical GPU.

use std::collections::HashMap;

use super::state::{PartitioningMode, VgpuSnapshot};

/// Measures contention impact on existing vGPU tenants.
pub struct ContentionMeasurer {
    /// Per-instance performance baselines (instance_id → snapshot).
    baselines: HashMap<String, VgpuSnapshot>,
    /// Threshold for reporting contention (fraction, e.g. 0.05 = 5% drop).
    threshold: f64,
}

impl ContentionMeasurer {
    pub fn new(threshold: f64) -> Self {
        Self {
            baselines: HashMap::new(),
            threshold,
        }
    }

    /// Record a baseline snapshot for an instance.
    pub fn record_baseline(&mut self, instance_id: &str, snapshot: VgpuSnapshot) {
        self.baselines.insert(instance_id.to_string(), snapshot);
    }

    /// Remove baseline when instance is destroyed.
    pub fn remove_baseline(&mut self, instance_id: &str) {
        self.baselines.remove(instance_id);
    }

    /// Compare current snapshots against baselines to detect contention.
    ///
    /// Returns `(affected_ids, bandwidth_impacts, compute_impacts)` for any
    /// instances that experienced a drop exceeding the threshold.
    /// Only meaningful for `TimeSliced` partitioning — hardware-partitioned
    /// (MIG) instances are isolated and won't show contention.
    pub fn detect_contention(
        &self,
        current_snapshots: &HashMap<String, VgpuSnapshot>,
        partitioning: PartitioningMode,
    ) -> Option<(Vec<String>, Vec<f64>, Vec<f64>)> {
        // Hardware-partitioned modes are isolated — no contention possible
        if partitioning == PartitioningMode::HardwarePartitioned {
            return None;
        }

        let mut affected = Vec::new();
        let mut bw_impacts = Vec::new();
        let mut compute_impacts = Vec::new();

        for (id, baseline) in &self.baselines {
            if let Some(current) = current_snapshots.get(id) {
                let bw_ratio = if baseline.bandwidth_gbps > 0.0 {
                    current.bandwidth_gbps / baseline.bandwidth_gbps
                } else {
                    1.0
                };
                let compute_ratio = if baseline.gflops > 0.0 {
                    current.gflops / baseline.gflops
                } else {
                    1.0
                };

                if (1.0 - bw_ratio) > self.threshold || (1.0 - compute_ratio) > self.threshold {
                    affected.push(id.clone());
                    bw_impacts.push(bw_ratio);
                    compute_impacts.push(compute_ratio);
                }
            }
        }

        if affected.is_empty() {
            None
        } else {
            Some((affected, bw_impacts, compute_impacts))
        }
    }

    /// Get the current baseline for an instance.
    pub fn baseline(&self, instance_id: &str) -> Option<&VgpuSnapshot> {
        self.baselines.get(instance_id)
    }

    /// Number of tracked baselines.
    pub fn tracked_count(&self) -> usize {
        self.baselines.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(bw: f64, gflops: f64) -> VgpuSnapshot {
        VgpuSnapshot {
            bandwidth_gbps: bw,
            gflops,
            memory_used_bytes: 1_000_000_000,
            memory_allocated_bytes: 4_000_000_000,
            utilization_pct: 80.0,
            temperature_c: 65,
            power_watts: 200.0,
            encoder_utilization_pct: None,
            decoder_utilization_pct: None,
        }
    }

    #[test]
    fn test_no_contention_when_hardware_partitioned() {
        let mut measurer = ContentionMeasurer::new(0.05);
        measurer.record_baseline("a", make_snapshot(500.0, 10000.0));

        let mut current = HashMap::new();
        current.insert("a".to_string(), make_snapshot(400.0, 8000.0)); // 20% drop

        // Should return None because MIG is hardware-partitioned
        let result = measurer.detect_contention(&current, PartitioningMode::HardwarePartitioned);
        assert!(result.is_none());
    }

    #[test]
    fn test_contention_detected_time_sliced() {
        let mut measurer = ContentionMeasurer::new(0.05);
        measurer.record_baseline("a", make_snapshot(500.0, 10000.0));
        measurer.record_baseline("b", make_snapshot(500.0, 10000.0));

        let mut current = HashMap::new();
        current.insert("a".to_string(), make_snapshot(400.0, 8000.0)); // 20% drop
        current.insert("b".to_string(), make_snapshot(490.0, 9900.0)); // 2% drop (below threshold)

        let result = measurer.detect_contention(&current, PartitioningMode::TimeSliced);
        assert!(result.is_some());
        let (affected, bw, compute) = result.unwrap();
        assert_eq!(affected.len(), 1);
        assert_eq!(affected[0], "a");
        assert!((bw[0] - 0.8).abs() < 0.01);
        assert!((compute[0] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_no_contention_within_threshold() {
        let mut measurer = ContentionMeasurer::new(0.05);
        measurer.record_baseline("a", make_snapshot(500.0, 10000.0));

        let mut current = HashMap::new();
        current.insert("a".to_string(), make_snapshot(490.0, 9800.0)); // 2% drop

        let result = measurer.detect_contention(&current, PartitioningMode::TimeSliced);
        assert!(result.is_none());
    }

    #[test]
    fn test_remove_baseline() {
        let mut measurer = ContentionMeasurer::new(0.05);
        measurer.record_baseline("a", make_snapshot(500.0, 10000.0));
        assert_eq!(measurer.tracked_count(), 1);

        measurer.remove_baseline("a");
        assert_eq!(measurer.tracked_count(), 0);
    }
}
