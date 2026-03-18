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
