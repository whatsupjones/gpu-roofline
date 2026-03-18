//! Teardown verification for vGPU lifecycle monitoring.
//!
//! Captures pre-teardown GPU state, then compares post-teardown to verify
//! that resources (memory, compute) were properly reclaimed.

use std::collections::HashMap;

use super::state::TeardownVerification;

/// Pre-teardown state captured for a vGPU instance.
#[derive(Debug, Clone)]
pub struct PreTeardownState {
    /// Physical GPU memory used (bytes) before teardown.
    pub memory_used_bytes: u64,
    /// VRAM allocated to the vGPU being torn down.
    pub vram_allocated_bytes: u64,
    /// Timestamp when teardown was initiated.
    pub teardown_started: std::time::Instant,
}

/// Verifies resource reclamation after vGPU teardown.
pub struct TeardownVerifier {
    pre_teardown: HashMap<String, PreTeardownState>,
}

impl TeardownVerifier {
    pub fn new() -> Self {
        Self {
            pre_teardown: HashMap::new(),
        }
    }

    /// Capture the pre-teardown state for a vGPU that is about to be destroyed.
    pub fn capture_pre_teardown(
        &mut self,
        instance_id: &str,
        memory_used_bytes: u64,
        vram_allocated_bytes: u64,
    ) {
        self.pre_teardown.insert(
            instance_id.to_string(),
            PreTeardownState {
                memory_used_bytes,
                vram_allocated_bytes,
                teardown_started: std::time::Instant::now(),
            },
        );
    }

    /// Verify that resources were reclaimed after the vGPU disappeared.
    ///
    /// `post_memory_used_bytes` is the physical GPU's memory_used after the
    /// vGPU has been destroyed.
    pub fn verify_teardown(
        &mut self,
        instance_id: &str,
        post_memory_used_bytes: u64,
    ) -> Option<TeardownVerification> {
        let pre = self.pre_teardown.remove(instance_id)?;

        let reclaim_latency_ms = pre.teardown_started.elapsed().as_secs_f64() * 1000.0;

        let actual_free_bytes = pre.memory_used_bytes.saturating_sub(post_memory_used_bytes);
        let ghost_allocations_bytes = pre.vram_allocated_bytes.saturating_sub(actual_free_bytes);

        let memory_reclaimed = actual_free_bytes >= pre.vram_allocated_bytes;

        Some(TeardownVerification {
            memory_reclaimed,
            expected_free_bytes: pre.vram_allocated_bytes,
            actual_free_bytes,
            reclaim_latency_ms,
            ghost_allocations_bytes,
            compute_reclaimed: memory_reclaimed, // Approximate: if memory is reclaimed, compute likely is too
        })
    }

    /// Number of pending teardown verifications.
    pub fn pending_count(&self) -> usize {
        self.pre_teardown.len()
    }
}

impl Default for TeardownVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_teardown() {
        let mut verifier = TeardownVerifier::new();

        // Pre-teardown: GPU using 20GB, vGPU allocated 10GB
        verifier.capture_pre_teardown("vgpu-1", 20_000_000_000, 10_000_000_000);
        assert_eq!(verifier.pending_count(), 1);

        // Post-teardown: GPU now using 10GB (full reclamation)
        let result = verifier.verify_teardown("vgpu-1", 10_000_000_000);
        assert!(result.is_some());

        let v = result.unwrap();
        assert!(v.memory_reclaimed);
        assert_eq!(v.expected_free_bytes, 10_000_000_000);
        assert_eq!(v.actual_free_bytes, 10_000_000_000);
        assert_eq!(v.ghost_allocations_bytes, 0);
        assert!(v.compute_reclaimed);
        assert_eq!(verifier.pending_count(), 0);
    }

    #[test]
    fn test_ghost_allocation_detected() {
        let mut verifier = TeardownVerifier::new();

        verifier.capture_pre_teardown("vgpu-2", 20_000_000_000, 10_000_000_000);

        // Only 9.5GB freed instead of 10GB
        let result = verifier.verify_teardown("vgpu-2", 10_500_000_000);
        assert!(result.is_some());

        let v = result.unwrap();
        assert!(!v.memory_reclaimed);
        assert_eq!(v.actual_free_bytes, 9_500_000_000);
        assert_eq!(v.ghost_allocations_bytes, 500_000_000);
    }

    #[test]
    fn test_unknown_instance_returns_none() {
        let mut verifier = TeardownVerifier::new();
        let result = verifier.verify_teardown("nonexistent", 0);
        assert!(result.is_none());
    }
}
