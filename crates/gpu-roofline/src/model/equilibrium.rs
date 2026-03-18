/// Equilibrium detector using rolling coefficient of variation (CV).
///
/// Monitors a stream of performance samples and detects when the GPU
/// has reached thermal steady-state (CV drops below threshold over
/// a configurable window).
#[derive(Debug)]
pub struct EquilibriumDetector {
    /// CV threshold — below this means stable.
    threshold: f64,
    /// Window size in number of samples.
    window_size: usize,
    /// Rolling buffer of recent measurements.
    samples: Vec<f64>,
    /// Whether equilibrium has been detected.
    detected: bool,
    /// Sample index when equilibrium was first detected.
    detection_index: Option<usize>,
    /// Total samples seen.
    total_samples: usize,
}

impl EquilibriumDetector {
    /// Create a new detector.
    ///
    /// `threshold` is the CV below which we consider the system stable.
    /// `window_size` is how many recent samples to consider.
    pub fn new(threshold: f64, window_size: usize) -> Self {
        Self {
            threshold,
            window_size: window_size.max(3), // Need at least 3 for meaningful CV
            samples: Vec::with_capacity(window_size),
            detected: false,
            detection_index: None,
            total_samples: 0,
        }
    }

    /// Add a new measurement and check for equilibrium.
    ///
    /// Returns `true` if equilibrium is detected for the first time.
    pub fn observe(&mut self, value: f64) -> bool {
        self.total_samples += 1;

        // Maintain rolling window
        if self.samples.len() >= self.window_size {
            self.samples.remove(0);
        }
        self.samples.push(value);

        // Need a full window before detecting
        if self.samples.len() < self.window_size {
            return false;
        }

        let cv = self.current_cv();

        if !self.detected && cv < self.threshold {
            self.detected = true;
            self.detection_index = Some(self.total_samples);
            return true; // First detection
        }

        false
    }

    /// Current coefficient of variation of the rolling window.
    pub fn current_cv(&self) -> f64 {
        if self.samples.len() < 2 {
            return f64::MAX;
        }

        let n = self.samples.len() as f64;
        let mean = self.samples.iter().sum::<f64>() / n;

        if mean.abs() < 1e-12 {
            return f64::MAX;
        }

        let variance = self.samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt() / mean.abs()
    }

    /// Whether equilibrium has been detected.
    pub fn is_stable(&self) -> bool {
        self.detected
    }

    /// The sample index at which equilibrium was detected, if any.
    pub fn detection_index(&self) -> Option<usize> {
        self.detection_index
    }

    /// Current mean of the rolling window.
    pub fn current_mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_signal_detects_equilibrium() {
        let mut detector = EquilibriumDetector::new(0.02, 5);

        // Feed a very stable signal
        for _ in 0..10 {
            detector.observe(100.0);
        }

        assert!(
            detector.is_stable(),
            "stable signal should reach equilibrium"
        );
        assert!(detector.current_cv() < 0.02);
    }

    #[test]
    fn test_noisy_signal_no_equilibrium() {
        let mut detector = EquilibriumDetector::new(0.02, 5);

        // Feed an alternating signal (high CV)
        for i in 0..20 {
            let val = if i % 2 == 0 { 100.0 } else { 50.0 };
            detector.observe(val);
        }

        assert!(
            !detector.is_stable(),
            "noisy signal should not reach equilibrium"
        );
    }

    #[test]
    fn test_converging_signal() {
        let mut detector = EquilibriumDetector::new(0.02, 5);

        // Start noisy, converge to stable
        let values = [
            100.0, 80.0, 90.0, 85.0, 87.0, // Noisy early phase
            86.0, 86.2, 85.9, 86.1, 86.0, // Converging
            86.0, 86.0, 86.1, 85.9, 86.0, // Stable
        ];

        let mut first_detection = None;
        for (i, &v) in values.iter().enumerate() {
            if detector.observe(v) {
                first_detection = Some(i);
            }
        }

        assert!(detector.is_stable(), "should eventually detect equilibrium");
        assert!(
            first_detection.unwrap() >= 5,
            "should not detect too early: detected at sample {}",
            first_detection.unwrap()
        );
    }

    #[test]
    fn test_needs_full_window() {
        let mut detector = EquilibriumDetector::new(0.02, 10);

        // Even a constant signal shouldn't trigger before window is full
        for _ in 0..9 {
            assert!(
                !detector.observe(100.0),
                "should not detect before window is full"
            );
        }

        // 10th sample should trigger
        assert!(detector.observe(100.0), "should detect at window size");
    }

    #[test]
    fn test_detection_index() {
        let mut detector = EquilibriumDetector::new(0.02, 3);

        detector.observe(100.0);
        detector.observe(100.0);
        assert_eq!(detector.detection_index(), None);

        detector.observe(100.0); // 3rd sample = full window
        assert!(detector.detection_index().is_some());
    }

    #[test]
    fn test_thermal_ramp_then_stabilize() {
        let mut detector = EquilibriumDetector::new(0.02, 5);

        // Simulate thermal ramp: performance decreases then stabilizes
        let ramp: Vec<f64> = (0..30)
            .map(|i| {
                let t = i as f64;
                // Exponential decay from 100 to 80, stabilizing around 80
                80.0 + 20.0 * (-t / 8.0).exp()
            })
            .collect();

        for &v in &ramp {
            detector.observe(v);
        }

        assert!(
            detector.is_stable(),
            "should detect equilibrium after thermal ramp. CV: {:.4}",
            detector.current_cv()
        );

        let mean = detector.current_mean();
        assert!(
            (mean - 80.0).abs() < 2.0,
            "equilibrium mean should be near 80, got {mean:.1}"
        );
    }
}
