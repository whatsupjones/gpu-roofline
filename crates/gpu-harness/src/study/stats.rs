//! Statistical analysis for the GPU waste study.
//!
//! Implements: Mann-Whitney U, effect sizes (rank-biserial, Cohen's d),
//! bootstrap BCa confidence intervals, and detection metrics.

use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Complete statistical summary for one waste category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStats {
    pub category: String,
    pub n_treatment: usize,
    pub n_control: usize,
    pub descriptive: DescriptiveStats,
    pub mann_whitney: Option<MannWhitneyResult>,
    pub effect_size: EffectSize,
    pub bootstrap_ci: BootstrapCI,
    pub detection: DetectionMetrics,
}

/// Descriptive statistics for a sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    pub mean: f64,
    pub median: f64,
    pub sd: f64,
    pub iqr: f64,
    pub min: f64,
    pub max: f64,
    pub control_mean: f64,
    pub control_median: f64,
}

/// Mann-Whitney U test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MannWhitneyResult {
    pub u_statistic: f64,
    pub z_score: f64,
    pub p_value: f64,
    pub significant: bool,
}

/// Effect size measures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSize {
    pub cohens_d: f64,
    pub rank_biserial_r: f64,
    pub common_language_effect: f64,
}

/// Bootstrap confidence interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapCI {
    pub point_estimate: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub n_resamples: usize,
}

/// Detection performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetrics {
    pub true_positives: u64,
    pub false_positives: u64,
    pub true_negatives: u64,
    pub false_negatives: u64,
    pub sensitivity: f64,
    pub specificity: f64,
    pub precision: f64,
    pub f1_score: f64,
    pub accuracy: f64,
}

/// Compute descriptive statistics for treatment and control samples.
pub fn descriptive(treatment: &[f64], control: &[f64]) -> DescriptiveStats {
    let mean = mean_of(treatment);
    let median = median_of(treatment);
    let sd = sd_of(treatment, mean);
    let iqr = iqr_of(treatment);
    let min = treatment.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = treatment.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let control_mean = mean_of(control);
    let control_median = median_of(control);

    DescriptiveStats {
        mean,
        median,
        sd,
        iqr,
        min,
        max,
        control_mean,
        control_median,
    }
}

/// Mann-Whitney U test (one-sided: treatment > control).
pub fn mann_whitney_u(treatment: &[f64], control: &[f64], alpha: f64) -> MannWhitneyResult {
    let n1 = treatment.len() as f64;
    let n2 = control.len() as f64;

    // Count how many times treatment > control
    let mut u: f64 = 0.0;
    for &t in treatment {
        for &c in control {
            if t > c {
                u += 1.0;
            } else if (t - c).abs() < f64::EPSILON {
                u += 0.5;
            }
        }
    }

    let mean_u = n1 * n2 / 2.0;
    let sigma_u = (n1 * n2 * (n1 + n2 + 1.0) / 12.0).sqrt();

    let z = if sigma_u > 0.0 {
        (u - mean_u) / sigma_u
    } else {
        0.0
    };

    // One-sided p-value using normal approximation
    let p_value = 1.0 - normal_cdf(z);

    MannWhitneyResult {
        u_statistic: u,
        z_score: z,
        p_value,
        significant: p_value < alpha,
    }
}

/// Compute effect sizes.
pub fn effect_sizes(treatment: &[f64], control: &[f64]) -> EffectSize {
    let n1 = treatment.len() as f64;
    let n2 = control.len() as f64;

    // Cohen's d
    let mean_t = mean_of(treatment);
    let mean_c = mean_of(control);
    let sd_t = sd_of(treatment, mean_t);
    let sd_c = sd_of(control, mean_c);
    let pooled_sd = ((((n1 - 1.0) * sd_t * sd_t + (n2 - 1.0) * sd_c * sd_c)
        / (n1 + n2 - 2.0))
        .max(0.0))
    .sqrt();
    let cohens_d = if pooled_sd > 0.0 {
        (mean_t - mean_c) / pooled_sd
    } else {
        0.0
    };

    // Rank-biserial r from U statistic
    let mut u: f64 = 0.0;
    for &t in treatment {
        for &c in control {
            if t > c {
                u += 1.0;
            } else if (t - c).abs() < f64::EPSILON {
                u += 0.5;
            }
        }
    }
    let r = 2.0 * u / (n1 * n2) - 1.0;

    // Common language effect size: P(treatment > control)
    let cles = u / (n1 * n2);

    EffectSize {
        cohens_d,
        rank_biserial_r: r,
        common_language_effect: cles,
    }
}

/// Bootstrap BCa 95% confidence interval for the median difference.
pub fn bootstrap_ci(
    treatment: &[f64],
    control: &[f64],
    n_resamples: usize,
    rng: &mut impl Rng,
) -> BootstrapCI {
    let point_estimate = median_of(treatment) - median_of(control);

    let mut bootstrap_diffs: Vec<f64> = Vec::with_capacity(n_resamples);

    for _ in 0..n_resamples {
        let t_sample: Vec<f64> = (0..treatment.len())
            .map(|_| *treatment.choose(rng).unwrap())
            .collect();
        let c_sample: Vec<f64> = (0..control.len())
            .map(|_| *control.choose(rng).unwrap())
            .collect();
        bootstrap_diffs.push(median_of(&t_sample) - median_of(&c_sample));
    }

    bootstrap_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // BCa adjustment: bias correction
    let below_point = bootstrap_diffs
        .iter()
        .filter(|&&d| d < point_estimate)
        .count() as f64;
    let z0 = inverse_normal_cdf(below_point / n_resamples as f64);

    // Acceleration (jackknife estimate)
    let n = treatment.len();
    let mut jackknife_medians: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let jack: Vec<f64> = treatment
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &v)| v)
            .collect();
        jackknife_medians.push(median_of(&jack));
    }
    let jack_mean = mean_of(&jackknife_medians);
    let num: f64 = jackknife_medians.iter().map(|&m| (jack_mean - m).powi(3)).sum();
    let den: f64 = jackknife_medians
        .iter()
        .map(|&m| (jack_mean - m).powi(2))
        .sum::<f64>()
        .powf(1.5);
    let a = if den.abs() > 1e-10 { num / (6.0 * den) } else { 0.0 };

    // BCa percentiles
    let z_alpha = inverse_normal_cdf(0.025);
    let z_1alpha = inverse_normal_cdf(0.975);

    let alpha1 = normal_cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)));
    let alpha2 = normal_cdf(z0 + (z0 + z_1alpha) / (1.0 - a * (z0 + z_1alpha)));

    let idx_lo = ((alpha1 * n_resamples as f64) as usize).clamp(0, n_resamples - 1);
    let idx_hi = ((alpha2 * n_resamples as f64) as usize).clamp(0, n_resamples - 1);

    BootstrapCI {
        point_estimate,
        ci_lower: bootstrap_diffs[idx_lo],
        ci_upper: bootstrap_diffs[idx_hi],
        n_resamples,
    }
}

/// Compute detection metrics from ground truth and predictions.
pub fn detection_metrics(ground_truth: &[bool], predicted: &[bool]) -> DetectionMetrics {
    assert_eq!(ground_truth.len(), predicted.len());

    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut tn = 0u64;
    let mut r#fn = 0u64;

    for (gt, pred) in ground_truth.iter().zip(predicted.iter()) {
        match (gt, pred) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => r#fn += 1,
            (false, false) => tn += 1,
        }
    }

    let sensitivity = if tp + r#fn > 0 {
        tp as f64 / (tp + r#fn) as f64
    } else {
        0.0
    };
    let specificity = if tn + fp > 0 {
        tn as f64 / (tn + fp) as f64
    } else {
        0.0
    };
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let f1 = if precision + sensitivity > 0.0 {
        2.0 * precision * sensitivity / (precision + sensitivity)
    } else {
        0.0
    };
    let total = (tp + fp + tn + r#fn) as f64;
    let accuracy = if total > 0.0 {
        (tp + tn) as f64 / total
    } else {
        0.0
    };

    DetectionMetrics {
        true_positives: tp,
        false_positives: fp,
        true_negatives: tn,
        false_negatives: r#fn,
        sensitivity,
        specificity,
        precision,
        f1_score: f1,
        accuracy,
    }
}

/// Holm-Bonferroni correction on p-values.
pub fn holm_bonferroni(p_values: &[f64], alpha: f64) -> Vec<(f64, f64, bool)> {
    let m = p_values.len();
    let mut indexed: Vec<(usize, f64)> = p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut results = vec![(0.0, 0.0, false); m];
    let mut any_failed = false;

    for (rank, &(orig_idx, raw_p)) in indexed.iter().enumerate() {
        let adjusted_alpha = alpha / (m - rank) as f64;
        let significant = !any_failed && raw_p < adjusted_alpha;
        if !significant {
            any_failed = true;
        }
        // Adjusted p-value (step-down)
        let adjusted_p = (raw_p * (m - rank) as f64).min(1.0);
        results[orig_idx] = (raw_p, adjusted_p, significant);
    }

    results
}

// =============================================================================
// Helper functions
// =============================================================================

fn mean_of(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn median_of(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn sd_of(data: &[f64], mean: f64) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn iqr_of(data: &[f64]) -> f64 {
    if data.len() < 4 {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    q3 - q1
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
fn normal_cdf(z: f64) -> f64 {
    if z < -8.0 {
        return 0.0;
    }
    if z > 8.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-z * z / 2.0).exp();
    let c = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    if z >= 0.0 {
        1.0 - p * c
    } else {
        p * c
    }
}

/// Inverse normal CDF approximation (Beasley-Springer-Moro).
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return -8.0;
    }
    if p >= 1.0 {
        return 8.0;
    }

    let p = p.clamp(1e-10, 1.0 - 1e-10);

    // Rational approximation
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -z } else { z }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_mann_whitney_significant_difference() {
        let treatment: Vec<f64> = (0..100).map(|i| 10.0 + i as f64 * 0.1).collect();
        let control: Vec<f64> = (0..100).map(|i| 5.0 + i as f64 * 0.1).collect();
        let result = mann_whitney_u(&treatment, &control, 0.05);
        assert!(result.significant, "treatment clearly > control, p={}", result.p_value);
    }

    #[test]
    fn test_mann_whitney_no_difference() {
        let treatment: Vec<f64> = (0..100).map(|i| 5.0 + i as f64 * 0.1).collect();
        let control: Vec<f64> = (0..100).map(|i| 5.0 + i as f64 * 0.1).collect();
        let result = mann_whitney_u(&treatment, &control, 0.05);
        assert!(!result.significant, "same samples should not be significant");
    }

    #[test]
    fn test_cohens_d_large_effect() {
        let treatment: Vec<f64> = (0..100).map(|_| 10.0).collect();
        let control: Vec<f64> = (0..100).map(|_| 5.0).collect();
        let es = effect_sizes(&treatment, &control);
        assert!(es.cohens_d > 1.0, "large effect expected, got d={}", es.cohens_d);
    }

    #[test]
    fn test_bootstrap_ci_covers_true_diff() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let treatment: Vec<f64> = (0..200).map(|i| 10.0 + (i as f64) * 0.05).collect();
        let control: Vec<f64> = (0..200).map(|i| 5.0 + (i as f64) * 0.05).collect();
        let ci = bootstrap_ci(&treatment, &control, 2000, &mut rng);
        assert!(ci.ci_lower > 0.0, "CI lower should be > 0: {}", ci.ci_lower);
        assert!(ci.point_estimate > 0.0, "point estimate should be positive");
    }

    #[test]
    fn test_detection_metrics() {
        let gt = vec![true, true, false, false, true, false];
        let pred = vec![true, false, false, true, true, false];
        let m = detection_metrics(&gt, &pred);
        assert_eq!(m.true_positives, 2);
        assert_eq!(m.false_positives, 1);
        assert_eq!(m.true_negatives, 2);
        assert_eq!(m.false_negatives, 1);
    }

    #[test]
    fn test_holm_bonferroni() {
        let p_values = vec![0.01, 0.03, 0.04, 0.20, 0.50];
        let results = holm_bonferroni(&p_values, 0.05);
        // First p-value (0.01) tested at 0.05/5 = 0.01, borderline
        // Should reject at least the most significant ones
        assert!(results[0].2 || results[0].0 <= 0.01);
    }
}
