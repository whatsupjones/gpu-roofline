//! Batch runner: executes raw simulation trials for each study category.
//!
//! The simulation phase intentionally emits per-trial JSON so downstream
//! statistical analysis can happen in Python, not inside the Rust binary.

use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::noise::NoiseModel;
use super::scenarios::*;

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub scale: f64,
    pub seed: u64,
    pub target_trials_per_category: usize,
    pub ghost_detection_threshold: u64,
    pub contention_threshold: f64,
    pub straggler_threshold: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            seed: 42,
            target_trials_per_category: 20_000,
            ghost_detection_threshold: 1024 * 1024,
            contention_threshold: 0.05,
            straggler_threshold: 0.90,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialRecord {
    pub trial_id: u32,
    pub category: u8,
    pub category_name: String,
    pub arm: String,
    pub condition: String,
    pub primary_metric_name: String,
    pub primary_metric_value: f64,
    pub control_metric_name: Option<String>,
    pub control_metric_value: Option<f64>,
    pub independent_variables: serde_json::Value,
    pub dependent_variables: serde_json::Value,
    pub ground_truth_waste: bool,
    pub gpu_roofline_detected: bool,
    pub nvidia_smi_detected: bool,
    pub dcgm_detected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryResult {
    pub category: String,
    pub category_index: u8,
    pub total_trials: usize,
    pub treatment_observations: usize,
    pub control_observations: usize,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResults {
    pub protocol_version: String,
    pub phase: String,
    pub generator: String,
    pub seed: u64,
    pub target_trials_per_category: usize,
    pub total_trials: usize,
    pub categories: Vec<CategoryResult>,
    pub trials: Vec<TrialRecord>,
}

pub fn run_simulation(config: &SimulationConfig) -> SimulationResults {
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let noise = NoiseModel::default();
    let mut next_trial_id = 1u32;

    let mut categories = Vec::new();
    let mut trials = Vec::new();

    let (cat1, mut cat1_trials) = run_ghost_category(config, &noise, &mut rng, &mut next_trial_id);
    categories.push(cat1);
    trials.append(&mut cat1_trials);

    let (cat2, mut cat2_trials) =
        run_contention_category(config, &noise, &mut rng, &mut next_trial_id);
    categories.push(cat2);
    trials.append(&mut cat2_trials);

    let (cat3, mut cat3_trials) =
        run_provision_category(config, &noise, &mut rng, &mut next_trial_id);
    categories.push(cat3);
    trials.append(&mut cat3_trials);

    let (cat4, mut cat4_trials) =
        run_burst_sustained_category(config, &noise, &mut rng, &mut next_trial_id);
    categories.push(cat4);
    trials.append(&mut cat4_trials);

    let (cat5, mut cat5_trials) =
        run_straggler_category(config, &noise, &mut rng, &mut next_trial_id);
    categories.push(cat5);
    trials.append(&mut cat5_trials);

    let (cat6, mut cat6_trials) =
        run_oversub_category(config, &noise, &mut rng, &mut next_trial_id);
    categories.push(cat6);
    trials.append(&mut cat6_trials);

    SimulationResults {
        protocol_version: "1.0".to_string(),
        phase: "simulation".to_string(),
        generator: "gpu-harness::study::runner".to_string(),
        seed: config.seed,
        target_trials_per_category: scaled_target(config),
        total_trials: trials.len(),
        categories,
        trials,
    }
}

fn scaled_target(config: &SimulationConfig) -> usize {
    ((config.target_trials_per_category as f64 * config.scale).round() as usize).max(1)
}

fn split_target(total: usize) -> (usize, usize) {
    let treatment = total / 2;
    let control = total - treatment;
    (treatment, control)
}

fn distribute_evenly(total: usize, cells: usize) -> Vec<usize> {
    if cells == 0 {
        return Vec::new();
    }
    let base = total / cells;
    let remainder = total % cells;
    (0..cells)
        .map(|idx| if idx < remainder { base + 1 } else { base })
        .collect()
}

fn next_id(next_trial_id: &mut u32) -> u32 {
    let id = *next_trial_id;
    *next_trial_id += 1;
    id
}

fn run_ghost_category(
    config: &SimulationConfig,
    noise: &NoiseModel,
    rng: &mut impl Rng,
    next_trial_id: &mut u32,
) -> (CategoryResult, Vec<TrialRecord>) {
    let target = scaled_target(config);
    let (treatment_target, control_target) = split_target(target);

    let treatment_cells: Vec<(TeardownMethod, MigProfile)> = TeardownMethod::all()
        .iter()
        .flat_map(|method| MigProfile::all().iter().map(move |profile| (*method, *profile)))
        .collect();
    let treatment_counts = distribute_evenly(treatment_target, treatment_cells.len());
    let control_counts = distribute_evenly(control_target, MigProfile::all().len());

    let mut trials = Vec::with_capacity(target);
    let mut conditions = Vec::new();

    for ((method, profile), count) in treatment_cells.iter().zip(treatment_counts.iter()) {
        let condition = format!("{}_{}", method.name(), profile.name());
        conditions.push(condition.clone());

        for _ in 0..*count {
            let params = ghost_trial_params(*method, *profile, false, rng);
            let result =
                run_ghost_trial(&params, noise, config.ghost_detection_threshold, rng);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::GhostAllocation.index(),
                category_name: WasteCategory::GhostAllocation.name().to_string(),
                arm: "treatment".to_string(),
                condition: condition.clone(),
                primary_metric_name: "ghost_bytes_measured".to_string(),
                primary_metric_value: result.ghost_bytes_measured as f64,
                control_metric_name: None,
                control_metric_value: None,
                independent_variables: json!({
                    "teardown_method": params.teardown_method,
                    "mig_profile": params.mig_profile,
                    "vram_allocated_bytes": params.vram_allocated_bytes,
                }),
                dependent_variables: json!({
                    "ghost_bytes_injected": params.ghost_bytes_injected,
                    "ghost_bytes_measured": result.ghost_bytes_measured,
                    "reclaim_latency_ms": result.reclaim_latency_ms,
                    "detection_threshold_bytes": result.detection_threshold_bytes,
                }),
                ground_truth_waste: params.ghost_bytes_injected > 0,
                gpu_roofline_detected: result.detected,
                nvidia_smi_detected: false,
                dcgm_detected: false,
            });
        }
    }

    for (profile, count) in MigProfile::all().iter().zip(control_counts.iter()) {
        let condition = format!("control_{}", profile.name());
        conditions.push(condition.clone());

        for _ in 0..*count {
            let params = ghost_trial_params(TeardownMethod::Clean, *profile, true, rng);
            let result =
                run_ghost_trial(&params, noise, config.ghost_detection_threshold, rng);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::GhostAllocation.index(),
                category_name: WasteCategory::GhostAllocation.name().to_string(),
                arm: "control".to_string(),
                condition: condition.clone(),
                primary_metric_name: "ghost_bytes_measured".to_string(),
                primary_metric_value: result.ghost_bytes_measured as f64,
                control_metric_name: None,
                control_metric_value: None,
                independent_variables: json!({
                    "teardown_method": params.teardown_method,
                    "mig_profile": params.mig_profile,
                    "vram_allocated_bytes": params.vram_allocated_bytes,
                }),
                dependent_variables: json!({
                    "ghost_bytes_injected": params.ghost_bytes_injected,
                    "ghost_bytes_measured": result.ghost_bytes_measured,
                    "reclaim_latency_ms": result.reclaim_latency_ms,
                    "detection_threshold_bytes": result.detection_threshold_bytes,
                }),
                ground_truth_waste: false,
                gpu_roofline_detected: result.detected,
                nvidia_smi_detected: false,
                dcgm_detected: false,
            });
        }
    }

    (
        CategoryResult {
            category: WasteCategory::GhostAllocation.name().to_string(),
            category_index: WasteCategory::GhostAllocation.index(),
            total_trials: trials.len(),
            treatment_observations: treatment_target,
            control_observations: control_target,
            conditions,
        },
        trials,
    )
}

fn run_contention_category(
    config: &SimulationConfig,
    noise: &NoiseModel,
    rng: &mut impl Rng,
    next_trial_id: &mut u32,
) -> (CategoryResult, Vec<TrialRecord>) {
    let target = scaled_target(config);
    let (treatment_target, control_target) = split_target(target);

    let treatment_tenants = [2u32, 3, 4];
    let control_timesliced_baseline = [1u32];
    let control_mig = [1u32, 2, 3, 4];

    let treatment_counts = distribute_evenly(treatment_target, treatment_tenants.len());
    let control_baseline_counts =
        distribute_evenly(control_target / 2, control_timesliced_baseline.len());
    let remaining_control = control_target - control_baseline_counts.iter().sum::<usize>();
    let control_mig_counts = distribute_evenly(remaining_control, control_mig.len());

    let mut trials = Vec::with_capacity(target);
    let mut conditions = Vec::new();

    for (&tenant_count, &count) in treatment_tenants.iter().zip(treatment_counts.iter()) {
        let condition = format!("timesliced_{}_tenants", tenant_count);
        conditions.push(condition.clone());

        for _ in 0..count {
            let params = ContentionTrialParams {
                tenant_count,
                partitioning_mode: "TimeSliced".to_string(),
                is_hardware_partitioned: false,
            };
            let result =
                run_contention_trial(&params, noise, config.contention_threshold, rng);
            let bandwidth_loss_pct = ((1.0 - result.mean_bandwidth_ratio) * 100.0).max(0.0);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::ContentionSqueeze.index(),
                category_name: WasteCategory::ContentionSqueeze.name().to_string(),
                arm: "treatment".to_string(),
                condition: condition.clone(),
                primary_metric_name: "bandwidth_loss_pct".to_string(),
                primary_metric_value: bandwidth_loss_pct,
                control_metric_name: Some("baseline_bandwidth_loss_pct".to_string()),
                control_metric_value: Some(0.0),
                independent_variables: json!({
                    "tenant_count": params.tenant_count,
                    "partitioning_mode": params.partitioning_mode,
                }),
                dependent_variables: json!({
                    "bandwidth_ratios": result.bandwidth_ratios,
                    "compute_ratios": result.compute_ratios,
                    "mean_bandwidth_ratio": result.mean_bandwidth_ratio,
                    "mean_compute_ratio": result.mean_compute_ratio,
                    "contention_threshold": result.contention_threshold,
                }),
                ground_truth_waste: true,
                gpu_roofline_detected: result.detected,
                nvidia_smi_detected: false,
                dcgm_detected: false,
            });
        }
    }

    for (&tenant_count, &count) in control_timesliced_baseline
        .iter()
        .zip(control_baseline_counts.iter())
    {
        let condition = format!("timesliced_{}_tenants", tenant_count);
        conditions.push(condition.clone());

        for _ in 0..count {
            let params = ContentionTrialParams {
                tenant_count,
                partitioning_mode: "TimeSliced".to_string(),
                is_hardware_partitioned: false,
            };
            let result =
                run_contention_trial(&params, noise, config.contention_threshold, rng);
            let bandwidth_loss_pct = ((1.0 - result.mean_bandwidth_ratio) * 100.0).max(0.0);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::ContentionSqueeze.index(),
                category_name: WasteCategory::ContentionSqueeze.name().to_string(),
                arm: "control".to_string(),
                condition: condition.clone(),
                primary_metric_name: "bandwidth_loss_pct".to_string(),
                primary_metric_value: bandwidth_loss_pct,
                control_metric_name: Some("baseline_bandwidth_loss_pct".to_string()),
                control_metric_value: Some(0.0),
                independent_variables: json!({
                    "tenant_count": params.tenant_count,
                    "partitioning_mode": params.partitioning_mode,
                }),
                dependent_variables: json!({
                    "bandwidth_ratios": result.bandwidth_ratios,
                    "compute_ratios": result.compute_ratios,
                    "mean_bandwidth_ratio": result.mean_bandwidth_ratio,
                    "mean_compute_ratio": result.mean_compute_ratio,
                    "contention_threshold": result.contention_threshold,
                }),
                ground_truth_waste: false,
                gpu_roofline_detected: result.detected,
                nvidia_smi_detected: false,
                dcgm_detected: false,
            });
        }
    }

    for (&tenant_count, &count) in control_mig.iter().zip(control_mig_counts.iter()) {
        let condition = format!("mig_{}_tenants", tenant_count);
        conditions.push(condition.clone());

        for _ in 0..count {
            let params = ContentionTrialParams {
                tenant_count,
                partitioning_mode: "HardwarePartitioned".to_string(),
                is_hardware_partitioned: true,
            };
            let result =
                run_contention_trial(&params, noise, config.contention_threshold, rng);
            let bandwidth_loss_pct = ((1.0 - result.mean_bandwidth_ratio) * 100.0).max(0.0);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::ContentionSqueeze.index(),
                category_name: WasteCategory::ContentionSqueeze.name().to_string(),
                arm: "control".to_string(),
                condition: condition.clone(),
                primary_metric_name: "bandwidth_loss_pct".to_string(),
                primary_metric_value: bandwidth_loss_pct,
                control_metric_name: Some("baseline_bandwidth_loss_pct".to_string()),
                control_metric_value: Some(0.0),
                independent_variables: json!({
                    "tenant_count": params.tenant_count,
                    "partitioning_mode": params.partitioning_mode,
                }),
                dependent_variables: json!({
                    "bandwidth_ratios": result.bandwidth_ratios,
                    "compute_ratios": result.compute_ratios,
                    "mean_bandwidth_ratio": result.mean_bandwidth_ratio,
                    "mean_compute_ratio": result.mean_compute_ratio,
                    "contention_threshold": result.contention_threshold,
                }),
                ground_truth_waste: false,
                gpu_roofline_detected: result.detected,
                nvidia_smi_detected: false,
                dcgm_detected: false,
            });
        }
    }

    (
        CategoryResult {
            category: WasteCategory::ContentionSqueeze.name().to_string(),
            category_index: WasteCategory::ContentionSqueeze.index(),
            total_trials: trials.len(),
            treatment_observations: treatment_target,
            control_observations: control_target,
            conditions,
        },
        trials,
    )
}

fn run_provision_category(
    config: &SimulationConfig,
    noise: &NoiseModel,
    rng: &mut impl Rng,
    next_trial_id: &mut u32,
) -> (CategoryResult, Vec<TrialRecord>) {
    let target = scaled_target(config);
    let cells = provision_valid_cells();
    let counts = distribute_evenly(target, cells.len());

    let mut trials = Vec::with_capacity(target);
    let mut conditions = Vec::new();

    for ((profile, load_state, concurrent_partitions), count) in cells.iter().zip(counts.iter()) {
        let condition = format!(
            "{}_{}_{}",
            profile.name(),
            load_state.name(),
            concurrent_partitions
        );
        conditions.push(condition.clone());

        let base_latency = profile.base_latency_ms() + load_state.latency_penalty_ms();

        for _ in 0..*count {
            let params = ProvisionTrialParams {
                profile: profile.name().to_string(),
                load_state: load_state.name().to_string(),
                concurrent_partitions: *concurrent_partitions,
                base_latency_ms: base_latency,
            };
            let result = run_provision_trial(&params, noise, rng);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::ProvisioningOverhead.index(),
                category_name: WasteCategory::ProvisioningOverhead.name().to_string(),
                arm: "paired".to_string(),
                condition: condition.clone(),
                primary_metric_name: "spin_up_latency_ms".to_string(),
                primary_metric_value: result.spin_up_latency_ms,
                control_metric_name: Some("nvidia_smi_reported_ms".to_string()),
                control_metric_value: Some(result.nvidia_smi_reported_ms),
                independent_variables: json!({
                    "profile": params.profile,
                    "load_state": params.load_state,
                    "concurrent_partitions": params.concurrent_partitions,
                }),
                dependent_variables: json!({
                    "spin_up_latency_ms": result.spin_up_latency_ms,
                    "nvidia_smi_reported_ms": result.nvidia_smi_reported_ms,
                    "dead_time_fraction": result.dead_time_fraction,
                }),
                ground_truth_waste: true,
                gpu_roofline_detected: result.detected,
                nvidia_smi_detected: false,
                dcgm_detected: false,
            });
        }
    }

    (
        CategoryResult {
            category: WasteCategory::ProvisioningOverhead.name().to_string(),
            category_index: WasteCategory::ProvisioningOverhead.index(),
            total_trials: trials.len(),
            treatment_observations: trials.len(),
            control_observations: trials.len(),
            conditions,
        },
        trials,
    )
}

fn run_burst_sustained_category(
    config: &SimulationConfig,
    noise: &NoiseModel,
    rng: &mut impl Rng,
    next_trial_id: &mut u32,
) -> (CategoryResult, Vec<TrialRecord>) {
    let target = scaled_target(config);
    let profiles = burst_sustained_profiles();
    let mut cells = Vec::new();
    for (profile_name, profile) in profiles {
        for workload in WorkloadType::all() {
            cells.push((profile_name, profile.clone(), *workload));
        }
    }
    let counts = distribute_evenly(target, cells.len());

    let mut trials = Vec::with_capacity(target);
    let mut conditions = Vec::new();

    for ((profile_name, profile, workload), count) in cells.iter().zip(counts.iter()) {
        let condition = format!("{}_{}", profile_name, workload.name());
        conditions.push(condition.clone());

        for _ in 0..*count {
            let params = BurstSustainedTrialParams {
                gpu_profile: (*profile_name).to_string(),
                workload_type: workload.name().to_string(),
                intensity: workload.intensity(),
            };
            let result = run_burst_sustained_trial(&params, profile, noise, rng);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::BurstSustainedGap.index(),
                category_name: WasteCategory::BurstSustainedGap.name().to_string(),
                arm: "paired".to_string(),
                condition: condition.clone(),
                primary_metric_name: "gap_pct".to_string(),
                primary_metric_value: result.gap_pct,
                control_metric_name: Some("ideal_gap_pct".to_string()),
                control_metric_value: Some(0.0),
                independent_variables: json!({
                    "gpu_profile": params.gpu_profile,
                    "workload_type": params.workload_type,
                    "intensity": params.intensity,
                }),
                dependent_variables: json!({
                    "burst_gflops": result.burst_gflops,
                    "sustained_gflops": result.sustained_gflops,
                    "burst_bandwidth_gbps": result.burst_bandwidth_gbps,
                    "sustained_bandwidth_gbps": result.sustained_bandwidth_gbps,
                    "burst_clock_mhz": result.burst_clock_mhz,
                    "sustained_clock_mhz": result.sustained_clock_mhz,
                    "burst_temp_c": result.burst_temp_c,
                    "sustained_temp_c": result.sustained_temp_c,
                    "gap_pct": result.gap_pct,
                    "equilibrium_time_secs": result.equilibrium_time_secs,
                }),
                ground_truth_waste: true,
                gpu_roofline_detected: result.detected,
                nvidia_smi_detected: false,
                dcgm_detected: false,
            });
        }
    }

    (
        CategoryResult {
            category: WasteCategory::BurstSustainedGap.name().to_string(),
            category_index: WasteCategory::BurstSustainedGap.index(),
            total_trials: trials.len(),
            treatment_observations: trials.len(),
            control_observations: trials.len(),
            conditions,
        },
        trials,
    )
}

fn run_straggler_category(
    config: &SimulationConfig,
    noise: &NoiseModel,
    rng: &mut impl Rng,
    next_trial_id: &mut u32,
) -> (CategoryResult, Vec<TrialRecord>) {
    let target = scaled_target(config);
    let (treatment_target, control_target) = split_target(target);
    let fleet_sizes = [8u32, 32, 128];
    let severities = [0u8, 1, 2];

    let mut treatment_cells: Vec<(StragglerDegradationType, u8, u32)> = Vec::new();
    for degradation_type in StragglerDegradationType::all() {
        for severity in severities {
            for fleet_size in fleet_sizes {
                treatment_cells.push((*degradation_type, severity, fleet_size));
            }
        }
    }
    let treatment_counts = distribute_evenly(treatment_target, treatment_cells.len());
    let control_counts = distribute_evenly(control_target, fleet_sizes.len());

    let mut trials = Vec::with_capacity(target);
    let mut conditions = Vec::new();

    for (&fleet_size, &count) in fleet_sizes.iter().zip(control_counts.iter()) {
        let condition = format!("control_fleet{}", fleet_size);
        conditions.push(condition.clone());

        for _ in 0..count {
            let params = StragglerTrialParams {
                fleet_size,
                degradation_type: "none".to_string(),
                severity: 0,
                is_control: true,
            };
            let result =
                run_straggler_trial(&params, noise, config.straggler_threshold, rng);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::StragglerTax.index(),
                category_name: WasteCategory::StragglerTax.name().to_string(),
                arm: "control".to_string(),
                condition: condition.clone(),
                primary_metric_name: "straggler_tax_pct".to_string(),
                primary_metric_value: result.straggler_tax_pct,
                control_metric_name: None,
                control_metric_value: None,
                independent_variables: json!({
                    "fleet_size": params.fleet_size,
                    "degradation_type": params.degradation_type,
                    "severity": params.severity,
                }),
                dependent_variables: json!({
                    "fleet_median_gflops": result.fleet_median_gflops,
                    "straggler_gflops": result.straggler_gflops,
                    "effective_fleet_throughput": result.effective_fleet_throughput,
                    "ideal_fleet_throughput": result.ideal_fleet_throughput,
                    "straggler_tax_pct": result.straggler_tax_pct,
                    "detection_threshold": result.detection_threshold,
                }),
                ground_truth_waste: false,
                gpu_roofline_detected: result.straggler_detected,
                nvidia_smi_detected: false,
                dcgm_detected: false,
            });
        }
    }

    for ((degradation_type, severity, fleet_size), count) in
        treatment_cells.iter().zip(treatment_counts.iter())
    {
        let condition = format!(
            "{}_sev{}_fleet{}",
            degradation_type.name(),
            severity,
            fleet_size
        );
        conditions.push(condition.clone());

        for _ in 0..*count {
            let params = StragglerTrialParams {
                fleet_size: *fleet_size,
                degradation_type: degradation_type.name().to_string(),
                severity: *severity,
                is_control: false,
            };
            let result =
                run_straggler_trial(&params, noise, config.straggler_threshold, rng);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::StragglerTax.index(),
                category_name: WasteCategory::StragglerTax.name().to_string(),
                arm: "treatment".to_string(),
                condition: condition.clone(),
                primary_metric_name: "straggler_tax_pct".to_string(),
                primary_metric_value: result.straggler_tax_pct,
                control_metric_name: None,
                control_metric_value: None,
                independent_variables: json!({
                    "fleet_size": params.fleet_size,
                    "degradation_type": params.degradation_type,
                    "severity": params.severity,
                }),
                dependent_variables: json!({
                    "fleet_median_gflops": result.fleet_median_gflops,
                    "straggler_gflops": result.straggler_gflops,
                    "effective_fleet_throughput": result.effective_fleet_throughput,
                    "ideal_fleet_throughput": result.ideal_fleet_throughput,
                    "straggler_tax_pct": result.straggler_tax_pct,
                    "detection_threshold": result.detection_threshold,
                }),
                ground_truth_waste: true,
                gpu_roofline_detected: result.straggler_detected,
                nvidia_smi_detected: false,
                dcgm_detected: false,
            });
        }
    }

    (
        CategoryResult {
            category: WasteCategory::StragglerTax.name().to_string(),
            category_index: WasteCategory::StragglerTax.index(),
            total_trials: trials.len(),
            treatment_observations: treatment_target,
            control_observations: control_target,
            conditions,
        },
        trials,
    )
}

fn run_oversub_category(
    config: &SimulationConfig,
    noise: &NoiseModel,
    rng: &mut impl Rng,
    next_trial_id: &mut u32,
) -> (CategoryResult, Vec<TrialRecord>) {
    let target = scaled_target(config);
    let (treatment_target, control_target) = split_target(target);

    let h100_vram_bytes: u64 = 80 * 1024 * 1024 * 1024;
    let control_instance_counts = [2u32, 4, 8, 16];
    let treatment_ratios = [1.25, 1.5, 2.0];
    let treatment_instance_counts = [2u32, 4, 8, 16];

    let control_counts = distribute_evenly(control_target, control_instance_counts.len());
    let treatment_cells: Vec<(f64, u32)> = treatment_ratios
        .iter()
        .flat_map(|ratio| {
            treatment_instance_counts
                .iter()
                .map(move |instance_count| (*ratio, *instance_count))
        })
        .collect();
    let treatment_counts = distribute_evenly(treatment_target, treatment_cells.len());

    let mut trials = Vec::with_capacity(target);
    let mut conditions = Vec::new();

    for (&instance_count, &count) in control_instance_counts.iter().zip(control_counts.iter()) {
        let condition = format!("ratio1.00_instances{}", instance_count);
        conditions.push(condition.clone());

        for _ in 0..count {
            let params = OversubTrialParams {
                overcommit_ratio: 1.0,
                instance_count,
                physical_vram_bytes: h100_vram_bytes,
            };
            let result = run_oversub_trial(&params, noise, rng);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::Oversubscription.index(),
                category_name: WasteCategory::Oversubscription.name().to_string(),
                arm: "control".to_string(),
                condition: condition.clone(),
                primary_metric_name: "performance_degradation_pct".to_string(),
                primary_metric_value: result.performance_degradation_pct,
                control_metric_name: None,
                control_metric_value: None,
                independent_variables: json!({
                    "overcommit_ratio": params.overcommit_ratio,
                    "instance_count": params.instance_count,
                    "physical_vram_bytes": params.physical_vram_bytes,
                }),
                dependent_variables: json!({
                    "total_allocated_bytes": result.total_allocated_bytes,
                    "physical_vram_bytes": result.physical_vram_bytes,
                    "actual_overcommit_ratio": result.actual_overcommit_ratio,
                    "allocation_failure_rate": result.allocation_failure_rate,
                    "performance_degradation_pct": result.performance_degradation_pct,
                }),
                ground_truth_waste: false,
                gpu_roofline_detected: result.gpu_roofline_detected,
                nvidia_smi_detected: result.nvidia_smi_detected,
                dcgm_detected: result.dcgm_detected,
            });
        }
    }

    for ((ratio, instance_count), count) in treatment_cells.iter().zip(treatment_counts.iter()) {
        let condition = format!("ratio{ratio:.2}_instances{instance_count}");
        conditions.push(condition.clone());

        for _ in 0..*count {
            let params = OversubTrialParams {
                overcommit_ratio: *ratio,
                instance_count: *instance_count,
                physical_vram_bytes: h100_vram_bytes,
            };
            let result = run_oversub_trial(&params, noise, rng);

            trials.push(TrialRecord {
                trial_id: next_id(next_trial_id),
                category: WasteCategory::Oversubscription.index(),
                category_name: WasteCategory::Oversubscription.name().to_string(),
                arm: "treatment".to_string(),
                condition: condition.clone(),
                primary_metric_name: "performance_degradation_pct".to_string(),
                primary_metric_value: result.performance_degradation_pct,
                control_metric_name: None,
                control_metric_value: None,
                independent_variables: json!({
                    "overcommit_ratio": params.overcommit_ratio,
                    "instance_count": params.instance_count,
                    "physical_vram_bytes": params.physical_vram_bytes,
                }),
                dependent_variables: json!({
                    "total_allocated_bytes": result.total_allocated_bytes,
                    "physical_vram_bytes": result.physical_vram_bytes,
                    "actual_overcommit_ratio": result.actual_overcommit_ratio,
                    "allocation_failure_rate": result.allocation_failure_rate,
                    "performance_degradation_pct": result.performance_degradation_pct,
                }),
                ground_truth_waste: true,
                gpu_roofline_detected: result.gpu_roofline_detected,
                nvidia_smi_detected: result.nvidia_smi_detected,
                dcgm_detected: result.dcgm_detected,
            });
        }
    }

    (
        CategoryResult {
            category: WasteCategory::Oversubscription.name().to_string(),
            category_index: WasteCategory::Oversubscription.index(),
            total_trials: trials.len(),
            treatment_observations: treatment_target,
            control_observations: control_target,
            conditions,
        },
        trials,
    )
}
