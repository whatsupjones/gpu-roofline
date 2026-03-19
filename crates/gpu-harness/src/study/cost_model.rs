//! Parametric cost model from study protocol Section 13.
//!
//! Projects aggregate annual waste at 3 fleet scales using
//! measured effect sizes from the simulation phase.

use serde::{Deserialize, Serialize};

/// Fleet scale for cost projections.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FleetScale {
    Small,  // 8 GPUs (DGX)
    Medium, // 100 GPUs (startup)
    Large,  // 10,000 GPUs (cloud provider)
}

impl FleetScale {
    pub fn all() -> &'static [FleetScale] {
        &[FleetScale::Small, FleetScale::Medium, FleetScale::Large]
    }

    pub fn gpu_count(&self) -> u32 {
        match self {
            FleetScale::Small => 8,
            FleetScale::Medium => 100,
            FleetScale::Large => 10_000,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            FleetScale::Small => "small_8gpu",
            FleetScale::Medium => "medium_100gpu",
            FleetScale::Large => "large_10000gpu",
        }
    }
}

/// Parameters for the cost model, per fleet scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModelParams {
    pub fleet_scale: String,
    pub gpu_count: u32,
    pub gpu_hourly_rate: f64,

    // Cat 1: Ghost allocations
    pub ghost_rate: f64,     // Fraction of teardowns with ghosts
    pub avg_ghost_frac: f64, // avg_ghost_bytes / total_gpu_bytes
    pub teardowns_per_day: f64,
    pub ghost_lifetime_hours: f64,

    // Cat 2: Contention squeeze
    pub contention_drop_pct: f64, // Bandwidth loss from contention
    pub avg_tenants: f64,
    pub active_tenant_hours_per_day: f64,

    // Cat 3: Provisioning overhead
    pub provisions_per_day: f64,
    pub avg_spin_up_secs: f64,

    // Cat 4: Burst-to-sustained gap
    pub burst_sustained_gap_pct: f64,
    pub sustained_hours_per_day: f64,

    // Cat 5: Straggler tax
    pub straggler_tax_pct: f64,
    pub training_job_hours: f64,
    pub jobs_per_day: f64,

    // Cat 6: Oversubscription
    pub oversub_waste_pct: f64,
    pub oversubscribed_hours_per_day: f64,
}

/// Cost projection for one fleet scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostProjection {
    pub fleet_scale: String,
    pub gpu_count: u32,
    pub w_ghost: f64,
    pub w_contention: f64,
    pub w_provisioning: f64,
    pub w_burst_gap: f64,
    pub w_straggler: f64,
    pub w_oversub: f64,
    pub total_per_gpu: f64,
    pub total_fleet: f64,
}

impl CostModelParams {
    /// Default parameters for a given fleet scale, with measured values plugged in.
    pub fn for_scale(
        scale: FleetScale,
        ghost_rate: f64,
        avg_ghost_frac: f64,
        contention_drop_pct: f64,
        avg_spin_up_secs: f64,
        burst_sustained_gap_pct: f64,
        straggler_tax_pct: f64,
        oversub_waste_pct: f64,
    ) -> Self {
        let (
            teardowns,
            provisions,
            tenants,
            tenant_hours,
            sustained_hours,
            training_hours,
            jobs,
            oversub_hours,
            ghost_lifetime,
        ) = match scale {
            FleetScale::Small => (5.0, 5.0, 2.0, 16.0, 12.0, 8.0, 1.0, 4.0, 24.0),
            FleetScale::Medium => (20.0, 20.0, 4.0, 20.0, 18.0, 16.0, 2.0, 12.0, 24.0),
            FleetScale::Large => (50.0, 50.0, 7.0, 22.0, 20.0, 20.0, 4.0, 16.0, 12.0),
        };

        Self {
            fleet_scale: scale.name().to_string(),
            gpu_count: scale.gpu_count(),
            gpu_hourly_rate: 2.50,
            ghost_rate,
            avg_ghost_frac,
            teardowns_per_day: teardowns,
            ghost_lifetime_hours: ghost_lifetime,
            contention_drop_pct,
            avg_tenants: tenants,
            active_tenant_hours_per_day: tenant_hours,
            provisions_per_day: provisions,
            avg_spin_up_secs,
            burst_sustained_gap_pct,
            sustained_hours_per_day: sustained_hours,
            straggler_tax_pct,
            training_job_hours: training_hours,
            jobs_per_day: jobs,
            oversub_waste_pct,
            oversubscribed_hours_per_day: oversub_hours,
        }
    }

    /// Compute cost projection from parameters.
    pub fn project(&self) -> CostProjection {
        // W_ghost = ghost_rate * avg_ghost_frac * teardowns/day * 365 * ghost_lifetime * rate
        let w_ghost = self.ghost_rate
            * self.avg_ghost_frac
            * self.teardowns_per_day
            * 365.0
            * self.ghost_lifetime_hours
            * self.gpu_hourly_rate;

        // W_contention = contention_drop * tenant_hours * 365 * rate * (1 - 1/tenants)
        let w_contention = (self.contention_drop_pct / 100.0)
            * self.active_tenant_hours_per_day
            * 365.0
            * self.gpu_hourly_rate
            * (1.0 - 1.0 / self.avg_tenants);

        // W_provisioning = provisions/day * spin_up_secs / 3600 * rate * 365
        let w_provisioning = self.provisions_per_day
            * (self.avg_spin_up_secs / 3600.0)
            * self.gpu_hourly_rate
            * 365.0;

        // W_burst_gap = gap_pct * sustained_hours * 365 * rate
        let w_burst_gap = (self.burst_sustained_gap_pct / 100.0)
            * self.sustained_hours_per_day
            * 365.0
            * self.gpu_hourly_rate;

        // W_straggler = tax_pct * training_hours * jobs/day * 365 * (N-1)/N * rate
        let fleet_factor = (self.gpu_count as f64 - 1.0) / self.gpu_count as f64;
        let w_straggler = (self.straggler_tax_pct / 100.0)
            * self.training_job_hours
            * self.jobs_per_day
            * 365.0
            * fleet_factor
            * self.gpu_hourly_rate;

        // W_oversub = waste_pct * oversub_hours * 365 * rate
        let w_oversub = (self.oversub_waste_pct / 100.0)
            * self.oversubscribed_hours_per_day
            * 365.0
            * self.gpu_hourly_rate;

        let total_per_gpu =
            w_ghost + w_contention + w_provisioning + w_burst_gap + w_straggler + w_oversub;

        CostProjection {
            fleet_scale: self.fleet_scale.clone(),
            gpu_count: self.gpu_count,
            w_ghost,
            w_contention,
            w_provisioning,
            w_burst_gap,
            w_straggler,
            w_oversub,
            total_per_gpu,
            total_fleet: total_per_gpu * self.gpu_count as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_projection_positive() {
        let params = CostModelParams::for_scale(
            FleetScale::Medium,
            0.03,  // 3% ghost rate
            0.003, // 0.3% of GPU capacity
            15.0,  // 15% contention drop
            0.3,   // 300ms spin-up
            20.0,  // 20% burst-sustained gap
            5.0,   // 5% straggler tax
            8.0,   // 8% oversub waste
        );
        let proj = params.project();
        assert!(proj.total_per_gpu > 0.0, "total waste should be positive");
        assert!(proj.total_fleet > proj.total_per_gpu, "fleet > per-gpu");
        assert!(proj.w_burst_gap > 0.0, "burst gap waste should be positive");
    }

    #[test]
    fn test_fleet_scaling() {
        let small =
            CostModelParams::for_scale(FleetScale::Small, 0.03, 0.003, 15.0, 0.3, 20.0, 5.0, 8.0)
                .project();
        let medium =
            CostModelParams::for_scale(FleetScale::Medium, 0.03, 0.003, 15.0, 0.3, 20.0, 5.0, 8.0)
                .project();
        let large =
            CostModelParams::for_scale(FleetScale::Large, 0.03, 0.003, 15.0, 0.3, 20.0, 5.0, 8.0)
                .project();

        assert!(
            large.total_fleet > medium.total_fleet,
            "large fleet > medium"
        );
        assert!(
            medium.total_fleet > small.total_fleet,
            "medium fleet > small"
        );
    }
}
