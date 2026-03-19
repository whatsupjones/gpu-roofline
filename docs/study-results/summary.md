# The Hidden Cost of GPU Virtualization

## Simulation Dataset

- Input: `docs/study-results/simulation-raw.json`
- Phase: `simulation`
- Protocol version: `1.0`
- Seed: `42`
- Total trials: `120000`
- Target trials/category: `20000`

## Table 1. Category-Level Statistical Results

| Category | Treatment N | Control N | Metric | Treatment Median | Control Median | Mann-Whitney U | p-value | Cohen's d | 95% CI | gpu-roofline Detect % | nvidia-smi Detect % | DCGM Detect % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ghost allocations | 10000 | 10000 | ghost_bytes_measured | 536350426.50 | 8316.00 | 99937086 | <1e-300 | 2.46 | [527280091.27, 545228612.40] | 99.9 | 0.0 | 0.0 |
| Contention squeeze | 10000 | 10000 | bandwidth_loss_pct | 66.66 | 0.00 | 100000000 | <1e-300 | 8.55 | [66.61, 66.68] | 100.0 | 0.0 | 0.0 |
| Provisioning overhead | 20000 | 20000 | spin_up_latency_ms | 246.59 | 0.50 | 400000000 | <1e-300 | 2.85 | [244.29, 248.29] | 100.0 | 0.0 | 0.0 |
| Burst-to-sustained gap | 20000 | 20000 | gap_pct | 1.69 | 0.00 | 331920000 | <1e-300 | 1.03 | [1.62, 1.77] | 56.5 | 0.0 | 0.0 |
| Straggler tax | 10000 | 10000 | straggler_tax_pct | 19.09 | 4.28 | 99805998 | <1e-300 | 2.46 | [14.56, 15.09] | 94.7 | 0.0 | 0.0 |
| Oversubscription | 10000 | 10000 | performance_degradation_pct | 33.33 | 0.00 | 100000000 | <1e-300 | 3.97 | [33.33, 33.33] | 100.0 | 0.0 | 0.0 |

## Table 2. Annual Cost Projection at $2.50/GPU/hr

| Scale | GPUs | Ghost | Contention | Provisioning | Burst gap | Straggler | Oversubscription | Total / GPU / year | Total fleet / year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_8gpu | 8 | $683 | $4,866 | $0 | $185 | $1,219 | $1,217 | $8,171 | $65,366 |
| medium_100gpu | 100 | $2,733 | $9,124 | $1 | $278 | $5,518 | $3,650 | $21,304 | $2,130,430 |
| large_10000gpu | 10000 | $3,417 | $11,470 | $3 | $308 | $13,934 | $4,867 | $33,999 | $339,985,250 |
