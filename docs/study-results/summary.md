# The Hidden Cost of GPU Virtualization

## Simulation Dataset

- Input: `D:/study-output-v2/raw/simulation-raw.json`
- Phase: `simulation`
- Protocol version: `1.0`
- Seed: `42`
- Total trials: `120000`
- Target trials/category: `20000`
- Claim framing: simulation evidence, modeled tool visibility, and scenario-based economic impact. Hardware validation remains future work.

## Table 1. Annual Cost of Invisible GPU Waste by Category and Fleet Scale

| Waste Category | Effect Size (d/d_z) | Waste per Event | Events/Day/GPU | Annual $/GPU | 8-GPU Fleet | 100-GPU Fleet | 10,000-GPU Fleet |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ghost allocations | 2.46 | 511.5 MiB | 20.00 | $2,733 | $5,466 | $273,321 | $34,165,064 |
| Contention squeeze | 8.55 | 66.66% | 0.83 | $9,124 | $38,929 | $912,394 | $114,700,996 |
| Provisioning overhead | 2.01 | 246.09 ms | 20.00 | $1 | $2 | $125 | $31,190 |
| Burst-to-sustained gap | 0.73 | 1.69% | 0.75 | $278 | $1,481 | $27,762 | $3,084,702 |
| Straggler tax | 2.46 | 14.81% | 2.00 | $5,518 | $9,755 | $551,828 | $139,336,568 |
| Oversubscription | 3.97 | 33.33% | 0.50 | $3,650 | $9,733 | $365,000 | $48,666,667 |
| Total |  |  |  | $21,304 | $65,366 | $2,130,430 | $339,985,187 |

## Table 2. Detection Rates by Tool Across All 6 Waste Categories

| Category | nvidia-smi Detection Rate | DCGM Detection Rate | gpu-roofline Detection Rate | McNemar p (smi vs roofline) | McNemar p (dcgm vs roofline) |
| --- | --- | --- | --- | --- | --- |
| Ghost allocations | 0.0 | 0.0 | 99.9 | <1e-300 | <1e-300 |
| Contention squeeze | 0.0 | 0.0 | 100.0 | <1e-300 | <1e-300 |
| Provisioning overhead | 0.0 | 0.0 | 100.0 | <1e-300 | <1e-300 |
| Burst-to-sustained gap | 0.0 | 0.0 | 56.5 | <1e-300 | <1e-300 |
| Straggler tax | 0.0 | 0.0 | 94.7 | <1e-300 | <1e-300 |
| Oversubscription | 0.0 | 0.0 | 100.0 | <1e-300 | <1e-300 |
| Overall | 0.0 | 0.0 | 88.4 | <1e-300 | <1e-300 |

## Table 3. Cross-Category Omnibus Statistical Tests

| Category | N (a) | N (b) | Design | Primary Test | Metric | Median (a) | Median (b) | Statistic | Raw p-value | Holm p-value | Cohen's d / d_z | Rank-biserial r | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ghost allocations | 10000 | 10000 | independent | Mann-Whitney U | ghost_bytes_measured | 536350426.50 | 8316.00 | 99937086 | <1e-300 | <1e-300 | 2.46 | 1.00 | [527280091.27, 545228612.40] |
| Contention squeeze | 10000 | 10000 | independent | Mann-Whitney U | bandwidth_loss_pct | 66.66 | 0.00 | 100000000 | <1e-300 | <1e-300 | 8.55 | 1.00 | [66.61, 66.68] |
| Provisioning overhead | 20000 | 20000 | paired | Wilcoxon signed-rank | spin_up_latency_ms | 246.59 | 0.50 | 200010000 | <1e-300 | <1e-300 | 2.01 | 1.00 | [244.32, 248.26] |
| Burst-to-sustained gap | 20000 | 20000 | paired | Wilcoxon signed-rank | gap_pct | 1.69 | 0.00 | 87021028 | <1e-300 | <1e-300 | 0.73 | 1.00 | [1.63, 1.77] |
| Straggler tax | 10000 | 10000 | independent | Mann-Whitney U | straggler_tax_pct | 19.09 | 4.28 | 99805998 | <1e-300 | <1e-300 | 2.46 | 1.00 | [14.56, 15.09] |
| Oversubscription | 10000 | 10000 | independent | Mann-Whitney U | performance_degradation_pct | 33.33 | 0.00 | 100000000 | <1e-300 | <1e-300 | 3.97 | 1.00 | [33.33, 33.33] |

## Cost Model Bootstrap Confidence Intervals

| Scale | Point estimate / GPU / year | 95% CI low / GPU / year | 95% CI high / GPU / year |
| --- | --- | --- | --- |
| 8-GPU Fleet | $8,171 | $8,149 | $8,192 |
| 100-GPU Fleet | $21,304 | $21,215 | $21,389 |
| 10,000-GPU Fleet | $33,999 | $33,809 | $34,193 |

## Interpretation Guardrails

- These costs are scenario-model outputs derived from simulation, not direct hardware measurements.
- The raw data, derived tables, hashes, and execution provenance are preserved in this package for auditability.
- Category-specific primary tests and sensitivity analyses are reported in `derived/supplement.md`.
