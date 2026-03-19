# Submission Plan

## Phase 1: Simulation Paper

Primary target:

- The Journal of Supercomputing

Positioning:

- simulation and methodology paper
- reproducibility-first
- taxonomy plus benchmarking framework
- scenario-based economics

Required package before submission:

1. `simulation-raw.json` frozen and hash-locked
2. `summary.md` and `supplement.md`
3. `analysis-results.json`
4. `PROVENANCE.md`, `provenance.json`, and `SHA256SUMS.txt`
5. manuscript draft and cover letter

## Phase 2: Hardware Validation Paper

Stretch target:

- IEEE TPDS

Positioning:

- bare-metal validation of the six-category framework
- simulation-to-hardware calibration
- stronger claims about observability and economics

## Decision Rule

Combine the studies into one paper only if the hardware phase can be completed quickly and cleanly enough to avoid delaying publication of the simulation package. Otherwise:

1. publish the simulation/methodology paper first
2. publish the hardware validation as a follow-on paper that cites the first

## Risks To Watch

- overclaiming from simulation-only evidence
- reviewer pushback on modeled tool visibility
- cost model assumptions presented without enough caveat
- insufficient provenance for raw data and derived outputs
