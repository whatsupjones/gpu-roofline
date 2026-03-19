# Provenance

- Generated at: `2026-03-19T19:45:14+00:00`
- Protocol version: `1.0`
- Phase: `simulation`
- Archive root: `D:/study-output-v4`
- Original input path: `docs/study-results/simulation-raw.json`
- Archived raw copy: `D:/study-output-v4/raw/simulation-raw.json`
- Input SHA256: `e386a3d599a238901abff636d421dbcb095cca31616e425d05e5653744fd3912`
- Input bytes: `83712562`
- Raw generator: `gpu-harness::study::runner`
- Raw dataset seed: `42`
- Raw total trials: `120000`
- Bootstrap resamples: `10000`
- Cost bootstrap resamples: `2000`
- Seed: `42`

## Commands

- Analysis: `D:\gpu-study-venv\Scripts\python.exe D:/study-archives/gpu-waste-simulation-20260319T1130Z/source_snapshot/analysis/analyze_study.py --input docs/study-results/simulation-raw.json --output-root D:/study-output-v4 --repo-root . --bootstrap-resamples 10000 --cost-bootstrap-resamples 2000 --seed 42`
- Recommended simulation rerun: `CARGO_TARGET_DIR=D:/cargo-target/gpu-tools TMPDIR=D:/tmp cargo run -p gpu-harness --release --bin study_sim -- --out docs/study-results/simulation-raw.json`

## Software

- Python: `3.11.4 (tags/v3.11.4:d2340ef, Jun  7 2023, 05:45:37) [MSC v.1934 64 bit (AMD64)]`
- Platform: `Windows-10-10.0.19045-SP0`
- numpy: `2.4.3`
- pandas: `3.0.1`
- scipy: `1.17.1`

## Source Snapshot

- `analysis/analyze_study.py` sha256 `c984860c5f31eb3bc2871d3c97b2c2a4a5c2c43226ca22d602c196ad6cc6fdb8` (66550 bytes)
- `repo/docs/study-protocol-gpu-waste.md` sha256 `c96c7f5b74618932d578b7a661c3eac93991136f9b079a9e980000cfb20834bc` (69264 bytes)
- `repo/docs/study-simulation-manuscript.md` sha256 `04b8017572f11c8e993cb987780de4dcd9b8b491b66ccc5b2cdfad1be80ead61` (28398 bytes)
- `repo/docs/study-submission-plan.md` sha256 `f99035429d01739ce383b702bbe171902880aae79f1ba13a367d90775755ee95` (1260 bytes)
- `repo/crates/gpu-harness/src/bin/study_sim.rs` sha256 `00e0918908c671a208ba7e9bd192c8b0980d735f1acd6c9dcfa94b5ff3a17516` (2197 bytes)
- `repo/crates/gpu-harness/src/study/mod.rs` sha256 `d42eff055b1e1e0799a02de3f1c186fb92a96f426d29727ba7fa9e07712b2c85` (558 bytes)
- `repo/crates/gpu-harness/src/study/runner.rs` sha256 `3be9f6ed5e664f49df5751a3fdab8c9cbbeebd767e8b9c12b136274b6d7f9253` (34010 bytes)
- `repo/crates/gpu-harness/src/study/noise.rs` sha256 `3ffcb09be44dd6c65513a049d66cc0fb72f2410eb161da9856cb99da13a4d611` (5148 bytes)
- `repo/crates/gpu-harness/src/study/stats.rs` sha256 `2e938d79c10bcae87fe26f7134bb252727647fbc5a27c4911d657de7ecf65fe0` (14323 bytes)
- `repo/crates/gpu-harness/src/study/cost_model.rs` sha256 `a4128c5b4cb08ec0e8ba0da81d31ac2728b9e8ee59b1c8be2b9b3a019014a7f5` (7877 bytes)
- `repo/crates/gpu-harness/src/study/scenarios.rs` sha256 `61f67705edf8f8a5970ff99b7e8a560ef85f374eef1ab889c3d0dce5d6cdf0f6` (29203 bytes)

## Derived Outputs

- `derived/summary.md` sha256 `8df837642c6a419b6115e15e6847380d5e1dc3e54485ae3962a7107adedd9b69` (6751 bytes)
- `derived/supplement.md` sha256 `f62be1cbed750af43ede98ab09e09da91c6e08e61e6b27ae72c0c4a25df724b9` (22153 bytes)
- `derived/analysis-results.json` sha256 `5a88d4d6a199d966c30cd5975492ee5ab13a8227f2c4b3d0bd172057856d6576` (21118 bytes)

## Best-Practice Retention

- Keep `raw/simulation-raw.json` immutable once cited.
- Preserve this provenance file, `provenance.json`, and `SHA256SUMS.txt` together with any journal submission package.
- Treat markdown tables as derived artifacts that are reproducible from the archived raw JSON and the archived analysis script.
