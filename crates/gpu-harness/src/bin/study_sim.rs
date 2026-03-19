use gpu_harness::study::runner::{run_simulation, SimulationConfig};
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut scale = 1.0;
    let mut seed = 42u64;
    let mut output = PathBuf::from("docs/study-results/simulation-raw.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scale" => {
                i += 1;
                scale = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(42);
            }
            "--out" => {
                i += 1;
                output = args
                    .get(i)
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("docs/study-results/simulation-raw.json"));
            }
            "--help" | "-h" => {
                eprintln!("GPU Waste Study simulation runner");
                eprintln!("Usage: study_sim [--scale FACTOR] [--seed N] [--out PATH]");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    let config = SimulationConfig {
        scale,
        seed,
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let results = run_simulation(&config);
    let elapsed = start.elapsed();

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).expect("create output directory");
    }

    let json = serde_json::to_string(&results).expect("serialize simulation results");
    fs::write(&output, json).expect("write simulation results");

    eprintln!("Simulation complete");
    eprintln!("  output: {}", output.display());
    eprintln!("  total trials: {}", results.total_trials);
    eprintln!(
        "  target/category: {}",
        results.target_trials_per_category
    );
    eprintln!("  elapsed: {:.1}s", elapsed.as_secs_f64());
    for category in &results.categories {
        eprintln!(
            "  cat {} {}: {} trials",
            category.category_index, category.category, category.total_trials
        );
    }
}
