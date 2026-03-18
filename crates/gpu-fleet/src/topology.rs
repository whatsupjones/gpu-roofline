//! Fleet topology discovery — PCIe/NVLink tree and P2P bandwidth matrix.

use gpu_harness::device::GpuDevice;
use gpu_harness::error::HarnessError;
use gpu_harness::GpuBackend;
use serde::{Deserialize, Serialize};

/// Discovered fleet topology with P2P bandwidth measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyView {
    pub gpus: Vec<GpuDevice>,
    pub p2p_bandwidth_gbps: Vec<Vec<f64>>,
}

/// Discover fleet topology: enumerate GPUs and measure P2P bandwidth.
pub fn discover_topology(backend: &dyn GpuBackend) -> Result<TopologyView, HarnessError> {
    let gpus = backend.discover_devices()?;
    let n = gpus.len();

    let mut p2p = vec![vec![0.0; n]; n];
    for (i, row) in p2p.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if i != j {
                match backend.p2p_bandwidth(i as u32, j as u32) {
                    Ok(result) => *cell = result.bandwidth_gbps,
                    Err(_) => *cell = 0.0,
                }
            }
        }
    }

    Ok(TopologyView {
        gpus,
        p2p_bandwidth_gbps: p2p,
    })
}

/// Print topology as an ASCII tree with P2P bandwidth matrix.
pub fn print_topology_tree(view: &TopologyView, _no_color: bool) {
    let n = view.gpus.len();
    println!(
        "\nFleet: {}x {}",
        n,
        view.gpus
            .first()
            .map(|g| g.name.as_str())
            .unwrap_or("Unknown")
    );

    // GPU list
    for gpu in &view.gpus {
        let arch = format!("{:?}", gpu.architecture);
        let vram_gb = gpu.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "  GPU {} | {} | {} | {:.0} GB",
            gpu.index, gpu.name, arch, vram_gb
        );
    }

    // P2P bandwidth matrix
    if n > 1
        && view
            .p2p_bandwidth_gbps
            .iter()
            .any(|row| row.iter().any(|&v| v > 0.0))
    {
        println!("\nP2P Bandwidth (GB/s):");
        print!("        ");
        for i in 0..n {
            print!("GPU{:<5}", i);
        }
        println!();

        for i in 0..n {
            print!("  GPU{} ", i);
            for j in 0..n {
                if i == j {
                    print!("  ---   ");
                } else {
                    print!("{:>6.0}  ", view.p2p_bandwidth_gbps[i][j]);
                }
            }
            println!();
        }

        // Detect interconnect type
        let max_bw = view
            .p2p_bandwidth_gbps
            .iter()
            .flat_map(|r| r.iter())
            .cloned()
            .fold(0.0_f64, f64::max);
        let interconnect = if max_bw > 400.0 {
            "NVLink"
        } else if max_bw > 20.0 {
            "PCIe P2P"
        } else {
            "None"
        };
        println!(
            "\n  Interconnect: {} (peak {:.0} GB/s)",
            interconnect, max_bw
        );
    } else if n > 1 {
        println!("\n  No P2P bandwidth measured (single-GPU or P2P not supported)");
    }
}

/// Print topology as JSON.
pub fn print_topology_json(view: &TopologyView) {
    if let Ok(json) = serde_json::to_string_pretty(view) {
        println!("{json}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_harness::sim::{fleet::SimulatedFleet, profiles, SimulatedBackend};

    #[test]
    fn test_topology_homogeneous_fleet() {
        let fleet = SimulatedFleet::homogeneous(profiles::h100_sxm(), 4);
        let backend = SimulatedBackend::with_fleet(fleet);
        let view = discover_topology(&backend).unwrap();

        assert_eq!(view.gpus.len(), 4);
        // P2P has simulated jitter, so check approximate symmetry
        for i in 0..4 {
            for j in 0..4 {
                let a = view.p2p_bandwidth_gbps[i][j];
                let b = view.p2p_bandwidth_gbps[j][i];
                if i != j {
                    // Both should be > 0 (NVLink)
                    assert!(a > 0.0, "P2P[{i}][{j}] should be > 0");
                    assert!(b > 0.0, "P2P[{j}][{i}] should be > 0");
                    // Within 10% of each other (jitter)
                    let ratio = a / b;
                    assert!(
                        (0.9..=1.1).contains(&ratio),
                        "P2P[{i}][{j}]={a:.0} vs P2P[{j}][{i}]={b:.0} differ by more than 10%"
                    );
                } else {
                    assert_eq!(a, 0.0, "self BW should be 0");
                }
            }
        }
    }

    #[test]
    fn test_topology_single_gpu() {
        let backend = SimulatedBackend::new(profiles::h100_sxm());
        let view = discover_topology(&backend).unwrap();
        assert_eq!(view.gpus.len(), 1);
        assert_eq!(view.p2p_bandwidth_gbps.len(), 1);
    }

    #[test]
    fn test_topology_consumer_fleet_pcie() {
        let fleet = SimulatedFleet::homogeneous(profiles::rtx_5090(), 2);
        let backend = SimulatedBackend::with_fleet(fleet);
        let view = discover_topology(&backend).unwrap();

        // Consumer GPUs use PCIe P2P (no NVLink)
        assert!(
            view.p2p_bandwidth_gbps[0][1] < 100.0,
            "consumer GPUs should use PCIe (< 100 GB/s)"
        );
    }
}
