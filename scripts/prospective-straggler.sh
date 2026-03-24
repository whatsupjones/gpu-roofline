#!/bin/bash
# =============================================================================
# Category 5: Straggler Tax — Pre-Registered Protocol Execution
# =============================================================================
# Implements docs/study-protocol-straggler-validation.md
# Requires: 8x GPU node with NVLink, gpu-roofline + gpu-fleet in PATH
# =============================================================================

set -euo pipefail

OUTDIR="${1:-straggler_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)

echo "============================================================================="
echo "Category 5: Straggler Tax — Pre-Registered Protocol"
echo "============================================================================="
echo "GPUs:     $GPU_COUNT x $GPU_NAME"
echo "Driver:   $DRIVER"
echo "Output:   $OUTDIR"
echo "Started:  $TIMESTAMP"
echo "============================================================================="

if [ "$GPU_COUNT" -lt 4 ]; then
    echo "ERROR: Need at least 4 GPUs. Found $GPU_COUNT."
    exit 1
fi

# --- Environment verification ---
echo ""
echo "===== Environment Verification ====="
nvidia-smi -L > "$OUTDIR/gpu_list.txt"
nvidia-smi topo -m > "$OUTDIR/topology.txt"
cat "$OUTDIR/gpu_list.txt"
echo ""
echo "Topology saved to $OUTDIR/topology.txt"

# --- Build if needed ---
if ! command -v gpu-roofline &>/dev/null; then
    echo ""
    echo "Building gpu-roofline from source..."
    cd /tmp
    git clone --depth 1 https://github.com/whatsupjones/gpu-roofline.git gpu-roofline-build 2>/dev/null || true
    cd gpu-roofline-build
    . "$HOME/.cargo/env" 2>/dev/null || true
    cargo build --release --features cuda -p gpu-roofline -p gpu-fleet 2>&1 | tail -3
    sudo cp target/release/gpu-roofline target/release/gpu-fleet /usr/local/bin/
    cd /
fi

gpu-roofline --version
gpu-fleet --version 2>/dev/null || echo "gpu-fleet not available (will use manual measurement)"

# =============================================================================
# BASELINE PHASE (10 replications)
# =============================================================================
echo ""
echo "===== Baseline Phase (10 replications) ====="

echo "[" > "$OUTDIR/h6_baseline.json"

for rep in $(seq 1 10); do
    echo "  Baseline rep $rep/10..."

    # Measure all GPUs in parallel
    RESULTS=""
    for gpu in $(seq 0 $((GPU_COUNT - 1))); do
        CUDA_VISIBLE_DEVICES=$gpu gpu-roofline measure --burst --format json 2>/dev/null > "$OUTDIR/tmp_gpu${gpu}.json" &
    done
    wait

    # Collect per-GPU results
    GPU_BWS=""
    GPU_GFS=""
    GPU_RECORDS=""
    for gpu in $(seq 0 $((GPU_COUNT - 1))); do
        BW=$(python3 -c "import json; d=json.load(open('$OUTDIR/tmp_gpu${gpu}.json')); print(f'{d[\"peak_bandwidth_gbps\"]:.1f}')" 2>/dev/null || echo "0")
        GF=$(python3 -c "import json; d=json.load(open('$OUTDIR/tmp_gpu${gpu}.json')); print(f'{d[\"peak_gflops\"]:.1f}')" 2>/dev/null || echo "0")
        GPU_BWS="$GPU_BWS $BW"
        GPU_GFS="$GPU_GFS $GF"
        if [ $gpu -gt 0 ]; then GPU_RECORDS="$GPU_RECORDS,"; fi
        GPU_RECORDS="$GPU_RECORDS{\"gpu\":$gpu,\"bw\":$BW,\"gf\":$GF}"
    done

    # Compute fleet metrics
    FLEET=$(python3 -c "
bws = [float(x) for x in '$GPU_BWS'.split()]
mn = min(bws)
md = sorted(bws)[len(bws)//2]
eff = mn * len(bws)
ideal = md * len(bws)
tax = 1 - (eff/ideal) if ideal > 0 else 0
print(f'{mn:.1f} {md:.1f} {eff:.1f} {ideal:.1f} {tax:.4f}')
" 2>/dev/null || echo "0 0 0 0 0")
    FLEET_MIN=$(echo $FLEET | awk '{print $1}')
    FLEET_MED=$(echo $FLEET | awk '{print $2}')
    FLEET_EFF=$(echo $FLEET | awk '{print $3}')
    FLEET_IDEAL=$(echo $FLEET | awk '{print $4}')
    FLEET_TAX=$(echo $FLEET | awk '{print $5}')

    # nvidia-smi utilization snapshot
    NVIDIA_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//')

    SEP=""; if [ $rep -gt 1 ]; then SEP=","; fi
    echo "${SEP}{\"rep\":$rep,\"phase\":\"baseline\",\"gpus\":[$GPU_RECORDS],\"fleet_min_bw\":$FLEET_MIN,\"fleet_median_bw\":$FLEET_MED,\"effective_throughput\":$FLEET_EFF,\"ideal_throughput\":$FLEET_IDEAL,\"straggler_tax\":$FLEET_TAX,\"nvidia_util\":[$NVIDIA_UTIL],\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUTDIR/h6_baseline.json"

    echo "    fleet_min=$FLEET_MIN fleet_med=$FLEET_MED tax=$FLEET_TAX"
    rm -f "$OUTDIR"/tmp_gpu*.json
    sleep 10
done

echo "]" >> "$OUTDIR/h6_baseline.json"
echo "  Baseline: COMPLETE"

# =============================================================================
# STRAGGLER PHASE (10 replications)
# GPU 0 runs competing workload to degrade its bandwidth
# =============================================================================
echo ""
echo "===== Straggler Phase (10 replications, GPU 0 degraded) ====="

echo "[" > "$OUTDIR/h6_straggler.json"

for rep in $(seq 1 10); do
    echo "  Straggler rep $rep/10..."

    # Launch competing bandwidth workload on GPU 0
    CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch, signal
x = torch.randn(8192, 8192, device='cuda')
while True: torch.mm(x, x); torch.cuda.synchronize()
" &>/dev/null &
    STRAGGLER_PID=$!
    sleep 3

    # Measure all GPUs in parallel
    for gpu in $(seq 0 $((GPU_COUNT - 1))); do
        CUDA_VISIBLE_DEVICES=$gpu gpu-roofline measure --burst --format json 2>/dev/null > "$OUTDIR/tmp_gpu${gpu}.json" &
    done
    wait $! 2>/dev/null || true  # Wait for measurements, not straggler process

    # Small sleep to ensure all measurement processes complete
    sleep 2

    # Kill straggler workload
    kill $STRAGGLER_PID 2>/dev/null
    wait $STRAGGLER_PID 2>/dev/null || true

    # Collect per-GPU results
    GPU_BWS=""
    GPU_GFS=""
    GPU_RECORDS=""
    for gpu in $(seq 0 $((GPU_COUNT - 1))); do
        BW=$(python3 -c "import json; d=json.load(open('$OUTDIR/tmp_gpu${gpu}.json')); print(f'{d[\"peak_bandwidth_gbps\"]:.1f}')" 2>/dev/null || echo "0")
        GF=$(python3 -c "import json; d=json.load(open('$OUTDIR/tmp_gpu${gpu}.json')); print(f'{d[\"peak_gflops\"]:.1f}')" 2>/dev/null || echo "0")
        GPU_BWS="$GPU_BWS $BW"
        GPU_GFS="$GPU_GFS $GF"
        if [ $gpu -gt 0 ]; then GPU_RECORDS="$GPU_RECORDS,"; fi
        GPU_RECORDS="$GPU_RECORDS{\"gpu\":$gpu,\"bw\":$BW,\"gf\":$GF}"
    done

    # Compute fleet metrics
    FLEET=$(python3 -c "
bws = [float(x) for x in '$GPU_BWS'.split()]
mn = min(bws)
md = sorted(bws)[len(bws)//2]
eff = mn * len(bws)
ideal = md * len(bws)
tax = 1 - (eff/ideal) if ideal > 0 else 0
print(f'{mn:.1f} {md:.1f} {eff:.1f} {ideal:.1f} {tax:.4f}')
" 2>/dev/null || echo "0 0 0 0 0")
    FLEET_MIN=$(echo $FLEET | awk '{print $1}')
    FLEET_MED=$(echo $FLEET | awk '{print $2}')
    FLEET_EFF=$(echo $FLEET | awk '{print $3}')
    FLEET_IDEAL=$(echo $FLEET | awk '{print $4}')
    FLEET_TAX=$(echo $FLEET | awk '{print $5}')

    # nvidia-smi utilization snapshot
    NVIDIA_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//')

    SEP=""; if [ $rep -gt 1 ]; then SEP=","; fi
    echo "${SEP}{\"rep\":$rep,\"phase\":\"straggler\",\"degraded_gpu\":0,\"gpus\":[$GPU_RECORDS],\"fleet_min_bw\":$FLEET_MIN,\"fleet_median_bw\":$FLEET_MED,\"effective_throughput\":$FLEET_EFF,\"ideal_throughput\":$FLEET_IDEAL,\"straggler_tax\":$FLEET_TAX,\"nvidia_util\":[$NVIDIA_UTIL],\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUTDIR/h6_straggler.json"

    echo "    fleet_min=$FLEET_MIN fleet_med=$FLEET_MED tax=$FLEET_TAX"
    rm -f "$OUTDIR"/tmp_gpu*.json
    sleep 10
done

echo "]" >> "$OUTDIR/h6_straggler.json"
echo "  Straggler: COMPLETE"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================================================="
echo "STRAGGLER VALIDATION COMPLETE"
echo "============================================================================="

FINISHED=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat > "$OUTDIR/summary.json" << EOF
{
  "protocol": "study-protocol-straggler-validation.md v1.0",
  "gpu": "$GPU_NAME",
  "gpu_count": $GPU_COUNT,
  "driver": "$DRIVER",
  "started": "$TIMESTAMP",
  "finished": "$FINISHED",
  "baseline_reps": 10,
  "straggler_reps": 10,
  "degraded_gpu": 0,
  "degradation_method": "competing_bandwidth_workload"
}
EOF

echo "Output: $OUTDIR/"
echo "Finished: $FINISHED"
ls -la "$OUTDIR/"
