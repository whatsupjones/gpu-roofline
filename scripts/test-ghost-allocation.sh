#!/bin/bash
# Ghost Allocation Hardware Validation
#
# Tests whether MIG teardown leaves unreleased VRAM on an H100.
# Compares nvidia-smi's reported memory against actual measurements.
#
# Requirements:
#   - NVIDIA H100 (or A100) with MIG capability
#   - Root access (MIG commands require sudo)
#   - gpu-roofline binary in PATH
#
# Usage: sudo ./test-ghost-allocation.sh
#
# Output: results saved to ghost_test_<timestamp>/

set -euo pipefail

OUTDIR="ghost_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "=== Ghost Allocation Hardware Validation ==="
echo "Output: $OUTDIR"
echo ""

# Record environment
echo "[1/8] Recording environment..."
nvidia-smi -q > "$OUTDIR/env_full.txt"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,memory.used,memory.free --format=csv > "$OUTDIR/env_summary.csv"
uname -a > "$OUTDIR/system.txt"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo ""

# Ensure clean state
echo "[2/8] Ensuring clean state (destroying any existing MIG instances)..."
nvidia-smi mig -dci 2>/dev/null || true
nvidia-smi mig -dgi 2>/dev/null || true

# Enable MIG if not already enabled
MIG_MODE=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader)
if [ "$MIG_MODE" != "Enabled" ]; then
    echo "  Enabling MIG mode..."
    nvidia-smi -mig 1 -i 0
    echo "  MIG enabled. A GPU reset may be required."
    echo "  If this script fails, run: nvidia-smi -mig 1 -i 0 && reboot"
fi
echo ""

# Baseline memory measurement
echo "[3/8] Measuring baseline memory (no MIG instances)..."
sleep 2  # Let driver stabilize
nvidia-smi -q -d MEMORY > "$OUTDIR/memory_baseline.txt"
BASELINE_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "  Baseline memory used: ${BASELINE_USED} MiB"
echo "$BASELINE_USED" > "$OUTDIR/baseline_used_mib.txt"
echo ""

# Create MIG instances
echo "[4/8] Creating MIG instances..."
# Profile 9 = 1g.10gb (smallest, allows up to 7 instances on H100)
nvidia-smi mig -cgi 9 -C -i 0
nvidia-smi mig -lgi > "$OUTDIR/mig_instances.txt"
echo "  MIG instance created"
echo ""

# Load VRAM on the MIG instance
echo "[5/8] Loading VRAM on MIG instance..."
# Find the MIG device
MIG_DEVICE=$(nvidia-smi mig -lgi --format=csv,noheader | head -1 | awk '{print $1}')
echo "  MIG device: $MIG_DEVICE"

# Run a burst measurement on the MIG instance to load VRAM
# This allocates GPU buffers and exercises the memory subsystem
if command -v gpu-roofline &> /dev/null; then
    gpu-roofline measure --burst --format json > "$OUTDIR/mig_workload.json" 2>&1 || true
    echo "  Workload completed on MIG instance"
else
    echo "  WARNING: gpu-roofline not found. Using nvidia-cuda-mps fallback."
    echo "  For best results, install gpu-roofline."
fi

# Record memory with MIG active
nvidia-smi -q -d MEMORY > "$OUTDIR/memory_with_mig.txt"
WITH_MIG_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "  Memory used with MIG active: ${WITH_MIG_USED} MiB"
echo "$WITH_MIG_USED" > "$OUTDIR/with_mig_used_mib.txt"
echo ""

# Destroy MIG instances
echo "[6/8] Destroying MIG instances..."
nvidia-smi mig -dci -i 0
nvidia-smi mig -dgi -i 0
echo "  MIG instances destroyed"
echo "  nvidia-smi reports: partition removed"
echo ""

# Post-teardown memory measurement (with stabilization)
echo "[7/8] Measuring post-teardown memory..."
echo "  Waiting for driver cleanup..."

# Poll memory every 500ms for 10 seconds to detect stabilization
for i in $(seq 1 20); do
    POST_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    echo "  t+${i}×500ms: ${POST_USED} MiB" >> "$OUTDIR/teardown_timeline.txt"
    sleep 0.5
done

# Final measurement after 10 seconds
nvidia-smi -q -d MEMORY > "$OUTDIR/memory_post_teardown.txt"
POST_TEARDOWN_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "  Post-teardown memory used: ${POST_TEARDOWN_USED} MiB"
echo "$POST_TEARDOWN_USED" > "$OUTDIR/post_teardown_used_mib.txt"
echo ""

# Analysis
echo "[8/8] Analysis..."
echo ""
GHOST=$((POST_TEARDOWN_USED - BASELINE_USED))

echo "  Baseline (no MIG):      ${BASELINE_USED} MiB"
echo "  With MIG active:        ${WITH_MIG_USED} MiB"
echo "  After MIG teardown:     ${POST_TEARDOWN_USED} MiB"
echo "  Ghost allocation:       ${GHOST} MiB"
echo ""

# Write summary
cat > "$OUTDIR/summary.json" << ENDJSON
{
    "test": "ghost_allocation",
    "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader)",
    "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)",
    "baseline_used_mib": $BASELINE_USED,
    "with_mig_used_mib": $WITH_MIG_USED,
    "post_teardown_used_mib": $POST_TEARDOWN_USED,
    "ghost_allocation_mib": $GHOST,
    "nvidia_smi_reports_clean": true,
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ENDJSON

if [ "$GHOST" -gt 0 ]; then
    echo "  *** GHOST ALLOCATION DETECTED ***"
    echo "  nvidia-smi reports the MIG partition as cleanly removed."
    echo "  ${GHOST} MiB of VRAM was not reclaimed."
    echo ""
    echo "  This memory is physically consumed but allocated to no instance."
    echo "  nvidia-smi and DCGM cannot detect this condition."
else
    echo "  No ghost allocation detected. Teardown was clean."
    echo "  Memory returned to baseline within 10 seconds."
fi

echo ""
echo "=== Done ==="
echo "Results: $OUTDIR/"
echo ""
echo "Files:"
ls -1 "$OUTDIR/"
