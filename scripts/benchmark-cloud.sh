#!/bin/bash
# Cloud GPU provider benchmark script.
# Runs a standardized set of measurements for cross-provider comparison.
#
# Usage: ./benchmark-cloud.sh <provider-name>
# Example: ./benchmark-cloud.sh lambda-labs
#
# Requires gpu-roofline in PATH. Total runtime: ~10 minutes.

set -euo pipefail

PROVIDER=${1:?Usage: ./benchmark-cloud.sh <provider-name>}
OUTDIR="results/${PROVIDER}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "=== Cloud GPU Benchmark ==="
echo "Provider: $PROVIDER"
echo "Output:   $OUTDIR"
echo ""

# 1. Diagnostic: what's wrong (if anything)?
echo "[1/5] Running diagnostic..."
gpu-roofline diagnose --format json > "$OUTDIR/diagnose.json"
gpu-roofline diagnose
echo ""

# 2. Burst roofline: peak cold-start performance
echo "[2/5] Measuring burst roofline..."
gpu-roofline measure --burst --format json --save-baseline "$OUTDIR/burst.json"
echo ""

# 3. Dynamic roofline: sustained performance after thermal equilibrium
echo "[3/5] Measuring dynamic roofline (120s)..."
gpu-roofline measure --format json --save-baseline "$OUTDIR/dynamic.json"
echo ""

# 4. Validation against known baselines
echo "[4/5] Validating against known baselines..."
gpu-roofline validate --strict --format json > "$OUTDIR/validate.json"
gpu-roofline validate --strict
echo ""

# 5. Variance: three consecutive burst runs
echo "[5/5] Measuring run-to-run variance (3 burst runs)..."
for i in 1 2 3; do
  gpu-roofline measure --burst --format json --save-baseline "$OUTDIR/burst_run${i}.json"
done
echo ""

echo "=== Done ==="
echo "Results saved to $OUTDIR/"
echo ""
echo "Files:"
ls -1 "$OUTDIR/"
