#!/bin/bash
# =============================================================================
# Prospective Hardware Validation — Pre-Registered Protocol Execution
# =============================================================================
# Implements the protocol in docs/study-protocol-hardware-validation.md
# This script must be committed and tagged BEFORE execution.
# No manual modifications during execution.
# =============================================================================

set -euo pipefail

OUTDIR="${1:-prospective_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "unknown")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "============================================================================="
echo "Prospective Hardware Validation — Pre-Registered Protocol"
echo "============================================================================="
echo "GPU:      $GPU_NAME"
echo "Driver:   $DRIVER"
echo "CUDA:     $CUDA_VER"
echo "Output:   $OUTDIR"
echo "Started:  $TIMESTAMP"
echo "============================================================================="

# --- Helpers ---
ensure_clean() {
    nvidia-smi mig -dci -i 0 2>/dev/null || true
    nvidia-smi mig -dgi -i 0 2>/dev/null || true
    sleep 1
}

ensure_mig() {
    local S=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i 0)
    if [ "$S" != "Enabled" ]; then
        nvidia-smi -mig 1 -i 0 2>/dev/null
        sleep 2
    fi
}

get_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i 0; }
get_power() { nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0; }
get_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; }
get_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }
get_total() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0; }

# =============================================================================
# CATEGORY 3: Provisioning Overhead (H1)
# 30 cycles x 2 profiles = 60 trials
# =============================================================================
echo ""
echo "===== H1: Provisioning Overhead (60 trials) ====="

ensure_mig
ensure_clean

echo "[" > "$OUTDIR/h1_provisioning.json"
FIRST=true

for PROF in "19:1g.12gb" "9:3g.48gb"; do
    PID=$(echo $PROF | cut -d: -f1)
    PNAME=$(echo $PROF | cut -d: -f2)
    echo "  Profile: $PNAME (30 cycles)"

    for cycle in $(seq 1 30); do
        ensure_clean
        TEMP=$(get_temp)
        POWER=$(get_power)

        S=$(date +%s%N)
        nvidia-smi mig -cgi $PID -C -i 0 >/dev/null 2>&1
        E=$(date +%s%N)
        CREATE=$(( (E - S) / 1000000 ))

        S=$(date +%s%N)
        nvidia-smi mig -dci -i 0 >/dev/null 2>&1
        nvidia-smi mig -dgi -i 0 >/dev/null 2>&1
        E=$(date +%s%N)
        DESTROY=$(( (E - S) / 1000000 ))

        SEP=""; if [ "$FIRST" = false ]; then SEP=","; fi; FIRST=false
        echo "${SEP}{\"profile\":\"$PNAME\",\"cycle\":$cycle,\"create_ms\":$CREATE,\"destroy_ms\":$DESTROY,\"temp_c\":$TEMP,\"power_w\":$POWER,\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUTDIR/h1_provisioning.json"
    done
    echo "    Done: $PNAME"
done

echo "]" >> "$OUTDIR/h1_provisioning.json"
echo "  H1: COMPLETE"

# =============================================================================
# CATEGORY 6: Divergence Validation (H2)
# 7 measurements (n=1..7)
# =============================================================================
echo ""
echo "===== H2: Divergence Validation (7 measurements) ====="

ensure_mig
ensure_clean
sleep 2
M_OVERHEAD=$(get_used)
M_TOTAL=$(get_total)
echo "  M_overhead=$M_OVERHEAD MiB, M_total=$M_TOTAL MiB"

echo "[" > "$OUTDIR/h2_divergence.json"

for n in 1 2 3 4 5 6 7; do
    ensure_clean
    for i in $(seq 1 $n); do
        nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1
    done
    sleep 2

    NV_FREE=$(get_free)
    NV_USED=$(get_used)
    MIG_COMMITTED=$((n * 11264))
    M_AVAIL=$((M_TOTAL - MIG_COMMITTED - M_OVERHEAD))
    DIVERGENCE=$((NV_FREE - M_AVAIL))

    SEP=""; if [ $n -gt 1 ]; then SEP=","; fi
    echo "${SEP}{\"n\":$n,\"nvidia_free\":$NV_FREE,\"nvidia_used\":$NV_USED,\"nvidia_total\":$M_TOTAL,\"mig_committed\":$MIG_COMMITTED,\"m_available\":$M_AVAIL,\"divergence\":$DIVERGENCE,\"m_overhead\":$M_OVERHEAD,\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUTDIR/h2_divergence.json"

    echo "  n=$n: D=$DIVERGENCE MiB"
done

ensure_clean
echo "]" >> "$OUTDIR/h2_divergence.json"
echo "  H2: COMPLETE"

# =============================================================================
# CATEGORY 1: Ghost Allocation (H3)
# Test A: 50 capacity-based cycles
# Test B: 5 rapid-churn replications
# =============================================================================
echo ""
echo "===== H3: Ghost Allocation (50 + 5*20 trials) ====="

ensure_mig
ensure_clean

echo "{\"test_a\":[" > "$OUTDIR/h3_ghost.json"

echo "  Test A: Capacity-based (50 cycles)"
GHOSTS_A=0
for cycle in $(seq 1 50); do
    ensure_clean
    for i in $(seq 1 7); do nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1; done

    nvidia-smi mig -dci -ci 0 -gi 7 -i 0 >/dev/null 2>&1
    nvidia-smi mig -dgi -gi 7 -i 0 >/dev/null 2>&1
    sleep 1

    RECREATE=false
    if nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1; then
        RECREATE=true
        nvidia-smi mig -dci -ci 0 -gi 7 -i 0 >/dev/null 2>&1
        nvidia-smi mig -dgi -gi 7 -i 0 >/dev/null 2>&1
    else
        GHOSTS_A=$((GHOSTS_A + 1))
    fi

    SEP=""; if [ $cycle -gt 1 ]; then SEP=","; fi
    echo "${SEP}{\"cycle\":$cycle,\"recreate\":$RECREATE,\"ghost\":$([ "$RECREATE" = false ] && echo true || echo false),\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUTDIR/h3_ghost.json"

    if [ $((cycle % 10)) -eq 0 ]; then echo "    Cycle $cycle/50: $GHOSTS_A ghosts so far"; fi
done

echo "],\"test_b\":[" >> "$OUTDIR/h3_ghost.json"

echo "  Test B: Rapid churn (5 reps x 20 cycles)"
for rep in $(seq 1 5); do
    ensure_clean
    INIT=0
    for i in $(seq 1 7); do nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1 && INIT=$((INIT+1)); done
    ensure_clean

    for churn in $(seq 1 20); do
        nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1
        nvidia-smi mig -dci -i 0 >/dev/null 2>&1
        nvidia-smi mig -dgi -i 0 >/dev/null 2>&1
    done
    sleep 2

    FINAL=0
    for i in $(seq 1 7); do nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1 && FINAL=$((FINAL+1)); done
    ensure_clean

    LOSS=$((INIT - FINAL))
    SEP=""; if [ $rep -gt 1 ]; then SEP=","; fi
    echo "${SEP}{\"rep\":$rep,\"initial_max\":$INIT,\"final_max\":$FINAL,\"capacity_loss\":$LOSS,\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUTDIR/h3_ghost.json"
    echo "    Rep $rep: init=$INIT final=$FINAL loss=$LOSS"
done

echo "]}" >> "$OUTDIR/h3_ghost.json"
echo "  H3: COMPLETE (Test A ghosts: $GHOSTS_A/50)"

# =============================================================================
# CATEGORY 2: MIG Contention (H4)
# 30 paired measurements
# =============================================================================
echo ""
echo "===== H4: MIG Contention (30 paired trials) ====="

ensure_mig
ensure_clean

echo "[" > "$OUTDIR/h4_contention.json"

for trial in $(seq 1 30); do
    # Phase A: baseline (single instance)
    ensure_clean
    nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1
    UUID_A=$(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[a-f0-9-]*' | head -1)

    BW_BASE="0"; GF_BASE="0"
    if [ -n "$UUID_A" ]; then
        RESULT=$(CUDA_VISIBLE_DEVICES=$UUID_A gpu-roofline measure --burst --format json 2>/dev/null || echo '{}')
        BW_BASE=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('peak_bandwidth_gbps',0))" 2>/dev/null || echo "0")
        GF_BASE=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('peak_gflops',0))" 2>/dev/null || echo "0")
    fi

    # Phase B: with 6 co-tenants under load
    ensure_clean
    for i in $(seq 1 7); do nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1; done

    TARGET_UUID=$(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[a-f0-9-]*' | head -1)
    ALL_UUIDS=$(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[a-f0-9-]*')
    PIDS=""

    for uuid in $ALL_UUIDS; do
        if [ "$uuid" != "$TARGET_UUID" ]; then
            CUDA_VISIBLE_DEVICES=$uuid python3 -c "
import torch
x = torch.randn(1024,1024,device='cuda')
for _ in range(999999): torch.mm(x,x)
" &>/dev/null &
            PIDS="$PIDS $!"
        fi
    done
    sleep 3

    BW_CONT="0"; GF_CONT="0"
    if [ -n "$TARGET_UUID" ]; then
        RESULT=$(CUDA_VISIBLE_DEVICES=$TARGET_UUID gpu-roofline measure --burst --format json 2>/dev/null || echo '{}')
        BW_CONT=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('peak_bandwidth_gbps',0))" 2>/dev/null || echo "0")
        GF_CONT=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('peak_gflops',0))" 2>/dev/null || echo "0")
    fi

    for pid in $PIDS; do kill $pid 2>/dev/null; done
    wait 2>/dev/null || true

    SEP=""; if [ $trial -gt 1 ]; then SEP=","; fi
    echo "${SEP}{\"trial\":$trial,\"bw_baseline\":$BW_BASE,\"bw_contention\":$BW_CONT,\"gf_baseline\":$GF_BASE,\"gf_contention\":$GF_CONT,\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUTDIR/h4_contention.json"

    echo "  Trial $trial/30: base=${BW_BASE} cont=${BW_CONT}"
    ensure_clean
done

echo "]" >> "$OUTDIR/h4_contention.json"
echo "  H4: COMPLETE"

# =============================================================================
# CATEGORY 4: Burst-to-Sustained (H5)
# 15 replications of 120s dynamic roofline
# =============================================================================
echo ""
echo "===== H5: Burst-to-Sustained (15 x 120s) ====="

ensure_clean
nvidia-smi -mig 0 -i 0 >/dev/null 2>&1 || true
sleep 5

echo "[" > "$OUTDIR/h5_tension.json"

for rep in $(seq 1 15); do
    TEMP=$(get_temp)
    POWER=$(get_power)

    RESULT=$(gpu-roofline measure --format json 2>/dev/null || echo '{"error":"failed"}')

    SEP=""; if [ $rep -gt 1 ]; then SEP=","; fi
    echo "${SEP}{\"rep\":$rep,\"temp_pre\":$TEMP,\"power_pre\":$POWER,\"data\":$RESULT,\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUTDIR/h5_tension.json"

    echo "  Rep $rep/15 complete"
    sleep 30
done

echo "]" >> "$OUTDIR/h5_tension.json"
echo "  H5: COMPLETE"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================================================="
echo "PROSPECTIVE VALIDATION COMPLETE"
echo "============================================================================="

GHOST_A=$(grep -c '"ghost":true' "$OUTDIR/h3_ghost.json" 2>/dev/null || echo 0)
FINISHED=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat > "$OUTDIR/summary.json" << EOF
{
  "protocol": "study-protocol-hardware-validation.md v1.0",
  "gpu": "$GPU_NAME",
  "driver": "$DRIVER",
  "cuda": "$CUDA_VER",
  "started": "$TIMESTAMP",
  "finished": "$FINISHED",
  "hypotheses": {
    "H1_provisioning": {"trials": 60, "file": "h1_provisioning.json"},
    "H2_divergence": {"trials": 7, "file": "h2_divergence.json"},
    "H3_ghost": {"test_a_trials": 50, "test_a_ghosts": $GHOST_A, "test_b_reps": 5, "file": "h3_ghost.json"},
    "H4_contention": {"trials": 30, "file": "h4_contention.json"},
    "H5_tension": {"trials": 15, "file": "h5_tension.json"}
  }
}
EOF

echo "Ghost detections (H3 Test A): $GHOST_A / 50"
echo "Output: $OUTDIR/"
echo "Finished: $FINISHED"
ls -la "$OUTDIR/"
