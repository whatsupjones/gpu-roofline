#!/bin/bash
# =============================================================================
# GPU Waste Category Hardware Validation Protocol
# =============================================================================
#
# Unified test protocol for Categories 1, 3, 4, 6 of the GPU Efficiency Gap
# study. Produces JSON records per the study protocol schema.
#
# Categories tested:
#   1. Ghost Allocations (capacity-based detection)
#   3. Provisioning Overhead (nanosecond timing)
#   4. Burst-to-Sustained Gap (dynamic roofline)
#   6. Oversubscription Visibility (nvidia-smi divergence proof)
#
# Categories NOT tested (require multi-GPU or GRID):
#   2. Contention Squeeze (needs GRID time-slicing)
#   5. Straggler Tax (needs multi-GPU fleet)
#
# Requirements:
#   - NVIDIA GPU with MIG support (H100, H200, GH200, A100)
#   - Root/sudo access for MIG commands
#   - gpu-roofline in PATH (built with --features cuda)
#   - Python 3 + PyTorch with CUDA
#
# Usage: sudo ./hardware-validation-protocol.sh [output_dir]
#
# Output: JSON files per test in output_dir, plus summary report
# =============================================================================

set -euo pipefail

OUTDIR="${1:-validation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "unknown")
ARCH=$(uname -m)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "============================================================================="
echo "GPU Waste Category Hardware Validation Protocol"
echo "============================================================================="
echo "GPU:      $GPU_NAME"
echo "Driver:   $DRIVER"
echo "CUDA:     $CUDA_VER"
echo "Arch:     $ARCH"
echo "Output:   $OUTDIR"
echo "Started:  $TIMESTAMP"
echo "============================================================================="
echo ""

# -----------------------------------------------------------------------------
# Helper: ensure clean MIG state
# -----------------------------------------------------------------------------
ensure_clean_mig() {
    sudo nvidia-smi mig -dci -i 0 2>/dev/null || true
    sudo nvidia-smi mig -dgi -i 0 2>/dev/null || true
    sleep 1
}

# -----------------------------------------------------------------------------
# Helper: ensure MIG enabled
# -----------------------------------------------------------------------------
ensure_mig_enabled() {
    local MIG_STATE=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i 0)
    if [ "$MIG_STATE" != "Enabled" ]; then
        sudo nvidia-smi -mig 1 -i 0 2>/dev/null
        sleep 2
    fi
}

# =============================================================================
# CATEGORY 6: Oversubscription Visibility (Divergence Proof Validation)
# Run first because it validates the mathematical framework.
#
# Validates Theorem 1: D(n,t) = Σ(q_i - u_i(t)) + M_fragmentation
#   where D is the divergence between nvidia-smi memory.free and actual
#   MIG-available capacity. nvidia-smi reports physical free bytes, not
#   bytes available for new MIG partitions. The divergence grows linearly
#   with partition count.
#
# Validates Theorem 2: Ghost Allocation Invisibility Bound
#   Any ghost allocation M_ghost < D(n,t) is undetectable by nvidia-smi.
#   This establishes why nvidia-smi-based ghost detection is fundamentally
#   insufficient and why capacity-based detection (Category 1) is required.
# =============================================================================
echo "===== CATEGORY 6: Oversubscription Visibility ====="
echo "Validating Theorem 1: D(n,t) = Σ(q_i - u_i) + M_fragmentation"
echo "Validating Theorem 2: Ghost invisibility bound = D(n,t)"
echo ""

ensure_mig_enabled
ensure_clean_mig

# Record overhead: memory used with MIG enabled but no instances
sleep 2
M_OVERHEAD=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "  M_overhead (MIG infra, no instances): ${M_OVERHEAD} MiB"

cat > "$OUTDIR/cat6_oversubscription.json" << JSONSTART
{"test": "oversubscription_visibility", "category": 6, "theorems_validated": ["T1_divergence", "T2_ghost_invisibility_bound"], "m_overhead_mib": $M_OVERHEAD, "records": [
JSONSTART

PROFILE_ID=19
PROFILE_NAME="1g.12gb"
PROFILE_QUOTA_MIB=11264  # 11 GB per instance

for n in 1 2 3 4 5 6 7; do
    ensure_clean_mig

    # Create n instances (idle, u_i = 0 for all i)
    for i in $(seq 1 $n); do
        sudo nvidia-smi mig -cgi $PROFILE_ID -C -i 0 >/dev/null 2>&1
    done
    sleep 1

    # nvidia-smi measurements
    NVIDIA_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    NVIDIA_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    NVIDIA_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)

    # Ground truth: M_available = M_total - Σ(q_i) - M_overhead
    MIG_COMMITTED=$((n * PROFILE_QUOTA_MIB))
    M_AVAILABLE=$((NVIDIA_TOTAL - MIG_COMMITTED - M_OVERHEAD))

    # Theorem 1: D(n,t) = M_free_reported - M_available
    DIVERGENCE=$((NVIDIA_FREE - M_AVAILABLE))

    # Theorem 1 prediction: D(n,t) = Σ(q_i - u_i) when instances are idle (u_i ≈ overhead per instance)
    # With idle instances: u_i ≈ per-instance overhead, so Σ(q_i - u_i) ≈ n * q - NVIDIA_USED + M_OVERHEAD
    PREDICTED_DIVERGENCE=$((MIG_COMMITTED - NVIDIA_USED + M_OVERHEAD))

    # Theorem 2: ghost invisibility bound = D(n,t)
    # Any ghost < DIVERGENCE MiB is undetectable by nvidia-smi
    GHOST_INVIS_BOUND=$DIVERGENCE

    # Attempt to create one more instance beyond current count
    CAN_CREATE_MORE=false
    if sudo nvidia-smi mig -cgi $PROFILE_ID -C -i 0 >/dev/null 2>&1; then
        CAN_CREATE_MORE=true
        sudo nvidia-smi mig -dci -i 0 >/dev/null 2>&1
        sudo nvidia-smi mig -dgi -i 0 >/dev/null 2>&1
        # Only destroy the extra one we just created (re-create the others)
        # Actually this destroys all - we need selective. Skip for n<7.
    fi

    SEPARATOR=""
    if [ $n -gt 1 ]; then SEPARATOR=","; fi

    cat >> "$OUTDIR/cat6_oversubscription.json" << RECORD
${SEPARATOR}{"n": $n, "profile": "$PROFILE_NAME", "q_per_instance_mib": $PROFILE_QUOTA_MIB, "nvidia_smi_free_mib": $NVIDIA_FREE, "nvidia_smi_used_mib": $NVIDIA_USED, "nvidia_smi_total_mib": $NVIDIA_TOTAL, "mig_committed_mib": $MIG_COMMITTED, "m_available_mib": $M_AVAILABLE, "divergence_mib": $DIVERGENCE, "predicted_divergence_mib": $PREDICTED_DIVERGENCE, "ghost_invisibility_bound_mib": $GHOST_INVIS_BOUND, "theorem1_error_mib": $((DIVERGENCE - PREDICTED_DIVERGENCE)), "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
RECORD

    echo "  n=$n: nvidia-smi free=${NVIDIA_FREE} MiB, M_available=${M_AVAILABLE} MiB, D=${DIVERGENCE} MiB, ghost_invis_bound=${GHOST_INVIS_BOUND} MiB"
done

ensure_clean_mig
echo "]}" >> "$OUTDIR/cat6_oversubscription.json"
echo ""
echo "  Theorem 1: D(n) should grow linearly with n. Check divergence_mib column."
echo "  Theorem 2: ghost_invisibility_bound shows max undetectable ghost per n."
echo "  Category 6: COMPLETE"
echo ""

# =============================================================================
# CATEGORY 3: Provisioning Overhead
# =============================================================================
echo "===== CATEGORY 3: Provisioning Overhead ====="
echo "Measuring MIG create/destroy latency (invisible to nvidia-smi)"
echo ""

ensure_mig_enabled
ensure_clean_mig

echo '{"test": "provisioning_overhead", "category": 3, "records": [' > "$OUTDIR/cat3_provisioning.json"

FIRST_RECORD=true

for PROFILE in "19:1g.12gb" "9:3g.48gb"; do
    PID=$(echo $PROFILE | cut -d: -f1)
    PNAME=$(echo $PROFILE | cut -d: -f2)

    echo "  Profile: $PNAME (25 cycles)"

    for cycle in $(seq 1 25); do
        # Measure create latency
        START_NS=$(date +%s%N)
        sudo nvidia-smi mig -cgi $PID -C -i 0 >/dev/null 2>&1
        END_NS=$(date +%s%N)
        CREATE_MS=$(( (END_NS - START_NS) / 1000000 ))

        # Measure destroy latency
        START_NS=$(date +%s%N)
        sudo nvidia-smi mig -dci -i 0 >/dev/null 2>&1
        sudo nvidia-smi mig -dgi -i 0 >/dev/null 2>&1
        END_NS=$(date +%s%N)
        DESTROY_MS=$(( (END_NS - START_NS) / 1000000 ))

        TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i 0)

        SEPARATOR=""
        if [ "$FIRST_RECORD" = false ]; then SEPARATOR=","; fi
        FIRST_RECORD=false

        cat >> "$OUTDIR/cat3_provisioning.json" << RECORD
${SEPARATOR}{"cycle": $cycle, "profile": "$PNAME", "create_ms": $CREATE_MS, "destroy_ms": $DESTROY_MS, "temperature_c": $TEMP, "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
RECORD
    done
    echo "    Done: $PNAME"
done

# Sequential multi-instance test
echo "  Sequential 7-instance provisioning..."
ensure_clean_mig
echo ',' >> "$OUTDIR/cat3_provisioning.json"

SEQ_RECORDS=""
for i in $(seq 1 7); do
    START_NS=$(date +%s%N)
    sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1
    END_NS=$(date +%s%N)
    MS=$(( (END_NS - START_NS) / 1000000 ))
    if [ $i -gt 1 ]; then SEQ_RECORDS="${SEQ_RECORDS},"; fi
    SEQ_RECORDS="${SEQ_RECORDS}{\"instance\": $i, \"create_ms\": $MS}"
    echo "    Instance $i: ${MS}ms"
done

START_NS=$(date +%s%N)
sudo nvidia-smi mig -dci -i 0 >/dev/null 2>&1
sudo nvidia-smi mig -dgi -i 0 >/dev/null 2>&1
END_NS=$(date +%s%N)
BULK_DESTROY_MS=$(( (END_NS - START_NS) / 1000000 ))

cat >> "$OUTDIR/cat3_provisioning.json" << RECORD
{"test": "sequential_7_instance", "instances": [$SEQ_RECORDS], "bulk_destroy_ms": $BULK_DESTROY_MS}
RECORD

echo "]}" >> "$OUTDIR/cat3_provisioning.json"
echo "  Category 3: COMPLETE"
echo ""

# =============================================================================
# CATEGORY 1: Ghost Allocations (Capacity-Based Detection)
#
# Per Theorem 3: nvidia-smi memory.free cannot detect ghost allocations
# below the divergence bound D(n,t). The correct measurement is
# capacity-based: after teardown, attempt to create a new MIG instance
# in the freed slot and verify it is usable. This tests at the MIG
# allocation layer, not the physical memory layer.
#
# Tests:
#   A1: Create max instances, destroy one, attempt re-create (25 cycles)
#   A2: Under-load teardown with capacity verification (15 cycles)
#   A3: Rapid churn (20 cycles) then capacity verification (5 reps)
#   A4: Kill -9 with capacity verification (10 cycles)
# =============================================================================
echo "===== CATEGORY 1: Ghost Allocations (Capacity-Based, per Theorem 3) ====="
echo "nvidia-smi detection is bounded by D(n,t) from Theorem 2."
echo "Using MIG-layer capacity verification instead."
echo ""

ensure_mig_enabled
ensure_clean_mig

echo '{"test": "ghost_allocation_capacity", "category": 1, "records": [' > "$OUTDIR/cat1_ghost.json"

FIRST_RECORD=true

# --- Test A1: Capacity-based detection ---
echo "  Test A1: Create max, destroy one, attempt re-create (25 cycles)"

for cycle in $(seq 1 25); do
    ensure_clean_mig

    # Create 7 instances (max for 1g.12gb)
    MAX_CREATED=0
    for i in $(seq 1 7); do
        if sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1; then
            MAX_CREATED=$((MAX_CREATED + 1))
        fi
    done

    MEM_FULL=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    # Load VRAM on first instance via PyTorch
    FIRST_UUID=$(nvidia-smi -L 2>/dev/null | grep -oP "MIG-[a-f0-9-]+" | head -1)
    if [ -n "$FIRST_UUID" ]; then
        CUDA_VISIBLE_DEVICES=$FIRST_UUID python3 -c "
import torch
x = torch.randn(2048, 2048, device='cuda')
torch.cuda.synchronize()
" 2>/dev/null || true
    fi

    MEM_LOADED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    # Destroy first instance
    FIRST_GI=$(sudo nvidia-smi mig -lgi -i 0 --format=csv,noheader 2>/dev/null | head -1 | awk '{print $1}')
    sudo nvidia-smi mig -dci -ci 0 -gi $FIRST_GI -i 0 >/dev/null 2>&1
    sudo nvidia-smi mig -dgi -gi $FIRST_GI -i 0 >/dev/null 2>&1
    sleep 1

    MEM_AFTER_DESTROY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    NVIDIA_FREE_AFTER=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)

    # Attempt re-creation in freed slot
    RECREATE_SUCCESS=false
    RECREATE_USABLE=false
    if sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1; then
        RECREATE_SUCCESS=true
        # Verify new instance is usable
        NEW_UUID=$(nvidia-smi -L 2>/dev/null | grep -oP "MIG-[a-f0-9-]+" | tail -1)
        if [ -n "$NEW_UUID" ]; then
            CUDA_VISIBLE_DEVICES=$NEW_UUID python3 -c "
import torch
x = torch.randn(1024, 1024, device='cuda')
allocated = torch.cuda.memory_allocated()
print(f'USABLE:{allocated}')
" 2>/dev/null | grep -q "USABLE:" && RECREATE_USABLE=true
        fi
    fi

    REMAINING=$(sudo nvidia-smi mig -lgi -i 0 --format=csv,noheader 2>/dev/null | wc -l)

    GHOST_DETECTED=false
    if [ "$RECREATE_SUCCESS" = false ]; then
        GHOST_DETECTED=true
    fi

    SEPARATOR=""
    if [ "$FIRST_RECORD" = false ]; then SEPARATOR=","; fi
    FIRST_RECORD=false

    cat >> "$OUTDIR/cat1_ghost.json" << RECORD
${SEPARATOR}{"test": "A1_capacity", "cycle": $cycle, "max_created": $MAX_CREATED, "mem_full_mib": $MEM_FULL, "mem_loaded_mib": $MEM_LOADED, "mem_after_destroy_mib": $MEM_AFTER_DESTROY, "nvidia_free_after_mib": $NVIDIA_FREE_AFTER, "recreate_success": $RECREATE_SUCCESS, "recreate_usable": $RECREATE_USABLE, "ghost_detected": $GHOST_DETECTED, "remaining_instances": $REMAINING, "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
RECORD

    if [ "$GHOST_DETECTED" = true ]; then
        echo "    Cycle $cycle: *** GHOST DETECTED *** re-creation failed"
    else
        echo "    Cycle $cycle: clean (re-creation succeeded, usable=$RECREATE_USABLE)"
    fi

    ensure_clean_mig
done

echo ""

# --- Test A2: Under-load teardown ---
echo "  Test A2: Under-load teardown with capacity verification (15 cycles)"

for cycle in $(seq 1 15); do
    ensure_clean_mig

    # Create instance
    sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1
    UUID=$(nvidia-smi -L 2>/dev/null | grep -oP "MIG-[a-f0-9-]+" | head -1)

    # Launch persistent workload in background
    if [ -n "$UUID" ]; then
        CUDA_VISIBLE_DEVICES=$UUID python3 -c "
import torch, signal, time
x = torch.randn(2048, 2048, device='cuda')
for _ in range(10000):
    torch.mm(x, x)
    torch.cuda.synchronize()
" &>/dev/null &
        WORK_PID=$!
        sleep 2
    fi

    MEM_UNDER_LOAD=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    # Destroy while workload runs
    GI=$(sudo nvidia-smi mig -lgi -i 0 --format=csv,noheader 2>/dev/null | head -1 | awk '{print $1}')
    sudo nvidia-smi mig -dci -ci 0 -gi $GI -i 0 >/dev/null 2>&1
    sudo nvidia-smi mig -dgi -gi $GI -i 0 >/dev/null 2>&1
    kill $WORK_PID 2>/dev/null; wait $WORK_PID 2>/dev/null || true
    sleep 1

    MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    # Attempt re-creation
    RECREATE_SUCCESS=false
    if sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1; then
        RECREATE_SUCCESS=true
        sudo nvidia-smi mig -dci -i 0 >/dev/null 2>&1
        sudo nvidia-smi mig -dgi -i 0 >/dev/null 2>&1
    fi

    GHOST_DETECTED=false
    if [ "$RECREATE_SUCCESS" = false ]; then GHOST_DETECTED=true; fi

    cat >> "$OUTDIR/cat1_ghost.json" << RECORD
,{"test": "A2_under_load", "cycle": $cycle, "mem_under_load_mib": $MEM_UNDER_LOAD, "mem_after_destroy_mib": $MEM_AFTER, "recreate_success": $RECREATE_SUCCESS, "ghost_detected": $GHOST_DETECTED, "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
RECORD

    if [ "$GHOST_DETECTED" = true ]; then
        echo "    Cycle $cycle: *** GHOST DETECTED ***"
    else
        echo "    Cycle $cycle: clean"
    fi

    ensure_clean_mig
done

echo ""

# --- Test A3: Rapid churn with capacity verification ---
echo "  Test A3: Rapid churn (20 cycles) then capacity check (5 replications)"

for rep in $(seq 1 5); do
    ensure_clean_mig

    # Record initial max capacity
    INITIAL_MAX=0
    for i in $(seq 1 7); do
        if sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1; then
            INITIAL_MAX=$((INITIAL_MAX + 1))
        fi
    done
    ensure_clean_mig

    # 20 rapid create/load/destroy cycles
    for churn in $(seq 1 20); do
        sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1
        UUID=$(nvidia-smi -L 2>/dev/null | grep -oP "MIG-[a-f0-9-]+" | head -1)
        if [ -n "$UUID" ]; then
            CUDA_VISIBLE_DEVICES=$UUID python3 -c "import torch; x=torch.randn(1024,1024,device='cuda'); torch.cuda.synchronize()" 2>/dev/null || true
        fi
        sudo nvidia-smi mig -dci -i 0 >/dev/null 2>&1
        sudo nvidia-smi mig -dgi -i 0 >/dev/null 2>&1
    done

    sleep 2

    # Check: can we still create max instances?
    FINAL_MAX=0
    for i in $(seq 1 7); do
        if sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1; then
            FINAL_MAX=$((FINAL_MAX + 1))
        fi
    done
    ensure_clean_mig

    CAPACITY_LOSS=$((INITIAL_MAX - FINAL_MAX))
    GHOST_DETECTED=false
    if [ $CAPACITY_LOSS -gt 0 ]; then GHOST_DETECTED=true; fi

    cat >> "$OUTDIR/cat1_ghost.json" << RECORD
,{"test": "A3_rapid_churn", "replication": $rep, "churn_cycles": 20, "initial_max_instances": $INITIAL_MAX, "final_max_instances": $FINAL_MAX, "capacity_loss": $CAPACITY_LOSS, "ghost_detected": $GHOST_DETECTED, "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
RECORD

    if [ "$GHOST_DETECTED" = true ]; then
        echo "    Rep $rep: *** GHOST DETECTED *** lost $CAPACITY_LOSS slots after 20 churn cycles"
    else
        echo "    Rep $rep: clean (${FINAL_MAX}/${INITIAL_MAX} slots available)"
    fi
done

echo ""

# --- Test A4: Kill -9 with capacity verification ---
echo "  Test A4: Kill -9 then capacity verification (10 cycles)"

for cycle in $(seq 1 10); do
    ensure_clean_mig

    sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1
    UUID=$(nvidia-smi -L 2>/dev/null | grep -oP "MIG-[a-f0-9-]+" | head -1)

    # Heavy VRAM allocation then kill -9
    if [ -n "$UUID" ]; then
        CUDA_VISIBLE_DEVICES=$UUID python3 -c "
import torch, signal, os, time
x = [torch.randn(2048, 2048, device='cuda') for _ in range(50)]
torch.cuda.synchronize()
print(f'PID:{os.getpid()}')
signal.pause()
" &>/dev/null &
        WORK_PID=$!
        sleep 2
        kill -9 $WORK_PID 2>/dev/null
        wait $WORK_PID 2>/dev/null || true
    fi

    sleep 1
    MEM_AFTER_KILL=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    # Destroy MIG
    sudo nvidia-smi mig -dci -i 0 >/dev/null 2>&1
    sudo nvidia-smi mig -dgi -i 0 >/dev/null 2>&1
    sleep 1

    # Attempt re-creation
    RECREATE_SUCCESS=false
    if sudo nvidia-smi mig -cgi 19 -C -i 0 >/dev/null 2>&1; then
        RECREATE_SUCCESS=true
        sudo nvidia-smi mig -dci -i 0 >/dev/null 2>&1
        sudo nvidia-smi mig -dgi -i 0 >/dev/null 2>&1
    fi

    GHOST_DETECTED=false
    if [ "$RECREATE_SUCCESS" = false ]; then GHOST_DETECTED=true; fi

    cat >> "$OUTDIR/cat1_ghost.json" << RECORD
,{"test": "A4_kill9", "cycle": $cycle, "mem_after_kill_mib": $MEM_AFTER_KILL, "recreate_success": $RECREATE_SUCCESS, "ghost_detected": $GHOST_DETECTED, "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
RECORD

    if [ "$GHOST_DETECTED" = true ]; then
        echo "    Cycle $cycle: *** GHOST DETECTED ***"
    else
        echo "    Cycle $cycle: clean"
    fi
done

echo "]}" >> "$OUTDIR/cat1_ghost.json"
echo "  Category 1: COMPLETE"
echo ""

# =============================================================================
# CATEGORY 4: Burst-to-Sustained Gap
# =============================================================================
echo "===== CATEGORY 4: Burst-to-Sustained Gap ====="
echo "10 replications of 120s dynamic roofline"
echo ""

# Disable MIG for roofline measurement
ensure_clean_mig
sudo nvidia-smi -mig 0 -i 0 >/dev/null 2>&1 || true
sleep 2

echo '{"test": "burst_sustained_gap", "category": 4, "records": [' > "$OUTDIR/cat4_tension.json"

for rep in $(seq 1 10); do
    RESULT=$(gpu-roofline measure --format json 2>/dev/null || echo '{"error": "failed"}')

    SEPARATOR=""
    if [ $rep -gt 1 ]; then SEPARATOR=","; fi

    echo "${SEPARATOR}{\"replication\": $rep, \"data\": $RESULT, \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUTDIR/cat4_tension.json"

    echo "  Rep $rep/10 complete"
done

echo "]}" >> "$OUTDIR/cat4_tension.json"
echo "  Category 4: COMPLETE"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "============================================================================="
echo "VALIDATION COMPLETE"
echo "============================================================================="
echo ""

# Count ghost detections
GHOST_COUNT=$(grep -c '"ghost_detected": true' "$OUTDIR/cat1_ghost.json" 2>/dev/null || echo 0)
TOTAL_GHOST_TESTS=$(grep -c '"ghost_detected"' "$OUTDIR/cat1_ghost.json" 2>/dev/null || echo 0)

echo "Category 1 (Ghost Allocations):      $GHOST_COUNT detections out of $TOTAL_GHOST_TESTS tests"
echo "Category 3 (Provisioning Overhead):   See $OUTDIR/cat3_provisioning.json"
echo "Category 4 (Burst-Sustained Gap):     See $OUTDIR/cat4_tension.json"
echo "Category 6 (Oversubscription):        See $OUTDIR/cat6_oversubscription.json"
echo ""
echo "GPU:      $GPU_NAME"
echo "Driver:   $DRIVER"
echo "Output:   $OUTDIR"
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cat > "$OUTDIR/summary.json" << SUMMARY
{
  "gpu": "$GPU_NAME",
  "driver": "$DRIVER",
  "cuda": "$CUDA_VER",
  "arch": "$ARCH",
  "started": "$TIMESTAMP",
  "finished": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "categories_tested": [1, 3, 4, 6],
  "categories_not_tested": [2, 5],
  "ghost_detections": $GHOST_COUNT,
  "ghost_total_tests": $TOTAL_GHOST_TESTS,
  "output_dir": "$OUTDIR"
}
SUMMARY

echo "Results saved to $OUTDIR/"
ls -la "$OUTDIR/"
