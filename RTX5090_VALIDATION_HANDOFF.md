# RTX 5090 Validation Handoff

## Instance Access
```
ssh -o StrictHostKeyChecking=no -i ~/.ssh/jarvislabs_h200_key root@74.2.96.25 -p 10469
```

GPU: NVIDIA GeForce RTX 5090, 32GB GDDR7, Driver 570.195.03, sm_120 (Blackwell consumer)

## Problem
Rust install is corrupted from parallel install attempts. Needs clean reinstall.

## Step 1: Fix Rust Install
```bash
rm -rf ~/.rustup ~/.cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
rustc --version  # Should show 1.94.0
```

## Step 2: Clone and Build
```bash
cd /tmp
git clone https://github.com/whatsupjones/gpu-roofline.git
cd gpu-roofline
cargo build --release --features cuda
```

If NVRTC compilation fails for tensor kernels on sm_120, that's expected — the tensor kernels are compiled with `--arch=sm_90` (Hopper). You may need to check if sm_120 needs a different arch flag. The FP32 kernels should work fine without any arch flag.

## Step 3: Fix NVML (if needed)
```bash
ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so
ldconfig
```

## Step 4: Run Full Validation Suite

Run all 6 tests with nvidia-smi monitoring. Save everything to `/tmp/rtx5090_validation/`.

### Test 1: Burst Roofline (all 7 FP32 kernels)
```bash
mkdir -p /tmp/rtx5090_validation
nvidia-smi dmon -s pcut -d 1 -c 25 > /tmp/rtx5090_validation/burst_smi.txt &
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  /tmp/gpu-roofline/target/release/gpu-roofline measure --burst --backend cuda \
  --save-baseline /tmp/rtx5090_validation/burst.json
kill %1 2>/dev/null
```

Expected: ~1,700-1,800 GB/s bandwidth (GDDR7 512-bit), ~80-90 TFLOPS FP32 (21760 CUDA cores at 2.5-2.9 GHz)

### Test 2: Validate
```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  /tmp/gpu-roofline/target/release/gpu-roofline validate --backend cuda
```

NOTE: The RTX 5090 may not have a baseline yet in `validate/baselines.rs`. If validate says "No baseline found", that's expected. The burst results are still valid — we'll add the baseline from these results.

### Test 3: Tensor Core (FP16 + BF16)
Create the test example:
```bash
mkdir -p /tmp/gpu-roofline/crates/gpu-harness/examples
cat > /tmp/gpu-roofline/crates/gpu-harness/examples/tensor_test.rs << 'EOF'
use gpu_harness::backend::{KernelSpec, RunConfig};
use gpu_harness::GpuBackend;
fn main() {
    let backend = gpu_harness::CudaBackend::new().expect("CUDA init");
    let devices = backend.discover_devices().unwrap();
    eprintln!("GPU: {}", devices[0].name);
    let config = RunConfig { warmup_iterations: 5, measurement_iterations: 50, buffer_size_bytes: 256*1024*1024 };
    for (name, ai) in [("fma_heavy", 64.0), ("tensor_fp16", 512.0), ("tensor_bf16", 512.0)] {
        let spec = KernelSpec { name: name.to_string(), working_set_bytes: 256*1024*1024, arithmetic_intensity: ai, iterations: 1 };
        match backend.run_kernel(&spec, &config) {
            Ok(r) => eprintln!("{:<12} {:>8.1} TFLOPS | median {:>8.1} us | CV {:.1}%", name, r.gflops()/1000.0, r.median_us(), r.cv()*100.0),
            Err(e) => eprintln!("{:<12} FAILED: {}", name, e),
        }
    }
}
EOF
cargo run --release --features cuda -p gpu-harness --example tensor_test 2>&1 | tee /tmp/rtx5090_validation/tensor_results.txt
```

IMPORTANT: Tensor kernels compile with `--arch=sm_90`. The RTX 5090 is sm_120 (Blackwell). If tensor compilation fails, try editing line 366 of `crates/gpu-harness/src/cuda_backend.rs`:
```rust
// Change:
arch: Some("sm_90"),
// To:
arch: Some("sm_120"),
// Or try:
arch: Some("compute_90"),
```
The WMMA API should be backward compatible on sm_120.

### Test 4: Dynamic Roofline (120s sustained)
```bash
nvidia-smi dmon -s pcut -d 5 -c 30 > /tmp/rtx5090_validation/dynamic_smi.txt &
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  /tmp/gpu-roofline/target/release/gpu-roofline measure --duration 120 --backend cuda \
  --save-baseline /tmp/rtx5090_validation/dynamic.json
kill %1 2>/dev/null
```

The RTX 5090 with 575W TDP will likely show MORE thermal throttling than datacenter GPUs. This is interesting data — consumer GPUs have a bigger burst-to-sustained gap.

### Test 5: Monitor (60s)
```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  /tmp/gpu-roofline/target/release/gpu-roofline monitor --interval 10 --duration 60 \
  --daemon --backend cuda --log /tmp/rtx5090_validation/monitor.jsonl
```

### Test 6: Validate JSON + list artifacts
```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  /tmp/gpu-roofline/target/release/gpu-roofline validate --backend cuda --format json \
  > /tmp/rtx5090_validation/validate_json.json 2>/dev/null
ls -la /tmp/rtx5090_validation/
```

## Step 5: Pull Results Back Locally
```bash
mkdir -p /c/Users/Chris/Projects/gpu-tools/docs/validation/rtx5090
scp -o StrictHostKeyChecking=no -i /c/Users/Chris/.ssh/jarvislabs_h200_key \
  -P 10469 root@74.2.96.25:/tmp/rtx5090_validation/* \
  /c/Users/Chris/Projects/gpu-tools/docs/validation/rtx5090/
```

## Step 6: Create Validation Report
Create `docs/validation/RTX5090_VALIDATION.md` following the same format as H100_VALIDATION.md and H200_VALIDATION.md in the same directory.

## Step 7: Update README and Commit
1. Add RTX 5090 row to the hardware table in README.md
2. Update the multi-precision roofline chart if tensor results are available
3. Add RTX 5090 baseline to `crates/gpu-roofline/src/validate/baselines.rs` if one doesn't exist for Blackwell consumer (check compute_capability 12.0)
4. Commit with message: "Add RTX 5090 validation: <BW> GB/s, <FP32>T FP32, <FP16>T FP16 Tensor"
5. Push to origin main

## Key Numbers to Expect (RTX 5090 Blackwell)
- GDDR7 bandwidth: ~1,700-1,800 GB/s (theoretical 1,792 GB/s)
- FP32 CUDA cores: 21,760 cores → ~80-95 TFLOPS
- FP16 Tensor: ~380 TFLOPS (spec)
- Boost clock: ~2,407 MHz (may boost higher)
- TDP: 575W → expect visible thermal throttling in sustained test

## Important Notes
- Do NOT push to GitHub without explicit permission from the user
- Do NOT modify any code beyond what's needed for sm_120 compatibility
- Save ALL terminal output and nvidia-smi logs
- If anything fails, document the failure and move on to the next test
- Shut down the instance when done to save costs
