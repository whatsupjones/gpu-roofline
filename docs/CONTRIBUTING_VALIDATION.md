# Contributing Hardware Validation Results

Help us build the most comprehensive GPU performance database by validating your hardware.

## Quick Start

```bash
# Install
cargo install gpu-roofline --features cuda   # NVIDIA datacenter
cargo install gpu-roofline                    # Consumer (Vulkan/DX12/Metal)

# Run validation (takes ~30 seconds)
gpu-roofline validate

# Run full roofline (takes ~15 seconds)
gpu-roofline measure --burst

# Save results as JSON
gpu-roofline measure --burst --save-baseline my_gpu.json
gpu-roofline validate --format json > validate.json
```

## What We Need

1. Copy the **terminal output** from both commands above
2. Tell us your GPU model, driver version, and OS
3. If you have Tensor Core support (NVIDIA sm_70+), also run:
   ```bash
   cargo install gpu-roofline --features cuda
   # (Tensor Core kernels require CUDA backend)
   ```

## How to Submit

**Option A: Open an Issue**
Create an issue titled `[Validation] <GPU Model>` with your results pasted in.

**Option B: Submit a PR**
1. Fork the repo
2. Create `docs/validation/<gpu_name>/` with your JSON files
3. Add a `<GPU_NAME>_VALIDATION.md` report (see existing H100/H200 reports as templates)
4. Submit PR

## Currently Validated

| GPU | BW | FP32 | FP16 Tensor | Status |
|-----|-----|------|-------------|--------|
| NVIDIA H200 141GB | 4,028 GB/s | 59.5T | 684T | Full |
| NVIDIA H100 80GB | 2,905 GB/s | 59.1T | 495T | Full |
| Intel UHD Graphics | 7 GB/s | 0.15T | — | Burst only |

## Wanted

Priority GPUs we'd like community validation for:

**Consumer NVIDIA:**
- [ ] RTX 4090
- [ ] RTX 4080 / 4080 Super
- [ ] RTX 4070 Ti / 4070
- [ ] RTX 3090 / 3080 / 3070
- [ ] RTX 3060 (most common ML dev GPU)

**Datacenter:**
- [ ] A100 80GB / 40GB
- [ ] A10G (AWS g5 instances)
- [ ] L4 / L40S
- [ ] T4 (most deployed inference GPU)

**AMD:**
- [ ] MI300X
- [ ] RX 7900 XTX
- [ ] RX 9070 XT

**Intel:**
- [ ] Arc A770 / A750
- [ ] Data Center GPU Max (Ponte Vecchio)

**Apple (via Metal backend):**
- [ ] M4 Pro / M4 Max
- [ ] M3 Ultra
