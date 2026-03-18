// gpu-roofline CUDA kernels for datacenter GPU measurement
// Optimized for maximum hardware utilization on Hopper/Ampere/Ada architectures
//
// Key optimizations:
// - copy: __ldcs/__stcs streaming loads/stores bypass L2 for pure HBM bandwidth
// - FMA: multiple independent accumulator lanes for instruction-level parallelism
//   (avoids serial dependency chains that starve the FMA pipeline)

// ===========================================================================
// BANDWIDTH KERNELS
// ===========================================================================

// Copy kernel with grid-stride loop and streaming memory access.
// Thread coarsening: each thread processes multiple elements to amortize
// launch overhead and maximize memory controller utilization on HBM3.
// __ldcs/__stcs bypass L2 cache for pure HBM bandwidth measurement.
extern "C" __global__
void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, unsigned int n) {
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        float4 val = __ldcs(src + idx);
        __stcs(dst + idx, val);
    }
}

// ===========================================================================
// COMPUTE KERNELS — Independent Accumulator Lanes for Maximum ILP
// ===========================================================================
// The H100 can execute multiple FMA operations per cycle per SM, but only
// if they are INDEPENDENT (no data dependency between them). Using N
// independent accumulator lanes lets the compiler pipeline N FMAs in flight
// simultaneously, approaching peak FLOPS.

// FMA Light: 4 FMA ops per element, 4 independent lanes = 16 total FMAs
// Arithmetic intensity: ~1.0 FLOP/byte
extern "C" __global__
void fma_light_kernel(const float4* __restrict__ src, float4* __restrict__ dst, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 v = src[idx];

        // 4 independent accumulator lanes
        float4 a0 = v, a1 = v, a2 = v, a3 = v;

        a0.x = a0.x * 1.01f + 0.01f; a0.y = a0.y * 1.01f + 0.01f;
        a0.z = a0.z * 1.01f + 0.01f; a0.w = a0.w * 1.01f + 0.01f;
        a1.x = a1.x * 1.02f + 0.02f; a1.y = a1.y * 1.02f + 0.02f;
        a1.z = a1.z * 1.02f + 0.02f; a1.w = a1.w * 1.02f + 0.02f;
        a2.x = a2.x * 1.03f + 0.03f; a2.y = a2.y * 1.03f + 0.03f;
        a2.z = a2.z * 1.03f + 0.03f; a2.w = a2.w * 1.03f + 0.03f;
        a3.x = a3.x * 1.04f + 0.04f; a3.y = a3.y * 1.04f + 0.04f;
        a3.z = a3.z * 1.04f + 0.04f; a3.w = a3.w * 1.04f + 0.04f;

        // Reduce to prevent dead code elimination
        float4 result;
        result.x = a0.x + a1.x + a2.x + a3.x;
        result.y = a0.y + a1.y + a2.y + a3.y;
        result.z = a0.z + a1.z + a2.z + a3.z;
        result.w = a0.w + a1.w + a2.w + a3.w;
        dst[idx] = result;
    }
}

// FMA Medium: 32 FMA ops per element, 8 independent lanes
// Arithmetic intensity: ~8.0 FLOP/byte
extern "C" __global__
void fma_medium_kernel(const float4* __restrict__ src, float4* __restrict__ dst, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 v = src[idx];

        // 8 independent accumulator lanes, 4 FMAs each = 32 FMAs per component
        float4 a0 = v, a1 = v, a2 = v, a3 = v;
        float4 a4 = v, a5 = v, a6 = v, a7 = v;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a0.x = a0.x * 1.001f + 0.001f; a0.y = a0.y * 1.001f + 0.001f;
            a0.z = a0.z * 1.001f + 0.001f; a0.w = a0.w * 1.001f + 0.001f;
            a1.x = a1.x * 1.002f + 0.002f; a1.y = a1.y * 1.002f + 0.002f;
            a1.z = a1.z * 1.002f + 0.002f; a1.w = a1.w * 1.002f + 0.002f;
            a2.x = a2.x * 1.003f + 0.003f; a2.y = a2.y * 1.003f + 0.003f;
            a2.z = a2.z * 1.003f + 0.003f; a2.w = a2.w * 1.003f + 0.003f;
            a3.x = a3.x * 1.004f + 0.004f; a3.y = a3.y * 1.004f + 0.004f;
            a3.z = a3.z * 1.004f + 0.004f; a3.w = a3.w * 1.004f + 0.004f;
            a4.x = a4.x * 1.005f + 0.005f; a4.y = a4.y * 1.005f + 0.005f;
            a4.z = a4.z * 1.005f + 0.005f; a4.w = a4.w * 1.005f + 0.005f;
            a5.x = a5.x * 1.006f + 0.006f; a5.y = a5.y * 1.006f + 0.006f;
            a5.z = a5.z * 1.006f + 0.006f; a5.w = a5.w * 1.006f + 0.006f;
            a6.x = a6.x * 1.007f + 0.007f; a6.y = a6.y * 1.007f + 0.007f;
            a6.z = a6.z * 1.007f + 0.007f; a6.w = a6.w * 1.007f + 0.007f;
            a7.x = a7.x * 1.008f + 0.008f; a7.y = a7.y * 1.008f + 0.008f;
            a7.z = a7.z * 1.008f + 0.008f; a7.w = a7.w * 1.008f + 0.008f;
        }

        float4 result;
        result.x = a0.x + a1.x + a2.x + a3.x + a4.x + a5.x + a6.x + a7.x;
        result.y = a0.y + a1.y + a2.y + a3.y + a4.y + a5.y + a6.y + a7.y;
        result.z = a0.z + a1.z + a2.z + a3.z + a4.z + a5.z + a6.z + a7.z;
        result.w = a0.w + a1.w + a2.w + a3.w + a4.w + a5.w + a6.w + a7.w;
        dst[idx] = result;
    }
}

// FMA Heavy: 256 FMA ops per element, 8 independent lanes x 32 iterations
// Arithmetic intensity: ~64.0 FLOP/byte
extern "C" __global__
void fma_heavy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 v = src[idx];

        // 8 independent accumulator lanes, 32 FMAs each = 256 FMAs per component
        float4 a0 = v, a1 = v, a2 = v, a3 = v;
        float4 a4 = v, a5 = v, a6 = v, a7 = v;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a0.x = a0.x * 1.0001f + 0.0001f; a0.y = a0.y * 1.0001f + 0.0001f;
            a0.z = a0.z * 1.0001f + 0.0001f; a0.w = a0.w * 1.0001f + 0.0001f;
            a1.x = a1.x * 1.0002f + 0.0002f; a1.y = a1.y * 1.0002f + 0.0002f;
            a1.z = a1.z * 1.0002f + 0.0002f; a1.w = a1.w * 1.0002f + 0.0002f;
            a2.x = a2.x * 1.0003f + 0.0003f; a2.y = a2.y * 1.0003f + 0.0003f;
            a2.z = a2.z * 1.0003f + 0.0003f; a2.w = a2.w * 1.0003f + 0.0003f;
            a3.x = a3.x * 1.0004f + 0.0004f; a3.y = a3.y * 1.0004f + 0.0004f;
            a3.z = a3.z * 1.0004f + 0.0004f; a3.w = a3.w * 1.0004f + 0.0004f;
            a4.x = a4.x * 1.0005f + 0.0005f; a4.y = a4.y * 1.0005f + 0.0005f;
            a4.z = a4.z * 1.0005f + 0.0005f; a4.w = a4.w * 1.0005f + 0.0005f;
            a5.x = a5.x * 1.0006f + 0.0006f; a5.y = a5.y * 1.0006f + 0.0006f;
            a5.z = a5.z * 1.0006f + 0.0006f; a5.w = a5.w * 1.0006f + 0.0006f;
            a6.x = a6.x * 1.0007f + 0.0007f; a6.y = a6.y * 1.0007f + 0.0007f;
            a6.z = a6.z * 1.0007f + 0.0007f; a6.w = a6.w * 1.0007f + 0.0007f;
            a7.x = a7.x * 1.0008f + 0.0008f; a7.y = a7.y * 1.0008f + 0.0008f;
            a7.z = a7.z * 1.0008f + 0.0008f; a7.w = a7.w * 1.0008f + 0.0008f;
        }

        float4 result;
        result.x = a0.x + a1.x + a2.x + a3.x + a4.x + a5.x + a6.x + a7.x;
        result.y = a0.y + a1.y + a2.y + a3.y + a4.y + a5.y + a6.y + a7.y;
        result.z = a0.z + a1.z + a2.z + a3.z + a4.z + a5.z + a6.z + a7.z;
        result.w = a0.w + a1.w + a2.w + a3.w + a4.w + a5.w + a6.w + a7.w;
        dst[idx] = result;
    }
}
