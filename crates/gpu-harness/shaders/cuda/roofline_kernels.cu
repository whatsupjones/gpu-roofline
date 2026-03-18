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

// Scale kernel: dst[i] = scalar * src[i]
// Uses grid-stride loop like copy for consistency.
extern "C" __global__
void scale_kernel(const float4* __restrict__ src, float4* __restrict__ dst, float scalar, unsigned int n) {
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        float4 val = __ldcs(src + idx);
        float4 result;
        result.x = scalar * val.x;
        result.y = scalar * val.y;
        result.z = scalar * val.z;
        result.w = scalar * val.w;
        __stcs(dst + idx, result);
    }
}

// Add kernel: dst[i] = src_a[i] + src_b[i]
// Two source buffers, one destination.
extern "C" __global__
void add_kernel(const float4* __restrict__ src_a, const float4* __restrict__ src_b,
                float4* __restrict__ dst, unsigned int n) {
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        float4 a = __ldcs(src_a + idx);
        float4 b = __ldcs(src_b + idx);
        float4 result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        result.z = a.z + b.z;
        result.w = a.w + b.w;
        __stcs(dst + idx, result);
    }
}

// Triad kernel: dst[i] = src_a[i] + scalar * src_b[i]
// STREAM triad — the gold standard bandwidth benchmark.
extern "C" __global__
void triad_kernel(const float4* __restrict__ src_a, const float4* __restrict__ src_b,
                  float4* __restrict__ dst, float scalar, unsigned int n) {
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        float4 a = __ldcs(src_a + idx);
        float4 b = __ldcs(src_b + idx);
        float4 result;
        result.x = a.x + scalar * b.x;
        result.y = a.y + scalar * b.y;
        result.z = a.z + scalar * b.z;
        result.w = a.w + scalar * b.w;
        __stcs(dst + idx, result);
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

// ===========================================================================
// TENSOR CORE KERNELS — WMMA (Warp Matrix Multiply-Accumulate)
// ===========================================================================
// These kernels use NVIDIA's Tensor Cores via the WMMA API (mma.h).
// Tensor Cores perform D = A * B + C on 16x16x16 tiles, where:
//   - A, B are FP16 or BF16 (input precision)
//   - C, D are FP32 (accumulation precision)
//
// Each warp (32 threads) cooperatively computes one 16x16x16 WMMA op,
// producing 16*16*16*2 = 8192 FLOPs per op. Multiple independent WMMA
// ops per warp maximize Tensor Core utilization via ILP.
//
// On Hopper (sm_90):
//   FP16 Tensor: ~989 TFLOPS theoretical
//   BF16 Tensor: ~989 TFLOPS theoretical
//   FP32 CUDA:   ~67 TFLOPS theoretical (~59 measured)

#include <mma.h>
using namespace nvcuda;

// FP16 Tensor Core kernel — 32 WMMA ops per warp (8 lanes x 4 iterations)
// Each WMMA: 16*16*16*2 = 8192 FLOPs → 32 ops = 262,144 FLOPs per warp
extern "C" __global__
void tensor_fp16_kernel(const half* __restrict__ src, float* __restrict__ dst, unsigned int n) {
    unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= n) return;

    // Load input tile
    const half* tile_ptr = src + warp_id * 256;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, tile_ptr, 16);
    wmma::load_matrix_sync(b_frag, tile_ptr, 16);

    // 8 independent accumulator lanes for Tensor Core ILP
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1, acc2, acc3;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc4, acc5, acc6, acc7;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);
    wmma::fill_fragment(acc3, 0.0f);
    wmma::fill_fragment(acc4, 0.0f);
    wmma::fill_fragment(acc5, 0.0f);
    wmma::fill_fragment(acc6, 0.0f);
    wmma::fill_fragment(acc7, 0.0f);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::mma_sync(acc0, a_frag, b_frag, acc0);
        wmma::mma_sync(acc1, a_frag, b_frag, acc1);
        wmma::mma_sync(acc2, a_frag, b_frag, acc2);
        wmma::mma_sync(acc3, a_frag, b_frag, acc3);
        wmma::mma_sync(acc4, a_frag, b_frag, acc4);
        wmma::mma_sync(acc5, a_frag, b_frag, acc5);
        wmma::mma_sync(acc6, a_frag, b_frag, acc6);
        wmma::mma_sync(acc7, a_frag, b_frag, acc7);
    }

    // Reduce to prevent dead code elimination
    for (int i = 0; i < acc0.num_elements; i++) {
        acc0.x[i] += acc1.x[i] + acc2.x[i] + acc3.x[i]
                   + acc4.x[i] + acc5.x[i] + acc6.x[i] + acc7.x[i];
    }

    wmma::store_matrix_sync(dst + warp_id * 256, acc0, 16, wmma::mem_row_major);
}

// BF16 Tensor Core kernel — same structure, bfloat16 input precision
// BF16 has same dynamic range as FP32 but FP16's throughput — preferred
// for ML training where gradient magnitude matters more than mantissa bits.
extern "C" __global__
void tensor_bf16_kernel(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, unsigned int n) {
    unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= n) return;

    const __nv_bfloat16* tile_ptr = src + warp_id * 256;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, tile_ptr, 16);
    wmma::load_matrix_sync(b_frag, tile_ptr, 16);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1, acc2, acc3;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc4, acc5, acc6, acc7;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);
    wmma::fill_fragment(acc3, 0.0f);
    wmma::fill_fragment(acc4, 0.0f);
    wmma::fill_fragment(acc5, 0.0f);
    wmma::fill_fragment(acc6, 0.0f);
    wmma::fill_fragment(acc7, 0.0f);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::mma_sync(acc0, a_frag, b_frag, acc0);
        wmma::mma_sync(acc1, a_frag, b_frag, acc1);
        wmma::mma_sync(acc2, a_frag, b_frag, acc2);
        wmma::mma_sync(acc3, a_frag, b_frag, acc3);
        wmma::mma_sync(acc4, a_frag, b_frag, acc4);
        wmma::mma_sync(acc5, a_frag, b_frag, acc5);
        wmma::mma_sync(acc6, a_frag, b_frag, acc6);
        wmma::mma_sync(acc7, a_frag, b_frag, acc7);
    }

    for (int i = 0; i < acc0.num_elements; i++) {
        acc0.x[i] += acc1.x[i] + acc2.x[i] + acc3.x[i]
                   + acc4.x[i] + acc5.x[i] + acc6.x[i] + acc7.x[i];
    }

    wmma::store_matrix_sync(dst + warp_id * 256, acc0, 16, wmma::mem_row_major);
}
