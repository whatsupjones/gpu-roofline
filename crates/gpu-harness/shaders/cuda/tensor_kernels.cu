// gpu-roofline Tensor Core kernels for datacenter GPU measurement
// Requires sm_70+ (Volta) for WMMA, sm_80+ for BF16, sm_90 for Hopper optimizations
//
// These kernels use NVIDIA's Tensor Cores via the WMMA API (mma.h).
// Tensor Cores perform D = A * B + C on 16x16x16 tiles, where:
//   - A, B are FP16 or BF16 (input precision)
//   - C, D are FP32 (accumulation precision)
//
// Each warp (32 threads) cooperatively computes one 16x16x16 WMMA op,
// producing 16*16*16*2 = 8192 FLOPs per op. Multiple independent WMMA
// ops per warp maximize Tensor Core utilization via ILP — same strategy
// as the FP32 FMA kernels using independent accumulator lanes.
//
// On Hopper (sm_90):
//   FP16 Tensor: ~989 TFLOPS theoretical
//   BF16 Tensor: ~989 TFLOPS theoretical
//   FP32 CUDA:   ~67 TFLOPS theoretical (~59 measured)

#include <mma.h>
using namespace nvcuda;

// ===========================================================================
// FP16 TENSOR CORE — 32 WMMA ops per warp (8 ILP lanes x 4 iterations)
// ===========================================================================
// Each WMMA: 16*16*16*2 = 8192 FLOPs → 32 ops = 262,144 FLOPs per warp
// This is the primary precision for ML inference (FP16 accumulate to FP32).
extern "C" __global__
void tensor_fp16_kernel(const half* __restrict__ src, float* __restrict__ dst, unsigned int n) {
    unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= n) return;

    // Load input tile (16x16 = 256 half elements)
    const half* tile_ptr = src + warp_id * 256;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, tile_ptr, 16);
    wmma::load_matrix_sync(b_frag, tile_ptr, 16);

    // 8 independent accumulator lanes — same ILP strategy as FP32 FMA kernels
    // Each lane is a completely independent WMMA operation, allowing the
    // Tensor Core pipeline to overlap multiple matrix multiplies in flight.
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

    // 4 iterations x 8 lanes = 32 WMMA ops per warp
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

    // Reduce accumulators to prevent dead code elimination
    for (int i = 0; i < acc0.num_elements; i++) {
        acc0.x[i] += acc1.x[i] + acc2.x[i] + acc3.x[i]
                   + acc4.x[i] + acc5.x[i] + acc6.x[i] + acc7.x[i];
    }

    // Store FP32 accumulator result
    wmma::store_matrix_sync(dst + warp_id * 256, acc0, 16, wmma::mem_row_major);
}

// ===========================================================================
// BF16 TENSOR CORE — same ILP structure, bfloat16 input precision
// ===========================================================================
// BF16 has the same dynamic range as FP32 (8-bit exponent) but FP16's
// throughput on Tensor Cores. Preferred for ML training where gradient
// magnitude matters more than mantissa precision.
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
