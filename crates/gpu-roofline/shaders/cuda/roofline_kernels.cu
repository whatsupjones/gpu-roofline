// gpu-roofline CUDA kernels for datacenter GPU measurement
// Equivalent to the WGSL shaders but compiled via NVRTC at runtime

extern "C" __global__
void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

extern "C" __global__
void fma_light_kernel(const float4* __restrict__ src, float4* __restrict__ dst, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 val = src[idx];
        // 4 FMA operations per component
        val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
        val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
        val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
        val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
        val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
        val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
        val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
        val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
        dst[idx] = val;
    }
}

extern "C" __global__
void fma_medium_kernel(const float4* __restrict__ src, float4* __restrict__ dst, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 val = src[idx];
        // 32 FMA operations per component
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
            val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
            val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
            val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
            val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
            val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
            val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
            val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
        }
        dst[idx] = val;
    }
}

extern "C" __global__
void fma_heavy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 val = src[idx];
        // 256 FMA operations per component
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
            val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
            val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
            val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
            val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
            val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
            val.x = val.x * val.x + val.x; val.y = val.y * val.y + val.y;
            val.z = val.z * val.z + val.z; val.w = val.w * val.w + val.w;
        }
        dst[idx] = val;
    }
}
