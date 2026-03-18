// FMA Medium kernel: 32 FMA operations per element.
// Medium arithmetic intensity — near the ridge point on modern GPUs.
// 64 FLOPs per element (32 FMA per component = 256 per vec4).
// 2 memory accesses (1 read + 1 write = 32 bytes per vec4).
// Arithmetic intensity: 256 FLOP / 32 bytes = 8.0 FLOP/byte.

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&src) {
        var val = src[idx];
        // 32 dependent FMA operations per component (8 groups of 4)
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        val = val * val + val; val = val * val + val;
        dst[idx] = val;
    }
}
