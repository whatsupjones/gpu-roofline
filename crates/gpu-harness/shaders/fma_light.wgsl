// FMA Light kernel: 4 FMA operations per element.
// Low arithmetic intensity — still memory-bound on most GPUs.
// 8 FLOPs per element (4 FMA = 4 mul + 4 add per component = 32 per vec4).
// 2 memory accesses (1 read + 1 write = 32 bytes per vec4).
// Arithmetic intensity: 32 FLOP / 32 bytes = 1.0 FLOP/byte.

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&src) {
        var val = src[idx];
        // 4 dependent FMA operations per component
        val = val * val + val;
        val = val * val + val;
        val = val * val + val;
        val = val * val + val;
        dst[idx] = val;
    }
}
