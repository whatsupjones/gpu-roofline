// FMA Heavy kernel: 256 FMA operations per element.
// High arithmetic intensity — compute-bound on all GPUs.
// 512 FLOPs per element (256 FMA per component = 2048 per vec4).
// 2 memory accesses (1 read + 1 write = 32 bytes per vec4).
// Arithmetic intensity: 2048 FLOP / 32 bytes = 64.0 FLOP/byte.

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&src) {
        var val = src[idx];
        // 256 dependent FMA operations per component
        // Unrolled in groups of 8 to help the compiler
        var i = 0u;
        loop {
            if i >= 32u { break; }
            val = val * val + val;
            val = val * val + val;
            val = val * val + val;
            val = val * val + val;
            val = val * val + val;
            val = val * val + val;
            val = val * val + val;
            val = val * val + val;
            i = i + 1u;
        }
        dst[idx] = val;
    }
}
