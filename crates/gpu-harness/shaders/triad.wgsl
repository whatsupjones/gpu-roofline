// Triad kernel: dst[i] = src_a[i] + scalar * src_b[i]
// Classic STREAM triad — the standard memory bandwidth benchmark.
// 2 FLOPs per element (1 multiply + 1 add per component = 8 FLOPs per vec4).
// 3 memory accesses per element (2 reads + 1 write = 48 bytes per vec4).
// Arithmetic intensity: 8 FLOP / 48 bytes ≈ 0.167 FLOP/byte.

@group(0) @binding(0) var<storage, read> src_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> src_b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    scalar: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&src_a) {
        dst[idx] = src_a[idx] + params.scalar * src_b[idx];
    }
}
