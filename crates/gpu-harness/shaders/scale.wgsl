// Scale kernel: dst[i] = scalar * src[i]
// 1 FLOP per element (1 multiply per vec4 component = 4 FLOPs per vec4).
// Arithmetic intensity: 4 FLOP / 32 bytes = 0.125 FLOP/byte.

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    scalar: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&src) {
        dst[idx] = src[idx] * params.scalar;
    }
}
