// Copy kernel: dst[i] = src[i]
// Pure memory bandwidth measurement — 0 FLOP per element.
// Arithmetic intensity: 0 FLOP/byte (memory-bound baseline).

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&src) {
        dst[idx] = src[idx];
    }
}
