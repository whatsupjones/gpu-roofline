use gpu_harness::KernelSpec;
use serde::{Deserialize, Serialize};

/// Categories of roofline kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelCategory {
    /// Memory bandwidth measurement (STREAM-equivalent).
    Bandwidth,
    /// Compute throughput measurement (FMA chains).
    Compute,
}

/// Built-in kernel identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuiltinKernel {
    Copy,
    Scale,
    Add,
    Triad,
    FmaLight,
    FmaMedium,
    FmaHeavy,
    /// FP16 Tensor Core (WMMA) — requires CUDA sm_70+
    TensorFp16,
    /// BF16 Tensor Core (WMMA) — requires CUDA sm_80+
    TensorBf16,
}

impl BuiltinKernel {
    /// All built-in kernels in order of increasing arithmetic intensity.
    /// Does NOT include Tensor Core kernels (use `all_with_tensor()` for those).
    pub fn all() -> &'static [BuiltinKernel] {
        &[
            Self::Copy,
            Self::Add,
            Self::Scale,
            Self::Triad,
            Self::FmaLight,
            Self::FmaMedium,
            Self::FmaHeavy,
        ]
    }

    /// All kernels including Tensor Core (for CUDA backends with sm_70+).
    pub fn all_with_tensor() -> &'static [BuiltinKernel] {
        &[
            Self::Copy,
            Self::Add,
            Self::Scale,
            Self::Triad,
            Self::FmaLight,
            Self::FmaMedium,
            Self::FmaHeavy,
            Self::TensorFp16,
            Self::TensorBf16,
        ]
    }

    /// Get the kernel definition.
    pub fn definition(self) -> KernelDefinition {
        match self {
            Self::Copy => KernelDefinition {
                name: "copy",
                category: KernelCategory::Bandwidth,
                wgsl_source: include_str!("../../shaders/copy.wgsl"),
                arithmetic_intensity: 0.0,
                flops_per_element: 0,
                bytes_per_element: 32, // vec4<f32> read + write = 16 + 16
                read_buffers: 1,
                write_buffers: 1,
                needs_uniform: false,
                description: "dst[i] = src[i] — pure memory bandwidth baseline",
            },
            Self::Scale => KernelDefinition {
                name: "scale",
                category: KernelCategory::Bandwidth,
                wgsl_source: include_str!("../../shaders/scale.wgsl"),
                arithmetic_intensity: 0.125,
                flops_per_element: 4, // 1 mul per component * 4 components
                bytes_per_element: 32,
                read_buffers: 1,
                write_buffers: 1,
                needs_uniform: true,
                description: "dst[i] = scalar * src[i] — scaled copy",
            },
            Self::Add => KernelDefinition {
                name: "add",
                category: KernelCategory::Bandwidth,
                wgsl_source: include_str!("../../shaders/add.wgsl"),
                arithmetic_intensity: 0.083,
                flops_per_element: 4,  // 1 add per component * 4
                bytes_per_element: 48, // 2 reads + 1 write = 16 + 16 + 16
                read_buffers: 2,
                write_buffers: 1,
                needs_uniform: false,
                description: "dst[i] = src_a[i] + src_b[i] — vector addition",
            },
            Self::Triad => KernelDefinition {
                name: "triad",
                category: KernelCategory::Bandwidth,
                wgsl_source: include_str!("../../shaders/triad.wgsl"),
                arithmetic_intensity: 0.167,
                flops_per_element: 8, // (1 mul + 1 add) per component * 4
                bytes_per_element: 48,
                read_buffers: 2,
                write_buffers: 1,
                needs_uniform: true,
                description: "dst[i] = src_a[i] + scalar * src_b[i] — STREAM triad",
            },
            Self::FmaLight => KernelDefinition {
                name: "fma_light",
                category: KernelCategory::Compute,
                wgsl_source: include_str!("../../shaders/fma_light.wgsl"),
                arithmetic_intensity: 1.0,
                flops_per_element: 32, // 4 FMA * 2 ops * 4 components
                bytes_per_element: 32,
                read_buffers: 1,
                write_buffers: 1,
                needs_uniform: false,
                description: "4 FMA ops/element — low compute, still memory-bound",
            },
            Self::FmaMedium => KernelDefinition {
                name: "fma_medium",
                category: KernelCategory::Compute,
                wgsl_source: include_str!("../../shaders/fma_medium.wgsl"),
                arithmetic_intensity: 8.0,
                flops_per_element: 256, // 32 FMA * 2 ops * 4 components
                bytes_per_element: 32,
                read_buffers: 1,
                write_buffers: 1,
                needs_uniform: false,
                description: "32 FMA ops/element — near ridge point",
            },
            Self::FmaHeavy => KernelDefinition {
                name: "fma_heavy",
                category: KernelCategory::Compute,
                wgsl_source: include_str!("../../shaders/fma_heavy.wgsl"),
                arithmetic_intensity: 64.0,
                flops_per_element: 2048, // 256 FMA * 2 ops * 4 components
                bytes_per_element: 32,
                read_buffers: 1,
                write_buffers: 1,
                needs_uniform: false,
                description: "256 FMA ops/element — compute-bound on all GPUs",
            },
            Self::TensorFp16 => KernelDefinition {
                name: "tensor_fp16",
                category: KernelCategory::Compute,
                wgsl_source: "", // CUDA-only, no WGSL equivalent
                // 32 WMMA ops per tile, each 8192 FLOPs, tile = 512 bytes input
                arithmetic_intensity: 512.0, // (32 * 8192) / 512 = 512 FLOP/byte
                flops_per_element: 32 * 8192, // per 16x16 tile
                bytes_per_element: 512,      // 16x16 * 2 bytes (fp16)
                read_buffers: 1,
                write_buffers: 1,
                needs_uniform: false,
                description: "FP16 Tensor Core WMMA — 8 independent 16x16x16 matmul lanes",
            },
            Self::TensorBf16 => KernelDefinition {
                name: "tensor_bf16",
                category: KernelCategory::Compute,
                wgsl_source: "", // CUDA-only, no WGSL equivalent
                arithmetic_intensity: 512.0,
                flops_per_element: 32 * 8192,
                bytes_per_element: 512,
                read_buffers: 1,
                write_buffers: 1,
                needs_uniform: false,
                description: "BF16 Tensor Core WMMA — 8 independent 16x16x16 matmul lanes",
            },
        }
    }

    /// Convert to a KernelSpec for the GpuBackend trait.
    pub fn to_spec(self, buffer_size_bytes: usize) -> KernelSpec {
        let def = self.definition();
        KernelSpec {
            name: def.name.to_string(),
            working_set_bytes: buffer_size_bytes,
            arithmetic_intensity: def.arithmetic_intensity,
            iterations: 1,
        }
    }
}

impl std::fmt::Display for BuiltinKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.definition().name)
    }
}

/// Complete definition of a roofline micro-kernel.
#[derive(Debug, Clone)]
pub struct KernelDefinition {
    /// Short name (e.g., "copy", "triad", "fma_heavy").
    pub name: &'static str,
    /// Whether this measures bandwidth or compute.
    pub category: KernelCategory,
    /// WGSL shader source (embedded at compile time).
    pub wgsl_source: &'static str,
    /// Theoretical arithmetic intensity (FLOP/byte).
    pub arithmetic_intensity: f64,
    /// FLOPs per vec4 element.
    pub flops_per_element: u64,
    /// Bytes transferred per vec4 element (reads + writes).
    pub bytes_per_element: u64,
    /// Number of read-only storage buffers.
    pub read_buffers: u32,
    /// Number of read-write storage buffers.
    pub write_buffers: u32,
    /// Whether a uniform buffer with scalar param is needed.
    pub needs_uniform: bool,
    /// Human-readable description.
    pub description: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_kernels_have_wgsl() {
        for kernel in BuiltinKernel::all() {
            let def = kernel.definition();
            assert!(
                !def.wgsl_source.is_empty(),
                "{} should have WGSL source",
                def.name
            );
            assert!(
                def.wgsl_source.contains("@compute"),
                "{} WGSL should contain @compute",
                def.name
            );
            assert!(
                def.wgsl_source.contains("@workgroup_size(256)"),
                "{} WGSL should have workgroup_size(256)",
                def.name
            );
        }
    }

    #[test]
    fn test_arithmetic_intensity_ordering() {
        // Bandwidth kernels should have low AI
        let copy_ai = BuiltinKernel::Copy.definition().arithmetic_intensity;
        let fma_heavy_ai = BuiltinKernel::FmaHeavy.definition().arithmetic_intensity;
        assert!(
            fma_heavy_ai > copy_ai,
            "fma_heavy AI ({fma_heavy_ai}) should be > copy AI ({copy_ai})"
        );

        // FMA kernels should increase in intensity
        let light = BuiltinKernel::FmaLight.definition().arithmetic_intensity;
        let medium = BuiltinKernel::FmaMedium.definition().arithmetic_intensity;
        let heavy = BuiltinKernel::FmaHeavy.definition().arithmetic_intensity;
        assert!(medium > light, "medium > light");
        assert!(heavy > medium, "heavy > medium");
    }

    #[test]
    fn test_kernel_spec_conversion() {
        let spec = BuiltinKernel::Triad.to_spec(16 * 1024 * 1024);
        assert_eq!(spec.name, "triad");
        assert_eq!(spec.working_set_bytes, 16 * 1024 * 1024);
        assert!((spec.arithmetic_intensity - 0.167).abs() < 0.01);
    }

    #[test]
    fn test_bandwidth_kernels_low_ai() {
        for kernel in &[
            BuiltinKernel::Copy,
            BuiltinKernel::Scale,
            BuiltinKernel::Add,
            BuiltinKernel::Triad,
        ] {
            let def = kernel.definition();
            assert_eq!(def.category, KernelCategory::Bandwidth);
            assert!(
                def.arithmetic_intensity < 1.0,
                "{} should have AI < 1.0, got {}",
                def.name,
                def.arithmetic_intensity
            );
        }
    }

    #[test]
    fn test_compute_kernels_high_ai() {
        for kernel in &[
            BuiltinKernel::FmaLight,
            BuiltinKernel::FmaMedium,
            BuiltinKernel::FmaHeavy,
        ] {
            let def = kernel.definition();
            assert_eq!(def.category, KernelCategory::Compute);
            assert!(
                def.arithmetic_intensity >= 1.0,
                "{} should have AI >= 1.0, got {}",
                def.name,
                def.arithmetic_intensity
            );
        }
    }
}
