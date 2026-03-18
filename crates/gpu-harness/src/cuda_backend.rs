//! CUDA compute backend for datacenter GPUs (H100, H200, A100).
//!
//! Uses cudarc for dynamic CUDA driver loading — no CUDA toolkit needed
//! at compile time. Kernels compiled at runtime via NVRTC (PTX).

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;
    use std::time::Instant;

    use crate::backend::{
        BandwidthResult, DeviceState, GpuBackend, KernelResult, KernelSpec, RunConfig,
    };
    use crate::device::{GpuArchitecture, GpuDevice, GpuFeatures, GpuLimits, GpuVendor};
    use crate::error::HarnessError;
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

    const CUDA_KERNEL_SOURCE: &str = include_str!("../shaders/cuda/roofline_kernels.cu");
    const TENSOR_KERNEL_SOURCE: &str = include_str!("../shaders/cuda/tensor_kernels.cu");
    const BLOCK_SIZE: u32 = 256;

    /// CUDA backend for datacenter GPU compute.
    pub struct CudaBackend {
        devices: Vec<CudaDeviceInfo>,
        #[cfg(feature = "nvml")]
        nvml: Option<crate::nvml_telemetry::NvmlTelemetry>,
    }

    struct CudaDeviceInfo {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        name: String,
        compute_capability: (i32, i32),
        total_memory_bytes: usize,
    }

    impl CudaBackend {
        /// Create a new CUDA backend, discovering all CUDA-capable devices.
        pub fn new() -> Result<Self, HarnessError> {
            let result = std::panic::catch_unwind(|| Self::init());
            match result {
                Ok(Ok(backend)) => Ok(backend),
                Ok(Err(e)) => Err(e),
                Err(_) => Err(HarnessError::BackendUnavailable(
                    "CUDA initialization panicked. Ensure NVIDIA drivers are installed \
                     and libcuda.so is accessible."
                        .to_string(),
                )),
            }
        }

        fn init() -> Result<Self, HarnessError> {
            let device_count = CudaContext::device_count().map_err(|e| {
                HarnessError::BackendUnavailable(format!("CUDA device count failed: {e}"))
            })? as u32;

            if device_count == 0 {
                return Err(HarnessError::NoDevice);
            }

            let mut devices = Vec::new();
            for ordinal in 0..device_count {
                let ctx = CudaContext::new(ordinal as usize).map_err(|e| {
                    HarnessError::BackendUnavailable(format!(
                        "Failed to create CUDA context on device {ordinal}: {e}"
                    ))
                })?;

                let stream = ctx.default_stream();

                let name = ctx
                    .name()
                    .unwrap_or_else(|_| "Unknown CUDA Device".to_string());

                let major = ctx
                    .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                    .unwrap_or(0);
                let minor = ctx
                    .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                    .unwrap_or(0);

                let (_, total_memory_bytes) =
                    cudarc::driver::result::mem_get_info().unwrap_or((0, 0));

                tracing::info!(
                    "CUDA device {ordinal}: {name} | sm_{major}{minor} | {:.1} GB VRAM",
                    total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );

                devices.push(CudaDeviceInfo {
                    ctx,
                    stream,
                    name,
                    compute_capability: (major, minor),
                    total_memory_bytes,
                });
            }

            if devices.is_empty() {
                return Err(HarnessError::NoDevice);
            }

            // Try to initialize NVML for telemetry (temp, clock, power)
            #[cfg(feature = "nvml")]
            let nvml = match crate::nvml_telemetry::NvmlTelemetry::new() {
                Ok(n) => {
                    tracing::info!(
                        "NVML initialized: driver {} | {} device(s)",
                        n.driver_version().unwrap_or_else(|| "unknown".into()),
                        n.device_count()
                    );
                    Some(n)
                }
                Err(e) => {
                    tracing::warn!("NVML not available (telemetry will be limited): {e}");
                    None
                }
            };

            Ok(Self {
                devices,
                #[cfg(feature = "nvml")]
                nvml,
            })
        }

        fn dispatch_kernel(
            dev_info: &CudaDeviceInfo,
            kernel_name: &str,
            buffer_size_bytes: usize,
            iterations: u32,
        ) -> Result<Vec<f64>, HarnessError> {
            let stream = &dev_info.stream;
            let element_count = buffer_size_bytes / 16; // float4 = 16 bytes

            // Compile PTX
            let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SOURCE).map_err(|e| {
                HarnessError::KernelFailed(format!("NVRTC compilation failed: {e}"))
            })?;

            // Load module and get function (cudarc 0.19 API)
            let module = dev_info.ctx.load_module(ptx).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to load PTX module: {e}"))
            })?;

            let func = module.load_function(kernel_name).map_err(|e| {
                HarnessError::KernelFailed(format!(
                    "Kernel function '{kernel_name}' not found: {e}"
                ))
            })?;

            // Allocate device buffers
            let src_data: Vec<f32> = (0..element_count * 4)
                .map(|i| (i as f32) * 0.001 + 1.0)
                .collect();

            let src: CudaSlice<f32> = stream.clone_htod(&src_data).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to copy src to device: {e}"))
            })?;

            let mut dst: CudaSlice<f32> = stream.alloc_zeros(element_count * 4).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to allocate dst on device: {e}"))
            })?;

            let n = element_count as u32;
            let grid_size = n.div_ceil(BLOCK_SIZE);

            let cfg = LaunchConfig {
                block_dim: (BLOCK_SIZE, 1, 1),
                grid_dim: (grid_size, 1, 1),
                shared_mem_bytes: 0,
            };

            // Warmup (cudarc 0.19 launch_builder API)
            {
                let mut builder = stream.launch_builder(&func);
                builder.arg(&src);
                builder.arg(&mut dst);
                builder.arg(&n);
                unsafe {
                    builder.launch(cfg).map_err(|e| {
                        HarnessError::KernelFailed(format!("Warmup launch failed: {e}"))
                    })?;
                }
            }
            stream
                .synchronize()
                .map_err(|e| HarnessError::KernelFailed(format!("Warmup sync failed: {e}")))?;

            // Timed iterations
            let mut timings = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                let start = Instant::now();
                {
                    let mut builder = stream.launch_builder(&func);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    builder.arg(&n);
                    unsafe {
                        builder.launch(cfg).map_err(|e| {
                            HarnessError::KernelFailed(format!("Launch failed: {e}"))
                        })?;
                    }
                }
                stream
                    .synchronize()
                    .map_err(|e| HarnessError::KernelFailed(format!("Sync failed: {e}")))?;
                let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
                timings.push(elapsed_us);
            }

            Ok(timings)
        }

        /// Dispatch multi-buffer kernels (add, scale, triad).
        fn dispatch_multi_buffer_kernel(
            dev_info: &CudaDeviceInfo,
            kernel_name: &str,
            kernel_type: &str,
            buffer_size_bytes: usize,
            iterations: u32,
        ) -> Result<Vec<f64>, HarnessError> {
            let stream = &dev_info.stream;
            let element_count = buffer_size_bytes / 16; // float4 = 16 bytes

            // Compile PTX
            let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SOURCE).map_err(|e| {
                HarnessError::KernelFailed(format!("NVRTC compilation failed: {e}"))
            })?;

            let module = dev_info.ctx.load_module(ptx).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to load PTX module: {e}"))
            })?;

            let func = module.load_function(kernel_name).map_err(|e| {
                HarnessError::KernelFailed(format!(
                    "Kernel function '{kernel_name}' not found: {e}"
                ))
            })?;

            // Allocate buffers
            let src_a_data: Vec<f32> = (0..element_count * 4)
                .map(|i| (i as f32) * 0.001 + 1.0)
                .collect();
            let src_b_data: Vec<f32> = (0..element_count * 4)
                .map(|i| (i as f32) * 0.002 + 0.5)
                .collect();

            let src_a: CudaSlice<f32> = stream.clone_htod(&src_a_data).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to copy src_a to device: {e}"))
            })?;
            let src_b: CudaSlice<f32> = stream.clone_htod(&src_b_data).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to copy src_b to device: {e}"))
            })?;
            let mut dst: CudaSlice<f32> = stream.alloc_zeros(element_count * 4).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to allocate dst on device: {e}"))
            })?;

            let n = element_count as u32;
            let grid_size = n.div_ceil(BLOCK_SIZE);
            let scalar: f32 = 2.0;

            let cfg = LaunchConfig {
                block_dim: (BLOCK_SIZE, 1, 1),
                grid_dim: (grid_size, 1, 1),
                shared_mem_bytes: 0,
            };

            // Warmup
            {
                let mut builder = stream.launch_builder(&func);
                match kernel_type {
                    "scale" => {
                        builder.arg(&src_a);
                        builder.arg(&mut dst);
                        builder.arg(&scalar);
                        builder.arg(&n);
                    }
                    "add" => {
                        builder.arg(&src_a);
                        builder.arg(&src_b);
                        builder.arg(&mut dst);
                        builder.arg(&n);
                    }
                    "triad" => {
                        builder.arg(&src_a);
                        builder.arg(&src_b);
                        builder.arg(&mut dst);
                        builder.arg(&scalar);
                        builder.arg(&n);
                    }
                    _ => unreachable!(),
                }
                unsafe {
                    builder.launch(cfg).map_err(|e| {
                        HarnessError::KernelFailed(format!("Warmup launch failed: {e}"))
                    })?;
                }
            }
            stream
                .synchronize()
                .map_err(|e| HarnessError::KernelFailed(format!("Warmup sync failed: {e}")))?;

            // Timed iterations
            let mut timings = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                let start = Instant::now();
                {
                    let mut builder = stream.launch_builder(&func);
                    match kernel_type {
                        "scale" => {
                            builder.arg(&src_a);
                            builder.arg(&mut dst);
                            builder.arg(&scalar);
                            builder.arg(&n);
                        }
                        "add" => {
                            builder.arg(&src_a);
                            builder.arg(&src_b);
                            builder.arg(&mut dst);
                            builder.arg(&n);
                        }
                        "triad" => {
                            builder.arg(&src_a);
                            builder.arg(&src_b);
                            builder.arg(&mut dst);
                            builder.arg(&scalar);
                            builder.arg(&n);
                        }
                        _ => unreachable!(),
                    }
                    unsafe {
                        builder.launch(cfg).map_err(|e| {
                            HarnessError::KernelFailed(format!("Launch failed: {e}"))
                        })?;
                    }
                }
                stream
                    .synchronize()
                    .map_err(|e| HarnessError::KernelFailed(format!("Sync failed: {e}")))?;
                let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
                timings.push(elapsed_us);
            }

            Ok(timings)
        }

        /// Dispatch Tensor Core kernels (FP16/BF16 WMMA).
        /// These use 16x16 tile-based warp operations instead of per-element float4.
        fn dispatch_tensor_kernel(
            dev_info: &CudaDeviceInfo,
            kernel_name: &str,
            _kernel_type: &str,
            buffer_size_bytes: usize,
            iterations: u32,
        ) -> Result<Vec<f64>, HarnessError> {
            let stream = &dev_info.stream;

            // Each tile is 16x16 = 256 elements, 2 bytes each (fp16/bf16) = 512 bytes
            let tile_bytes = 256 * 2; // 16x16 tiles, 2 bytes per element
            let num_tiles = buffer_size_bytes / tile_bytes;

            // Compile tensor kernels with device's actual compute capability + mma.h support
            // WMMA requires sm_70+; use device arch for best codegen (sm_120 for Blackwell, etc.)
            let (cc_major, cc_minor) = dev_info.compute_capability;
            let arch_str = format!("sm_{}{}", cc_major, cc_minor);
            let compile_opts = cudarc::nvrtc::CompileOptions {
                arch: Some(&arch_str),
                include_paths: vec!["/usr/local/cuda/include".to_string()],
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(TENSOR_KERNEL_SOURCE, compile_opts)
                .map_err(|e| {
                    HarnessError::KernelFailed(format!("NVRTC Tensor Core compilation failed: {e}"))
                })?;

            let module = dev_info.ctx.load_module(ptx).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to load PTX module: {e}"))
            })?;

            let func = module.load_function(kernel_name).map_err(|e| {
                HarnessError::KernelFailed(format!(
                    "Tensor Core kernel '{kernel_name}' not found: {e}"
                ))
            })?;

            // Allocate source buffer (fp16 = u16 on host side)
            // Fill with small values to avoid overflow in matrix multiply
            let src_data: Vec<u16> = (0..num_tiles * 256)
                .map(|i| {
                    // Convert small float to fp16 bits
                    let val = ((i % 256) as f32) * 0.001 + 0.1;
                    half_from_f32(val)
                })
                .collect();

            let src: CudaSlice<u16> = stream.clone_htod(&src_data).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to copy tensor src to device: {e}"))
            })?;

            // Output is FP32 (accumulator)
            let mut dst: CudaSlice<f32> = stream.alloc_zeros(num_tiles * 256).map_err(|e| {
                HarnessError::KernelFailed(format!("Failed to allocate tensor dst on device: {e}"))
            })?;

            let n = num_tiles as u32;
            // Each warp (32 threads) processes one tile
            let warps_per_block = BLOCK_SIZE / 32;
            let grid_size = n.div_ceil(warps_per_block);

            let cfg = LaunchConfig {
                block_dim: (BLOCK_SIZE, 1, 1),
                grid_dim: (grid_size, 1, 1),
                shared_mem_bytes: 0,
            };

            // Warmup
            {
                let mut builder = stream.launch_builder(&func);
                builder.arg(&src);
                builder.arg(&mut dst);
                builder.arg(&n);
                unsafe {
                    builder.launch(cfg).map_err(|e| {
                        HarnessError::KernelFailed(format!("Tensor warmup failed: {e}"))
                    })?;
                }
            }
            stream
                .synchronize()
                .map_err(|e| HarnessError::KernelFailed(format!("Warmup sync failed: {e}")))?;

            // Timed iterations
            let mut timings = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                let start = Instant::now();
                {
                    let mut builder = stream.launch_builder(&func);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    builder.arg(&n);
                    unsafe {
                        builder.launch(cfg).map_err(|e| {
                            HarnessError::KernelFailed(format!("Tensor launch failed: {e}"))
                        })?;
                    }
                }
                stream
                    .synchronize()
                    .map_err(|e| HarnessError::KernelFailed(format!("Sync failed: {e}")))?;
                let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
                timings.push(elapsed_us);
            }

            Ok(timings)
        }
    }

    /// Convert f32 to IEEE 754 half-precision (fp16) bits.
    fn half_from_f32(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mantissa = (bits >> 13) & 0x3FF;
        if exp <= 0 {
            sign as u16 // Flush to zero for subnormals
        } else if exp >= 31 {
            (sign | 0x7C00) as u16 // Infinity
        } else {
            (sign | ((exp as u32) << 10) | mantissa) as u16
        }
    }

    impl GpuBackend for CudaBackend {
        fn run_kernel(
            &self,
            kernel: &KernelSpec,
            config: &RunConfig,
        ) -> Result<KernelResult, HarnessError> {
            let dev_info = self.devices.first().ok_or(HarnessError::NoDevice)?;

            let cuda_kernel_name = match kernel.name.as_str() {
                "copy" => "copy_kernel",
                "fma_light" => "fma_light_kernel",
                "fma_medium" => "fma_medium_kernel",
                "fma_heavy" => "fma_heavy_kernel",
                "scale" => "scale_kernel",
                "add" => "add_kernel",
                "triad" => "triad_kernel",
                "tensor_fp16" => "tensor_fp16_kernel",
                "tensor_bf16" => "tensor_bf16_kernel",
                other => {
                    return Err(HarnessError::KernelFailed(format!(
                        "Unknown kernel: {other}"
                    )));
                }
            };

            let timings = match kernel.name.as_str() {
                "scale" | "add" | "triad" => Self::dispatch_multi_buffer_kernel(
                    dev_info,
                    cuda_kernel_name,
                    &kernel.name,
                    config.buffer_size_bytes,
                    config.measurement_iterations,
                )?,
                "tensor_fp16" | "tensor_bf16" => Self::dispatch_tensor_kernel(
                    dev_info,
                    cuda_kernel_name,
                    &kernel.name,
                    config.buffer_size_bytes,
                    config.measurement_iterations,
                )?,
                _ => Self::dispatch_kernel(
                    dev_info,
                    cuda_kernel_name,
                    config.buffer_size_bytes,
                    config.measurement_iterations,
                )?,
            };

            let total_flops = match kernel.name.as_str() {
                "tensor_fp16" | "tensor_bf16" => {
                    // Tensor Core: each tile = 16x16x16 WMMA, 2 FLOPs per multiply-add
                    // = 8192 FLOPs per WMMA op. 32 ops per warp (8 lanes x 4 iterations)
                    let tile_bytes = 256 * 2; // 16x16 elements, 2 bytes each
                    let num_tiles = config.buffer_size_bytes / tile_bytes;
                    let wmma_ops_per_tile = 32u64; // 8 lanes x 4 iterations
                    let flops_per_wmma = 16 * 16 * 16 * 2; // 8192
                    num_tiles as u64 * wmma_ops_per_tile * flops_per_wmma
                }
                _ => {
                    let elements = config.buffer_size_bytes / 16; // float4 = 16 bytes
                                                                  // AI = FLOP / bytes_transferred. Bytes = read (16) + write (16) = 32
                    let flops_per_element = (kernel.arithmetic_intensity * 32.0) as u64;
                    elements as u64 * flops_per_element
                }
            };

            Ok(KernelResult {
                kernel_name: kernel.name.clone(),
                elapsed_us: timings,
                bytes_processed: config.buffer_size_bytes,
                flops_executed: total_flops,
            })
        }

        fn device_state(&self, device_index: u32) -> Result<DeviceState, HarnessError> {
            let dev = self
                .devices
                .get(device_index as usize)
                .ok_or(HarnessError::DeviceIndexOutOfRange(device_index))?;

            // Use NVML for real telemetry if available
            #[cfg(feature = "nvml")]
            if let Some(nvml) = &self.nvml {
                if let Ok(state) = nvml.query_state(device_index) {
                    return Ok(state);
                }
            }

            // Fallback: only memory total from CUDA API
            Ok(DeviceState {
                clock_mhz: 0,
                temperature_c: 0,
                power_watts: 0.0,
                memory_used_bytes: 0,
                memory_total_bytes: dev.total_memory_bytes as u64,
                utilization_pct: 0.0,
            })
        }

        fn discover_devices(&self) -> Result<Vec<GpuDevice>, HarnessError> {
            // Get NVML-enriched data if available
            #[cfg(feature = "nvml")]
            let driver_version = self.nvml.as_ref().and_then(|n| n.driver_version());

            #[cfg(not(feature = "nvml"))]
            let driver_version: Option<String> = None;

            Ok(self
                .devices
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let arch = GpuArchitecture::from_compute_capability(
                        d.compute_capability.0 as u32,
                        d.compute_capability.1 as u32,
                    );

                    // Get PCI bus ID from NVML if available
                    #[cfg(feature = "nvml")]
                    let pci_bus_id = self.nvml.as_ref().and_then(|n| n.pci_bus_id(i as u32));

                    #[cfg(not(feature = "nvml"))]
                    let pci_bus_id: Option<String> = None;

                    GpuDevice {
                        index: i as u32,
                        name: d.name.clone(),
                        vendor: GpuVendor::Nvidia,
                        architecture: arch,
                        memory_bytes: d.total_memory_bytes as u64,
                        pci_bus_id,
                        driver_version: driver_version.clone(),
                        features: GpuFeatures {
                            timestamp_queries: true,
                            shader_f16: d.compute_capability.0 >= 7,
                            shader_int64: d.compute_capability.0 >= 6,
                            compute_capability: Some((
                                d.compute_capability.0 as u32,
                                d.compute_capability.1 as u32,
                            )),
                            tensor_cores: d.compute_capability.0 >= 7,
                            rt_cores: d.compute_capability.0 >= 7,
                        },
                        limits: GpuLimits::default(),
                    }
                })
                .collect())
        }

        fn p2p_bandwidth(&self, _src: u32, _dst: u32) -> Result<BandwidthResult, HarnessError> {
            Err(HarnessError::FeatureNotSupported(
                "P2P bandwidth measurement not yet implemented for CUDA backend".to_string(),
            ))
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::CudaBackend;
