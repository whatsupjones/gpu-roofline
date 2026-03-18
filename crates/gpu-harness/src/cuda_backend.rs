//! CUDA compute backend for datacenter GPUs (H100, H200, A100).
//!
//! Uses cudarc for dynamic CUDA driver loading — no CUDA toolkit needed
//! at compile time. Kernels compiled at runtime via NVRTC (PTX).

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;
    use std::time::Instant;

    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    use crate::backend::{
        BandwidthResult, DeviceState, GpuBackend, KernelResult, KernelSpec, RunConfig,
    };
    use crate::device::{GpuArchitecture, GpuDevice, GpuFeatures, GpuLimits, GpuVendor};
    use crate::error::HarnessError;

    const CUDA_KERNEL_SOURCE: &str =
        include_str!("../../gpu-roofline/shaders/cuda/roofline_kernels.cu");
    const BLOCK_SIZE: u32 = 256;

    /// CUDA backend for datacenter GPU compute.
    pub struct CudaBackend {
        devices: Vec<CudaDeviceInfo>,
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

            Ok(Self { devices })
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
                "scale" | "add" | "triad" => {
                    tracing::info!(
                        "Skipping kernel '{}' (multi-buffer not yet supported in CUDA backend)",
                        kernel.name
                    );
                    return Ok(KernelResult {
                        kernel_name: kernel.name.clone(),
                        elapsed_us: vec![0.0; config.measurement_iterations as usize],
                        bytes_processed: config.buffer_size_bytes,
                        flops_executed: 0,
                    });
                }
                other => {
                    return Err(HarnessError::KernelFailed(format!(
                        "Unknown kernel: {other}"
                    )));
                }
            };

            let timings = Self::dispatch_kernel(
                dev_info,
                cuda_kernel_name,
                config.buffer_size_bytes,
                config.measurement_iterations,
            )?;

            let elements = config.buffer_size_bytes / 16; // float4 = 16 bytes
                                                          // AI = FLOP / bytes_transferred. Bytes = read (16) + write (16) = 32
            let flops_per_element = (kernel.arithmetic_intensity * 32.0) as u64;
            let total_flops = elements as u64 * flops_per_element;

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
            Ok(self
                .devices
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let arch = GpuArchitecture::from_compute_capability(
                        d.compute_capability.0 as u32,
                        d.compute_capability.1 as u32,
                    );

                    GpuDevice {
                        index: i as u32,
                        name: d.name.clone(),
                        vendor: GpuVendor::Nvidia,
                        architecture: arch,
                        memory_bytes: d.total_memory_bytes as u64,
                        pci_bus_id: None,
                        driver_version: None,
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
