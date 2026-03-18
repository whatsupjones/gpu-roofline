//! Real GPU backend using wgpu for cross-vendor compute shader dispatch.
//!
//! Supports multiple graphics API backends (Vulkan, DX12, Metal) with
//! auto-detection and graceful fallback for headless/datacenter environments.

#[cfg(feature = "wgpu-backend")]
mod inner {
    use std::time::Instant;

    use wgpu::{self, util::DeviceExt};

    use crate::backend::{
        BandwidthResult, DeviceState, GpuBackend, KernelResult, KernelSpec, RunConfig,
    };
    use crate::device::{GpuArchitecture, GpuDevice, GpuFeatures, GpuLimits, GpuVendor};
    use crate::error::HarnessError;

    /// Graphics API backend selection.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum GpuApiBackend {
        /// Auto-detect best available backend.
        Auto,
        /// Force Vulkan (Linux default, also available on Windows).
        Vulkan,
        /// Force DirectX 12 (Windows only).
        Dx12,
        /// Force Metal (macOS/iOS only).
        Metal,
        /// Force OpenGL (broad compatibility fallback).
        Gl,
    }

    impl GpuApiBackend {
        fn to_wgpu_backends(self) -> wgpu::Backends {
            match self {
                Self::Auto => wgpu::Backends::all(),
                Self::Vulkan => wgpu::Backends::VULKAN,
                Self::Dx12 => wgpu::Backends::DX12,
                Self::Metal => wgpu::Backends::METAL,
                Self::Gl => wgpu::Backends::GL,
            }
        }
    }

    /// Detected environment metadata for a GPU adapter.
    #[derive(Debug, Clone)]
    pub struct AdapterMetadata {
        /// GPU name from driver.
        pub name: String,
        /// Graphics API backend in use (Vulkan, DX12, Metal, GL).
        pub backend: String,
        /// Driver version string.
        pub driver: String,
        /// Driver description.
        pub driver_info: String,
        /// Device type (Discrete, Integrated, Virtual, CPU).
        pub device_type: String,
        /// Whether this appears to be a virtual/cloud GPU.
        pub is_virtual: bool,
        /// Vendor ID.
        pub vendor_id: u32,
        /// Device ID.
        pub device_id: u32,
    }

    /// Real GPU backend using wgpu for compute shader dispatch.
    pub struct WgpuBackend {
        adapters: Vec<AdapterEntry>,
        /// Metadata about each detected adapter.
        pub metadata: Vec<AdapterMetadata>,
        /// NVML telemetry for NVIDIA GPUs (enriches device_state).
        #[cfg(feature = "nvml")]
        nvml: Option<crate::nvml_telemetry::NvmlTelemetry>,
    }

    struct AdapterEntry {
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
    }

    impl WgpuBackend {
        /// Create a new wgpu backend with auto-detected API backend.
        pub fn new() -> Result<Self, HarnessError> {
            Self::with_backend(GpuApiBackend::Auto)
        }

        /// Create a new wgpu backend with a specific API backend.
        pub fn with_backend(api: GpuApiBackend) -> Result<Self, HarnessError> {
            // Catch panics from wgpu initialization (e.g., missing Vulkan ICD)
            let result = std::panic::catch_unwind(|| Self::init_backend(api));

            match result {
                Ok(Ok(backend)) => Ok(backend),
                Ok(Err(e)) => Err(e),
                Err(_panic) => {
                    let hint = match api {
                        GpuApiBackend::Vulkan => {
                            "Vulkan initialization failed. This often happens on headless \
                             datacenter GPUs without Vulkan ICD installed.\n\
                             Try: --backend dx12 (Windows), --backend auto, or run inside a \
                             Docker container with nvidia/vulkan image."
                        }
                        GpuApiBackend::Dx12 => {
                            "DirectX 12 initialization failed. Ensure you're on Windows 10+ \
                             with a compatible GPU driver."
                        }
                        _ => {
                            "GPU backend initialization failed. Try --backend auto to let \
                             gpu-roofline detect the best available backend, or use \
                             --sim <profile> for simulation mode."
                        }
                    };
                    Err(HarnessError::BackendUnavailable(hint.to_string()))
                }
            }
        }

        fn init_backend(api: GpuApiBackend) -> Result<Self, HarnessError> {
            let backends = api.to_wgpu_backends();

            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                ..Default::default()
            });

            let adapters_raw = instance.enumerate_adapters(backends);

            if adapters_raw.is_empty() {
                return Err(HarnessError::NoDevice);
            }

            let mut adapters = Vec::new();
            let mut metadata = Vec::new();

            for adapter in adapters_raw {
                let info = adapter.get_info();

                // Skip CPU/software adapters
                if info.device_type == wgpu::DeviceType::Cpu {
                    continue;
                }

                // Detect virtual/cloud GPU
                let is_virtual = info.device_type == wgpu::DeviceType::VirtualGpu
                    || info.name.to_lowercase().contains("virtual")
                    || info.name.to_lowercase().contains("grid")
                    || info.driver_info.to_lowercase().contains("virtual");

                let backend_name = format!("{:?}", info.backend);
                let device_type_name = match info.device_type {
                    wgpu::DeviceType::DiscreteGpu => "Discrete",
                    wgpu::DeviceType::IntegratedGpu => "Integrated",
                    wgpu::DeviceType::VirtualGpu => "Virtual",
                    wgpu::DeviceType::Cpu => "CPU",
                    wgpu::DeviceType::Other => "Other",
                };

                metadata.push(AdapterMetadata {
                    name: info.name.clone(),
                    backend: backend_name,
                    driver: info.driver.clone(),
                    driver_info: info.driver_info.clone(),
                    device_type: device_type_name.to_string(),
                    is_virtual,
                    vendor_id: info.vendor,
                    device_id: info.device,
                });

                // Request device with timestamp queries if available
                let features = adapter.features();
                let required = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                };

                // Configure wgpu to NOT panic on errors
                match pollster::block_on(adapter.request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some(&format!("gpu-roofline-{}", info.name)),
                        required_features: required,
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )) {
                    Ok((device, queue)) => {
                        // Set error handler to log instead of panic
                        device.on_uncaptured_error(Box::new(|err| {
                            tracing::error!("wgpu device error: {}", err);
                        }));

                        adapters.push(AdapterEntry {
                            adapter,
                            device,
                            queue,
                            info,
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create device for {}: {}", info.name, e);
                    }
                }
            }

            if adapters.is_empty() {
                return Err(HarnessError::NoDevice);
            }

            // Log what we found
            for meta in &metadata {
                tracing::info!(
                    "Detected GPU: {} | {} | {} | Driver: {} | {}",
                    meta.name,
                    meta.backend,
                    meta.device_type,
                    meta.driver,
                    if meta.is_virtual {
                        "VIRTUAL (cloud passthrough detected)"
                    } else {
                        "bare metal"
                    }
                );
            }

            // Try NVML for NVIDIA GPU telemetry
            #[cfg(feature = "nvml")]
            let nvml = {
                // Only init NVML if we have an NVIDIA adapter
                let has_nvidia = metadata.iter().any(|m| m.vendor_id == 0x10DE);
                if has_nvidia {
                    match crate::nvml_telemetry::NvmlTelemetry::new() {
                        Ok(n) => {
                            tracing::info!("NVML telemetry available for NVIDIA GPUs");
                            Some(n)
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                }
            };

            Ok(Self {
                adapters,
                metadata,
                #[cfg(feature = "nvml")]
                nvml,
            })
        }

        /// Run a compute shader on a specific device.
        fn dispatch_kernel(
            adapter_entry: &AdapterEntry,
            shader_source: &str,
            buffer_size_bytes: usize,
            iterations: u32,
        ) -> Result<Vec<f64>, HarnessError> {
            let device = &adapter_entry.device;
            let queue = &adapter_entry.queue;

            // Create shader module
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("roofline-kernel"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            // Create buffers — src (read) + dst (read-write)
            let element_count = buffer_size_bytes / 16; // vec4<f32> = 16 bytes
            let src_data: Vec<f32> = (0..element_count * 4)
                .map(|i| (i as f32) * 0.001 + 1.0)
                .collect();

            let src_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("src"),
                contents: bytemuck::cast_slice(&src_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dst"),
                size: buffer_size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Use auto-layout — let wgpu derive the bind group layout from the shader
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("roofline-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let bind_group_layout = pipeline.get_bind_group_layout(0);

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("roofline-bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buffer.as_entire_binding(),
                    },
                ],
            });

            let workgroups = (element_count as u32).div_ceil(256);

            // Warmup dispatch
            {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
            }

            // Timed iterations
            let mut timings = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }

                let start = Instant::now();
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
                let elapsed_us = start.elapsed().as_secs_f64() * 1e6;

                timings.push(elapsed_us);
            }

            Ok(timings)
        }
    }

    impl GpuBackend for WgpuBackend {
        fn run_kernel(
            &self,
            kernel: &KernelSpec,
            config: &RunConfig,
        ) -> Result<KernelResult, HarnessError> {
            let adapter_entry = self.adapters.first().ok_or(HarnessError::NoDevice)?;

            // Look up WGSL source for built-in kernels.
            // Only 2-binding kernels (src + dst) supported in v0.1.
            let wgsl_source = match kernel.name.as_str() {
                "copy" => include_str!("../shaders/copy.wgsl"),
                "fma_light" => include_str!("../shaders/fma_light.wgsl"),
                "fma_medium" => include_str!("../shaders/fma_medium.wgsl"),
                "fma_heavy" => include_str!("../shaders/fma_heavy.wgsl"),
                "scale" | "add" | "triad" => {
                    tracing::info!(
                        "Skipping kernel '{}' (multi-binding not yet supported on real GPU)",
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
                adapter_entry,
                wgsl_source,
                config.buffer_size_bytes,
                config.measurement_iterations,
            )?;

            let elements = config.buffer_size_bytes / 16; // float4/vec4 = 16 bytes
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
            let _adapter = self
                .adapters
                .get(device_index as usize)
                .ok_or(HarnessError::DeviceIndexOutOfRange(device_index))?;

            // Use NVML for NVIDIA GPUs if available
            #[cfg(feature = "nvml")]
            if let Some(nvml) = &self.nvml {
                // NVML device index may differ from wgpu adapter index.
                // For single-GPU systems this is fine; multi-GPU would need PCI matching.
                if let Ok(state) = nvml.query_state(device_index) {
                    return Ok(state);
                }
            }

            // Fallback: wgpu doesn't expose thermal/power telemetry
            Ok(DeviceState {
                clock_mhz: 0,
                temperature_c: 0,
                power_watts: 0.0,
                memory_used_bytes: 0,
                memory_total_bytes: 0,
                utilization_pct: 0.0,
            })
        }

        fn discover_devices(&self) -> Result<Vec<GpuDevice>, HarnessError> {
            Ok(self
                .adapters
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let vendor = GpuVendor::from_vendor_id(a.info.vendor);
                    let arch = GpuArchitecture::detect(vendor, &a.info.name);
                    let features = a.adapter.features();
                    let limits = a.adapter.limits();

                    GpuDevice {
                        index: i as u32,
                        name: a.info.name.clone(),
                        vendor,
                        architecture: arch,
                        memory_bytes: 0,
                        pci_bus_id: None,
                        driver_version: Some(a.info.driver.clone()),
                        features: GpuFeatures {
                            timestamp_queries: features.contains(wgpu::Features::TIMESTAMP_QUERY),
                            shader_f16: features.contains(wgpu::Features::SHADER_F16),
                            shader_int64: false,
                            compute_capability: None,
                            tensor_cores: matches!(
                                arch,
                                GpuArchitecture::Blackwell
                                    | GpuArchitecture::Hopper
                                    | GpuArchitecture::Ada
                                    | GpuArchitecture::Ampere
                            ),
                            rt_cores: matches!(
                                arch,
                                GpuArchitecture::Blackwell
                                    | GpuArchitecture::Ada
                                    | GpuArchitecture::Ampere
                            ),
                        },
                        limits: GpuLimits {
                            max_buffer_size: limits.max_buffer_size,
                            max_workgroup_size: [
                                limits.max_compute_workgroup_size_x,
                                limits.max_compute_workgroup_size_y,
                                limits.max_compute_workgroup_size_z,
                            ],
                            max_workgroups: [
                                limits.max_compute_workgroups_per_dimension,
                                limits.max_compute_workgroups_per_dimension,
                                limits.max_compute_workgroups_per_dimension,
                            ],
                            max_storage_buffers: limits.max_storage_buffers_per_shader_stage,
                        },
                    }
                })
                .collect())
        }

        fn p2p_bandwidth(&self, _src: u32, _dst: u32) -> Result<BandwidthResult, HarnessError> {
            Err(HarnessError::FeatureNotSupported(
                "P2P bandwidth measurement requires CUDA or platform-specific API".to_string(),
            ))
        }
    }
}

#[cfg(feature = "wgpu-backend")]
pub use inner::{AdapterMetadata, GpuApiBackend, WgpuBackend};
