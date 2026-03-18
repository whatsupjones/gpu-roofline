//! Real GPU backend using wgpu for cross-vendor compute shader dispatch.

#[cfg(feature = "wgpu-backend")]
mod inner {
    use std::time::Instant;

    use wgpu::{self, util::DeviceExt};

    use crate::backend::{
        BandwidthResult, DeviceState, GpuBackend, KernelResult, KernelSpec, RunConfig,
    };
    use crate::device::{GpuArchitecture, GpuDevice, GpuFeatures, GpuLimits, GpuVendor};
    use crate::error::HarnessError;

    /// Real GPU backend using wgpu for compute shader dispatch.
    pub struct WgpuBackend {
        _instance: wgpu::Instance,
        adapters: Vec<AdapterInfo>,
    }

    struct AdapterInfo {
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
    }

    impl WgpuBackend {
        /// Create a new wgpu backend, discovering all available GPUs.
        pub fn new() -> Result<Self, HarnessError> {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapters_raw = instance.enumerate_adapters(wgpu::Backends::all());

            if adapters_raw.is_empty() {
                return Err(HarnessError::NoDevice);
            }

            let mut adapters = Vec::new();
            for adapter in adapters_raw {
                let info = adapter.get_info();

                // Skip CPU/software adapters
                if info.device_type == wgpu::DeviceType::Cpu {
                    continue;
                }

                // Request device with timestamp queries if available
                let features = adapter.features();
                let required = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                };

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
                        adapters.push(AdapterInfo {
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

            Ok(Self {
                _instance: instance,
                adapters,
            })
        }

        /// Run a compute shader on a specific device.
        fn dispatch_kernel(
            adapter_info: &AdapterInfo,
            shader_source: &str,
            buffer_size_bytes: usize,
            iterations: u32,
        ) -> Result<Vec<f64>, HarnessError> {
            let device = &adapter_info.device;
            let queue = &adapter_info.queue;

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
                layout: None, // Auto-layout from shader reflection
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
            let adapter_info = self.adapters.first().ok_or(HarnessError::NoDevice)?;

            // Look up WGSL source for built-in kernels.
            // Only 2-binding kernels (src + dst) are supported in v0.1.
            // 3-binding kernels (add, triad) need separate bind group layout — v0.2.
            let wgsl_source = match kernel.name.as_str() {
                "copy" => include_str!("../../gpu-roofline/shaders/copy.wgsl"),
                "fma_light" => include_str!("../../gpu-roofline/shaders/fma_light.wgsl"),
                "fma_medium" => include_str!("../../gpu-roofline/shaders/fma_medium.wgsl"),
                "fma_heavy" => include_str!("../../gpu-roofline/shaders/fma_heavy.wgsl"),
                "scale" | "add" | "triad" => {
                    // These kernels need >2 bindings (uniform or 3rd buffer).
                    // Skip gracefully — return synthetic result so measurement continues.
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
                adapter_info,
                wgsl_source,
                config.buffer_size_bytes,
                config.measurement_iterations,
            )?;

            let elements = config.buffer_size_bytes / 16; // vec4<f32>
            let flops_per_element = (kernel.arithmetic_intensity * 16.0) as u64; // AI * bytes_per_element
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

            // wgpu doesn't expose thermal/power telemetry — return defaults.
            // NVML feature would enrich this.
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
                        memory_bytes: 0, // wgpu doesn't expose VRAM size directly
                        pci_bus_id: None,
                        driver_version: Some(a.info.driver.clone()),
                        features: GpuFeatures {
                            timestamp_queries: features.contains(wgpu::Features::TIMESTAMP_QUERY),
                            shader_f16: features.contains(wgpu::Features::SHADER_F16),
                            shader_int64: false, // wgpu doesn't expose this directly
                            compute_capability: None, // CUDA-specific
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
pub use inner::WgpuBackend;
