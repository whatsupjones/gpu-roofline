use thiserror::Error;

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("no GPU device found")]
    NoDevice,

    #[error("device index {0} out of range")]
    DeviceIndexOutOfRange(u32),

    #[error("feature not supported: {0}")]
    FeatureNotSupported(String),

    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("kernel execution failed: {0}")]
    KernelFailed(String),

    #[error("measurement failed: {0}")]
    MeasurementFailed(String),

    #[error("simulation error: {0}")]
    SimulationError(String),

    #[error("vGPU detection failed: {0}")]
    VgpuDetectionFailed(String),

    #[error("vGPU not supported on this platform/hardware")]
    VgpuNotSupported,

    #[error("vGPU instance not found: {0}")]
    VgpuInstanceNotFound(String),
}
