//! vGPU lifecycle monitoring: detect provisioning, measure contention,
//! verify teardown — from birth to death.

pub mod contention;
pub mod detect;
pub mod sim;
pub mod state;
pub mod teardown;

pub use state::{
    PartitioningMode, TeardownVerification, VgpuEvent, VgpuEventType, VgpuInstance, VgpuPhase,
    VgpuSnapshot, VgpuState, VgpuTechnology,
};

pub use detect::{
    auto_detect, CloudPassthroughDetector, CompositeDetector, K8sDevicePluginDetector,
    NvidiaGridDetector, NvidiaMigDetector, SrIovDetector, VgpuDetector,
};

pub use contention::ContentionMeasurer;
pub use teardown::TeardownVerifier;

pub use sim::{
    available_scenarios, scenario_by_name, ScheduledEvent, SimAction, SimulatedDetector,
    VgpuSimScenario,
};
