use serde::{Deserialize, Serialize};

/// Represents a discovered GPU device with all known properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub vendor: GpuVendor,
    pub architecture: GpuArchitecture,
    pub memory_bytes: u64,
    pub pci_bus_id: Option<String>,
    pub driver_version: Option<String>,
    pub features: GpuFeatures,
    pub limits: GpuLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

impl GpuVendor {
    /// Classify vendor from PCI vendor ID.
    pub fn from_vendor_id(id: u32) -> Self {
        match id {
            0x10DE => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            _ => Self::Other,
        }
    }
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvidia => write!(f, "NVIDIA"),
            Self::Amd => write!(f, "AMD"),
            Self::Intel => write!(f, "Intel"),
            Self::Apple => write!(f, "Apple"),
            Self::Other => write!(f, "Unknown"),
        }
    }
}

/// GPU architecture generation, identified from device name or compute capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuArchitecture {
    // NVIDIA
    Blackwell,
    Hopper,
    Ada,
    Ampere,
    NvidiaOther,

    // AMD
    Rdna4,
    Rdna3,
    Cdna3,
    AmdOther,

    // Intel
    Battlemage,
    ArcAlchemist,
    IntelOther,

    // Other
    AppleSilicon,
    Integrated,
    Unknown,
}

impl GpuArchitecture {
    /// Classify NVIDIA architecture from device name.
    pub fn detect_nvidia(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("5090")
            || lower.contains("5080")
            || lower.contains("5070")
            || lower.contains("gb2")
            || lower.contains("blackwell")
            || lower.contains("b200")
            || lower.contains("b100")
        {
            Self::Blackwell
        } else if lower.contains("h100")
            || lower.contains("h200")
            || lower.contains("h800")
            || lower.contains("hopper")
            || lower.contains("gh")
        {
            Self::Hopper
        } else if lower.contains("4090")
            || lower.contains("4080")
            || lower.contains("4070")
            || lower.contains("4060")
            || lower.contains("ada")
            || lower.contains("l40")
            || lower.contains("ad1")
        {
            Self::Ada
        } else if lower.contains("3090")
            || lower.contains("3080")
            || lower.contains("3070")
            || lower.contains("3060")
            || lower.contains("a100")
            || lower.contains("a30")
            || lower.contains("ampere")
        {
            Self::Ampere
        } else {
            Self::NvidiaOther
        }
    }

    /// Classify NVIDIA architecture from CUDA compute capability.
    pub fn from_compute_capability(major: u32, minor: u32) -> Self {
        match (major, minor) {
            (10, _) => Self::Blackwell,
            (9, _) => Self::Hopper,
            (8, 9) => Self::Ada,
            (8, _) => Self::Ampere,
            _ => Self::NvidiaOther,
        }
    }

    /// Classify AMD architecture from device name.
    pub fn detect_amd(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("rx 8") || lower.contains("rdna 4") {
            Self::Rdna4
        } else if lower.contains("rx 7") || lower.contains("rdna 3") {
            Self::Rdna3
        } else if lower.contains("mi300") || lower.contains("mi250") || lower.contains("cdna") {
            Self::Cdna3
        } else {
            Self::AmdOther
        }
    }

    /// Classify Intel architecture from device name.
    pub fn detect_intel(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("battlemage") || lower.contains("b5") || lower.contains("b7") {
            Self::Battlemage
        } else if lower.contains("arc") || lower.contains("a7") || lower.contains("a5") {
            Self::ArcAlchemist
        } else if lower.contains("uhd") || lower.contains("iris") {
            Self::Integrated
        } else {
            Self::IntelOther
        }
    }

    /// Detect architecture from vendor + device name.
    pub fn detect(vendor: GpuVendor, name: &str) -> Self {
        match vendor {
            GpuVendor::Nvidia => Self::detect_nvidia(name),
            GpuVendor::Amd => Self::detect_amd(name),
            GpuVendor::Intel => Self::detect_intel(name),
            GpuVendor::Apple => Self::AppleSilicon,
            GpuVendor::Other => Self::Unknown,
        }
    }
}

impl std::fmt::Display for GpuArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blackwell => write!(f, "Blackwell"),
            Self::Hopper => write!(f, "Hopper"),
            Self::Ada => write!(f, "Ada Lovelace"),
            Self::Ampere => write!(f, "Ampere"),
            Self::NvidiaOther => write!(f, "NVIDIA (older)"),
            Self::Rdna4 => write!(f, "RDNA 4"),
            Self::Rdna3 => write!(f, "RDNA 3"),
            Self::Cdna3 => write!(f, "CDNA 3"),
            Self::AmdOther => write!(f, "AMD (older)"),
            Self::Battlemage => write!(f, "Battlemage"),
            Self::ArcAlchemist => write!(f, "Arc Alchemist"),
            Self::IntelOther => write!(f, "Intel (older)"),
            Self::AppleSilicon => write!(f, "Apple Silicon"),
            Self::Integrated => write!(f, "Integrated"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// GPU feature capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuFeatures {
    pub timestamp_queries: bool,
    pub shader_f16: bool,
    pub shader_int64: bool,
    pub compute_capability: Option<(u32, u32)>,
    pub tensor_cores: bool,
    pub rt_cores: bool,
}

/// GPU hardware limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuLimits {
    pub max_buffer_size: u64,
    pub max_workgroup_size: [u32; 3],
    pub max_workgroups: [u32; 3],
    pub max_storage_buffers: u32,
}

impl Default for GpuLimits {
    fn default() -> Self {
        Self {
            max_buffer_size: 128 * 1024 * 1024,
            max_workgroup_size: [256, 256, 64],
            max_workgroups: [65535, 65535, 65535],
            max_storage_buffers: 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvidia_architecture_detection() {
        assert_eq!(
            GpuArchitecture::detect_nvidia("NVIDIA GeForce RTX 5090"),
            GpuArchitecture::Blackwell
        );
        assert_eq!(
            GpuArchitecture::detect_nvidia("NVIDIA H100 80GB HBM3"),
            GpuArchitecture::Hopper
        );
        assert_eq!(
            GpuArchitecture::detect_nvidia("NVIDIA GeForce RTX 4090"),
            GpuArchitecture::Ada
        );
        assert_eq!(
            GpuArchitecture::detect_nvidia("NVIDIA A100-SXM4-80GB"),
            GpuArchitecture::Ampere
        );
        assert_eq!(
            GpuArchitecture::detect_nvidia("NVIDIA GeForce GTX 1080"),
            GpuArchitecture::NvidiaOther
        );
    }

    #[test]
    fn test_compute_capability_mapping() {
        assert_eq!(
            GpuArchitecture::from_compute_capability(10, 0),
            GpuArchitecture::Blackwell
        );
        assert_eq!(
            GpuArchitecture::from_compute_capability(9, 0),
            GpuArchitecture::Hopper
        );
        assert_eq!(
            GpuArchitecture::from_compute_capability(8, 9),
            GpuArchitecture::Ada
        );
        assert_eq!(
            GpuArchitecture::from_compute_capability(8, 0),
            GpuArchitecture::Ampere
        );
    }

    #[test]
    fn test_amd_architecture_detection() {
        assert_eq!(
            GpuArchitecture::detect_amd("AMD Radeon RX 7900 XTX"),
            GpuArchitecture::Rdna3
        );
        assert_eq!(
            GpuArchitecture::detect_amd("AMD Instinct MI300X"),
            GpuArchitecture::Cdna3
        );
    }

    #[test]
    fn test_intel_architecture_detection() {
        assert_eq!(
            GpuArchitecture::detect_intel("Intel Arc A770"),
            GpuArchitecture::ArcAlchemist
        );
        assert_eq!(
            GpuArchitecture::detect_intel("Intel UHD Graphics 770"),
            GpuArchitecture::Integrated
        );
    }

    #[test]
    fn test_vendor_from_id() {
        assert_eq!(GpuVendor::from_vendor_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_vendor_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_vendor_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_vendor_id(0xFFFF), GpuVendor::Other);
    }
}
