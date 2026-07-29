pub const POINTS_PER_CHUNK: usize = 32 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FieldType {
    Unknown = 0,
    Int8 = 1,
    Uint8 = 2,
    Int16 = 3,
    Uint16 = 4,
    Int32 = 5,
    Uint32 = 6,
    Float32 = 7,
    Float64 = 8,
    Int64 = 9,
    Uint64 = 10,
}

impl FieldType {
    pub fn size_of(self) -> usize {
        match self {
            Self::Int8 | Self::Uint8 => 1,
            Self::Int16 | Self::Uint16 => 2,
            Self::Int32 | Self::Uint32 | Self::Float32 => 4,
            Self::Float64 | Self::Int64 | Self::Uint64 => 8,
            Self::Unknown => 0,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "UNKNOWN",
            Self::Int8 => "INT8",
            Self::Uint8 => "UINT8",
            Self::Int16 => "INT16",
            Self::Uint16 => "UINT16",
            Self::Int32 => "INT32",
            Self::Uint32 => "UINT32",
            Self::Float32 => "FLOAT32",
            Self::Float64 => "FLOAT64",
            Self::Int64 => "INT64",
            Self::Uint64 => "UINT64",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_uppercase().as_str() {
            "UNKNOWN" | "0" => Some(Self::Unknown),
            "INT8" | "1" => Some(Self::Int8),
            "UINT8" | "2" => Some(Self::Uint8),
            "INT16" | "3" => Some(Self::Int16),
            "UINT16" | "4" => Some(Self::Uint16),
            "INT32" | "5" => Some(Self::Int32),
            "UINT32" | "6" => Some(Self::Uint32),
            "FLOAT32" | "7" => Some(Self::Float32),
            "FLOAT64" | "8" => Some(Self::Float64),
            "INT64" | "9" => Some(Self::Int64),
            "UINT64" | "10" => Some(Self::Uint64),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EncodingOptions {
    None = 0,
    Lossy = 1,
    Lossless = 2,
}

impl EncodingOptions {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Lossy => "LOSSY",
            Self::Lossless => "LOSSLESS",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_uppercase().as_str() {
            "NONE" | "0" => Some(Self::None),
            "LOSSY" | "1" => Some(Self::Lossy),
            "LOSSLESS" | "2" => Some(Self::Lossless),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CompressionOption {
    None = 0,
    Lz4 = 1,
    Zstd = 2,
}

impl CompressionOption {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Lz4 => "LZ4",
            Self::Zstd => "ZSTD",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_uppercase().as_str() {
            "NONE" | "0" => Some(Self::None),
            "LZ4" | "1" => Some(Self::Lz4),
            "ZSTD" | "2" => Some(Self::Zstd),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PointField {
    pub name: String,
    pub offset: u32,
    pub field_type: FieldType,
    pub resolution: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodingInfo {
    pub fields: Vec<PointField>,
    pub width: u32,
    pub height: u32,
    pub point_step: u32,
    pub encoding_opt: EncodingOptions,
    pub compression_opt: CompressionOption,
    pub encoding_config: String,
    pub version: u8,
}

impl Default for EncodingInfo {
    fn default() -> Self {
        Self {
            fields: Vec::new(),
            width: 0,
            height: 1,
            point_step: 0,
            encoding_opt: EncodingOptions::Lossy,
            compression_opt: CompressionOption::Zstd,
            encoding_config: String::new(),
            version: crate::codec::ENCODING_VERSION,
        }
    }
}
