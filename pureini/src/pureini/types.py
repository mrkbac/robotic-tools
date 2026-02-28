"""
Core data types for Pureini point cloud compression.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

from dataclasses import dataclass, field
from enum import IntEnum


class FieldType(IntEnum):
    """Field types for point cloud data. Values 1-8 match sensor_msgs/PointField.msg."""

    UNKNOWN = 0
    INT8 = 1
    UINT8 = 2
    INT16 = 3
    UINT16 = 4
    INT32 = 5
    UINT32 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    INT64 = 9
    UINT64 = 10


class EncodingOptions(IntEnum):
    """First stage of encoding using custom field encoding."""

    # No encoding, compression done in second stage only (benchmarking only)
    NONE = 0
    # Lossy compression for FLOAT32 fields
    LOSSY = 1
    # Lossless compression (e.g. XOR for FLOAT32) - currently poor performance
    LOSSLESS = 2


class CompressionOption(IntEnum):
    """Second stage of encoding using general-purpose compression."""

    NONE = 0
    LZ4 = 1
    ZSTD = 2


# Constants
ENCODING_VERSION = 3
MAGIC_HEADER = b"CLOUDINI_V"
MAGIC_HEADER_LENGTH = 10
DECODE_BUT_SKIP_STORE = 0xFFFFFFFF  # uint32_t max value
POINTS_PER_CHUNK = 32768


def sizeof_field_type(field_type: FieldType) -> int:
    """Return size in bytes for a given field type."""
    size_map = {
        FieldType.INT8: 1,
        FieldType.UINT8: 1,
        FieldType.INT16: 2,
        FieldType.UINT16: 2,
        FieldType.INT32: 4,
        FieldType.UINT32: 4,
        FieldType.FLOAT32: 4,
        FieldType.FLOAT64: 8,
        FieldType.INT64: 8,
        FieldType.UINT64: 8,
    }
    return size_map.get(field_type, 0)


@dataclass(slots=True, eq=True)
class PointField:
    """
    Field definition for a point cloud.

    Attributes:
        name: Name of the field
        offset: Offset in bytes from the start of the point
        type: Data type of the field
        resolution: Optional resolution for lossy compression (max error = 0.5 * resolution)
    """

    name: str
    offset: int = 0
    type: FieldType = FieldType.UNKNOWN
    resolution: float | None = None


@dataclass(slots=True, eq=True)
class EncodingInfo:
    """
    Complete encoding configuration for a point cloud.

    Attributes:
        fields: List of field definitions
        width: Number of points (when height == 1) or width of organized cloud
        height: Height of organized cloud (1 for unorganized clouds)
        point_step: Size in bytes of a single point
        encoding_opt: First-stage encoding option
        compression_opt: Second-stage compression option
        version: Encoding format version
    """

    fields: list[PointField] = field(default_factory=list)
    width: int = 0
    height: int = 1
    point_step: int = 0
    encoding_opt: EncodingOptions = EncodingOptions.LOSSY
    compression_opt: CompressionOption = CompressionOption.ZSTD
    version: int = ENCODING_VERSION
