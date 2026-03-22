"""Manual type definitions for MCAP CLI (enums, constants, and CLI parameters)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import Group, Parameter
from small_mcap import CompressionType as SmallMcapCompressionType


class CompressionType(str, Enum):
    """Compression algorithm types."""

    ZSTD = "zstd"
    LZ4 = "lz4"
    NONE = "none"


def str_to_compression_type(compression: str) -> SmallMcapCompressionType:
    """Convert compression string to small_mcap CompressionType enum."""
    compression_lower = compression.lower()
    if compression_lower in ("none", "", "off"):
        return SmallMcapCompressionType.NONE
    if compression_lower == "lz4":
        return SmallMcapCompressionType.LZ4
    if compression_lower == "zstd":
        return SmallMcapCompressionType.ZSTD
    raise ValueError(f"Unknown compression type: {compression}")


# MCAP processing constants
MIN_CHUNK_SIZE = 1024  # 1 KiB minimum chunk size
DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB default chunk size
DEFAULT_COMPRESSION = CompressionType.ZSTD  # Default compression algorithm

# Parameter groups
OUTPUT_OPTIONS_GROUP = Group("Output Options")

ChunkSizeOption = Annotated[
    int,
    Parameter(
        name=["--chunk-size"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]

CompressionOption = Annotated[
    CompressionType,
    Parameter(
        name=["--compression"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]

OutputPathOption = Annotated[
    Path,
    Parameter(
        name=["-o", "--output"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]

ForceOverwriteOption = Annotated[
    bool,
    Parameter(
        name=["-f", "--force"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]
