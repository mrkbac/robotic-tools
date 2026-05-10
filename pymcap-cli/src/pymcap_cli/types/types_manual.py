"""Manual type definitions for MCAP CLI parameter annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Group, Parameter
from small_mcap import CompressionType as SmallMcapCompressionType

CompressionName = Literal["zstd", "lz4", "none"]


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
    CompressionName,
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

NoClobberOption = Annotated[
    bool,
    Parameter(
        name=["--no-clobber"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]

DeleteSourceOption = Annotated[
    bool,
    Parameter(
        name=["--delete-source"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]
