"""Manual type definitions for MCAP CLI parameter annotations."""

from __future__ import annotations

from typing import Literal

from small_mcap import CompressionType as SmallMcapCompressionType

CompressionName = Literal["zstd", "lz4", "none"]

OrderName = Literal["preserve", "log_time", "topic"]


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
