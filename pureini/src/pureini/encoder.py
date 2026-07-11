"""
Point cloud encoder for Pureini compression.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

import struct
from typing import TYPE_CHECKING, Protocol, cast

import lz4.block
import numpy as np

from .encoding_utils import build_field_metadata
from .header import HeaderEncoding, encode_header
from .jit_codec import encode_chunk_jit
from .types import POINTS_PER_CHUNK, CompressionOption, EncodingInfo, EncodingOptions

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

# Python 3.14 promoted zstd into the stdlib; older interpreters use the
# third-party `zstandard` package (declared in pyproject as a 3.13-only dep).
try:
    from compression import zstd

    _ZSTD_IS_STDLIB = True
except ImportError:
    import zstandard as zstd

    _ZSTD_IS_STDLIB = False


class _StdlibZstdCompressor(Protocol):
    """The 3.14 stdlib ``compression.zstd.ZstdCompressor`` surface we rely on.

    ty resolves ``zstd`` to the third-party ``zstandard`` class on the 3.10
    target, which lacks ``FLUSH_FRAME`` and the two-arg ``compress``.
    """

    FLUSH_FRAME: int

    def compress(self, data: "ReadableBuffer", mode: int = ..., /) -> bytes: ...


class PointcloudEncoder:
    """
    Point cloud encoder using two-stage compression.

    Stage 1: Field-specific encoding (delta, quantization, etc.) via Numba JIT
    Stage 2: General-purpose compression (LZ4 or ZSTD)
    """

    def __init__(self, info: EncodingInfo) -> None:
        """
        Initialize the encoder with encoding configuration.

        Args:
            info: Encoding configuration
        """
        self.info = info
        self.header = bytes(encode_header(info, HeaderEncoding.YAML))

        # Build field metadata arrays for JIT
        self.field_offsets, self.field_types, self.field_resolutions = build_field_metadata(info)

        # Reuse compressor across chunks
        if info.compression_opt == CompressionOption.ZSTD:
            self._zstd_cctx = zstd.ZstdCompressor(level=1)

    def encode(self, cloud_data: "ReadableBuffer") -> bytes:
        """
        Encode point cloud data.

        Args:
            cloud_data: Raw point cloud bytes

        Returns:
            Compressed point cloud data with header
        """
        # Convert to numpy array for JIT
        cloud_view = memoryview(cloud_data)
        point_data = np.frombuffer(cloud_view, dtype=np.uint8)

        # Calculate points and chunks
        points_count = len(cloud_view) // self.info.point_step

        output_parts: list[bytes] = [self.header]
        self._encode_chunks(point_data, points_count, output_parts)
        return b"".join(output_parts)

    def _encode_chunks(
        self, point_data: np.ndarray, points_count: int, output_parts: list[bytes]
    ) -> None:
        """
        Encode point cloud data in chunks using JIT.

        Args:
            point_data: Raw point cloud data as numpy array
            points_count: Total number of points
            output_parts: Header/chunk parts joined once after encoding
        """
        # Allocate temporary buffer for encoded chunk (before compression)
        chunk_buffer = np.empty(POINTS_PER_CHUNK * self.info.point_step, dtype=np.uint8)

        chunk_start = 0
        while chunk_start < points_count:
            # Determine chunk size
            chunk_points = min(POINTS_PER_CHUNK, points_count - chunk_start)

            # Fast path for NONE encoding: copy raw bytes without transformation
            if self.info.encoding_opt == EncodingOptions.NONE:
                # Calculate byte offsets
                byte_start = chunk_start * self.info.point_step
                byte_end = byte_start + (chunk_points * self.info.point_step)
                chunk_data = point_data.data[byte_start:byte_end]
            else:
                # Call JIT encoder for this chunk
                bytes_written = encode_chunk_jit(
                    point_data,
                    chunk_start,
                    chunk_points,
                    self.info.point_step,
                    self.field_offsets,
                    self.field_types,
                    self.field_resolutions,
                    chunk_buffer,
                )
                chunk_data = chunk_buffer.data[:bytes_written]

            # Compress and write chunk
            self._append_chunk(chunk_data, output_parts)

            chunk_start += chunk_points

    def _append_chunk(self, chunk_data: memoryview, output_parts: list[bytes]) -> None:
        """
        Compress and write a chunk to output.

        Args:
            chunk_data: Encoded chunk data
            output_parts: Header/chunk parts joined once after encoding
        """
        if self.info.compression_opt == CompressionOption.LZ4:
            payload = lz4.block.compress(chunk_data, store_size=False)
        elif self.info.compression_opt == CompressionOption.ZSTD:
            if _ZSTD_IS_STDLIB:
                # compression.zstd's streaming ZstdCompressor needs an explicit frame
                # terminator (FLUSH_FRAME); see _StdlibZstdCompressor for why we cast.
                cctx = cast("_StdlibZstdCompressor", self._zstd_cctx)
                payload = cctx.compress(chunk_data, cctx.FLUSH_FRAME)
            else:
                payload = self._zstd_cctx.compress(chunk_data)
        elif self.info.compression_opt == CompressionOption.NONE:
            payload = bytes(chunk_data)
        else:
            raise RuntimeError(f"Unknown compression option: {self.info.compression_opt}")

        output_parts.append(struct.pack("<I", len(payload)))
        output_parts.append(payload)
