"""
Point cloud encoder for Pureini compression.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

import struct

import lz4.block
import numpy as np
import zstandard as zstd

from .encoding_utils import BufferView, build_field_metadata
from .header import HeaderEncoding, encode_header
from .jit_codec import encode_chunk_jit
from .types import CompressionOption, EncodingInfo, EncodingOptions, POINTS_PER_CHUNK


def _zstd_compress_bound(src_size: int) -> int:
    """
    Calculate maximum compressed size for zstd compression.

    Implements the ZSTD_compressBound formula from the C API:
    srcSize + (srcSize >> 8) + margin_for_small_inputs

    The Python zstandard library doesn't expose ZSTD_compressBound,
    so we implement it based on the official zstd specification.

    Args:
        src_size: Size of uncompressed data in bytes

    Returns:
        Maximum size needed for compressed output buffer
    """
    base = src_size + (src_size >> 8)
    # Add extra margin for inputs smaller than 128 KB
    if src_size < (128 << 10):  # 128 KB = 131072 bytes
        margin = ((128 << 10) - src_size) >> 11
        return base + margin
    return base


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
        self.header = encode_header(info, HeaderEncoding.YAML)

        # Build field metadata arrays for JIT
        self.field_offsets, self.field_types, self.field_resolutions = build_field_metadata(info)

        # Reuse compressor across chunks
        if info.compression_opt == CompressionOption.ZSTD:
            self._zstd_cctx = zstd.ZstdCompressor(level=1)

    def encode(self, cloud_data: bytes) -> bytes:
        """
        Encode point cloud data.

        Args:
            cloud_data: Raw point cloud bytes

        Returns:
            Compressed point cloud data with header
        """
        # Convert to numpy array for JIT
        point_data = np.frombuffer(cloud_data, dtype=np.uint8)

        # Calculate points and chunks
        points_count = len(cloud_data) // self.info.point_step
        chunks_count = (points_count + POINTS_PER_CHUNK - 1) // POINTS_PER_CHUNK

        # Worst-case compression bound
        if self.info.compression_opt == CompressionOption.ZSTD:
            max_compressed = _zstd_compress_bound(len(cloud_data))
        else:
            max_compressed = len(cloud_data) * 2  # Conservative estimate for LZ4

        # Allocate output buffer: header + compressed data + chunk headers (4 bytes each)
        output_size = len(self.header) + max_compressed + (4 * chunks_count)
        output = bytearray(output_size)
        output_view = BufferView(output)

        # Write header
        output_view.write_bytes(self.header)

        # Encode and compress chunks
        self._encode_chunks(point_data, points_count, output_view)

        # Trim to actual size
        actual_size = output_size - output_view.size()
        return bytes(output[:actual_size])

    def _encode_chunks(
        self, point_data: np.ndarray, points_count: int, output_view: BufferView
    ) -> None:
        """
        Encode point cloud data in chunks using JIT.

        Args:
            point_data: Raw point cloud data as numpy array
            points_count: Total number of points
            output_view: Output buffer for compressed chunks
        """
        # Allocate temporary buffer for encoded chunk (before compression)
        chunk_buffer = np.zeros(POINTS_PER_CHUNK * self.info.point_step, dtype=np.uint8)

        chunk_start = 0
        while chunk_start < points_count:
            # Determine chunk size
            chunk_points = min(POINTS_PER_CHUNK, points_count - chunk_start)

            # Fast path for NONE encoding: copy raw bytes without transformation
            if self.info.encoding_opt == EncodingOptions.NONE:
                # Calculate byte offsets
                byte_start = chunk_start * self.info.point_step
                byte_end = byte_start + (chunk_points * self.info.point_step)
                chunk_data = bytes(point_data[byte_start:byte_end])
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
                chunk_data = bytes(chunk_buffer[:bytes_written])

            # Compress and write chunk
            self._write_chunk(chunk_data, output_view)

            chunk_start += chunk_points

    def _write_chunk(self, chunk_data: bytes, output_view: BufferView) -> None:
        """
        Compress and write a chunk to output.

        Args:
            chunk_data: Encoded chunk data
            output_view: Output buffer
        """
        if self.info.compression_opt == CompressionOption.LZ4:
            payload = lz4.block.compress(chunk_data, store_size=False)
        elif self.info.compression_opt == CompressionOption.ZSTD:
            payload = self._zstd_cctx.compress(chunk_data)
        elif self.info.compression_opt == CompressionOption.NONE:
            payload = chunk_data
        else:
            raise RuntimeError(f"Unknown compression option: {self.info.compression_opt}")

        # Write uint32 size + payload
        struct.pack_into("<I", output_view.data, 0, len(payload))
        output_view.trim_front(4)
        output_view.write_bytes(payload)
