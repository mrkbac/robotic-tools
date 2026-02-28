"""
Point cloud encoder for Pureini compression.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

import lz4.block
import numpy as np
import zstandard as zstd

from .encoding_utils import BufferView, encode
from .header import HeaderEncoding, encode_header
from .jit_codec import encode_chunk_jit
from .types import CompressionOption, EncodingInfo, EncodingOptions, FieldType

# Default chunk size: 32,768 points per chunk
POINTS_PER_CHUNK = 32768


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
        self._build_field_metadata()

    def _build_field_metadata(self) -> None:
        """
        Build numpy arrays for field metadata to pass to JIT functions.

        Creates three arrays:
        - field_offsets: byte offset of each field
        - field_types: type code for each field
        - field_resolutions: resolution for each field (0.0 = lossless)
        """
        num_fields = len(self.info.fields)

        self.field_offsets = np.zeros(num_fields, dtype=np.int32)
        self.field_types = np.zeros(num_fields, dtype=np.int32)
        self.field_resolutions = np.zeros(num_fields, dtype=np.float64)

        for idx, field in enumerate(self.info.fields):
            self.field_offsets[idx] = field.offset
            self.field_types[idx] = int(field.type)

            # Set resolution based on encoding option
            # Sentinel values:
            #   > 0.0  → lossy quantize+delta+varint
            #   == 0.0 → XOR lossless (FLOAT64 only)
            #   -1.0   → raw 4-byte copy (FLOAT32 without resolution)
            if self.info.encoding_opt == EncodingOptions.LOSSY and field.resolution is not None:
                self.field_resolutions[idx] = field.resolution
            elif field.type == FieldType.FLOAT64:
                # FLOAT64 without resolution → XOR lossless
                self.field_resolutions[idx] = 0.0
            elif field.type == FieldType.FLOAT32:
                # FLOAT32 without resolution → raw copy (C++ uses FieldEncoderCopy)
                self.field_resolutions[idx] = -1.0
            else:
                # Integer types: resolution doesn't matter (always delta+varint)
                self.field_resolutions[idx] = 0.0

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
        if self.info.compression_opt == CompressionOption.NONE:
            # No compression: write size + data
            size_buffer = BufferView(memoryview(output_view.data[:4]))
            encode(len(chunk_data), size_buffer, "I")
            output_view.trim_front(4)
            output_view.write_bytes(chunk_data)

        elif self.info.compression_opt == CompressionOption.LZ4:
            # LZ4 compression
            compressed = lz4.block.compress(chunk_data, store_size=False)
            # Write chunk size (uint32)
            size_buffer = BufferView(memoryview(output_view.data[:4]))
            encode(len(compressed), size_buffer, "I")
            output_view.trim_front(4)
            output_view.write_bytes(compressed)

        elif self.info.compression_opt == CompressionOption.ZSTD:
            # ZSTD compression
            cctx = zstd.ZstdCompressor(level=1)
            compressed = cctx.compress(chunk_data)
            # Write chunk size (uint32)
            size_buffer = BufferView(memoryview(output_view.data[:4]))
            encode(len(compressed), size_buffer, "I")
            output_view.trim_front(4)
            output_view.write_bytes(compressed)

        else:
            raise RuntimeError(f"Unknown compression option: {self.info.compression_opt}")
