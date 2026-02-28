"""
Point cloud decoder for Pureini compression.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

import lz4.block
import numpy as np
import zstandard as zstd

from .encoding_utils import ConstBufferView, decode
from .header import HeaderEncoding, decode_header, encode_header
from .jit_codec import decode_chunk_jit
from .types import (
    MAGIC_HEADER_LENGTH,
    CompressionOption,
    EncodingInfo,
    EncodingOptions,
    FieldType,
)

POINTS_PER_CHUNK = 32768


class PointcloudDecoder:
    """
    Point cloud decoder for two-stage decompression.

    Stage 1: General-purpose decompression (LZ4 or ZSTD)
    Stage 2: Field-specific decoding (delta, dequantization, etc.) via Numba JIT
    """

    def __init__(self) -> None:
        """Initialize the decoder."""

    def decode(self, compressed_data: bytes) -> tuple[bytes, EncodingInfo]:
        """
        Decode compressed point cloud data.

        Args:
            compressed_data: Compressed data including header

        Returns:
            Tuple of (decompressed_data, encoding_info)
        """
        # Create a mutable view for header decoding
        input_view = ConstBufferView(compressed_data)

        # Decode header (returns EncodingInfo)
        info = decode_header(bytes(compressed_data))

        # Calculate header size by encoding it again and checking length
        # Check if it's YAML or binary by looking at the format
        if len(compressed_data) > MAGIC_HEADER_LENGTH + 2:
            if compressed_data[MAGIC_HEADER_LENGTH + 2] == ord("\n"):
                # YAML format
                test_header = encode_header(info, HeaderEncoding.YAML)
            else:
                # Binary format
                test_header = encode_header(info, HeaderEncoding.BINARY)
            header_size = len(test_header)
        else:
            raise RuntimeError("Invalid compressed data: too short")

        # Skip the header
        input_view.trim_front(header_size)

        # Build field metadata arrays for JIT
        field_offsets, field_types, field_resolutions = self._build_field_metadata(info)

        # Allocate output buffer
        total_points = info.width * info.height
        output_size = total_points * info.point_step
        output = np.zeros(output_size, dtype=np.uint8)
        output_offset = 0

        # Decode chunks (version 3+)
        if info.version >= 3:
            while not input_view.empty():
                # Read chunk size
                chunk_size = decode(input_view, "I")  # uint32

                if chunk_size > input_view.size():
                    raise RuntimeError("Invalid chunk size found while decoding")

                # Extract chunk
                chunk_data = input_view.read_bytes(chunk_size)

                # Decompress chunk
                decompressed = self._decompress_chunk(info, chunk_data)

                # Fast path for NONE encoding: copy raw bytes without transformation
                if info.encoding_opt == EncodingOptions.NONE:
                    # Copy raw bytes directly to output
                    chunk_bytes = len(decompressed)
                    points_decoded = chunk_bytes // info.point_step
                    output[output_offset : output_offset + chunk_bytes] = np.frombuffer(
                        decompressed, dtype=np.uint8
                    )
                    output_offset += chunk_bytes
                else:
                    # Decode chunk using JIT
                    # Calculate max points we can decode into remaining output space
                    remaining_output_bytes = output_size - output_offset
                    max_points_to_decode = remaining_output_bytes // info.point_step

                    output_chunk = output[output_offset:]
                    points_decoded = decode_chunk_jit(
                        memoryview(decompressed),
                        max_points_to_decode,
                        info.point_step,
                        field_offsets,
                        field_types,
                        field_resolutions,
                        output_chunk,
                    )

                    output_offset += points_decoded * info.point_step
        else:
            # Version 2: single chunk
            decompressed = self._decompress_chunk(info, bytes(input_view.data))

            # Fast path for NONE encoding: copy raw bytes without transformation
            if info.encoding_opt == EncodingOptions.NONE:
                chunk_bytes = len(decompressed)
                output[:chunk_bytes] = np.frombuffer(decompressed, dtype=np.uint8)
                output_offset = chunk_bytes
            else:
                max_points_to_decode = total_points
                points_decoded = decode_chunk_jit(
                    memoryview(decompressed),
                    max_points_to_decode,
                    info.point_step,
                    field_offsets,
                    field_types,
                    field_resolutions,
                    output,
                )
                output_offset = points_decoded * info.point_step

        return bytes(output[:output_offset]), info

    def _build_field_metadata(
        self, info: EncodingInfo
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build numpy arrays for field metadata to pass to JIT functions.

        Args:
            info: Encoding information

        Returns:
            Tuple of (field_offsets, field_types, field_resolutions)
        """
        num_fields = len(info.fields)

        field_offsets = np.zeros(num_fields, dtype=np.int32)
        field_types = np.zeros(num_fields, dtype=np.int32)
        field_resolutions = np.zeros(num_fields, dtype=np.float64)

        for idx, field in enumerate(info.fields):
            field_offsets[idx] = field.offset
            field_types[idx] = int(field.type)

            # Set resolution based on encoding option
            # Sentinel values:
            #   > 0.0  → lossy quantize+delta+varint
            #   == 0.0 → XOR lossless (FLOAT64 only)
            #   -1.0   → raw 4-byte copy (FLOAT32 without resolution)
            if info.encoding_opt == EncodingOptions.LOSSY and field.resolution is not None:
                field_resolutions[idx] = field.resolution
            elif field.type == FieldType.FLOAT64:
                # FLOAT64 without resolution → XOR lossless
                field_resolutions[idx] = 0.0
            elif field.type == FieldType.FLOAT32:
                # FLOAT32 without resolution → raw copy (C++ uses FieldEncoderCopy)
                field_resolutions[idx] = -1.0
            else:
                # Integer types: resolution doesn't matter (always delta+varint)
                field_resolutions[idx] = 0.0

        return field_offsets, field_types, field_resolutions

    def _decompress_chunk(self, info: EncodingInfo, chunk_data: bytes) -> bytes:
        """
        Decompress a single chunk.

        Args:
            info: Encoding information
            chunk_data: Compressed chunk data

        Returns:
            Decompressed chunk data
        """
        if info.compression_opt == CompressionOption.LZ4:
            try:
                max_size = POINTS_PER_CHUNK * info.point_step
                return lz4.block.decompress(chunk_data, uncompressed_size=max_size)
            except (ValueError, RuntimeError) as e:
                raise RuntimeError(f"LZ4 decompression failed: {e}") from e

        elif info.compression_opt == CompressionOption.ZSTD:
            try:
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(chunk_data)
            except (ValueError, RuntimeError, zstd.ZstdError) as e:
                raise RuntimeError(f"ZSTD decompression failed: {e}") from e

        elif info.compression_opt == CompressionOption.NONE:
            # No decompression needed
            return chunk_data

        else:
            raise RuntimeError(f"Unknown compression option: {info.compression_opt}")
