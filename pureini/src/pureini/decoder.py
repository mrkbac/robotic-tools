"""
Point cloud decoder for Pureini compression.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

import lz4.block
import numpy as np
import zstandard as zstd

from .encoding_utils import BufferView, build_field_metadata, decode
from .header import decode_header
from .jit_codec import decode_chunk_jit
from .types import (
    POINTS_PER_CHUNK,
    CompressionOption,
    EncodingInfo,
    EncodingOptions,
)


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
        # Decode header and get bytes consumed
        info, header_size = decode_header(bytes(compressed_data))

        # Create view starting after header
        input_view = BufferView(compressed_data)
        input_view.trim_front(header_size)

        # Build field metadata arrays for JIT
        field_offsets, field_types, field_resolutions = build_field_metadata(info)

        # Allocate output buffer
        total_points = info.width * info.height
        output_size = total_points * info.point_step
        output = np.zeros(output_size, dtype=np.uint8)
        output_offset = 0

        # Decode chunks (version 3+)
        if info.version >= 3:
            while not input_view.empty():
                # Read chunk size
                chunk_size = int(decode(input_view, "I"))  # uint32

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
