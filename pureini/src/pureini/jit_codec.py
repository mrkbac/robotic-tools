"""
Unified Numba-JIT encoders/decoders for pureini.

This module contains the core encoding/decoding logic implemented entirely
in Numba JIT for maximum performance. All field types are handled in unified
functions, eliminating special-case fast paths.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

import numba as nb
import numpy as np

from .encoding_utils import decode_varint, encode_varint64_to_buffer

# Field type constants (from FieldType IntEnum)
FIELD_INT8 = 1
FIELD_UINT8 = 2
FIELD_INT16 = 3
FIELD_UINT16 = 4
FIELD_INT32 = 5
FIELD_UINT32 = 6
FIELD_FLOAT32 = 7
FIELD_FLOAT64 = 8
FIELD_INT64 = 9
FIELD_UINT64 = 10


@nb.njit(cache=True, fastmath=False)  # Must disable fastmath for NaN handling
def encode_chunk_jit(
    point_data: np.ndarray,  # Raw point cloud bytes (uint8)
    chunk_start: int,  # Starting point index
    chunk_points: int,  # Number of points in this chunk
    point_step: int,  # Bytes per point
    field_offsets: np.ndarray,  # int32 array of field offsets
    field_types: np.ndarray,  # int32 array of field type codes
    field_resolutions: np.ndarray,  # float64 array of resolutions (0.0 = lossless)
    output: np.ndarray,  # Output buffer (uint8)
) -> int:
    """
    Encode a single chunk of points (all fields).

    This is the unified encoder that handles all field types:
    - FLOAT32/FLOAT64 with resolution: quantize + delta + varint
    - FLOAT32/FLOAT64 without resolution: XOR encoding (lossless)
    - INT8-UINT64: delta + varint (except INT8/UINT8 which are copied)

    Args:
        point_data: Raw point cloud data as uint8 array
        chunk_start: Starting point index in the cloud
        chunk_points: Number of points to encode in this chunk
        point_step: Size of one point in bytes
        field_offsets: Byte offset of each field within a point
        field_types: Type code for each field (FIELD_INT8, etc.)
        field_resolutions: Resolution for lossy float encoding (0.0 = lossless/XOR)
        output: Output buffer for encoded data

    Returns:
        Number of bytes written to output
    """
    num_fields = len(field_offsets)
    out_offset = 0

    # State for delta encoding (one per field)
    prev_int_values = np.zeros(num_fields, dtype=np.int64)
    prev_float_bits = np.zeros(num_fields, dtype=np.uint64)

    # Scratch buffers for float reinterpretation (reused across all points)
    scratch_f32 = np.empty(1, dtype=np.float32)
    scratch_u32 = scratch_f32.view(np.uint32)
    scratch_f64 = np.empty(1, dtype=np.float64)
    scratch_u64 = scratch_f64.view(np.uint64)

    # Encode each point
    for point_idx in range(chunk_points):
        point_offset = (chunk_start + point_idx) * point_step

        # Encode each field
        for field_idx in range(num_fields):
            field_offset = field_offsets[field_idx]
            field_type = field_types[field_idx]
            resolution = field_resolutions[field_idx]

            data_offset = point_offset + field_offset

            # Handle each field type
            if field_type == FIELD_FLOAT32:
                if resolution < 0.0:
                    # Raw copy (C++ FieldEncoderCopy for FLOAT32 without resolution)
                    output[out_offset] = point_data[data_offset]
                    output[out_offset + 1] = point_data[data_offset + 1]
                    output[out_offset + 2] = point_data[data_offset + 2]
                    output[out_offset + 3] = point_data[data_offset + 3]
                    out_offset += 4
                elif resolution > 0.0:
                    # Read as bits
                    float_bits = (
                        np.uint32(point_data[data_offset])
                        | (np.uint32(point_data[data_offset + 1]) << 8)
                        | (np.uint32(point_data[data_offset + 2]) << 16)
                        | (np.uint32(point_data[data_offset + 3]) << 24)
                    )

                    # Lossy: reinterpret bits as float using pre-allocated scratch buffer
                    scratch_u32[0] = float_bits
                    value_real = scratch_f32[0]

                    # Lossy: quantize + delta + varint
                    if np.isnan(value_real):
                        output[out_offset] = 0
                        prev_int_values[field_idx] = 0
                        out_offset += 1
                    else:
                        multiplier = 1.0 / resolution
                        quantized = int(np.round(value_real * multiplier))
                        delta = quantized - prev_int_values[field_idx]
                        prev_int_values[field_idx] = quantized

                        # Inline varint encoding for speed
                        val = (delta << 1) if delta >= 0 else ((-delta - 1) << 1) | 1
                        val += 1
                        if val <= 0x7F:
                            output[out_offset] = val & 0xFF
                            out_offset += 1
                        elif val <= 0x3FFF:
                            output[out_offset] = (val & 0x7F) | 0x80
                            output[out_offset + 1] = (val >> 7) & 0xFF
                            out_offset += 2
                        elif val <= 0x1FFFFF:
                            output[out_offset] = (val & 0x7F) | 0x80
                            output[out_offset + 1] = ((val >> 7) & 0x7F) | 0x80
                            output[out_offset + 2] = (val >> 14) & 0xFF
                            out_offset += 3
                        else:
                            count = encode_varint64_to_buffer(delta, output[out_offset:], 0)
                            out_offset += count
                else:
                    # Read as bits for XOR
                    float_bits = (
                        np.uint32(point_data[data_offset])
                        | (np.uint32(point_data[data_offset + 1]) << 8)
                        | (np.uint32(point_data[data_offset + 2]) << 16)
                        | (np.uint32(point_data[data_offset + 3]) << 24)
                    )

                    # Lossless: XOR encoding
                    residual = float_bits ^ np.uint32(prev_float_bits[field_idx])
                    prev_float_bits[field_idx] = float_bits

                    # Write as uint32
                    output[out_offset] = residual & 0xFF
                    output[out_offset + 1] = (residual >> 8) & 0xFF
                    output[out_offset + 2] = (residual >> 16) & 0xFF
                    output[out_offset + 3] = (residual >> 24) & 0xFF
                    out_offset += 4

            elif field_type == FIELD_FLOAT64:
                # Read as bits first (unified for both lossy and lossless)
                float_bits = (
                    np.uint64(point_data[data_offset])
                    | (np.uint64(point_data[data_offset + 1]) << 8)
                    | (np.uint64(point_data[data_offset + 2]) << 16)
                    | (np.uint64(point_data[data_offset + 3]) << 24)
                    | (np.uint64(point_data[data_offset + 4]) << 32)
                    | (np.uint64(point_data[data_offset + 5]) << 40)
                    | (np.uint64(point_data[data_offset + 6]) << 48)
                    | (np.uint64(point_data[data_offset + 7]) << 56)
                )

                if resolution > 0.0:
                    # Lossy: reinterpret bits as float using pre-allocated scratch buffer
                    scratch_u64[0] = float_bits
                    value_real = scratch_f64[0]

                    # Lossy: quantize + delta + varint
                    if np.isnan(value_real):
                        output[out_offset] = 0
                        prev_int_values[field_idx] = 0
                        out_offset += 1
                    else:
                        multiplier = 1.0 / resolution
                        quantized = int(np.round(value_real * multiplier))
                        delta = quantized - prev_int_values[field_idx]
                        prev_int_values[field_idx] = quantized

                        # Inline varint encoding for speed
                        val = (delta << 1) if delta >= 0 else ((-delta - 1) << 1) | 1
                        val += 1
                        if val <= 0x7F:
                            output[out_offset] = val & 0xFF
                            out_offset += 1
                        elif val <= 0x3FFF:
                            output[out_offset] = (val & 0x7F) | 0x80
                            output[out_offset + 1] = (val >> 7) & 0xFF
                            out_offset += 2
                        elif val <= 0x1FFFFF:
                            output[out_offset] = (val & 0x7F) | 0x80
                            output[out_offset + 1] = ((val >> 7) & 0x7F) | 0x80
                            output[out_offset + 2] = (val >> 14) & 0xFF
                            out_offset += 3
                        else:
                            count = encode_varint64_to_buffer(delta, output[out_offset:], 0)
                            out_offset += count
                else:
                    # Lossless: XOR encoding
                    residual = float_bits ^ prev_float_bits[field_idx]
                    prev_float_bits[field_idx] = float_bits

                    # Write as uint64
                    output[out_offset] = residual & 0xFF
                    output[out_offset + 1] = (residual >> 8) & 0xFF
                    output[out_offset + 2] = (residual >> 16) & 0xFF
                    output[out_offset + 3] = (residual >> 24) & 0xFF
                    output[out_offset + 4] = (residual >> 32) & 0xFF
                    output[out_offset + 5] = (residual >> 40) & 0xFF
                    output[out_offset + 6] = (residual >> 48) & 0xFF
                    output[out_offset + 7] = (residual >> 56) & 0xFF
                    out_offset += 8

            elif field_type == FIELD_UINT8:
                # No compression for uint8
                output[out_offset] = point_data[data_offset]
                out_offset += 1

            elif field_type == FIELD_INT8:
                # No compression for int8
                output[out_offset] = point_data[data_offset]
                out_offset += 1

            elif field_type == FIELD_UINT16:
                # Delta + varint - read directly
                value = np.uint16(point_data[data_offset]) | (
                    np.uint16(point_data[data_offset + 1]) << 8
                )
                delta = np.int64(value) - prev_int_values[field_idx]
                prev_int_values[field_idx] = value

                # Inline varint encoding for speed
                val = (delta << 1) if delta >= 0 else ((-delta - 1) << 1) | 1
                val += 1
                if val <= 0x7F:
                    output[out_offset] = val & 0xFF
                    out_offset += 1
                elif val <= 0x3FFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = (val >> 7) & 0xFF
                    out_offset += 2
                elif val <= 0x1FFFFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = ((val >> 7) & 0x7F) | 0x80
                    output[out_offset + 2] = (val >> 14) & 0xFF
                    out_offset += 3
                else:
                    count = encode_varint64_to_buffer(delta, output[out_offset:], 0)
                    out_offset += count

            elif field_type == FIELD_INT16:
                # Delta + varint - read directly
                uval = np.uint16(point_data[data_offset]) | (
                    np.uint16(point_data[data_offset + 1]) << 8
                )
                # Sign extend from 16-bit to int64
                value = np.int64(np.int16(uval))
                delta = value - prev_int_values[field_idx]
                prev_int_values[field_idx] = value

                # Inline varint encoding for speed
                val = (delta << 1) if delta >= 0 else ((-delta - 1) << 1) | 1
                val += 1
                if val <= 0x7F:
                    output[out_offset] = val & 0xFF
                    out_offset += 1
                elif val <= 0x3FFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = (val >> 7) & 0xFF
                    out_offset += 2
                elif val <= 0x1FFFFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = ((val >> 7) & 0x7F) | 0x80
                    output[out_offset + 2] = (val >> 14) & 0xFF
                    out_offset += 3
                else:
                    count = encode_varint64_to_buffer(delta, output[out_offset:], 0)
                    out_offset += count

            elif field_type == FIELD_UINT32:
                # Delta + varint - read directly
                value = (
                    np.uint32(point_data[data_offset])
                    | (np.uint32(point_data[data_offset + 1]) << 8)
                    | (np.uint32(point_data[data_offset + 2]) << 16)
                    | (np.uint32(point_data[data_offset + 3]) << 24)
                )
                delta = np.int64(value) - prev_int_values[field_idx]
                prev_int_values[field_idx] = value

                # Inline varint encoding for speed
                # Zigzag encoding
                val = (delta << 1) if delta >= 0 else ((-delta - 1) << 1) | 1
                val += 1  # Reserve 0 for NaN

                # Varint encoding (inlined)
                if val <= 0x7F:
                    # 1 byte (common case)
                    output[out_offset] = val & 0xFF
                    out_offset += 1
                elif val <= 0x3FFF:
                    # 2 bytes
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = (val >> 7) & 0xFF
                    out_offset += 2
                elif val <= 0x1FFFFF:
                    # 3 bytes
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = ((val >> 7) & 0x7F) | 0x80
                    output[out_offset + 2] = (val >> 14) & 0xFF
                    out_offset += 3
                else:
                    # Fall back to loop for larger values
                    count = encode_varint64_to_buffer(delta, output[out_offset:], 0)
                    out_offset += count

            elif field_type == FIELD_INT32:
                # Delta + varint - read directly
                uval = (
                    np.uint32(point_data[data_offset])
                    | (np.uint32(point_data[data_offset + 1]) << 8)
                    | (np.uint32(point_data[data_offset + 2]) << 16)
                    | (np.uint32(point_data[data_offset + 3]) << 24)
                )
                # Sign extend from 32-bit to int64
                value = np.int64(np.int32(uval))
                delta = value - prev_int_values[field_idx]
                prev_int_values[field_idx] = value

                # Inline varint encoding for speed
                val = (delta << 1) if delta >= 0 else ((-delta - 1) << 1) | 1
                val += 1
                if val <= 0x7F:
                    output[out_offset] = val & 0xFF
                    out_offset += 1
                elif val <= 0x3FFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = (val >> 7) & 0xFF
                    out_offset += 2
                elif val <= 0x1FFFFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = ((val >> 7) & 0x7F) | 0x80
                    output[out_offset + 2] = (val >> 14) & 0xFF
                    out_offset += 3
                else:
                    count = encode_varint64_to_buffer(delta, output[out_offset:], 0)
                    out_offset += count

            elif field_type == FIELD_UINT64:
                # Delta + varint - read directly
                value = (
                    np.uint64(point_data[data_offset])
                    | (np.uint64(point_data[data_offset + 1]) << 8)
                    | (np.uint64(point_data[data_offset + 2]) << 16)
                    | (np.uint64(point_data[data_offset + 3]) << 24)
                    | (np.uint64(point_data[data_offset + 4]) << 32)
                    | (np.uint64(point_data[data_offset + 5]) << 40)
                    | (np.uint64(point_data[data_offset + 6]) << 48)
                    | (np.uint64(point_data[data_offset + 7]) << 56)
                )
                delta = np.int64(value) - prev_int_values[field_idx]
                prev_int_values[field_idx] = np.int64(value)

                # Inline varint encoding for speed
                val = (delta << 1) if delta >= 0 else ((-delta - 1) << 1) | 1
                val += 1
                if val <= 0x7F:
                    output[out_offset] = val & 0xFF
                    out_offset += 1
                elif val <= 0x3FFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = (val >> 7) & 0xFF
                    out_offset += 2
                elif val <= 0x1FFFFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = ((val >> 7) & 0x7F) | 0x80
                    output[out_offset + 2] = (val >> 14) & 0xFF
                    out_offset += 3
                else:
                    count = encode_varint64_to_buffer(delta, output[out_offset:], 0)
                    out_offset += count

            elif field_type == FIELD_INT64:
                # Delta + varint - read directly
                uval = (
                    np.uint64(point_data[data_offset])
                    | (np.uint64(point_data[data_offset + 1]) << 8)
                    | (np.uint64(point_data[data_offset + 2]) << 16)
                    | (np.uint64(point_data[data_offset + 3]) << 24)
                    | (np.uint64(point_data[data_offset + 4]) << 32)
                    | (np.uint64(point_data[data_offset + 5]) << 40)
                    | (np.uint64(point_data[data_offset + 6]) << 48)
                    | (np.uint64(point_data[data_offset + 7]) << 56)
                )
                # Reinterpret as signed
                value = np.int64(uval)
                delta = value - prev_int_values[field_idx]
                prev_int_values[field_idx] = value

                # Inline varint encoding for speed
                val = (delta << 1) if delta >= 0 else ((-delta - 1) << 1) | 1
                val += 1
                if val <= 0x7F:
                    output[out_offset] = val & 0xFF
                    out_offset += 1
                elif val <= 0x3FFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = (val >> 7) & 0xFF
                    out_offset += 2
                elif val <= 0x1FFFFF:
                    output[out_offset] = (val & 0x7F) | 0x80
                    output[out_offset + 1] = ((val >> 7) & 0x7F) | 0x80
                    output[out_offset + 2] = (val >> 14) & 0xFF
                    out_offset += 3
                else:
                    count = encode_varint64_to_buffer(delta, output[out_offset:], 0)
                    out_offset += count

    return out_offset


@nb.njit(cache=True, fastmath=False)  # Must disable fastmath for NaN handling
def decode_chunk_jit(
    encoded_data: memoryview,  # Encoded chunk data
    num_points: int,  # Number of points to decode
    point_step: int,  # Bytes per point
    field_offsets: np.ndarray,  # int32 array of field offsets
    field_types: np.ndarray,  # int32 array of field type codes
    field_resolutions: np.ndarray,  # float64 array of resolutions
    output: np.ndarray,  # Output buffer for decoded points (uint8)
) -> int:
    """
    Decode a single chunk of points (all fields).

    Args:
        encoded_data: Encoded chunk data
        num_points: Number of points to decode
        point_step: Size of one point in bytes
        field_offsets: Byte offset of each field within a point
        field_types: Type code for each field
        field_resolutions: Resolution for lossy float decoding (0.0 = lossless/XOR)
        output: Output buffer for decoded point data

    Returns:
        Number of points decoded
    """
    num_fields = len(field_offsets)
    in_offset = 0

    # State for delta decoding (one per field)
    prev_int_values = np.zeros(num_fields, dtype=np.int64)
    prev_float_bits = np.zeros(num_fields, dtype=np.uint64)

    # Scratch buffers for float reinterpretation (hoisted out of loop)
    scratch_f32 = np.empty(1, dtype=np.float32)
    scratch_u32 = scratch_f32.view(np.uint32)
    scratch_f64 = np.empty(1, dtype=np.float64)
    scratch_u64 = scratch_f64.view(np.uint64)

    # Decode each point
    points_decoded = 0
    for point_idx in range(num_points):
        if in_offset >= len(encoded_data):
            break

        point_offset = point_idx * point_step

        # Decode each field
        for field_idx in range(num_fields):
            field_offset = field_offsets[field_idx]
            field_type = field_types[field_idx]
            resolution = field_resolutions[field_idx]

            data_offset = point_offset + field_offset

            # Handle each field type
            if field_type == FIELD_FLOAT32:
                if resolution < 0.0:
                    # Raw copy (C++ FieldEncoderCopy for FLOAT32 without resolution)
                    output[data_offset] = encoded_data[in_offset]
                    output[data_offset + 1] = encoded_data[in_offset + 1]
                    output[data_offset + 2] = encoded_data[in_offset + 2]
                    output[data_offset + 3] = encoded_data[in_offset + 3]
                    in_offset += 4
                elif resolution > 0.0:
                    # Lossy: varint + delta + dequantize
                    if encoded_data[in_offset] == 0:
                        # NaN
                        value_real = np.nan
                        prev_int_values[field_idx] = 0
                        in_offset += 1
                    else:
                        # Decode varint
                        delta, count = decode_varint(encoded_data, in_offset)
                        in_offset += count

                        # Reconstruct quantized value
                        value = prev_int_values[field_idx] + delta
                        prev_int_values[field_idx] = value

                        # Dequantize
                        value_real = float(value) * resolution

                    # Write float32 - convert to bits using numpy view
                    scratch_f32[0] = value_real
                    value_bits = scratch_u32[0]
                    output[data_offset] = value_bits & 0xFF
                    output[data_offset + 1] = (value_bits >> 8) & 0xFF
                    output[data_offset + 2] = (value_bits >> 16) & 0xFF
                    output[data_offset + 3] = (value_bits >> 24) & 0xFF
                else:
                    # Lossless: XOR decoding
                    residual = (
                        np.uint32(encoded_data[in_offset])
                        | (np.uint32(encoded_data[in_offset + 1]) << 8)
                        | (np.uint32(encoded_data[in_offset + 2]) << 16)
                        | (np.uint32(encoded_data[in_offset + 3]) << 24)
                    )
                    in_offset += 4

                    current_bits = residual ^ np.uint32(prev_float_bits[field_idx])
                    prev_float_bits[field_idx] = current_bits

                    # Write uint32 as float32
                    output[data_offset] = current_bits & 0xFF
                    output[data_offset + 1] = (current_bits >> 8) & 0xFF
                    output[data_offset + 2] = (current_bits >> 16) & 0xFF
                    output[data_offset + 3] = (current_bits >> 24) & 0xFF

            elif field_type == FIELD_FLOAT64:
                if resolution > 0.0:
                    # Lossy: varint + delta + dequantize
                    if encoded_data[in_offset] == 0:
                        # NaN
                        value_real = np.nan
                        prev_int_values[field_idx] = 0
                        in_offset += 1
                    else:
                        # Decode varint
                        delta, count = decode_varint(encoded_data, in_offset)
                        in_offset += count

                        # Reconstruct quantized value
                        value = prev_int_values[field_idx] + delta
                        prev_int_values[field_idx] = value

                        # Dequantize
                        value_real = float(value) * resolution

                    # Write float64 - convert to bits using numpy view
                    scratch_f64[0] = value_real
                    value_bits = scratch_u64[0]
                    output[data_offset] = value_bits & 0xFF
                    output[data_offset + 1] = (value_bits >> 8) & 0xFF
                    output[data_offset + 2] = (value_bits >> 16) & 0xFF
                    output[data_offset + 3] = (value_bits >> 24) & 0xFF
                    output[data_offset + 4] = (value_bits >> 32) & 0xFF
                    output[data_offset + 5] = (value_bits >> 40) & 0xFF
                    output[data_offset + 6] = (value_bits >> 48) & 0xFF
                    output[data_offset + 7] = (value_bits >> 56) & 0xFF
                else:
                    # Lossless: XOR decoding
                    residual = (
                        np.uint64(encoded_data[in_offset])
                        | (np.uint64(encoded_data[in_offset + 1]) << 8)
                        | (np.uint64(encoded_data[in_offset + 2]) << 16)
                        | (np.uint64(encoded_data[in_offset + 3]) << 24)
                        | (np.uint64(encoded_data[in_offset + 4]) << 32)
                        | (np.uint64(encoded_data[in_offset + 5]) << 40)
                        | (np.uint64(encoded_data[in_offset + 6]) << 48)
                        | (np.uint64(encoded_data[in_offset + 7]) << 56)
                    )
                    in_offset += 8

                    current_bits = residual ^ prev_float_bits[field_idx]
                    prev_float_bits[field_idx] = current_bits

                    # Write uint64 as float64
                    output[data_offset] = current_bits & 0xFF
                    output[data_offset + 1] = (current_bits >> 8) & 0xFF
                    output[data_offset + 2] = (current_bits >> 16) & 0xFF
                    output[data_offset + 3] = (current_bits >> 24) & 0xFF
                    output[data_offset + 4] = (current_bits >> 32) & 0xFF
                    output[data_offset + 5] = (current_bits >> 40) & 0xFF
                    output[data_offset + 6] = (current_bits >> 48) & 0xFF
                    output[data_offset + 7] = (current_bits >> 56) & 0xFF

            elif field_type in (FIELD_UINT8, FIELD_INT8):
                # No decompression
                output[data_offset] = encoded_data[in_offset]
                in_offset += 1

            elif field_type == FIELD_UINT16:
                # Varint + delta
                delta, count = decode_varint(encoded_data, in_offset)
                in_offset += count

                value = prev_int_values[field_idx] + delta
                prev_int_values[field_idx] = value

                # Write uint16 directly
                output[data_offset] = value & 0xFF
                output[data_offset + 1] = (value >> 8) & 0xFF

            elif field_type == FIELD_INT16:
                # Varint + delta
                delta, count = decode_varint(encoded_data, in_offset)
                in_offset += count

                value = prev_int_values[field_idx] + delta
                prev_int_values[field_idx] = value

                # Write int16 directly
                output[data_offset] = value & 0xFF
                output[data_offset + 1] = (value >> 8) & 0xFF

            elif field_type == FIELD_UINT32:
                # Varint + delta
                delta, count = decode_varint(encoded_data, in_offset)
                in_offset += count

                value = prev_int_values[field_idx] + delta
                prev_int_values[field_idx] = value

                # Write uint32 directly
                output[data_offset] = value & 0xFF
                output[data_offset + 1] = (value >> 8) & 0xFF
                output[data_offset + 2] = (value >> 16) & 0xFF
                output[data_offset + 3] = (value >> 24) & 0xFF

            elif field_type == FIELD_INT32:
                # Varint + delta
                delta, count = decode_varint(encoded_data, in_offset)
                in_offset += count

                value = prev_int_values[field_idx] + delta
                prev_int_values[field_idx] = value

                # Write int32 directly
                output[data_offset] = value & 0xFF
                output[data_offset + 1] = (value >> 8) & 0xFF
                output[data_offset + 2] = (value >> 16) & 0xFF
                output[data_offset + 3] = (value >> 24) & 0xFF

            elif field_type == FIELD_UINT64:
                # Varint + delta
                delta, count = decode_varint(encoded_data, in_offset)
                in_offset += count

                value = prev_int_values[field_idx] + delta
                prev_int_values[field_idx] = value

                # Write uint64 directly
                output[data_offset] = value & 0xFF
                output[data_offset + 1] = (value >> 8) & 0xFF
                output[data_offset + 2] = (value >> 16) & 0xFF
                output[data_offset + 3] = (value >> 24) & 0xFF
                output[data_offset + 4] = (value >> 32) & 0xFF
                output[data_offset + 5] = (value >> 40) & 0xFF
                output[data_offset + 6] = (value >> 48) & 0xFF
                output[data_offset + 7] = (value >> 56) & 0xFF

            elif field_type == FIELD_INT64:
                # Varint + delta
                delta, count = decode_varint(encoded_data, in_offset)
                in_offset += count

                value = prev_int_values[field_idx] + delta
                prev_int_values[field_idx] = value

                # Write int64 directly
                output[data_offset] = value & 0xFF
                output[data_offset + 1] = (value >> 8) & 0xFF
                output[data_offset + 2] = (value >> 16) & 0xFF
                output[data_offset + 3] = (value >> 24) & 0xFF
                output[data_offset + 4] = (value >> 32) & 0xFF
                output[data_offset + 5] = (value >> 40) & 0xFF
                output[data_offset + 6] = (value >> 48) & 0xFF
                output[data_offset + 7] = (value >> 56) & 0xFF

        # Successfully decoded this point
        points_decoded += 1

    return points_decoded
