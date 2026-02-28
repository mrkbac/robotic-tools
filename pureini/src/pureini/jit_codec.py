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
from .types import FieldType

# Field type constants derived from FieldType enum (Numba can't use Python enums)
FIELD_INT8 = int(FieldType.INT8)
FIELD_UINT8 = int(FieldType.UINT8)
FIELD_INT16 = int(FieldType.INT16)
FIELD_UINT16 = int(FieldType.UINT16)
FIELD_INT32 = int(FieldType.INT32)
FIELD_UINT32 = int(FieldType.UINT32)
FIELD_FLOAT32 = int(FieldType.FLOAT32)
FIELD_FLOAT64 = int(FieldType.FLOAT64)
FIELD_INT64 = int(FieldType.INT64)
FIELD_UINT64 = int(FieldType.UINT64)


# ---------------------------------------------------------------------------
# Shared helpers (always inlined by Numba)
# ---------------------------------------------------------------------------


@nb.njit(cache=True, inline="always")
def _read_le_u32(data: np.ndarray, offset: int) -> np.uint32:
    """Read a little-endian uint32 from 4 bytes."""
    return (
        np.uint32(data[offset])
        | (np.uint32(data[offset + 1]) << 8)
        | (np.uint32(data[offset + 2]) << 16)
        | (np.uint32(data[offset + 3]) << 24)
    )


@nb.njit(cache=True, inline="always")
def _read_le_u64(data: np.ndarray, offset: int) -> np.uint64:
    """Read a little-endian uint64 from 8 bytes."""
    return (
        np.uint64(data[offset])
        | (np.uint64(data[offset + 1]) << 8)
        | (np.uint64(data[offset + 2]) << 16)
        | (np.uint64(data[offset + 3]) << 24)
        | (np.uint64(data[offset + 4]) << 32)
        | (np.uint64(data[offset + 5]) << 40)
        | (np.uint64(data[offset + 6]) << 48)
        | (np.uint64(data[offset + 7]) << 56)
    )


@nb.njit(cache=True, inline="always")
def _write_le_u32(output: np.ndarray, offset: int, value: np.uint32) -> None:
    """Write a uint32 as 4 little-endian bytes."""
    output[offset] = value & 0xFF
    output[offset + 1] = (value >> 8) & 0xFF
    output[offset + 2] = (value >> 16) & 0xFF
    output[offset + 3] = (value >> 24) & 0xFF


@nb.njit(cache=True, inline="always")
def _write_le_u64(output: np.ndarray, offset: int, value: np.uint64) -> None:
    """Write a uint64 as 8 little-endian bytes."""
    output[offset] = value & 0xFF
    output[offset + 1] = (value >> 8) & 0xFF
    output[offset + 2] = (value >> 16) & 0xFF
    output[offset + 3] = (value >> 24) & 0xFF
    output[offset + 4] = (value >> 32) & 0xFF
    output[offset + 5] = (value >> 40) & 0xFF
    output[offset + 6] = (value >> 48) & 0xFF
    output[offset + 7] = (value >> 56) & 0xFF


@nb.njit(cache=True, inline="always")
def _encode_delta_varint(
    delta: int, output: np.ndarray, out_offset: int
) -> int:
    """Zigzag + varint encode a delta value. Returns new out_offset."""
    val = (delta << 1) if delta >= 0 else ((-delta - 1) << 1) | 1
    val += 1  # Reserve 0 for NaN
    if val <= 0x7F:
        output[out_offset] = val & 0xFF
        return out_offset + 1
    elif val <= 0x3FFF:
        output[out_offset] = (val & 0x7F) | 0x80
        output[out_offset + 1] = (val >> 7) & 0xFF
        return out_offset + 2
    elif val <= 0x1FFFFF:
        output[out_offset] = (val & 0x7F) | 0x80
        output[out_offset + 1] = ((val >> 7) & 0x7F) | 0x80
        output[out_offset + 2] = (val >> 14) & 0xFF
        return out_offset + 3
    else:
        count = encode_varint64_to_buffer(delta, output[out_offset:], 0)
        return out_offset + count


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


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
                    # Raw copy (FLOAT32 without resolution)
                    output[out_offset] = point_data[data_offset]
                    output[out_offset + 1] = point_data[data_offset + 1]
                    output[out_offset + 2] = point_data[data_offset + 2]
                    output[out_offset + 3] = point_data[data_offset + 3]
                    out_offset += 4
                elif resolution > 0.0:
                    float_bits = _read_le_u32(point_data, data_offset)
                    scratch_u32[0] = float_bits
                    value_real = scratch_f32[0]

                    if np.isnan(value_real):
                        output[out_offset] = 0
                        prev_int_values[field_idx] = 0
                        out_offset += 1
                    else:
                        quantized = int(np.round(value_real / resolution))
                        delta = quantized - prev_int_values[field_idx]
                        prev_int_values[field_idx] = quantized
                        out_offset = _encode_delta_varint(delta, output, out_offset)
                else:
                    # Lossless: XOR encoding
                    float_bits = _read_le_u32(point_data, data_offset)
                    residual = float_bits ^ np.uint32(prev_float_bits[field_idx])
                    prev_float_bits[field_idx] = float_bits
                    _write_le_u32(output, out_offset, residual)
                    out_offset += 4

            elif field_type == FIELD_FLOAT64:
                float_bits = _read_le_u64(point_data, data_offset)

                if resolution > 0.0:
                    scratch_u64[0] = float_bits
                    value_real = scratch_f64[0]

                    if np.isnan(value_real):
                        output[out_offset] = 0
                        prev_int_values[field_idx] = 0
                        out_offset += 1
                    else:
                        quantized = int(np.round(value_real / resolution))
                        delta = quantized - prev_int_values[field_idx]
                        prev_int_values[field_idx] = quantized
                        out_offset = _encode_delta_varint(delta, output, out_offset)
                else:
                    # Lossless: XOR encoding
                    residual = float_bits ^ prev_float_bits[field_idx]
                    prev_float_bits[field_idx] = float_bits
                    _write_le_u64(output, out_offset, residual)
                    out_offset += 8

            elif field_type == FIELD_UINT8:
                output[out_offset] = point_data[data_offset]
                out_offset += 1

            elif field_type == FIELD_INT8:
                output[out_offset] = point_data[data_offset]
                out_offset += 1

            elif field_type == FIELD_UINT16:
                value = np.uint16(point_data[data_offset]) | (
                    np.uint16(point_data[data_offset + 1]) << 8
                )
                delta = np.int64(value) - prev_int_values[field_idx]
                prev_int_values[field_idx] = value
                out_offset = _encode_delta_varint(delta, output, out_offset)

            elif field_type == FIELD_INT16:
                uval = np.uint16(point_data[data_offset]) | (
                    np.uint16(point_data[data_offset + 1]) << 8
                )
                value = np.int64(np.int16(uval))
                delta = value - prev_int_values[field_idx]
                prev_int_values[field_idx] = value
                out_offset = _encode_delta_varint(delta, output, out_offset)

            elif field_type == FIELD_UINT32:
                value = _read_le_u32(point_data, data_offset)
                delta = np.int64(value) - prev_int_values[field_idx]
                prev_int_values[field_idx] = value
                out_offset = _encode_delta_varint(delta, output, out_offset)

            elif field_type == FIELD_INT32:
                uval = _read_le_u32(point_data, data_offset)
                value = np.int64(np.int32(uval))
                delta = value - prev_int_values[field_idx]
                prev_int_values[field_idx] = value
                out_offset = _encode_delta_varint(delta, output, out_offset)

            elif field_type == FIELD_UINT64:
                value = _read_le_u64(point_data, data_offset)
                delta = np.int64(value) - prev_int_values[field_idx]
                prev_int_values[field_idx] = np.int64(value)
                out_offset = _encode_delta_varint(delta, output, out_offset)

            elif field_type == FIELD_INT64:
                uval = _read_le_u64(point_data, data_offset)
                value = np.int64(uval)
                delta = value - prev_int_values[field_idx]
                prev_int_values[field_idx] = value
                out_offset = _encode_delta_varint(delta, output, out_offset)

    return out_offset


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


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
                    # Raw copy
                    output[data_offset] = encoded_data[in_offset]
                    output[data_offset + 1] = encoded_data[in_offset + 1]
                    output[data_offset + 2] = encoded_data[in_offset + 2]
                    output[data_offset + 3] = encoded_data[in_offset + 3]
                    in_offset += 4
                elif resolution > 0.0:
                    # Lossy: varint + delta + dequantize
                    if encoded_data[in_offset] == 0:
                        value_real = np.nan
                        prev_int_values[field_idx] = 0
                        in_offset += 1
                    else:
                        delta, count = decode_varint(encoded_data, in_offset)
                        in_offset += count
                        value = prev_int_values[field_idx] + delta
                        prev_int_values[field_idx] = value
                        value_real = float(value) * resolution

                    scratch_f32[0] = value_real
                    _write_le_u32(output, data_offset, scratch_u32[0])
                else:
                    # Lossless: XOR decoding
                    residual = _read_le_u32(encoded_data, in_offset)
                    in_offset += 4
                    current_bits = residual ^ np.uint32(prev_float_bits[field_idx])
                    prev_float_bits[field_idx] = current_bits
                    _write_le_u32(output, data_offset, current_bits)

            elif field_type == FIELD_FLOAT64:
                if resolution > 0.0:
                    # Lossy: varint + delta + dequantize
                    if encoded_data[in_offset] == 0:
                        value_real = np.nan
                        prev_int_values[field_idx] = 0
                        in_offset += 1
                    else:
                        delta, count = decode_varint(encoded_data, in_offset)
                        in_offset += count
                        value = prev_int_values[field_idx] + delta
                        prev_int_values[field_idx] = value
                        value_real = float(value) * resolution

                    scratch_f64[0] = value_real
                    _write_le_u64(output, data_offset, scratch_u64[0])
                else:
                    # Lossless: XOR decoding
                    residual = _read_le_u64(encoded_data, in_offset)
                    in_offset += 8
                    current_bits = residual ^ prev_float_bits[field_idx]
                    prev_float_bits[field_idx] = current_bits
                    _write_le_u64(output, data_offset, current_bits)

            elif field_type in (FIELD_UINT8, FIELD_INT8):
                output[data_offset] = encoded_data[in_offset]
                in_offset += 1

            elif field_type in (FIELD_UINT16, FIELD_INT16):
                delta, count = decode_varint(encoded_data, in_offset)
                in_offset += count
                value = prev_int_values[field_idx] + delta
                prev_int_values[field_idx] = value
                output[data_offset] = value & 0xFF
                output[data_offset + 1] = (value >> 8) & 0xFF

            elif field_type in (FIELD_UINT32, FIELD_INT32):
                delta, count = decode_varint(encoded_data, in_offset)
                in_offset += count
                value = prev_int_values[field_idx] + delta
                prev_int_values[field_idx] = value
                _write_le_u32(output, data_offset, np.uint32(value))

            elif field_type in (FIELD_UINT64, FIELD_INT64):
                delta, count = decode_varint(encoded_data, in_offset)
                in_offset += count
                value = prev_int_values[field_idx] + delta
                prev_int_values[field_idx] = value
                _write_le_u64(output, data_offset, np.uint64(value))

        # Successfully decoded this point
        points_decoded += 1

    return points_decoded
