"""
Encoding utilities for Pureini point cloud compression.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

import struct

import numba as nb
import numpy as np

from .types import EncodingInfo, EncodingOptions, FieldType


class BufferView:
    """
    A view into a byte buffer that can be consumed as data is written/read.
    Accepts bytes, bytearray, or memoryview. Write operations require a
    writable underlying buffer (bytearray or writable memoryview).
    """

    def __init__(self, data: bytes | bytearray | memoryview) -> None:
        if isinstance(data, (bytes, bytearray)):
            self._view = memoryview(data)
        else:
            self._view = data
        self._offset = 0

    @property
    def data(self) -> memoryview:
        """Get the current view of remaining data."""
        return self._view[self._offset :]

    def size(self) -> int:
        """Get the size of remaining data."""
        return len(self._view) - self._offset

    def empty(self) -> bool:
        """Check if buffer is empty."""
        return self.size() == 0

    def trim_front(self, n: int) -> None:
        if n > self.size():
            raise RuntimeError(f"Cannot trim {n} bytes, only {self.size()} available")
        self._offset += n

    def write_bytes(self, data: bytes | bytearray) -> None:
        n = len(data)
        if n > self.size():
            raise RuntimeError(f"Cannot write {n} bytes, only {self.size()} available")
        self.data[:n] = data
        self.trim_front(n)

    def read_bytes(self, n: int) -> bytes:
        if n > self.size():
            raise RuntimeError(f"Cannot read {n} bytes, only {self.size()} available")
        result = bytes(self.data[:n])
        self.trim_front(n)
        return result


@nb.njit(cache=True, fastmath=True)
def encode_varint64_to_buffer(value: int, buffer: memoryview, offset: int = 0) -> int:
    """
    Encode a signed 64-bit integer directly to a buffer (zero-copy).
    Value 0 is reserved for NaN, so all values are shifted by +1.

    JIT-compiled with Numba for maximum performance.

    Args:
        value: The signed 64-bit integer to encode
        buffer: The target buffer (memoryview)
        offset: Starting offset in the buffer

    Returns:
        Number of bytes written
    """
    # Zigzag encoding
    val = value << 1 if value >= 0 else ((-value - 1) << 1) | 1

    # Reserve value 0 for NaN
    val += 1

    # Varint encoding - write directly to buffer
    ptr = offset
    while val > 0x7F:
        buffer[ptr] = (val & 0x7F) | 0x80
        val >>= 7
        ptr += 1
    buffer[ptr] = val & 0xFF
    ptr += 1

    return ptr - offset


def encode_varint64(value: int) -> bytes:
    """
    Encode a signed 64-bit integer using zigzag encoding + varint.
    Value 0 is reserved for NaN, so all values are shifted by +1.

    Args:
        value: The signed 64-bit integer to encode

    Returns:
        Varint-encoded bytes

    Note: This creates a new bytearray. For better performance,
    use encode_varint64_to_buffer() to write directly to a buffer.
    """
    # Zigzag encoding: (value << 1) ^ (value >> 63)
    # For Python, we need to handle the sign extension properly
    val = value << 1 if value >= 0 else ((-value - 1) << 1) | 1

    # Reserve value 0 for NaN
    val += 1

    # Varint encoding
    result = bytearray()
    while val > 0x7F:
        result.append((val & 0x7F) | 0x80)
        val >>= 7
    result.append(val & 0xFF)

    return bytes(result)


@nb.njit(cache=True, fastmath=True)
def decode_varint(data: bytes | memoryview, offset: int = 0) -> tuple[int, int]:
    """
    Decode a zigzag-encoded varint from bytes.
    Handles the NaN reservation (value 0 is NaN).

    JIT-compiled with Numba for maximum performance.

    Args:
        data: The byte buffer
        offset: Starting offset in the buffer

    Returns:
        Tuple of (decoded_value, bytes_consumed)
    """
    uval = 0
    shift = 0
    ptr = offset

    while True:
        if ptr >= len(data):
            raise RuntimeError("Incomplete varint in buffer")

        byte = data[ptr]
        ptr += 1
        uval |= (byte & 0x7F) << shift
        shift += 7

        if (byte & 0x80) == 0:
            break

    # Value 0 is reserved for NaN
    uval -= 1

    # Zigzag decoding (branchless)
    val = (uval >> 1) ^ -(uval & 1)

    return val, ptr - offset


def encode(value: float, buff: BufferView, format_char: str) -> None:
    """
    Encode a primitive value into the buffer using struct.pack.

    Args:
        value: The value to encode
        buff: The buffer view to write to
        format_char: struct format character (e.g., 'f' for float, 'I' for uint32)
    """
    data = struct.pack(f"<{format_char}", value)  # Little-endian
    buff.write_bytes(data)


def encode_string(s: str, buff: BufferView) -> None:
    """
    Encode a string as uint16 length + UTF-8 bytes.

    Args:
        s: The string to encode
        buff: The buffer view to write to
    """
    encoded = s.encode("utf-8")
    length = len(encoded)
    if length > 65535:
        raise ValueError(f"String too long: {length} bytes (max 65535)")

    # Write length as uint16
    encode(length, buff, "H")
    # Write string bytes
    buff.write_bytes(encoded)


def decode(buff: BufferView, format_char: str) -> int | float:
    """
    Decode a primitive value from the buffer using struct.unpack.

    Args:
        buff: The buffer view to read from
        format_char: struct format character

    Returns:
        The decoded value
    """
    size = struct.calcsize(f"<{format_char}")
    data = buff.read_bytes(size)
    return struct.unpack(f"<{format_char}", data)[0]


def decode_string(buff: BufferView) -> str:
    """
    Decode a string (uint16 length + UTF-8 bytes).

    Args:
        buff: The buffer view to read from

    Returns:
        The decoded string
    """
    length = int(decode(buff, "H"))
    encoded = buff.read_bytes(length)
    return encoded.decode("utf-8")


def to_int64(data: bytes | bytearray | memoryview, offset: int, dtype: str) -> int:
    """
    Read a value from bytes and convert to int64.

    Args:
        data: The byte buffer (bytes, bytearray, or memoryview)
        offset: Offset in the buffer
        dtype: Data type ('b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q')

    Returns:
        The value as int64
    """
    value = struct.unpack_from(f"<{dtype}", data, offset)[0]
    return int(value)


def build_field_metadata(
    info: EncodingInfo,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build numpy arrays for field metadata to pass to JIT functions.

    Creates three arrays:
    - field_offsets: byte offset of each field
    - field_types: type code for each field
    - field_resolutions: resolution for each field

    Resolution sentinel values:
      > 0.0  -> lossy quantize+delta+varint
      == 0.0 -> XOR lossless (FLOAT64 only)
      -1.0   -> raw 4-byte copy (FLOAT32 without resolution)

    Args:
        info: Encoding configuration

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

        if info.encoding_opt == EncodingOptions.LOSSY and field.resolution is not None:
            field_resolutions[idx] = field.resolution
        elif field.type == FieldType.FLOAT64:
            field_resolutions[idx] = 0.0
        elif field.type == FieldType.FLOAT32:
            field_resolutions[idx] = -1.0
        else:
            field_resolutions[idx] = 0.0

    return field_offsets, field_types, field_resolutions
