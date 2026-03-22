"""Tests for pureini.encoding_utils — BufferView, varint encoding, field metadata."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from pureini.encoding_utils import (
    BufferView,
    build_field_metadata,
    decode,
    decode_string,
    decode_varint,
    encode,
    encode_string,
    encode_varint64,
    to_int64,
)
from pureini.types import EncodingInfo, EncodingOptions, FieldType, PointField

# ---------------------------------------------------------------------------
# BufferView
# ---------------------------------------------------------------------------


class TestBufferView:
    def test_init_from_bytes(self):
        buf = BufferView(b"hello")
        assert buf.size() == 5
        assert not buf.empty()

    def test_init_from_bytearray(self):
        buf = BufferView(bytearray(10))
        assert buf.size() == 10

    def test_trim_front(self):
        buf = BufferView(b"hello world")
        buf.trim_front(6)
        assert buf.size() == 5
        assert bytes(buf.data) == b"world"

    def test_trim_front_overflow(self):
        buf = BufferView(b"hi")
        with pytest.raises(RuntimeError, match="Cannot trim"):
            buf.trim_front(5)

    def test_read_bytes(self):
        buf = BufferView(b"hello world")
        result = buf.read_bytes(5)
        assert result == b"hello"
        assert buf.size() == 6

    def test_read_bytes_overflow(self):
        buf = BufferView(b"hi")
        with pytest.raises(RuntimeError, match="Cannot read"):
            buf.read_bytes(5)

    def test_write_bytes(self):
        data = bytearray(10)
        buf = BufferView(data)
        buf.write_bytes(b"AB")
        assert data[:2] == b"AB"
        assert buf.size() == 8

    def test_write_bytes_overflow(self):
        data = bytearray(2)
        buf = BufferView(data)
        with pytest.raises(RuntimeError, match="Cannot write"):
            buf.write_bytes(b"hello")

    def test_empty(self):
        buf = BufferView(b"")
        assert buf.empty()

    def test_data_property(self):
        buf = BufferView(b"abcdef")
        buf.trim_front(3)
        assert bytes(buf.data) == b"def"


# ---------------------------------------------------------------------------
# Varint encode/decode roundtrip
# ---------------------------------------------------------------------------


class TestVarintRoundtrip:
    @pytest.mark.parametrize("value", [0, 1, -1, 127, -128, 1000, -1000, 2**31, -(2**31)])
    def test_roundtrip(self, value: int):
        encoded = encode_varint64(value)
        decoded, consumed = decode_varint(encoded, 0)
        assert decoded == value
        assert consumed == len(encoded)

    def test_zero(self):
        encoded = encode_varint64(0)
        decoded, _ = decode_varint(encoded, 0)
        assert decoded == 0

    def test_large_positive(self):
        value = 2**30  # within safe zigzag range
        encoded = encode_varint64(value)
        decoded, _ = decode_varint(encoded, 0)
        assert decoded == value

    def test_large_negative(self):
        value = -(2**30)
        encoded = encode_varint64(value)
        decoded, _ = decode_varint(encoded, 0)
        assert decoded == value

    def test_single_byte_values(self):
        # Small values should encode to few bytes
        encoded = encode_varint64(0)
        assert len(encoded) == 1

    def test_multi_byte_values(self):
        encoded = encode_varint64(10000)
        assert len(encoded) > 1


# ---------------------------------------------------------------------------
# Encode/decode primitives
# ---------------------------------------------------------------------------


class TestEncodeDecodePrimitive:
    def test_float32_roundtrip(self):
        data = bytearray(4)
        buf = BufferView(data)
        encode(3.14, buf, "f")
        buf2 = BufferView(data)
        result = decode(buf2, "f")
        assert abs(result - 3.14) < 1e-5

    def test_uint32_roundtrip(self):
        data = bytearray(4)
        buf = BufferView(data)
        encode(42, buf, "I")
        buf2 = BufferView(data)
        result = decode(buf2, "I")
        assert result == 42

    def test_int64_roundtrip(self):
        data = bytearray(8)
        buf = BufferView(data)
        encode(-999, buf, "q")
        buf2 = BufferView(data)
        result = decode(buf2, "q")
        assert result == -999


# ---------------------------------------------------------------------------
# String encode/decode
# ---------------------------------------------------------------------------


class TestStringEncoding:
    def test_roundtrip(self):
        data = bytearray(100)
        buf = BufferView(data)
        encode_string("hello", buf)
        buf2 = BufferView(data)
        result = decode_string(buf2)
        assert result == "hello"

    def test_empty_string(self):
        data = bytearray(10)
        buf = BufferView(data)
        encode_string("", buf)
        buf2 = BufferView(data)
        result = decode_string(buf2)
        assert result == ""

    def test_unicode(self):
        data = bytearray(100)
        buf = BufferView(data)
        encode_string("hello 🌍", buf)
        buf2 = BufferView(data)
        result = decode_string(buf2)
        assert result == "hello 🌍"


# ---------------------------------------------------------------------------
# to_int64
# ---------------------------------------------------------------------------


class TestToInt64:
    def test_uint8(self):
        data = bytes([42])
        assert to_int64(data, 0, "B") == 42

    def test_int32(self):
        data = struct.pack("<i", -1000)
        assert to_int64(data, 0, "i") == -1000

    def test_uint32(self):
        data = struct.pack("<I", 3_000_000_000)
        assert to_int64(data, 0, "I") == 3_000_000_000

    def test_with_offset(self):
        data = b"\x00\x00" + struct.pack("<i", 99)
        assert to_int64(data, 2, "i") == 99


# ---------------------------------------------------------------------------
# build_field_metadata
# ---------------------------------------------------------------------------


class TestBuildFieldMetadata:
    def test_basic_float32_fields(self):
        info = EncodingInfo(
            fields=[
                PointField(name="x", offset=0, type=FieldType.FLOAT32),
                PointField(name="y", offset=4, type=FieldType.FLOAT32),
                PointField(name="z", offset=8, type=FieldType.FLOAT32),
            ],
            width=10,
            height=1,
            point_step=12,
            encoding_opt=EncodingOptions.NONE,
        )
        offsets, types, resolutions = build_field_metadata(info)
        assert len(offsets) == 3
        np.testing.assert_array_equal(offsets, [0, 4, 8])
        np.testing.assert_array_equal(types, [int(FieldType.FLOAT32)] * 3)
        # FLOAT32 without resolution -> -1.0 sentinel
        np.testing.assert_array_equal(resolutions, [-1.0, -1.0, -1.0])

    def test_lossy_with_resolution(self):
        info = EncodingInfo(
            fields=[
                PointField(name="x", offset=0, type=FieldType.FLOAT32, resolution=0.01),
            ],
            width=10,
            height=1,
            point_step=4,
            encoding_opt=EncodingOptions.LOSSY,
        )
        _, _, resolutions = build_field_metadata(info)
        assert resolutions[0] == pytest.approx(0.01)

    def test_float64_gets_xor(self):
        info = EncodingInfo(
            fields=[
                PointField(name="t", offset=0, type=FieldType.FLOAT64),
            ],
            width=10,
            height=1,
            point_step=8,
            encoding_opt=EncodingOptions.NONE,
        )
        _, _, resolutions = build_field_metadata(info)
        assert resolutions[0] == 0.0  # XOR lossless sentinel

    def test_int_types_get_zero_resolution(self):
        info = EncodingInfo(
            fields=[
                PointField(name="intensity", offset=0, type=FieldType.UINT8),
                PointField(name="ring", offset=1, type=FieldType.UINT16),
            ],
            width=10,
            height=1,
            point_step=3,
            encoding_opt=EncodingOptions.NONE,
        )
        _, _, resolutions = build_field_metadata(info)
        np.testing.assert_array_equal(resolutions, [0.0, 0.0])
