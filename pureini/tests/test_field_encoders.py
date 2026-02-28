"""
Test field encoders and decoders.

Ported from cloudini_lib/test/test_field_encoders.cpp
"""

import math
import random
import struct

import numpy as np
import pytest
from pureini import CompressionOption, EncodingInfo, EncodingOptions, FieldType, PointField
from pureini.decoder import PointcloudDecoder
from pureini.encoder import PointcloudEncoder


def _roundtrip(info: EncodingInfo, point_cloud: bytes) -> bytes:
    """Encode then decode, returning decompressed bytes."""
    encoder = PointcloudEncoder(info)
    compressed = encoder.encode(point_cloud)
    decoder = PointcloudDecoder()
    decompressed, _ = decoder.decode(compressed)
    return decompressed


# ---------------------------------------------------------------------------
# Integer field tests
# ---------------------------------------------------------------------------


def test_int_field():
    """
    Test integer field encoding/decoding (TEST(FieldEncoders, IntField)).

    Exact port of C++ test:
    - 100 random uint32 values (0-999)
    - Delta + varint encoding
    - Lossless (exact match required)
    """
    random.seed(42)

    num_points = 100
    input_data = [random.randint(0, 999) for _ in range(num_points)]  # noqa: S311

    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<I", value))

    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.LOSSY
    info.compression_opt = CompressionOption.NONE
    info.fields = [PointField(name="value", offset=0, type=FieldType.UINT32, resolution=None)]

    decompressed = _roundtrip(info, bytes(point_cloud))

    for i in range(num_points):
        value = struct.unpack_from("<I", decompressed, i * 4)[0]
        assert value == input_data[i], f"Mismatch at index {i}"


def test_int16_field():
    """INT16 delta+varint roundtrip — exact."""
    random.seed(123)
    num_points = 200
    input_data = [random.randint(-500, 500) for _ in range(num_points)]  # noqa: S311

    point_cloud = bytearray()
    for v in input_data:
        point_cloud.extend(struct.pack("<h", v))

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=2,
        encoding_opt=EncodingOptions.LOSSY,
        compression_opt=CompressionOption.NONE,
        fields=[PointField(name="val", offset=0, type=FieldType.INT16)],
    )

    decompressed = _roundtrip(info, bytes(point_cloud))
    for i in range(num_points):
        assert struct.unpack_from("<h", decompressed, i * 2)[0] == input_data[i]


def test_int32_field():
    """INT32 delta+varint roundtrip — exact."""
    random.seed(456)
    num_points = 200
    input_data = [random.randint(-100_000, 100_000) for _ in range(num_points)]  # noqa: S311

    point_cloud = bytearray()
    for v in input_data:
        point_cloud.extend(struct.pack("<i", v))

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=4,
        encoding_opt=EncodingOptions.LOSSY,
        compression_opt=CompressionOption.NONE,
        fields=[PointField(name="val", offset=0, type=FieldType.INT32)],
    )

    decompressed = _roundtrip(info, bytes(point_cloud))
    for i in range(num_points):
        assert struct.unpack_from("<i", decompressed, i * 4)[0] == input_data[i]


def test_int64_field():
    """INT64 delta+varint roundtrip — exact."""
    random.seed(789)
    num_points = 200
    input_data = [random.randint(-(2**40), 2**40) for _ in range(num_points)]  # noqa: S311

    point_cloud = bytearray()
    for v in input_data:
        point_cloud.extend(struct.pack("<q", v))

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=8,
        encoding_opt=EncodingOptions.LOSSY,
        compression_opt=CompressionOption.NONE,
        fields=[PointField(name="val", offset=0, type=FieldType.INT64)],
    )

    decompressed = _roundtrip(info, bytes(point_cloud))
    for i in range(num_points):
        assert struct.unpack_from("<q", decompressed, i * 8)[0] == input_data[i]


def test_uint16_field():
    """UINT16 delta+varint roundtrip — exact."""
    random.seed(111)
    num_points = 200
    input_data = [random.randint(0, 65535) for _ in range(num_points)]  # noqa: S311

    point_cloud = bytearray()
    for v in input_data:
        point_cloud.extend(struct.pack("<H", v))

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=2,
        encoding_opt=EncodingOptions.LOSSY,
        compression_opt=CompressionOption.NONE,
        fields=[PointField(name="val", offset=0, type=FieldType.UINT16)],
    )

    decompressed = _roundtrip(info, bytes(point_cloud))
    for i in range(num_points):
        assert struct.unpack_from("<H", decompressed, i * 2)[0] == input_data[i]


def test_uint64_field():
    """UINT64 delta+varint roundtrip — exact."""
    random.seed(222)
    num_points = 200
    input_data = [random.randint(0, 2**48) for _ in range(num_points)]  # noqa: S311

    point_cloud = bytearray()
    for v in input_data:
        point_cloud.extend(struct.pack("<Q", v))

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=8,
        encoding_opt=EncodingOptions.LOSSY,
        compression_opt=CompressionOption.NONE,
        fields=[PointField(name="val", offset=0, type=FieldType.UINT64)],
    )

    decompressed = _roundtrip(info, bytes(point_cloud))
    for i in range(num_points):
        assert struct.unpack_from("<Q", decompressed, i * 8)[0] == input_data[i]


# ---------------------------------------------------------------------------
# Float tests
# ---------------------------------------------------------------------------


def test_float_lossy():
    """
    Test lossy float compression (TEST(FieldEncoders, FloatLossy)).

    Exact port of C++ test:
    - 1,000,000 random floats (0.0 to ~10.0)
    - Resolution: 0.01
    - NaN at indices 1, 15, 16
    - Tolerance: 0.010001
    """
    num_points = 1_000_000
    resolution = 0.01
    tolerance = resolution * 1.0001

    random.seed(42)
    input_data = [0.001 * random.randint(0, 10000) for _ in range(num_points)]  # noqa: S311

    input_data[1] = float("nan")
    input_data[15] = float("nan")
    input_data[16] = float("nan")

    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<f", value))

    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.LOSSY
    info.compression_opt = CompressionOption.NONE
    info.fields = [
        PointField(name="the_float", offset=0, type=FieldType.FLOAT32, resolution=resolution)
    ]

    decompressed = _roundtrip(info, bytes(point_cloud))

    for i in range(num_points):
        value = struct.unpack_from("<f", decompressed, i * 4)[0]
        if math.isnan(input_data[i]):
            assert math.isnan(value), f"NaN not preserved at index {i}"
        else:
            diff = abs(value - input_data[i])
            assert diff <= tolerance, f"Tolerance exceeded at index {i}: {diff} > {tolerance}"


def test_float32_lossless():
    """FLOAT32 lossless (raw copy roundtrip) — byte-exact."""
    random.seed(333)
    num_points = 500

    # Build raw float32 bytes including NaN and special values
    rng = np.random.default_rng(333)
    values = rng.standard_normal(num_points).astype(np.float32)
    values[0] = np.float32(0.0)
    values[1] = np.float32("nan")
    values[2] = np.float32("inf")
    values[3] = np.float32("-inf")

    point_cloud = values.tobytes()

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=4,
        encoding_opt=EncodingOptions.LOSSLESS,
        compression_opt=CompressionOption.NONE,
        fields=[PointField(name="val", offset=0, type=FieldType.FLOAT32)],
    )

    decompressed = _roundtrip(info, point_cloud)
    assert decompressed == point_cloud, "FLOAT32 lossless roundtrip must be byte-exact"


def test_float64_lossy():
    """FLOAT64 lossy with resolution — within tolerance."""
    num_points = 10_000
    resolution = 0.001
    tolerance = resolution * 1.0001

    rng = np.random.default_rng(444)
    values = (rng.random(num_points) * 100.0).astype(np.float64)
    values[5] = np.nan

    point_cloud = values.tobytes()

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=8,
        encoding_opt=EncodingOptions.LOSSY,
        compression_opt=CompressionOption.NONE,
        fields=[PointField(name="val", offset=0, type=FieldType.FLOAT64, resolution=resolution)],
    )

    decompressed = _roundtrip(info, point_cloud)

    out = np.frombuffer(decompressed, dtype=np.float64)
    for i in range(num_points):
        if np.isnan(values[i]):
            assert np.isnan(out[i]), f"NaN not preserved at {i}"
        else:
            assert abs(out[i] - values[i]) <= tolerance, f"Tolerance exceeded at {i}"


def test_float64_lossless():
    """FLOAT64 lossless (XOR) — byte-exact."""
    rng = np.random.default_rng(555)
    num_points = 500
    values = rng.standard_normal(num_points).astype(np.float64)
    values[0] = 0.0
    values[1] = np.nan
    values[2] = np.inf
    values[3] = -np.inf

    point_cloud = values.tobytes()

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=8,
        encoding_opt=EncodingOptions.LOSSLESS,
        compression_opt=CompressionOption.NONE,
        fields=[PointField(name="val", offset=0, type=FieldType.FLOAT64)],
    )

    decompressed = _roundtrip(info, point_cloud)
    assert decompressed == point_cloud, "FLOAT64 lossless roundtrip must be byte-exact"


# ---------------------------------------------------------------------------
# Mixed fields test (port of C++ test_ros_msg.cpp)
# ---------------------------------------------------------------------------


def test_mixed_fields():
    """
    Mixed field types: x,y,z FLOAT32 + intensity FLOAT32 + ring UINT16 + timestamp FLOAT64.
    64K points, LOSSY+ZSTD.

    Port of C++ test_ros_msg.cpp style test.
    """
    num_points = 65536
    # Layout: x(f32) y(f32) z(f32) intensity(f32) ring(u16) _pad(2) timestamp(f64)
    # offsets:  0      4      8      12              16       18      20
    point_step = 28

    rng = np.random.default_rng(777)

    point_cloud = bytearray(num_points * point_step)
    for i in range(num_points):
        base = i * point_step
        x = rng.uniform(-10.0, 10.0)
        y = rng.uniform(-10.0, 10.0)
        z = rng.uniform(-2.0, 5.0)
        intensity = rng.uniform(0.0, 100.0)
        ring = int(rng.integers(0, 128))
        timestamp = 1700000000.0 + i * 0.0001

        struct.pack_into("<f", point_cloud, base + 0, x)
        struct.pack_into("<f", point_cloud, base + 4, y)
        struct.pack_into("<f", point_cloud, base + 8, z)
        struct.pack_into("<f", point_cloud, base + 12, intensity)
        struct.pack_into("<H", point_cloud, base + 16, ring)
        # 2 bytes padding at 18-19 (left as zero)
        struct.pack_into("<d", point_cloud, base + 20, timestamp)

    xyz_res = 0.001
    intensity_res = 0.1
    ts_res = 0.000001

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=point_step,
        encoding_opt=EncodingOptions.LOSSY,
        compression_opt=CompressionOption.ZSTD,
        fields=[
            PointField(name="x", offset=0, type=FieldType.FLOAT32, resolution=xyz_res),
            PointField(name="y", offset=4, type=FieldType.FLOAT32, resolution=xyz_res),
            PointField(name="z", offset=8, type=FieldType.FLOAT32, resolution=xyz_res),
            PointField(
                name="intensity", offset=12, type=FieldType.FLOAT32, resolution=intensity_res
            ),
            PointField(name="ring", offset=16, type=FieldType.UINT16),
            PointField(name="timestamp", offset=20, type=FieldType.FLOAT64, resolution=ts_res),
        ],
    )

    encoder = PointcloudEncoder(info)
    compressed = encoder.encode(bytes(point_cloud))
    decoder = PointcloudDecoder()
    decompressed, decoded_info = decoder.decode(compressed)

    assert decoded_info.width == num_points
    assert len(decompressed) == len(point_cloud)

    for i in range(num_points):
        base = i * point_step
        orig_x = struct.unpack_from("<f", point_cloud, base + 0)[0]
        orig_y = struct.unpack_from("<f", point_cloud, base + 4)[0]
        orig_z = struct.unpack_from("<f", point_cloud, base + 8)[0]
        orig_int = struct.unpack_from("<f", point_cloud, base + 12)[0]
        orig_ring = struct.unpack_from("<H", point_cloud, base + 16)[0]
        orig_ts = struct.unpack_from("<d", point_cloud, base + 20)[0]

        dec_x = struct.unpack_from("<f", decompressed, base + 0)[0]
        dec_y = struct.unpack_from("<f", decompressed, base + 4)[0]
        dec_z = struct.unpack_from("<f", decompressed, base + 8)[0]
        dec_int = struct.unpack_from("<f", decompressed, base + 12)[0]
        dec_ring = struct.unpack_from("<H", decompressed, base + 16)[0]
        dec_ts = struct.unpack_from("<d", decompressed, base + 20)[0]

        assert abs(dec_x - orig_x) <= xyz_res * 1.0001, f"x mismatch at {i}"
        assert abs(dec_y - orig_y) <= xyz_res * 1.0001, f"y mismatch at {i}"
        assert abs(dec_z - orig_z) <= xyz_res * 1.0001, f"z mismatch at {i}"
        assert abs(dec_int - orig_int) <= intensity_res * 1.0001, f"intensity mismatch at {i}"
        assert dec_ring == orig_ring, f"ring mismatch at {i}"
        assert abs(dec_ts - orig_ts) <= ts_res * 1.0001, f"timestamp mismatch at {i}"

    print(f"\nMixed fields: {len(point_cloud)} -> {len(compressed)} bytes")
    print(f"Compression ratio: {len(point_cloud) / len(compressed):.2f}x")


# ---------------------------------------------------------------------------
# Multi-chunk with compression tests
# ---------------------------------------------------------------------------


def test_multi_chunk_lz4():
    """Multi-chunk (>32768 points) with LZ4 compression, LOSSY encoding."""
    num_points = 40000  # > POINTS_PER_CHUNK (32768) → 2 chunks
    resolution = 0.01
    tolerance = resolution * 1.0001

    rng = np.random.default_rng(888)
    values = (rng.random(num_points) * 50.0).astype(np.float32)
    point_cloud = values.tobytes()

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=4,
        encoding_opt=EncodingOptions.LOSSY,
        compression_opt=CompressionOption.LZ4,
        fields=[PointField(name="val", offset=0, type=FieldType.FLOAT32, resolution=resolution)],
    )

    decompressed = _roundtrip(info, point_cloud)
    out = np.frombuffer(decompressed, dtype=np.float32)

    assert len(out) == num_points
    for i in range(num_points):
        assert abs(out[i] - values[i]) <= tolerance, f"Mismatch at {i}"


def test_multi_chunk_zstd():
    """Multi-chunk (>32768 points) with ZSTD compression, LOSSY encoding."""
    num_points = 70000  # > 2 * POINTS_PER_CHUNK → 3 chunks
    resolution = 0.005
    tolerance = resolution * 1.0001

    rng = np.random.default_rng(999)
    values = (rng.random(num_points) * 20.0).astype(np.float32)
    point_cloud = values.tobytes()

    info = EncodingInfo(
        width=num_points,
        height=1,
        point_step=4,
        encoding_opt=EncodingOptions.LOSSY,
        compression_opt=CompressionOption.ZSTD,
        fields=[PointField(name="val", offset=0, type=FieldType.FLOAT32, resolution=resolution)],
    )

    decompressed = _roundtrip(info, point_cloud)
    out = np.frombuffer(decompressed, dtype=np.float32)

    assert len(out) == num_points
    for i in range(num_points):
        assert abs(out[i] - values[i]) <= tolerance, f"Mismatch at {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
