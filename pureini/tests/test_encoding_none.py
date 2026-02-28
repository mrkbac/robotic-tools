"""
Test EncodingOptions.NONE mode.

This mode should copy raw bytes without any field-specific transformations
(no delta encoding, no varint, no XOR, no quantization).
"""

import math
import random
import struct

import pytest
from pureini import CompressionOption, EncodingInfo, EncodingOptions, FieldType, PointField
from pureini.decoder import PointcloudDecoder
from pureini.encoder import PointcloudEncoder


def test_none_mode_uint32():
    """
    Test NONE encoding with UINT32 field.

    Should produce exact byte-for-byte copy without delta/varint encoding.
    """
    random.seed(42)

    # Generate random uint32 values
    num_points = 1000
    input_data = [random.randint(0, 999) for _ in range(num_points)]  # noqa: S311

    # Create point cloud data
    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<I", value))

    # Create encoding info with NONE mode
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.NONE  # No field-specific encoding
    info.compression_opt = CompressionOption.NONE  # No compression
    info.fields = [PointField(name="value", offset=0, type=FieldType.UINT32, resolution=None)]

    # Encode
    encoder = PointcloudEncoder(info)
    compressed = encoder.encode(bytes(point_cloud))

    # Decode
    decoder = PointcloudDecoder()
    decompressed, decoded_info = decoder.decode(compressed)

    # Verify exact match (byte-for-byte)
    assert len(decompressed) == len(point_cloud)
    assert decompressed == bytes(point_cloud)

    # Verify decoded info
    assert decoded_info.encoding_opt == EncodingOptions.NONE

    print(
        f"\nNONE mode UINT32: {len(point_cloud)} bytes -> {len(compressed)} bytes "
        f"(header overhead: {len(compressed) - len(point_cloud)} bytes)"
    )


def test_none_mode_float32():
    """
    Test NONE encoding with FLOAT32 field.

    Should produce exact byte-for-byte copy without XOR/quantization.
    """
    random.seed(42)

    # Generate random float values including NaN
    num_points = 1000
    input_data = [0.001 * random.randint(0, 10000) for _ in range(num_points)]  # noqa: S311
    input_data[10] = float("nan")
    input_data[100] = float("inf")
    input_data[200] = float("-inf")

    # Create point cloud data
    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<f", value))

    # Create encoding info with NONE mode
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.NONE  # No field-specific encoding
    info.compression_opt = CompressionOption.NONE  # No compression
    info.fields = [PointField(name="value", offset=0, type=FieldType.FLOAT32, resolution=0.01)]

    # Encode
    encoder = PointcloudEncoder(info)
    compressed = encoder.encode(bytes(point_cloud))

    # Decode
    decoder = PointcloudDecoder()
    decompressed, _decoded_info = decoder.decode(compressed)

    # Verify exact match (byte-for-byte)
    assert len(decompressed) == len(point_cloud)
    assert decompressed == bytes(point_cloud)

    # Verify special values preserved
    for i in [10, 100, 200]:
        input_val = input_data[i]
        output_val = struct.unpack_from("<f", decompressed, i * 4)[0]
        if math.isnan(input_val):
            assert math.isnan(output_val)
        else:
            assert input_val == output_val

    print(
        f"\nNONE mode FLOAT32: {len(point_cloud)} bytes -> {len(compressed)} bytes "
        f"(header overhead: {len(compressed) - len(point_cloud)} bytes)"
    )


def test_none_mode_multi_field():
    """
    Test NONE encoding with multiple fields.

    Should copy all fields exactly without any transformations.
    """
    random.seed(42)

    # Generate data: x, y, z (float32), intensity (uint16)
    num_points = 500
    point_step = 14  # 3*4 + 2 = 14 bytes per point

    point_cloud = bytearray()
    input_values = []

    for _ in range(num_points):
        x = random.uniform(-10.0, 10.0)  # noqa: S311
        y = random.uniform(-10.0, 10.0)  # noqa: S311
        z = random.uniform(-10.0, 10.0)  # noqa: S311
        intensity = random.randint(0, 65535)  # noqa: S311

        point_cloud.extend(struct.pack("<fffH", x, y, z, intensity))
        input_values.append((x, y, z, intensity))

    # Create encoding info with NONE mode
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = point_step
    info.encoding_opt = EncodingOptions.NONE  # No field-specific encoding
    info.compression_opt = CompressionOption.NONE  # No compression
    info.fields = [
        PointField(name="x", offset=0, type=FieldType.FLOAT32, resolution=0.01),
        PointField(name="y", offset=4, type=FieldType.FLOAT32, resolution=0.01),
        PointField(name="z", offset=8, type=FieldType.FLOAT32, resolution=0.01),
        PointField(name="intensity", offset=12, type=FieldType.UINT16, resolution=None),
    ]

    # Encode
    encoder = PointcloudEncoder(info)
    compressed = encoder.encode(bytes(point_cloud))

    # Decode
    decoder = PointcloudDecoder()
    decompressed, _decoded_info = decoder.decode(compressed)

    # Verify exact match (byte-for-byte)
    assert len(decompressed) == len(point_cloud)
    assert decompressed == bytes(point_cloud)

    # Spot check a few points
    for i in [0, 100, 499]:
        offset = i * point_step
        x, y, z, intensity = struct.unpack_from("<fffH", decompressed, offset)
        x_in, y_in, z_in, intensity_in = input_values[i]
        # For floats, check byte-level equality since NONE mode preserves exact bits
        assert struct.pack("<f", x) == struct.pack("<f", x_in)
        assert struct.pack("<f", y) == struct.pack("<f", y_in)
        assert struct.pack("<f", z) == struct.pack("<f", z_in)
        assert intensity == intensity_in

    print(
        f"\nNONE mode multi-field: {len(point_cloud)} bytes -> {len(compressed)} bytes "
        f"(header overhead: {len(compressed) - len(point_cloud)} bytes)"
    )


def test_none_mode_with_compression():
    """
    Test NONE encoding with LZ4 compression.

    NONE mode only disables field-specific encoding, not stage-2 compression.
    """
    random.seed(42)

    # Generate data with some redundancy (better compression)
    num_points = 10000
    input_data = [i % 100 for i in range(num_points)]  # Repeating pattern

    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<I", value))

    # Create encoding info with NONE + LZ4
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.NONE  # No field-specific encoding
    info.compression_opt = CompressionOption.LZ4  # Enable LZ4 compression
    info.fields = [PointField(name="value", offset=0, type=FieldType.UINT32, resolution=None)]

    # Encode
    encoder = PointcloudEncoder(info)
    compressed = encoder.encode(bytes(point_cloud))

    # Decode
    decoder = PointcloudDecoder()
    decompressed, _decoded_info = decoder.decode(compressed)

    # Verify exact match
    assert len(decompressed) == len(point_cloud)
    assert decompressed == bytes(point_cloud)

    # Verify compression worked (should be much smaller due to LZ4)
    compression_ratio = len(point_cloud) / len(compressed)
    assert compression_ratio > 2.0, "LZ4 should provide significant compression on repeating data"

    print(
        f"\nNONE mode + LZ4: {len(point_cloud)} bytes -> {len(compressed)} bytes "
        f"(compression ratio: {compression_ratio:.2f}x)"
    )


def test_none_mode_chunking():
    """
    Test NONE encoding with multiple chunks (> 32,768 points).

    Should handle chunking correctly without transformations.
    """
    random.seed(42)

    # Generate data spanning multiple chunks
    num_points = 100000  # Should create 4 chunks (32,768 points per chunk)
    input_data = [random.randint(0, 999) for _ in range(num_points)]  # noqa: S311

    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<I", value))

    # Create encoding info with NONE mode
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.NONE
    info.compression_opt = CompressionOption.NONE
    info.fields = [PointField(name="value", offset=0, type=FieldType.UINT32, resolution=None)]

    # Encode
    encoder = PointcloudEncoder(info)
    compressed = encoder.encode(bytes(point_cloud))

    # Decode
    decoder = PointcloudDecoder()
    decompressed, _decoded_info = decoder.decode(compressed)

    # Verify exact match
    assert len(decompressed) == len(point_cloud)
    assert decompressed == bytes(point_cloud)

    # Verify specific values at chunk boundaries
    chunk_size = 32768
    test_indices = [0, 1, chunk_size - 1, chunk_size, chunk_size + 1, num_points - 1]

    for idx in test_indices:
        input_val = input_data[idx]
        output_val = struct.unpack_from("<I", decompressed, idx * 4)[0]
        assert input_val == output_val, f"Mismatch at index {idx}"

    expected_chunks = (num_points + chunk_size - 1) // chunk_size
    print(
        f"\nNONE mode chunking: {num_points} points ({expected_chunks} chunks), "
        f"{len(point_cloud)} bytes -> {len(compressed)} bytes"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
