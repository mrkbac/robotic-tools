"""
Test header encoding and decoding.

Ported from cloudini_lib/test/test_header.cpp
"""

import pytest
from pureini import (
    CompressionOption,
    EncodingInfo,
    EncodingOptions,
    FieldType,
    HeaderEncoding,
    PointField,
    decode_header,
    encode_header,
)


def test_header_yaml_encoding():
    """Test YAML header encoding and decoding (TEST(Cloudini, Header))."""
    # Create EncodingInfo with same configuration as C++ test
    header = EncodingInfo()
    header.width = 10
    header.height = 20
    header.point_step = 16  # sizeof(float) * 4
    header.encoding_opt = EncodingOptions.LOSSY
    header.compression_opt = CompressionOption.ZSTD

    # Add 4 FLOAT32 fields
    header.fields = [
        PointField(name="x", offset=0, type=FieldType.FLOAT32, resolution=0.01),
        PointField(name="y", offset=4, type=FieldType.FLOAT32, resolution=0.01),
        PointField(name="z", offset=8, type=FieldType.FLOAT32, resolution=0.01),
        PointField(name="intensity", offset=12, type=FieldType.FLOAT32, resolution=0.01),
    ]

    # Encode header as YAML
    encoded = encode_header(header, HeaderEncoding.YAML)

    # Decode header
    decoded_header, header_size = decode_header(encoded)

    # Validate header size matches encoded length
    assert header_size == len(encoded)

    # Validate metadata
    assert decoded_header.width == header.width  # 10
    assert decoded_header.height == header.height  # 20
    assert decoded_header.point_step == header.point_step  # 16
    assert decoded_header.encoding_opt == header.encoding_opt  # LOSSY
    assert decoded_header.compression_opt == header.compression_opt  # ZSTD
    assert len(decoded_header.fields) == len(header.fields)  # 4

    # Validate each field
    for i in range(len(header.fields)):
        assert decoded_header.fields[i].name == header.fields[i].name
        assert decoded_header.fields[i].offset == header.fields[i].offset
        assert decoded_header.fields[i].type == header.fields[i].type
        # Float precision: compare with small tolerance
        if (
            header.fields[i].resolution is not None
            and decoded_header.fields[i].resolution is not None
        ):
            assert abs(decoded_header.fields[i].resolution - header.fields[i].resolution) < 1e-6
        else:
            assert decoded_header.fields[i].resolution == header.fields[i].resolution


def test_header_binary_encoding():
    """Test binary header encoding and decoding."""
    # Create EncodingInfo
    header = EncodingInfo()
    header.width = 100
    header.height = 1
    header.point_step = 32
    header.encoding_opt = EncodingOptions.LOSSLESS
    header.compression_opt = CompressionOption.LZ4

    # Add mixed field types
    header.fields = [
        PointField(name="x", offset=0, type=FieldType.FLOAT32, resolution=0.001),
        PointField(name="y", offset=4, type=FieldType.FLOAT32, resolution=0.001),
        PointField(name="z", offset=8, type=FieldType.FLOAT32, resolution=0.001),
        PointField(name="intensity", offset=16, type=FieldType.FLOAT32, resolution=None),
        PointField(name="ring", offset=20, type=FieldType.UINT16, resolution=None),
        PointField(name="timestamp", offset=24, type=FieldType.FLOAT64, resolution=None),
    ]

    # Encode header as BINARY
    encoded = encode_header(header, HeaderEncoding.BINARY)

    # Decode header
    decoded_header, header_size = decode_header(encoded)

    # Validate header size matches encoded length
    assert header_size == len(encoded)

    # Validate metadata
    assert decoded_header.width == header.width
    assert decoded_header.height == header.height
    assert decoded_header.point_step == header.point_step
    assert decoded_header.encoding_opt == header.encoding_opt
    assert decoded_header.compression_opt == header.compression_opt
    assert len(decoded_header.fields) == len(header.fields)

    # Validate each field
    for i in range(len(header.fields)):
        assert decoded_header.fields[i].name == header.fields[i].name
        assert decoded_header.fields[i].offset == header.fields[i].offset
        assert decoded_header.fields[i].type == header.fields[i].type
        # Float precision: compare with small tolerance
        if (
            header.fields[i].resolution is not None
            and decoded_header.fields[i].resolution is not None
        ):
            assert abs(decoded_header.fields[i].resolution - header.fields[i].resolution) < 1e-6
        else:
            assert decoded_header.fields[i].resolution == header.fields[i].resolution


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
