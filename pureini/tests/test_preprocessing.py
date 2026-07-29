from __future__ import annotations

import math
import struct

import pytest
from pureini import (
    CompressionOption,
    EncodingInfo,
    EncodingOptions,
    FieldType,
    PointcloudDecoder,
    PointcloudEncoder,
    PointField,
)

_POINT = struct.Struct("<fffHH")
_BIG_ENDIAN_POINT = struct.Struct(">fffHH")
_FLOAT64_POINT = struct.Struct("<dddB7x")


def _info(width: int) -> EncodingInfo:
    return EncodingInfo(
        fields=[
            PointField("x", 0, FieldType.FLOAT32, 0.01),
            PointField("y", 4, FieldType.FLOAT32, 0.01),
            PointField("z", 8, FieldType.FLOAT32, 0.01),
            PointField("line", 12, FieldType.UINT16),
            PointField("intensity", 14, FieldType.UINT16),
        ],
        width=width,
        height=1,
        point_step=_POINT.size,
        encoding_opt=EncodingOptions.LOSSLESS,
        compression_opt=CompressionOption.NONE,
    )


def _pack(points: list[tuple[float, float, float, int, int]]) -> bytes:
    return b"".join(_POINT.pack(*point) for point in points)


def _unpack(data: bytes) -> list[tuple[float, float, float, int, int]]:
    return [tuple(values) for values in struct.iter_unpack(_POINT.format, data)]


def test_encode_filters_and_stably_groups_in_one_call() -> None:
    points = [
        (3.0, 3.0, 3.0, 2, 30),
        (0.0, 0.0, 0.0, 0, 99),
        (1.0, 1.0, 1.0, 1, 10),
        (math.nan, 2.0, 2.0, 0, 98),
        (2.0, 2.0, 2.0, 1, 20),
    ]
    encoder = PointcloudEncoder(_info(len(points)))

    encoded, transformed_point_count, did_filter_invalid_xyz = encoder.encode(
        _pack(points),
        drop_invalid=True,
        sort_field="line",
        return_metadata=True,
    )
    decoded, decoded_info = PointcloudDecoder().decode(encoded)

    assert transformed_point_count == 3
    assert decoded_info.width == 3
    assert decoded_info.height == 1
    assert did_filter_invalid_xyz is True
    assert not hasattr(encoder, "encode_preprocessed")
    assert _unpack(decoded) == [
        (1.0, 1.0, 1.0, 1, 10),
        (2.0, 2.0, 2.0, 1, 20),
        (3.0, 3.0, 3.0, 2, 30),
    ]


def test_encode_can_filter_without_reordering() -> None:
    points = [
        (3.0, 3.0, 3.0, 2, 30),
        (0.0, 0.0, 0.0, 0, 99),
        (1.0, 1.0, 1.0, 1, 10),
    ]

    encoded, transformed_point_count, did_filter_invalid_xyz = PointcloudEncoder(
        _info(len(points))
    ).encode(
        _pack(points),
        drop_invalid=True,
        sort_field=None,
        return_metadata=True,
    )
    decoded, _ = PointcloudDecoder().decode(encoded)

    assert transformed_point_count == 2
    assert did_filter_invalid_xyz is True
    assert _unpack(decoded) == [points[0], points[2]]


def test_preprocess_returns_raw_points_and_updated_layout() -> None:
    points = [
        (3.0, 3.0, 3.0, 2, 30),
        (0.0, 0.0, 0.0, 0, 99),
        (1.0, 1.0, 1.0, 1, 10),
    ]

    prepared, transformed_point_count, did_filter_invalid_xyz = PointcloudEncoder(
        _info(len(points))
    ).preprocess(
        _pack(points),
        drop_invalid=True,
        sort_field="line",
    )

    assert transformed_point_count == 2
    assert did_filter_invalid_xyz is True
    assert _unpack(prepared) == [points[2], points[0]]


def test_preprocess_filters_float64_xyz() -> None:
    points = [
        (3.0, 3.0, 3.0, 2),
        (0.0, 0.0, 0.0, 0),
        (math.nan, 2.0, 2.0, 0),
        (1.0, 1.0, 1.0, 1),
    ]
    info = EncodingInfo(
        fields=[
            PointField("x", 0, FieldType.FLOAT64),
            PointField("y", 8, FieldType.FLOAT64),
            PointField("z", 16, FieldType.FLOAT64),
            PointField("line", 24, FieldType.UINT8),
        ],
        width=len(points),
        height=1,
        point_step=_FLOAT64_POINT.size,
        encoding_opt=EncodingOptions.LOSSLESS,
        compression_opt=CompressionOption.NONE,
    )

    prepared, transformed_point_count, did_filter_invalid_xyz = PointcloudEncoder(info).preprocess(
        b"".join(_FLOAT64_POINT.pack(*point) for point in points),
        drop_invalid=True,
        sort_field="line",
    )

    assert transformed_point_count == 2
    assert did_filter_invalid_xyz is True
    assert list(struct.iter_unpack(_FLOAT64_POINT.format, prepared)) == [points[3], points[0]]


def test_preprocess_filters_infinite_xyz() -> None:
    points = [
        (math.inf, 1.0, 1.0, 1, 10),
        (1.0, -math.inf, 1.0, 1, 20),
        (1.0, 1.0, 1.0, 1, 30),
    ]

    prepared, transformed_point_count, did_filter_invalid_xyz = PointcloudEncoder(
        _info(len(points))
    ).preprocess(
        _pack(points),
        drop_invalid=True,
        sort_field=None,
    )

    assert transformed_point_count == 1
    assert did_filter_invalid_xyz is True
    assert _unpack(prepared) == [points[2]]


def test_lossy_encode_maps_non_finite_values_to_nan() -> None:
    info = _info(1)
    info.encoding_opt = EncodingOptions.LOSSY

    encoded = PointcloudEncoder(info).encode(_pack([(-math.inf, math.inf, 1.0, 1, 10)]))
    decoded, _ = PointcloudDecoder().decode(encoded)
    point = _unpack(decoded)[0]

    assert math.isnan(point[0])
    assert math.isnan(point[1])
    assert point[2:] == (1.0, 1, 10)


def test_preprocess_noop_returns_original_bytes() -> None:
    source = _pack([(3.0, 3.0, 3.0, 2, 30)])

    prepared, transformed_point_count, did_filter_invalid_xyz = PointcloudEncoder(
        _info(1)
    ).preprocess(
        source,
        drop_invalid=True,
        sort_field=None,
    )

    assert prepared is source
    assert transformed_point_count is None
    assert did_filter_invalid_xyz is True


def test_encode_normalizes_big_endian_input() -> None:
    encoded = PointcloudEncoder(_info(1)).encode(
        _BIG_ENDIAN_POINT.pack(3.0, 3.0, 3.0, 2, 30),
        is_bigendian=True,
    )

    decoded, _ = PointcloudDecoder().decode(encoded)

    assert _unpack(decoded) == [(3.0, 3.0, 3.0, 2, 30)]


def test_preprocess_normalizes_big_endian_input() -> None:
    prepared, transformed_point_count, _ = PointcloudEncoder(_info(1)).preprocess(
        _BIG_ENDIAN_POINT.pack(3.0, 3.0, 3.0, 2, 30),
        drop_invalid=False,
        sort_field=None,
        is_bigendian=True,
    )

    assert _unpack(prepared) == [(3.0, 3.0, 3.0, 2, 30)]
    assert transformed_point_count == 1


def test_encode_without_preprocessing_preserves_organized_dimensions() -> None:
    info = _info(2)
    info.height = 2
    points = [
        (3.0, 3.0, 3.0, 2, 30),
        (0.0, 0.0, 0.0, 0, 99),
        (1.0, 1.0, 1.0, 1, 10),
        (2.0, 2.0, 2.0, 1, 20),
    ]

    encoded = PointcloudEncoder(info).encode(
        _pack(points),
        drop_invalid=False,
        sort_field=None,
    )
    decoded, decoded_info = PointcloudDecoder().decode(encoded)

    assert decoded_info.width == 2
    assert decoded_info.height == 2
    assert _unpack(decoded) == points


def test_encode_rejects_data_count_mismatch() -> None:
    with pytest.raises(RuntimeError, match="point count"):
        PointcloudEncoder(_info(2)).encode(
            _pack(
                [
                    (1.0, 1.0, 1.0, 1, 10),
                    (2.0, 2.0, 2.0, 1, 20),
                    (3.0, 3.0, 3.0, 1, 30),
                ]
            )
        )


def test_direct_lossless_xyz_rejects_data_count_mismatch() -> None:
    info = EncodingInfo(
        fields=[
            PointField("x", 0, FieldType.FLOAT32),
            PointField("y", 4, FieldType.FLOAT32),
            PointField("z", 8, FieldType.FLOAT32),
        ],
        width=1,
        height=1,
        point_step=12,
        encoding_opt=EncodingOptions.LOSSLESS,
        compression_opt=CompressionOption.NONE,
    )

    with pytest.raises(RuntimeError, match="point count"):
        PointcloudEncoder(info).encode(struct.pack("<ffffff", 1, 2, 3, 4, 5, 6))


def test_encode_parallel_preprocessing_is_stable() -> None:
    points = [
        (
            0.0 if index % 5 == 0 else float(index + 1),
            0.0 if index % 5 == 0 else float(index + 2),
            0.0 if index % 5 == 0 else float(index + 3),
            index % 128,
            index % 65536,
        )
        for index in range(70_000)
    ]

    encoded, transformed_point_count, did_filter_invalid_xyz = PointcloudEncoder(
        _info(len(points))
    ).encode(
        _pack(points),
        drop_invalid=True,
        sort_field="line",
        return_metadata=True,
    )
    decoded, _ = PointcloudDecoder().decode(encoded)
    decoded_points = _unpack(decoded)

    assert transformed_point_count == 56_000
    assert did_filter_invalid_xyz is True
    assert [point[3] for point in decoded_points] == sorted(point[3] for point in decoded_points)
    for line in range(128):
        intensities = [point[4] for point in decoded_points if point[3] == line]
        expected = [point[4] for point in points if point[0] != 0.0 and point[3] == line]
        assert intensities == expected
