"""Point-cloud compression metadata and preprocessing contracts."""

from __future__ import annotations

import math
import struct
from importlib import metadata
from types import SimpleNamespace

import numpy as np
from mcap_codec_support.pointcloud.compression import (
    CloudiniPointCloudCompressor,
    PointCloudCompressionResult,
    build_compressed_pointcloud2_message,
)
from pointcloud2 import PointField, create_cloud
from pureini import PointcloudDecoder


def _header() -> SimpleNamespace:
    return SimpleNamespace(
        frame_id="lidar",
        stamp=SimpleNamespace(sec=1, nanosec=2),
    )


def _float64_cloud():
    fields = [
        PointField("x", 0, PointField.FLOAT64),
        PointField("y", 8, PointField.FLOAT64),
        PointField("z", 16, PointField.FLOAT64),
        PointField("line", 24, PointField.UINT8),
    ]
    dtype = np.dtype(
        {
            "names": ["x", "y", "z", "line"],
            "formats": ["<f8", "<f8", "<f8", "u1"],
            "itemsize": 32,
        }
    )
    points = np.zeros(4, dtype=dtype)
    points["x"] = [3.0, 0.0, math.nan, 1.0]
    points["y"] = [3.0, 0.0, 2.0, 1.0]
    points["z"] = [3.0, 0.0, 2.0, 1.0]
    points["line"] = [2, 0, 0, 1]
    cloud = create_cloud(_header(), fields, points, step=32)
    cloud.is_dense = False
    return cloud


def test_pointcloud_extra_requires_native_pureini_release():
    requirements = metadata.requires("mcap-codec-support") or []
    pureini_version = metadata.version("pureini")

    assert f"pureini>={pureini_version} ; extra == 'pointcloud'" in requirements


def test_cloudini_compressor_filters_float64_xyz_and_reports_dense():
    compressor = CloudiniPointCloudCompressor(
        encoding="lossless",
        compression="none",
        drop_invalid=True,
        sort_field="line",
    )

    result = compressor.compress_result(_float64_cloud())
    decoded, info = PointcloudDecoder().decode(result.data)

    assert result.width == 2
    assert result.is_dense is True
    assert info.width == 2
    assert not np.isnan(np.ndarray((info.width,), "<f8", decoded, 0, (info.point_step,))).any()


def test_cloudini_compressor_does_not_claim_dense_without_xyz():
    fields = [PointField("intensity", 0, PointField.FLOAT32)]
    points = np.array([(1.0,), (2.0,)], dtype=[("intensity", "<f4")])
    cloud = create_cloud(_header(), fields, points)
    cloud.is_dense = False

    result = CloudiniPointCloudCompressor(
        encoding="lossless",
        compression="none",
        drop_invalid=True,
    ).compress_result(cloud)

    assert result.is_dense is False


def test_cloudini_compressor_uses_native_xyz_filter_applicability():
    fields = [
        PointField("x", 0, PointField.FLOAT32, count=2),
        PointField("y", 8, PointField.FLOAT32, count=2),
        PointField("z", 16, PointField.FLOAT32, count=2),
    ]
    cloud = SimpleNamespace(
        fields=fields,
        width=1,
        height=1,
        point_step=24,
        row_step=24,
        is_bigendian=False,
        is_dense=False,
        data=struct.pack("<ffffff", 1.0, 9.0, 2.0, 9.0, 3.0, 9.0),
    )

    result = CloudiniPointCloudCompressor(
        encoding="lossless",
        compression="none",
        drop_invalid=True,
    ).compress_result(cloud)

    assert result.is_dense is True


def test_cloudini_compressor_preserves_array_fields():
    fields = [
        PointField("x", 0, PointField.FLOAT32),
        PointField("normal", 4, PointField.FLOAT32, count=3),
    ]
    data = struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0) + struct.pack("<ffff", 5.0, 6.0, 7.0, 8.0)
    cloud = SimpleNamespace(
        fields=fields,
        width=2,
        height=1,
        point_step=16,
        row_step=32,
        is_bigendian=False,
        is_dense=True,
        data=data,
    )

    result = CloudiniPointCloudCompressor(
        encoding="lossless",
        compression="none",
    ).compress_result(cloud)
    decoded, _ = PointcloudDecoder().decode(result.data)

    assert decoded == data


def test_cloudini_compressor_removes_organized_row_padding():
    fields = [PointField("x", 0, PointField.FLOAT32)]
    data = struct.pack("<ff", 1.0, 2.0) + b"pad!" + struct.pack("<ff", 3.0, 4.0) + b"pad!"
    cloud = SimpleNamespace(
        fields=fields,
        width=2,
        height=2,
        point_step=4,
        row_step=12,
        is_bigendian=False,
        is_dense=True,
        data=data,
    )

    result = CloudiniPointCloudCompressor(
        encoding="lossless",
        compression="none",
    ).compress_result(cloud)
    decoded, _ = PointcloudDecoder().decode(result.data)

    assert decoded == struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
    assert result.width == 2
    assert result.height == 2
    assert result.row_step == 8


def test_cloudini_compressor_normalizes_big_endian_input():
    fields = [
        PointField("x", 0, PointField.FLOAT32),
        PointField("ring", 4, PointField.UINT16),
    ]
    cloud = SimpleNamespace(
        fields=fields,
        width=2,
        height=1,
        point_step=8,
        row_step=16,
        is_bigendian=True,
        is_dense=True,
        data=struct.pack(">fHxxfHxx", 1.5, 7, -2.25, 12),
    )

    result = CloudiniPointCloudCompressor(
        encoding="lossless",
        compression="none",
    ).compress_result(cloud)
    decoded, _ = PointcloudDecoder().decode(result.data)

    assert struct.unpack("<fHxxfHxx", decoded) == (1.5, 7, -2.25, 12)
    assert result.is_bigendian is False


def test_build_compressed_pointcloud2_message_preserves_padded_row_step():
    cloud = _float64_cloud()
    cloud.height = 2
    cloud.width = 2
    cloud.row_step = 80

    without_result = build_compressed_pointcloud2_message(
        cloud,
        b"compressed",
        fmt="draco",
    )
    with_result = build_compressed_pointcloud2_message(
        cloud,
        b"compressed",
        fmt="draco",
        result=PointCloudCompressionResult(
            data=b"compressed",
            width=2,
            height=2,
            point_step=32,
            row_step=80,
            is_dense=False,
        ),
    )

    assert without_result["row_step"] == 80
    assert with_result["row_step"] == 80
