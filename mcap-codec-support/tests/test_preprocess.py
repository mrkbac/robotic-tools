"""Tests for point cloud preprocessing (drop invalid points + group by line)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from mcap_codec_support.pointcloud import drop_invalid_and_reorder
from pointcloud2 import PointField, create_cloud, read_points


@dataclass
class _Stamp:
    sec: int = 1
    nanosec: int = 2


@dataclass
class _Header:
    frame_id: str = "lidar"
    stamp: _Stamp = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.stamp is None:
            self.stamp = _Stamp()


_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32),
    PointField(name="line", offset=12, datatype=PointField.UINT8),
]


def _make_cloud(xyz: np.ndarray, line: np.ndarray):
    dtype = np.dtype(
        {"names": ["x", "y", "z", "line"], "formats": ["<f4", "<f4", "<f4", "u1"], "itemsize": 16}
    )
    pts = np.zeros(len(xyz), dtype=dtype)
    pts["x"], pts["y"], pts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    pts["line"] = line
    return create_cloud(_Header(), _FIELDS, pts, step=16)


def test_drop_invalid_and_reorder_removes_zeros_and_nans():
    xyz = np.array(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [np.nan, 1.0, 1.0]],
        dtype=np.float32,
    )
    line = np.array([2, 0, 1, 3], dtype=np.uint8)
    out = drop_invalid_and_reorder(_make_cloud(xyz, line))

    pts = read_points(out)
    assert out.width == 2  # the (0,0,0) and NaN points are gone
    assert out.height == 1
    assert out.is_dense is True
    got = {(float(p["x"]), float(p["y"]), float(p["z"])) for p in pts}
    assert got == {(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)}


def test_drop_invalid_and_reorder_groups_by_line():
    # 6 valid points with interleaved line indices; result must be line-sorted.
    xyz = np.arange(18, dtype=np.float32).reshape(6, 3) + 1.0
    line = np.array([1, 0, 1, 0, 2, 0], dtype=np.uint8)
    out = drop_invalid_and_reorder(_make_cloud(xyz, line))

    pts = read_points(out)
    lines = pts["line"]
    assert list(lines) == sorted(lines)  # grouped by ring
    # stable within a ring: points keep their original relative order per line
    line0_x = [float(p["x"]) for p in pts if p["line"] == 0]
    assert line0_x == [4.0, 10.0, 16.0]  # rows 1,3,5 (x = 3*row+1) in original order


def test_drop_invalid_and_reorder_noop_returns_same_object():
    # Already-clean, single-ring cloud with a >1 point count but nothing to drop
    # and a constant sort key still short-circuits when order is a no-op is not
    # guaranteed; instead use a cloud with no xyz fields to hit the early return.
    dtype = np.dtype({"names": ["a"], "formats": ["<f4"], "itemsize": 4})
    pts = np.zeros(3, dtype=dtype)
    cloud = create_cloud(
        _Header(), [PointField(name="a", offset=0, datatype=PointField.FLOAT32)], pts, step=4
    )
    assert drop_invalid_and_reorder(cloud) is cloud


def test_drop_invalid_and_reorder_bigendian_skipped():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    cloud = _make_cloud(xyz, np.array([0, 1], dtype=np.uint8))
    cloud.is_bigendian = True
    assert drop_invalid_and_reorder(cloud) is cloud


@pytest.mark.parametrize("sort_field", [None, "line"])
def test_drop_invalid_preserves_all_field_values(sort_field):
    xyz = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    line = np.array([5, 9, 3], dtype=np.uint8)
    out = drop_invalid_and_reorder(_make_cloud(xyz, line), sort_field=sort_field)
    pts = read_points(out)
    # the line value must travel with its point through filtering/reordering
    mapping = {float(p["x"]): int(p["line"]) for p in pts}
    assert mapping == {1.0: 5, 2.0: 3}
