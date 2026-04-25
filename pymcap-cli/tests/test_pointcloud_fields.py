"""Smoke test — pymcap-cli's export path relies on pointcloud2.read_points.

This test exists to catch dependency/API drift with the pointcloud2 workspace
package (e.g. a rename or signature change would break export-duckdb).
"""

from __future__ import annotations

import numpy as np
from pointcloud2 import PointField, create_cloud, read_points


def test_pointcloud2_roundtrip_xyz_intensity() -> None:
    fields = [
        PointField("x", 0, PointField.FLOAT32),
        PointField("y", 4, PointField.FLOAT32),
        PointField("z", 8, PointField.FLOAT32),
        PointField("intensity", 12, PointField.FLOAT32),
    ]
    pts = np.array(
        [(1.0, 2.0, 3.0, 10.0), (4.0, 5.0, 6.0, 20.0), (7.0, 8.0, 9.0, 30.0)],
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")]),
    )
    cloud = create_cloud(header=None, fields=fields, points=pts)
    decoded = read_points(cloud)
    np.testing.assert_array_equal(decoded["x"], pts["x"])
    np.testing.assert_array_equal(decoded["intensity"], pts["intensity"])
