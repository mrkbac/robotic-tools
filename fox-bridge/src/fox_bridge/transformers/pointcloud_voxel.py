"""Voxel-based downsampling for sensor_msgs/PointCloud2 messages."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pointcloud2 import PointCloud2, PointField, dtype_from_fields, read_points

from . import Transformer, TransformError

logger = logging.getLogger(__name__)

XYZ_FIELDS: tuple[str, str, str] = ("x", "y", "z")


class PointCloudVoxelTransformer(Transformer):
    """Compress PointCloud2 messages by keeping a single point per voxel."""

    def __init__(self, voxel_size: float = 0.5, skip_nans: bool = True) -> None:
        """Initialize the transformer.

        Args:
            voxel_size: Size of the cubic voxel in meters. Must be positive.
            skip_nans: Drop points with NaN coordinates before voxelization.
        """
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")

        self.voxel_size = float(voxel_size)
        self.skip_nans = skip_nans

    def get_input_schema(self) -> str:
        """Input schema is sensor_msgs/PointCloud2."""
        return "sensor_msgs/msg/PointCloud2"

    def get_output_schema(self) -> str:
        """Output schema is also sensor_msgs/PointCloud2."""
        return "sensor_msgs/msg/PointCloud2"

    def transform(self, message: Any) -> dict[str, Any]:
        """Downsample a point cloud using a voxel grid and strip non XYZ fields."""
        try:
            cloud = message
            xyz = self._extract_xyz(cloud)

            if xyz.shape[0] == 0:
                logger.debug("Point cloud contained no XYZ data after filtering")
                return self._build_output(message, np.empty((0, 3), dtype=np.float32))

            downsampled = self._voxel_downsample(xyz)
            return self._build_output(message, downsampled)
        except TransformError:
            raise
        except Exception as exc:
            raise TransformError(f"Point cloud voxel compression failed: {exc}") from exc

    def _extract_xyz(self, cloud: PointCloud2) -> np.ndarray:
        """Read XYZ coordinates from the point cloud."""
        try:
            point_array = read_points(
                cloud,
                field_names=list(XYZ_FIELDS),
                skip_nans=self.skip_nans,
            )
        except AssertionError as exc:
            raise TransformError(f"Point cloud missing XYZ fields: {exc}") from exc

        if point_array.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        stacked = np.column_stack([point_array[name] for name in XYZ_FIELDS])
        return np.asarray(stacked, dtype=np.float32)

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """Keep one point per voxel cell."""
        if points.shape[0] == 0:
            return points

        scaled = np.floor(points / self.voxel_size)
        voxel_indices = scaled.astype(np.int64, copy=False)

        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        unique_indices.sort()
        return points[unique_indices]

    def _build_output(self, message: Any, points: np.ndarray) -> dict[str, Any]:
        """Package downsampled XYZ points into a PointCloud2 dictionary."""
        points = np.asarray(points, dtype=np.float32, order="C")

        xyz_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        dtype = dtype_from_fields(xyz_fields)
        structured = np.zeros(points.shape[0], dtype=dtype)
        if points.shape[0] > 0:
            structured["x"] = points[:, 0]
            structured["y"] = points[:, 1]
            structured["z"] = points[:, 2]

        data_bytes = structured.tobytes()

        return {
            "header": self._header_to_dict(getattr(message, "header", None)),
            "height": 1,
            "width": int(points.shape[0]),
            "fields": [
                {
                    "name": field.name,
                    "offset": int(field.offset),
                    "datatype": int(field.datatype),
                    "count": int(field.count),
                }
                for field in xyz_fields
            ],
            "is_bigendian": False,
            "point_step": int(dtype.itemsize),
            "row_step": int(dtype.itemsize * points.shape[0]),
            "data": list(data_bytes),
            "is_dense": True,
        }

    def _header_to_dict(self, header: Any) -> dict[str, Any]:
        """Convert std_msgs/Header-like object into a dictionary."""
        if header is None:
            return {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}

        stamp = getattr(header, "stamp", None)
        sec = int(getattr(stamp, "sec", 0)) if stamp is not None else 0
        nanosec = int(getattr(stamp, "nanosec", 0)) if stamp is not None else 0
        frame_id = str(getattr(header, "frame_id", ""))

        return {
            "stamp": {"sec": sec, "nanosec": nanosec},
            "frame_id": frame_id,
        }


__all__ = ["PointCloudVoxelTransformer"]
