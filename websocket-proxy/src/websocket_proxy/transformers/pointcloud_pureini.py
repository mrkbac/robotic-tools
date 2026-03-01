"""Pureini-based compression for sensor_msgs/PointCloud2 messages."""

from __future__ import annotations

import logging
from typing import Any

from pymcap_cli.image_utils import PointCloudCompressor

from . import Transformer, TransformError

logger = logging.getLogger(__name__)


class PointCloudPureiniTransformer(Transformer):
    """Compress PointCloud2 messages using pureini encoding."""

    def __init__(
        self,
        encoding: str = "lossy",
        compression: str = "zstd",
        resolution: float = 0.01,
    ) -> None:
        self._compressor = PointCloudCompressor(
            encoding=encoding, compression=compression, resolution=resolution
        )

    def get_input_schema(self) -> str:
        return "sensor_msgs/msg/PointCloud2"

    def get_output_schema(self) -> str:
        return "point_cloud_interfaces/msg/CompressedPointCloud2"

    def transform(self, message: Any) -> dict[str, Any]:
        try:
            compressed = self._compressor.compress(message)

            return {
                "header": {
                    "stamp": {
                        "sec": message.header.stamp.sec,
                        "nanosec": message.header.stamp.nanosec,
                    },
                    "frame_id": message.header.frame_id,
                },
                "height": message.height,
                "width": message.width,
                "fields": [
                    {
                        "name": f.name,
                        "offset": f.offset,
                        "datatype": f.datatype,
                        "count": f.count,
                    }
                    for f in message.fields
                ],
                "is_bigendian": message.is_bigendian,
                "point_step": message.point_step,
                "row_step": message.row_step,
                "compressed_data": list(compressed),
                "is_dense": message.is_dense,
                "format": "cloudini",
            }
        except TransformError:
            raise
        except Exception as exc:
            raise TransformError(f"Point cloud pureini compression failed: {exc}") from exc


__all__ = ["PointCloudPureiniTransformer"]
