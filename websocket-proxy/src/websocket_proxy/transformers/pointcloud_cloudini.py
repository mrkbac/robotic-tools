"""Cloudini compression for sensor_msgs/PointCloud2 messages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mcap_codec_support.pointcloud import (
    CLOUDINI_COMPRESSED_POINTCLOUD2,
    CloudiniPointCloudCompressor,
    build_compressed_pointcloud2_message,
)

from . import Transformer, TransformError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import SimpleNamespace

logger = logging.getLogger(__name__)


class PointCloudCloudiniTransformer(Transformer):
    """Compress PointCloud2 messages using Cloudini encoding."""

    def __init__(
        self,
        encoding: str = "lossy",
        compression: str = "zstd",
        resolution: float = 0.01,
    ) -> None:
        self._compressor = CloudiniPointCloudCompressor(
            encoding=encoding, compression=compression, resolution=resolution
        )

    def get_input_schema(self) -> str:
        return "sensor_msgs/msg/PointCloud2"

    def get_output_schema(self) -> str:
        return "point_cloud_interfaces/msg/CompressedPointCloud2"

    def get_output_schema_definition(self) -> str:
        return CLOUDINI_COMPRESSED_POINTCLOUD2

    def transform(self, message: SimpleNamespace) -> Mapping[str, object]:
        try:
            compressed = self._compressor.compress(message)
            return build_compressed_pointcloud2_message(message, compressed, fmt="cloudini")
        except TransformError:
            raise
        except Exception as exc:
            raise TransformError(f"Point cloud Cloudini compression failed: {exc}") from exc


PointCloudPureiniTransformer = PointCloudCloudiniTransformer

__all__ = ["PointCloudCloudiniTransformer", "PointCloudPureiniTransformer"]
