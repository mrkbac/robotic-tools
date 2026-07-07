"""Point-cloud compression as a pipeline processor.

Transcodes ``sensor_msgs/PointCloud2`` messages to Cloudini
``CompressedPointCloud2`` or Draco ``CompressedPointCloud`` in-place on the
same topic, via :class:`MessageTransformProcessor`. Synchronous — the codec
runs inline in ``transform`` — so there is no ordering hazard; per-channel and
per-message order are preserved by the base. Composes with the rest of the
pipeline (topic drop, rechunk, per-schema compression split, splitting, …).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Literal

from mcap_codec_support._schemas import normalize_schema_name
from mcap_codec_support.pointcloud import (
    COMPRESSED_POINTCLOUD2,
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    POINTCLOUD2_SCHEMAS,
    PointCloudCompressionError,
    build_compressed_pointcloud2_message,
    build_foxglove_compressed_pointcloud_message,
    drop_invalid_and_reorder,
)
from typing_extensions import override

from pymcap_cli.core.processors.message_transform import (
    MessageTransformProcessor,
    TransformOutput,
)

if TYPE_CHECKING:
    from mcap_codec_support.pointcloud import PointCloudCompressorProtocol
    from small_mcap import Channel, Schema

logger = logging.getLogger(__name__)


def _make_compressor(
    pc_format: str,
    pc_encoding: str,
    pc_compression: str,
    resolution: float,
    draco_compression_level: int,
) -> PointCloudCompressorProtocol:
    """Build the point-cloud compressor. Mirrors roscompress's factory.

    Raises ``ImportError`` if the optional codec dependency is missing; the CLI
    layer catches this to print an install hint.
    """
    if pc_format == "draco":
        from mcap_codec_support.pointcloud import DracoPointCloudCompressor  # noqa: PLC0415

        return DracoPointCloudCompressor(
            resolution=resolution, compression_level=draco_compression_level
        )
    from mcap_codec_support.pointcloud import CloudiniPointCloudCompressor  # noqa: PLC0415

    return CloudiniPointCloudCompressor(
        encoding=pc_encoding, compression=pc_compression, resolution=resolution
    )


class PointcloudCompressProcessor(MessageTransformProcessor):
    """Compress PointCloud2 messages to CompressedPointCloud2 / CompressedPointCloud."""

    def __init__(
        self,
        *,
        pc_format: Literal["cloudini", "draco"] = "cloudini",
        pc_schema: Literal["auto", "pointcloud2", "foxglove"] = "auto",
        pc_encoding: Literal["lossy", "lossless", "none"] = "lossy",
        pc_compression: Literal["zstd", "lz4", "none"] = "zstd",
        resolution: float = 0.01,
        draco_compression_level: int = 7,
        clean: bool = True,
        workers: int = 0,
    ) -> None:
        super().__init__(workers=workers)
        self._clean = clean
        self._compressor_args = (
            pc_format,
            pc_encoding,
            pc_compression,
            resolution,
            draco_compression_level,
        )
        # The native compressor is not thread-safe, so with workers > 0 each
        # worker thread keeps its own (via thread-local). Build one eagerly on
        # this thread too, which validates the optional dependency up front
        # (ImportError surfaces at construction, not mid-stream on a worker).
        self._tls = threading.local()
        self._tls.compressor = _make_compressor(*self._compressor_args)

        resolved = pc_schema
        if resolved == "auto":
            resolved = "foxglove" if pc_format == "draco" else "pointcloud2"
        self._pc_format = pc_format
        if resolved == "foxglove":
            self._out_schema_name = FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA
            self._out_schema_data = FOXGLOVE_COMPRESSED_POINTCLOUD.encode()
            self._foxglove = True
        else:
            self._out_schema_name = COMPRESSED_POINTCLOUD2_SCHEMA
            self._out_schema_data = COMPRESSED_POINTCLOUD2.encode()
            self._foxglove = False

    @override
    def matches(self, channel: Channel, schema: Schema | None) -> bool:
        return schema is not None and normalize_schema_name(schema.name) in POINTCLOUD2_SCHEMAS

    def _compressor(self) -> PointCloudCompressorProtocol:
        """The calling thread's compressor (built lazily; not shared across threads)."""
        compressor = getattr(self._tls, "compressor", None)
        if compressor is None:
            compressor = _make_compressor(*self._compressor_args)
            self._tls.compressor = compressor
        return compressor

    @override
    def transform(
        self, channel: Channel, schema: Schema, decoded: Any
    ) -> list[TransformOutput] | None:
        if self._clean:
            decoded = drop_invalid_and_reorder(decoded)
        try:
            compressed = self._compressor().compress(decoded)
        except PointCloudCompressionError as exc:
            logger.warning("Skipping point cloud compression for %s: %s", channel.topic, exc)
            return None  # keep the raw message rather than drop it
        if self._foxglove:
            data = build_foxglove_compressed_pointcloud_message(
                decoded, compressed, fmt=self._pc_format
            )
        else:
            data = build_compressed_pointcloud2_message(decoded, compressed, fmt=self._pc_format)
        return [
            TransformOutput(
                topic=channel.topic,
                schema_name=self._out_schema_name,
                schema_encoding="ros2msg",
                schema_data=self._out_schema_data,
                data=data,
            )
        ]
