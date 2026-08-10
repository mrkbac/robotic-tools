"""Shared roscompress configuration and processor construction for bridge commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from pymcap_cli.cmd._pointcloud_cleanup import resolve_pointcloud_cleanup
from pymcap_cli.core.message_filter import ALL_TOPICS, TopicSelection

if TYPE_CHECKING:
    from pymcap_cli.cmd._pointcloud_cleanup import PointcloudCleanupConfig
    from pymcap_cli.core.processors.message_transform import MessageTransformProcessor
    from pymcap_cli.core.processors.video_compress import VideoCompressProcessor

BackendName = Literal["auto", "pyav", "ffmpeg-cli", "gstreamer"]
ImageFormat = Literal["video", "jpeg", "png", "none"]
PointCloudFormat = Literal["cloudini", "draco"]
PointCloudSchema = Literal["auto", "pointcloud2", "foxglove"]
PointCloudEncoding = Literal["lossy", "lossless", "none"]
PointCloudCompression = Literal["zstd", "lz4", "none"]


@dataclass(frozen=True, slots=True)
class RoscompressConfig:
    image_format: ImageFormat = "video"
    codec: Literal["h264", "h265", "vp9", "av1"] = "h264"
    quality: int = 28
    adaptive_quality: bool = False
    encoder: str | None = None
    backend: BackendName = "auto"
    scale: int | None = None
    jpeg_quality: int = 90
    pointcloud: bool = True
    resolution: float = 0.01
    pc_format: PointCloudFormat = "cloudini"
    pc_schema: PointCloudSchema = "auto"
    pc_encoding: PointCloudEncoding = "lossy"
    pc_compression: PointCloudCompression = "zstd"
    draco_compression_level: int = 7
    pointcloud_drop_invalid: bool | None = None
    pointcloud_sort_field: str | None = None
    ffmpeg_args: tuple[str, ...] = ()


def resolve_cleanup(config: RoscompressConfig) -> PointcloudCleanupConfig:
    return resolve_pointcloud_cleanup(
        pointcloud_compression_enabled=config.pointcloud,
        pointcloud_drop_invalid=config.pointcloud_drop_invalid,
        pointcloud_sort_field=config.pointcloud_sort_field,
    )


def pointcloud_output_schema(config: RoscompressConfig) -> tuple[str, str]:
    from mcap_codec_support.pointcloud import (  # noqa: PLC0415
        COMPRESSED_POINTCLOUD2,
        COMPRESSED_POINTCLOUD2_SCHEMA,
        FOXGLOVE_COMPRESSED_POINTCLOUD,
        FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    )

    schema = config.pc_schema
    if schema == "auto":
        schema = "foxglove" if config.pc_format == "draco" else "pointcloud2"
    if schema == "foxglove":
        return FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA, FOXGLOVE_COMPRESSED_POINTCLOUD
    return COMPRESSED_POINTCLOUD2_SCHEMA, COMPRESSED_POINTCLOUD2


def create_image_processor(config: RoscompressConfig) -> MessageTransformProcessor:
    from pymcap_cli.core.processors.image_compress import ImageCompressProcessor  # noqa: PLC0415

    return ImageCompressProcessor(
        image_format=cast("Literal['jpeg', 'png']", config.image_format),
        jpeg_quality=config.jpeg_quality,
        scale=config.scale,
    )


def create_pointcloud_processors(
    config: RoscompressConfig,
    *,
    workers: int = 0,
) -> tuple[MessageTransformProcessor, ...]:
    processors: list[MessageTransformProcessor] = []
    cleanup_processor = (
        None if fuses_cloudini_cleanup(config) else create_pointcloud_cleanup_processor(config)
    )
    if cleanup_processor is not None:
        processors.append(cleanup_processor)
    if config.pointcloud:
        processors.append(create_pointcloud_compress_processor(config, workers=workers))
    return tuple(processors)


def create_pointcloud_cleanup_processor(
    config: RoscompressConfig,
    *,
    topics: TopicSelection = ALL_TOPICS,
) -> MessageTransformProcessor | None:
    cleanup = resolve_cleanup(config)
    if not cleanup.enabled:
        return None
    from pymcap_cli.core.processors.pointcloud_clean import (  # noqa: PLC0415
        PointcloudCleanProcessor,
    )

    return PointcloudCleanProcessor(
        drop_invalid=cleanup.drop_invalid,
        sort_field=cleanup.sort_field,
        topics=topics,
    )


def fuses_cloudini_cleanup(config: RoscompressConfig) -> bool:
    """Whether this config's compressor performs cleanup inside its native encode."""
    return config.pointcloud and config.pc_format == "cloudini"


def create_pointcloud_compress_processor(
    config: RoscompressConfig,
    *,
    workers: int = 0,
    topics: TopicSelection = ALL_TOPICS,
) -> MessageTransformProcessor:
    from pymcap_cli.core.processors.pointcloud_compress import (  # noqa: PLC0415
        PointcloudCompressProcessor,
    )

    cleanup = resolve_cleanup(config)
    use_fused_cleanup = fuses_cloudini_cleanup(config)
    return PointcloudCompressProcessor(
        pc_format=config.pc_format,
        pc_schema=config.pc_schema,
        pc_encoding=config.pc_encoding,
        pc_compression=config.pc_compression,
        resolution=config.resolution,
        draco_compression_level=config.draco_compression_level,
        drop_invalid=cleanup.drop_invalid if use_fused_cleanup else False,
        sort_field=cleanup.sort_field if use_fused_cleanup else None,
        workers=workers,
        topics=topics,
    )


def create_video_compress_processor(
    config: RoscompressConfig,
    *,
    topics: TopicSelection = ALL_TOPICS,
    shared_by: int = 1,
) -> VideoCompressProcessor:
    """Build a video compressor sharing its decode-worker budget with its peers."""
    from mcap_codec_support.video import EncoderMode  # noqa: PLC0415

    from pymcap_cli.core.processors.video_compress import (  # noqa: PLC0415
        VideoCompressProcessor,
        split_decode_workers,
    )

    return VideoCompressProcessor(
        codec=config.codec,
        quality=config.quality,
        encoder=config.encoder,
        scale=config.scale,
        backend=EncoderMode(config.backend),
        ffmpeg_args=config.ffmpeg_args,
        topics=topics,
        decode_workers=split_decode_workers(shared_by),
    )
