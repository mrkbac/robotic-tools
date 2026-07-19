"""Lossless JIT ROS transforms for MCAP bridge playback."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, cast

from cyclopts import Group, Parameter
from small_mcap import Channel, Schema

from pymcap_cli.cmd._cli_options import (
    ENCODING_GROUP,
    POINTCLOUD_GROUP,
)
from pymcap_cli.cmd._pointcloud_cleanup import resolve_pointcloud_cleanup
from pymcap_cli.cmd.bridge._playback import (
    PlaybackChannel,
    PlaybackError,
    PlaybackOutput,
    PlaybackTransformPlan,
    PlaybackTransformSession,
)

if TYPE_CHECKING:
    from pymcap_cli.cmd._pointcloud_cleanup import PointcloudCleanupConfig
    from pymcap_cli.core.processors.message_transform import (
        MessageTransformProcessor,
        TransformOutput,
    )
    from pymcap_cli.core.processors.video_compress import VideoEncoderSession

logger = logging.getLogger(__name__)

_ADAPTIVE_WINDOW_SECONDS = 2.0
_ADAPTIVE_CONGESTION_RATIO = 0.25
_ADAPTIVE_HOLD_SECONDS = 3.0
_ADAPTIVE_RECOVERY_SECONDS = 30.0
_MAX_VIDEO_FPS = 30.0
_ADAPTIVE_SCALE_FACTORS = (0.75, 0.5, 0.375)
_ADAPTIVE_FRAME_RATES = (20.0, 10.0, 5.0, 2.0)

TRANSFORM_GROUP = Group("ROS Transform")
Preset = Literal["compress", "decompress", "fast", "low"]
BackendName = Literal["auto", "pyav", "ffmpeg-cli", "gstreamer"]
ImageFormat = Literal["video", "jpeg", "png", "none"]
VideoFormat = Literal["compressed", "raw"]
PointCloudFormat = Literal["cloudini", "draco"]
PointCloudSchema = Literal["auto", "pointcloud2", "foxglove"]
PointCloudEncoding = Literal["lossy", "lossless", "none"]
PointCloudCompression = Literal["zstd", "lz4", "none"]

OptionalPresetOption = Annotated[
    Preset | None,
    Parameter(
        name=["--preset"],
        group=TRANSFORM_GROUP,
        help=(
            "JIT ROS payload preset: compress (full-resolution video), "
            "decompress, fast (960px video), low (480px video). "
            "Explicit encoding flags override the preset's defaults."
        ),
    ),
]
OptionalImageFormatOption = Annotated[
    ImageFormat | None,
    Parameter(
        name=["--image-format"],
        group=ENCODING_GROUP,
        help="roscompress image mode. Preset default: video.",
    ),
]
OptionalQualityOption = Annotated[
    int | None,
    Parameter(name=["-q", "--quality"], group=ENCODING_GROUP, help="Preset default: 28."),
]
OptionalAdaptiveQualityOption = Annotated[
    bool | None,
    Parameter(
        name=["--adaptive-quality"],
        negative="--no-adaptive-quality",
        group=ENCODING_GROUP,
        help=(
            "Cap compressed video at 30 FPS, then reduce frame rate before "
            "resolution when subscribers fall behind. Preserve the requested "
            "quality. Serve default: enabled."
        ),
    ),
]
OptionalEncoderOption = Annotated[
    str | None,
    Parameter(name=["--encoder"], group=ENCODING_GROUP),
]
OptionalScaleOption = Annotated[
    int | None,
    Parameter(name=["-s", "--scale"], group=ENCODING_GROUP),
]
OptionalJpegQualityOption = Annotated[
    int | None,
    Parameter(name=["--jpeg-quality"], group=ENCODING_GROUP, help="Preset default: 90."),
]
OptionalVideoOption = Annotated[
    bool | None,
    Parameter(
        name=["--video"],
        negative="--no-video",
        group=ENCODING_GROUP,
        help="Enable rosdecompress video decoding. Preset default: enabled.",
    ),
]
OptionalVideoFormatOption = Annotated[
    VideoFormat | None,
    Parameter(
        name=["--video-format"],
        group=ENCODING_GROUP,
        help="rosdecompress video output. Preset default: compressed.",
    ),
]
OptionalPointCloudOption = Annotated[
    bool | None,
    Parameter(
        name=["--pointcloud"],
        negative="--no-pointcloud",
        group=POINTCLOUD_GROUP,
        help="Enable point-cloud transform. Preset default: enabled.",
    ),
]
OptionalResolutionOption = Annotated[
    float | None,
    Parameter(name=["--resolution"], group=POINTCLOUD_GROUP, help="Preset default: 0.01."),
]
OptionalPointCloudFormatOption = Annotated[
    PointCloudFormat | None,
    Parameter(name=["--pc-format"], group=POINTCLOUD_GROUP, help="Preset default: cloudini."),
]
OptionalPointCloudSchemaOption = Annotated[
    PointCloudSchema | None,
    Parameter(name=["--pc-schema"], group=POINTCLOUD_GROUP, help="Preset default: auto."),
]
OptionalPointCloudEncodingOption = Annotated[
    PointCloudEncoding | None,
    Parameter(name=["--pc-encoding"], group=POINTCLOUD_GROUP, help="Preset default: lossy."),
]
OptionalPointCloudCompressionOption = Annotated[
    PointCloudCompression | None,
    Parameter(name=["--pc-compression"], group=POINTCLOUD_GROUP, help="Preset default: zstd."),
]
OptionalDracoCompressionLevelOption = Annotated[
    int | None,
    Parameter(
        name=["--draco-compression-level"],
        group=POINTCLOUD_GROUP,
        help="Preset default: 7.",
    ),
]


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


@dataclass(frozen=True, slots=True)
class RosdecompressConfig:
    video: bool = True
    video_format: VideoFormat = "compressed"
    jpeg_quality: int = 90
    backend: BackendName = "auto"
    pointcloud: bool = True


PlaybackTransformConfig = RoscompressConfig | RosdecompressConfig | None


@dataclass(frozen=True, slots=True)
class _PresetSpec:
    image_format: ImageFormat | None
    scale: int | None


_PRESETS: dict[Preset, _PresetSpec] = {
    "compress": _PresetSpec("video", None),
    "decompress": _PresetSpec(None, None),
    "fast": _PresetSpec("video", 960),
    "low": _PresetSpec("video", 480),
}


def apply_preset(
    preset: Preset | None,
    *,
    image_format: ImageFormat | None,
    scale: int | None,
) -> tuple[ImageFormat | None, int | None]:
    """Apply a preset's image format and scale defaults.

    Explicitly supplied encoding flags win over the preset's defaults. Without a
    preset no transform is applied and playback streams the raw payloads.
    """
    if preset is None:
        return image_format, scale
    spec = _PRESETS[preset]
    return (
        spec.image_format if image_format is None else image_format,
        spec.scale if scale is None else scale,
    )


def resolve_playback_transform_config(
    *,
    preset: Preset | None,
    adaptive_quality: bool | None,
    image_format: ImageFormat | None,
    codec: Literal["h264", "h265", "vp9", "av1"] | None,
    quality: int | None,
    encoder: str | None,
    backend: BackendName | None,
    scale: int | None,
    jpeg_quality: int | None,
    video: bool | None,
    video_format: VideoFormat | None,
    pointcloud: bool | None,
    resolution: float | None,
    pc_format: PointCloudFormat | None,
    pc_schema: PointCloudSchema | None,
    pc_encoding: PointCloudEncoding | None,
    pc_compression: PointCloudCompression | None,
    draco_compression_level: int | None,
    pointcloud_drop_invalid: bool | None,
    pointcloud_sort_field: str | None,
) -> PlaybackTransformConfig:
    compression_options = {
        "--adaptive-quality": adaptive_quality,
        "--image-format": image_format,
        "--codec": codec,
        "--quality": quality,
        "--encoder": encoder,
        "--scale": scale,
        "--resolution": resolution,
        "--pc-format": pc_format,
        "--pc-schema": pc_schema,
        "--pc-encoding": pc_encoding,
        "--pc-compression": pc_compression,
        "--draco-compression-level": draco_compression_level,
        "--pointcloud-drop-invalid": pointcloud_drop_invalid,
        "--pointcloud-sort-field": pointcloud_sort_field,
    }
    decompression_options = {"--video": video, "--video-format": video_format}
    shared_options = {
        "--backend": backend,
        "--jpeg-quality": jpeg_quality,
        "--pointcloud": pointcloud,
    }

    if preset is None:
        supplied = _supplied_options(compression_options)
        if supplied:
            raise ValueError(f"{supplied[0]} requires --preset compress, fast, or low")
        supplied = _supplied_options(decompression_options)
        if supplied:
            raise ValueError(f"{supplied[0]} requires --preset decompress")
        supplied = _supplied_options(shared_options)
        if supplied:
            raise ValueError(f"{supplied[0]} requires a --preset")
        return None
    if preset != "decompress":
        supplied = _supplied_options(decompression_options)
        if supplied:
            raise ValueError(f"{supplied[0]} requires --preset decompress")
        config = RoscompressConfig(
            image_format=image_format or "video",
            codec=codec or "h264",
            quality=28 if quality is None else quality,
            adaptive_quality=False if adaptive_quality is None else adaptive_quality,
            encoder=encoder,
            backend=backend or "auto",
            scale=scale,
            jpeg_quality=90 if jpeg_quality is None else jpeg_quality,
            pointcloud=True if pointcloud is None else pointcloud,
            resolution=0.01 if resolution is None else resolution,
            pc_format=pc_format or "cloudini",
            pc_schema=pc_schema or "auto",
            pc_encoding=pc_encoding or "lossy",
            pc_compression=pc_compression or "zstd",
            draco_compression_level=(
                7 if draco_compression_level is None else draco_compression_level
            ),
            pointcloud_drop_invalid=pointcloud_drop_invalid,
            pointcloud_sort_field=pointcloud_sort_field,
        )
        _validate_common(config.jpeg_quality)
        if not 0 <= config.draco_compression_level <= 10:
            raise ValueError("--draco-compression-level must be in [0, 10]")
        if config.scale is not None and config.scale <= 0:
            raise ValueError("--scale must be positive")
        if config.adaptive_quality and config.image_format != "video":
            raise ValueError("--adaptive-quality requires --image-format video")
        return config

    supplied = _supplied_options(compression_options)
    if supplied:
        raise ValueError(f"{supplied[0]} requires --preset compress, fast, or low")
    config = RosdecompressConfig(
        video=True if video is None else video,
        video_format=video_format or "compressed",
        jpeg_quality=90 if jpeg_quality is None else jpeg_quality,
        backend=backend or "auto",
        pointcloud=True if pointcloud is None else pointcloud,
    )
    _validate_common(config.jpeg_quality)
    return config


def _supplied_options(*groups: Mapping[str, object | None]) -> list[str]:
    return [name for group in groups for name, value in group.items() if value is not None]


def _validate_common(jpeg_quality: int) -> None:
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")


class _SyncChannelTransform(Protocol):
    def process(
        self, payload: bytes | memoryview, timestamp_ns: int
    ) -> tuple[PlaybackOutput, ...]: ...

    def finish(self) -> tuple[PlaybackOutput, ...]: ...

    def close(self) -> None: ...


TransformFactory = Callable[[], _SyncChannelTransform]


@dataclass(frozen=True, slots=True)
class _ChannelTransformSpec:
    source: PlaybackChannel
    output: PlaybackChannel
    factories: tuple[TransformFactory, ...]
    video_rungs: tuple[_AdaptiveVideoRung, ...] = ()


@dataclass(frozen=True, slots=True)
class _AdaptiveVideoRung:
    quality: int
    scale_factor: float
    max_fps: float | None = None


@dataclass(slots=True)
class _AdaptiveVideoController:
    max_rung: int = 3
    rung: int = 0
    _window_started: float | None = None
    _observations: int = 0
    _congested_observations: int = 0
    _last_change: float = -math.inf
    _clean_since: float | None = None

    def observe(self, *, is_congested: bool, now: float) -> int:
        if is_congested:
            self._clean_since = None
        elif self._clean_since is None:
            self._clean_since = now

        if (
            self.rung > 0
            and self._clean_since is not None
            and now - self._clean_since >= _ADAPTIVE_RECOVERY_SECONDS
            and now - self._last_change >= _ADAPTIVE_HOLD_SECONDS
        ):
            self.rung -= 1
            self._last_change = now
            self._clean_since = now
            self._reset_window(now, is_congested)
            return self.rung

        if self._window_started is None:
            self._reset_window(now, is_congested)
            return self.rung

        self._observations += 1
        self._congested_observations += int(is_congested)
        if now - self._window_started < _ADAPTIVE_WINDOW_SECONDS:
            return self.rung

        congestion_ratio = self._congested_observations / self._observations
        # Only degrade while congestion is still present: a stale window whose
        # single congested sample is seconds old must not keep stepping quality
        # down during an otherwise clean period.
        if (
            is_congested
            and congestion_ratio >= _ADAPTIVE_CONGESTION_RATIO
            and self.rung < self.max_rung
        ):
            if now - self._last_change >= _ADAPTIVE_HOLD_SECONDS:
                self.rung += 1
                self._last_change = now
                self._reset_window(now, is_congested)
            return self.rung

        self._reset_window(now, is_congested)
        return self.rung

    def _reset_window(self, now: float, is_congested: bool) -> None:
        self._window_started = now
        self._observations = 1
        self._congested_observations = int(is_congested)


class JitPlaybackTransformPlan(PlaybackTransformPlan):
    def __init__(self, mode: str, specs: tuple[_ChannelTransformSpec, ...]) -> None:
        self.mode = mode
        self._specs = specs
        self._outputs = {spec.source: spec.output for spec in specs}
        by_topic: dict[str, PlaybackChannel] = {}
        for spec in specs:
            previous = by_topic.get(spec.output.topic)
            if previous is not None and previous != spec.output:
                raise PlaybackError(
                    f"Transform produces incompatible channel definitions for {spec.output.topic!r}"
                )
            by_topic[spec.output.topic] = spec.output
        self.channels = tuple(sorted(by_topic.values(), key=lambda channel: channel.topic))

    def create_session(self) -> PlaybackTransformSession:
        return _JitPlaybackTransformSession(self._specs)

    def output_channel(self, source: PlaybackChannel) -> PlaybackChannel:
        return self._outputs[source]


class _JitPlaybackTransformSession(PlaybackTransformSession):
    def __init__(self, specs: tuple[_ChannelTransformSpec, ...]) -> None:
        self._factories = {spec.source: spec.factories for spec in specs}
        self._video_rungs = {spec.source: spec.video_rungs for spec in specs}
        self._active_rungs = {spec.source: 0 for spec in specs}
        self._last_video_frame: dict[PlaybackChannel, float] = {}
        self._controllers = {
            spec.source: _AdaptiveVideoController(max_rung=len(spec.factories) - 1)
            for spec in specs
            if len(spec.factories) > 1
        }
        self._transforms: dict[PlaybackChannel, _SyncChannelTransform] = {}

    async def observe_congestion(
        self,
        channel: PlaybackChannel,
        *,
        is_congested: bool,
        now: float,
    ) -> None:
        controller = self._controllers.get(channel)
        if controller is None:
            return
        previous_rung = self._active_rungs[channel]
        new_rung = controller.observe(is_congested=is_congested, now=now)
        if new_rung == previous_rung:
            return
        rungs = self._video_rungs[channel]
        previous = rungs[previous_rung]
        current = rungs[new_rung]
        if previous.quality != current.quality or previous.scale_factor != current.scale_factor:
            transform = self._transforms.pop(channel, None)
            if transform is not None:
                await asyncio.to_thread(transform.close)
        self._active_rungs[channel] = new_rung
        logger.info(
            "Adaptive video for %s: q%d @ %.1f%% / %s fps -> q%d @ %.1f%% / %s fps",
            channel.topic,
            previous.quality,
            previous.scale_factor * 100,
            "source" if previous.max_fps is None else f"{previous.max_fps:g}",
            current.quality,
            current.scale_factor * 100,
            "source" if current.max_fps is None else f"{current.max_fps:g}",
        )

    def should_drop_frame(self, channel: PlaybackChannel, *, now: float) -> bool:
        rungs = self._video_rungs.get(channel)
        if not rungs:
            return False
        max_fps = rungs[self._active_rungs[channel]].max_fps
        if max_fps is None:
            return False
        last_frame = self._last_video_frame.get(channel)
        if last_frame is not None and now - last_frame < 1 / max_fps:
            return True
        self._last_video_frame[channel] = now
        return False

    async def transform(
        self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
    ) -> tuple[PlaybackOutput, ...]:
        transform = self._transforms.get(channel)
        if transform is None:
            transform = self._factories[channel][self._active_rungs[channel]]()
            self._transforms[channel] = transform
        try:
            return await asyncio.to_thread(transform.process, payload, timestamp_ns)
        except PlaybackError:
            raise
        except Exception as exc:
            raise PlaybackError(f"Transform failed for {channel.topic!r}: {exc}") from exc

    async def finish(self) -> tuple[PlaybackOutput, ...]:
        outputs: list[PlaybackOutput] = []
        for transform in self._transforms.values():
            try:
                outputs.extend(await asyncio.to_thread(transform.finish))
            except PlaybackError:
                raise
            except Exception as exc:
                raise PlaybackError(f"Transform flush failed: {exc}") from exc
        outputs.sort(key=lambda output: output.timestamp_ns)
        return tuple(outputs)

    async def deactivate(self, channel: PlaybackChannel) -> None:
        self._last_video_frame.pop(channel, None)
        transform = self._transforms.pop(channel, None)
        if transform is not None:
            await asyncio.to_thread(transform.close)

    async def restart(self) -> None:
        # Close every live transform so the next message rebuilds it. Video
        # encoders then open a fresh GOP whose first packet is a keyframe,
        # which the client needs to decode after a timeline restart (loop or
        # seek). Keep the learned adaptive rung across timeline restarts.
        transforms = tuple(self._transforms.values())
        self._transforms.clear()
        self._last_video_frame.clear()
        for transform in transforms:
            await asyncio.to_thread(transform.close)

    async def close(self) -> None:
        for transform in self._transforms.values():
            await asyncio.to_thread(transform.close)


def create_playback_transform_plan(
    config: PlaybackTransformConfig,
    channels: tuple[PlaybackChannel, ...],
) -> JitPlaybackTransformPlan | None:
    if config is None:
        return None
    try:
        if isinstance(config, RoscompressConfig):
            return _create_roscompress_plan(config, channels)
        return _create_rosdecompress_plan(config, channels)
    except (ImportError, PlaybackError, ValueError):
        raise
    except Exception as exc:
        raise PlaybackError(str(exc)) from exc


def _create_roscompress_plan(
    config: RoscompressConfig,
    channels: tuple[PlaybackChannel, ...],
) -> JitPlaybackTransformPlan:
    from mcap_codec_support._schemas import normalize_schema_name  # noqa: PLC0415
    from mcap_codec_support.pointcloud import (  # noqa: PLC0415
        COMPRESSED_POINTCLOUD2,
        COMPRESSED_POINTCLOUD2_SCHEMA,
        FOXGLOVE_COMPRESSED_POINTCLOUD,
        FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
        POINTCLOUD2_SCHEMAS,
    )
    from mcap_codec_support.video import (  # noqa: PLC0415
        COMPRESSED_IMAGE,
        COMPRESSED_VIDEO_SCHEMA,
        FOXGLOVE_COMPRESSED_VIDEO,
        IMAGE_SCHEMAS,
        RAW_SCHEMAS,
    )

    cleanup = resolve_pointcloud_cleanup(
        pointcloud_compression_enabled=config.pointcloud,
        pointcloud_drop_invalid=config.pointcloud_drop_invalid,
        pointcloud_sort_field=config.pointcloud_sort_field,
    )
    specs: list[_ChannelTransformSpec] = []
    for source in channels:
        schema_name = normalize_schema_name(source.schema_name)
        output = source
        factory: TransformFactory = partial(_PassthroughTransform, source)
        is_video = (
            _is_ros_channel(source)
            and config.image_format == "video"
            and schema_name in IMAGE_SCHEMAS
        )
        if is_video:
            output = PlaybackChannel(
                topic=source.topic,
                message_encoding="cdr",
                schema_name=COMPRESSED_VIDEO_SCHEMA,
                schema_encoding="ros2msg",
                schema_text=FOXGLOVE_COMPRESSED_VIDEO,
            )
            video_rungs = (
                _adaptive_video_rungs(config.quality)
                if config.adaptive_quality
                else (_AdaptiveVideoRung(config.quality, 1.0, _MAX_VIDEO_FPS),)
            )
            factories = tuple(
                partial(
                    _VideoCompressTransform,
                    source,
                    output,
                    replace(config, quality=rung.quality, adaptive_quality=False),
                    scale_factor=rung.scale_factor,
                )
                for rung in video_rungs
            )
        elif (
            _is_ros_channel(source)
            and config.image_format in {"jpeg", "png"}
            and schema_name in RAW_SCHEMAS
        ):
            output = PlaybackChannel(
                topic=source.topic,
                message_encoding="cdr",
                schema_name="sensor_msgs/msg/CompressedImage",
                schema_encoding="ros2msg",
                schema_text=COMPRESSED_IMAGE,
            )
            factory = partial(_create_image_processor_transform, source, output, config)
        elif (
            _is_ros_channel(source)
            and schema_name in POINTCLOUD2_SCHEMAS
            and (config.pointcloud or cleanup.enabled)
        ):
            if config.pointcloud:
                resolved_schema = config.pc_schema
                if resolved_schema == "auto":
                    resolved_schema = "foxglove" if config.pc_format == "draco" else "pointcloud2"
                if resolved_schema == "foxglove":
                    out_name = FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA
                    out_schema = FOXGLOVE_COMPRESSED_POINTCLOUD
                else:
                    out_name = COMPRESSED_POINTCLOUD2_SCHEMA
                    out_schema = COMPRESSED_POINTCLOUD2
                output = PlaybackChannel(
                    topic=source.topic,
                    message_encoding="cdr",
                    schema_name=out_name,
                    schema_encoding="ros2msg",
                    schema_text=out_schema,
                )
            factory = partial(
                _create_pointcloud_processor_transform, source, output, config, cleanup
            )
        if is_video:
            specs.append(_ChannelTransformSpec(source, output, factories, video_rungs))
        else:
            specs.append(_ChannelTransformSpec(source, output, (factory,)))
    plan = JitPlaybackTransformPlan("roscompress", tuple(specs))
    _probe_factories(specs)
    return plan


def _create_rosdecompress_plan(
    config: RosdecompressConfig,
    channels: tuple[PlaybackChannel, ...],
) -> JitPlaybackTransformPlan:
    from mcap_codec_support.pointcloud import (  # noqa: PLC0415
        COMPRESSED_POINTCLOUD2_SCHEMA,
        FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
        POINTCLOUD2,
    )
    from mcap_codec_support.video import (  # noqa: PLC0415
        COMPRESSED_IMAGE,
        COMPRESSED_VIDEO_SCHEMA,
        IMAGE,
    )

    compressed_pc_schemas = {
        COMPRESSED_POINTCLOUD2_SCHEMA,
        FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    }
    specs: list[_ChannelTransformSpec] = []
    for source in channels:
        output = source
        factory: TransformFactory = partial(_PassthroughTransform, source)
        if (
            _is_ros_channel(source)
            and config.video
            and source.schema_name == COMPRESSED_VIDEO_SCHEMA
        ):
            if config.video_format == "compressed":
                out_name = "sensor_msgs/msg/CompressedImage"
                out_schema = COMPRESSED_IMAGE
            else:
                out_name = "sensor_msgs/msg/Image"
                out_schema = IMAGE
            output = PlaybackChannel(
                topic=source.topic,
                message_encoding="cdr",
                schema_name=out_name,
                schema_encoding="ros2msg",
                schema_text=out_schema,
            )
            factory = partial(_VideoDecompressTransform, source, output, config)
        elif (
            _is_ros_channel(source)
            and config.pointcloud
            and source.schema_name in compressed_pc_schemas
        ):
            output = PlaybackChannel(
                topic=source.topic,
                message_encoding="cdr",
                schema_name="sensor_msgs/msg/PointCloud2",
                schema_encoding="ros2msg",
                schema_text=POINTCLOUD2,
            )
            factory = partial(_PointcloudDecompressTransform, source, output)
        specs.append(_ChannelTransformSpec(source, output, (factory,)))
    plan = JitPlaybackTransformPlan("rosdecompress", tuple(specs))
    _probe_factories(specs)
    return plan


def _probe_factories(specs: Iterable[_ChannelTransformSpec]) -> None:
    for spec in specs:
        transform = spec.factories[0]()
        transform.close()


def _adaptive_video_rungs(quality: int) -> tuple[_AdaptiveVideoRung, ...]:
    frame_rate_rungs = tuple(
        _AdaptiveVideoRung(quality, 1.0, max_fps)
        for max_fps in (_MAX_VIDEO_FPS, *_ADAPTIVE_FRAME_RATES)
    )
    minimum_fps = _ADAPTIVE_FRAME_RATES[-1]
    resolution_rungs = tuple(
        _AdaptiveVideoRung(quality, scale_factor, minimum_fps)
        for scale_factor in _ADAPTIVE_SCALE_FACTORS
    )
    return frame_rate_rungs + resolution_rungs


def _adaptive_max_dimension(
    configured_scale: int | None,
    *,
    width: int,
    height: int,
    scale_factor: float,
) -> int | None:
    if scale_factor == 1.0:
        return configured_scale
    source_max_dimension = max(width, height)
    base_max_dimension = (
        source_max_dimension
        if configured_scale is None
        else min(source_max_dimension, configured_scale)
    )
    return max(2, round(base_max_dimension * scale_factor))


def _is_ros_channel(channel: PlaybackChannel) -> bool:
    return channel.message_encoding == "cdr" and channel.schema_encoding == "ros2msg"


class _PassthroughTransform:
    def __init__(self, channel: PlaybackChannel) -> None:
        self._channel = channel

    def process(self, payload: bytes | memoryview, timestamp_ns: int) -> tuple[PlaybackOutput, ...]:
        return (PlaybackOutput(self._channel, timestamp_ns, payload),)

    def finish(self) -> tuple[PlaybackOutput, ...]:
        return ()

    def close(self) -> None:
        return


class _ProcessorChannelTransform:
    def __init__(
        self,
        source: PlaybackChannel,
        output: PlaybackChannel,
        processors: tuple[object, ...],
    ) -> None:
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415
        from mcap_ros2_support_fast.writer import ROS2EncoderFactory  # noqa: PLC0415

        self._source = source
        self._output = output
        self._processors = cast("tuple[MessageTransformProcessor, ...]", processors)
        self._channel = Channel(
            id=1,
            schema_id=1,
            topic=source.topic,
            message_encoding=source.message_encoding,
            metadata={},
        )
        self._schema = Schema(
            id=1,
            name=source.schema_name,
            encoding=source.schema_encoding,
            data=source.schema_text.encode(),
        )
        decoder = DecoderFactory().decoder_for(source.message_encoding, self._schema)
        if decoder is None:
            raise PlaybackError(f"No CDR decoder for {source.topic!r} ({source.schema_name})")
        self._decoder: Callable[[bytes | memoryview], object] = decoder
        output_schema = Schema(
            id=2,
            name=output.schema_name,
            encoding=output.schema_encoding,
            data=output.schema_text.encode(),
        )
        encoder = ROS2EncoderFactory().encoder_for(output_schema)
        if encoder is None:
            raise PlaybackError(f"No CDR encoder for {output.schema_name!r}")
        self._encoder: Callable[[object], bytes | memoryview] = encoder
        self._warned_unconvertible = False

    def process(self, payload: bytes | memoryview, timestamp_ns: int) -> tuple[PlaybackOutput, ...]:
        current = self._decoder(payload)
        channel = self._channel
        schema = self._schema
        for processor in self._processors:
            if not processor.matches(channel, schema):
                continue
            output = _single_processor_output(processor.transform(channel, schema, current))
            if output is None:
                # The processor left the message unchanged (e.g. nothing to
                # clean, or compression bailed and kept the raw payload).
                continue
            current = output.data
            schema = Schema(
                id=2,
                name=output.schema_name,
                encoding=output.schema_encoding,
                data=output.schema_data,
            )
            channel = Channel(
                id=2,
                schema_id=2,
                topic=output.topic,
                message_encoding=output.message_encoding,
                metadata={},
            )
        if (
            schema.name != self._output.schema_name
            or schema.encoding != self._output.schema_encoding
            or schema.data != self._output.schema_text.encode()
        ):
            # The advertised channel promises the transformed schema; a message
            # the chain could not convert cannot be delivered on it.
            if not self._warned_unconvertible:
                self._warned_unconvertible = True
                logger.warning(
                    "Dropping %s messages the JIT transform could not convert to %s",
                    self._source.topic,
                    self._output.schema_name,
                )
            return ()
        return (PlaybackOutput(self._output, timestamp_ns, self._encoder(current)),)

    def finish(self) -> tuple[PlaybackOutput, ...]:
        for processor in self._processors:
            if tuple(processor.finalize()):
                raise PlaybackError("JIT message transform produced unexpected buffered output")
        return ()

    def close(self) -> None:
        return


def _single_processor_output(
    outputs: Iterable[TransformOutput] | None,
) -> TransformOutput | None:
    if outputs is None:
        return None
    iterator = iter(outputs)
    try:
        output = next(iterator)
    except StopIteration as exc:
        raise PlaybackError("JIT transforms must preserve every message") from exc
    try:
        next(iterator)
    except StopIteration:
        return output
    raise PlaybackError("JIT transforms must produce exactly one message per input")


def _create_image_processor_transform(
    source: PlaybackChannel,
    output: PlaybackChannel,
    config: RoscompressConfig,
) -> _ProcessorChannelTransform:
    from pymcap_cli.core.processors.image_compress import ImageCompressProcessor  # noqa: PLC0415

    image_format = cast("Literal['jpeg', 'png']", config.image_format)
    return _ProcessorChannelTransform(
        source,
        output,
        (
            ImageCompressProcessor(
                image_format=image_format,
                jpeg_quality=config.jpeg_quality,
                scale=config.scale,
            ),
        ),
    )


def _create_pointcloud_processor_transform(
    source: PlaybackChannel,
    output: PlaybackChannel,
    config: RoscompressConfig,
    cleanup: PointcloudCleanupConfig,
) -> _ProcessorChannelTransform:
    processors: list[object] = []
    if cleanup.enabled:
        from pymcap_cli.core.processors.pointcloud_clean import (  # noqa: PLC0415
            PointcloudCleanProcessor,
        )

        processors.append(
            PointcloudCleanProcessor(
                drop_invalid=cleanup.drop_invalid,
                sort_field=cleanup.sort_field,
            )
        )
    if config.pointcloud:
        from pymcap_cli.core.processors.pointcloud_compress import (  # noqa: PLC0415
            PointcloudCompressProcessor,
        )

        processors.append(
            PointcloudCompressProcessor(
                pc_format=config.pc_format,
                pc_schema=config.pc_schema,
                pc_encoding=config.pc_encoding,
                pc_compression=config.pc_compression,
                resolution=config.resolution,
                draco_compression_level=config.draco_compression_level,
            )
        )
    return _ProcessorChannelTransform(source, output, tuple(processors))


@dataclass(frozen=True, slots=True)
class _ImageMessage:
    decoded_message: object
    channel: Channel


@dataclass(frozen=True, slots=True)
class _VideoMeta:
    timestamp_ns: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str


class _HeaderStamp(Protocol):
    sec: int
    nanosec: int


class _Header(Protocol):
    stamp: _HeaderStamp
    frame_id: str


class _HeaderMessage(Protocol):
    header: _Header


class _CompressedVideoTimestamp(Protocol):
    sec: int
    nanosec: int


class _CompressedVideoMessage(Protocol):
    timestamp: _CompressedVideoTimestamp
    frame_id: str
    data: bytes | bytearray | memoryview
    format: str


class _DecodedFrame(Protocol):
    is_jpeg: bool
    data: bytes
    width: int
    height: int


class _VideoCompressTransform:
    def __init__(
        self,
        source: PlaybackChannel,
        output: PlaybackChannel,
        config: RoscompressConfig,
        *,
        scale_factor: float = 1.0,
    ) -> None:
        from mcap_codec_support.video import EncoderMode  # noqa: PLC0415
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415
        from mcap_ros2_support_fast.writer import ROS2EncoderFactory  # noqa: PLC0415

        from pymcap_cli.core.processors.video_compress import (  # noqa: PLC0415
            VideoEncoderSession,
            build_compressed_video_payload,
            resolve_video_compression_backend,
        )

        self._source = source
        self._output = output
        self._config = config
        from mcap_codec_support._schemas import normalize_schema_name  # noqa: PLC0415

        self._normalized_schema_name = normalize_schema_name(source.schema_name)
        self._channel = Channel(1, 1, source.topic, source.message_encoding, {})
        source_schema = Schema(
            1, source.schema_name, source.schema_encoding, source.schema_text.encode()
        )
        decoder = DecoderFactory().decoder_for(source.message_encoding, source_schema)
        if decoder is None:
            raise PlaybackError(f"No CDR decoder for {source.topic!r} ({source.schema_name})")
        self._decoder: Callable[[bytes | memoryview], object] = decoder
        output_schema = Schema(
            2, output.schema_name, output.schema_encoding, output.schema_text.encode()
        )
        encoder = ROS2EncoderFactory().encoder_for(output_schema)
        if encoder is None:
            raise PlaybackError(f"No CDR encoder for {output.schema_name!r}")
        self._encoder: Callable[[object], bytes | memoryview] = encoder
        resolved = resolve_video_compression_backend(
            codec=config.codec,
            encoder=config.encoder,
            backend=EncoderMode(config.backend),
        )
        self._backend = resolved.backend
        self._encoder_name = resolved.encoder_name
        self._session_type = VideoEncoderSession
        self._session: VideoEncoderSession | None = None
        self._scale_factor = scale_factor
        self._build_payload = build_compressed_video_payload
        self._pending: deque[_VideoMeta] = deque()

    def process(self, payload: bytes | memoryview, timestamp_ns: int) -> tuple[PlaybackOutput, ...]:
        decoded = self._decoder(payload)
        header = cast("_HeaderMessage", decoded).header
        self._pending.append(
            _VideoMeta(
                timestamp_ns,
                header.stamp.sec,
                header.stamp.nanosec,
                header.frame_id,
            )
        )
        frame, width, height = self._backend.decode_image(
            _ImageMessage(decoded, self._channel), self._normalized_schema_name
        )
        session = self._session
        if session is None:
            session = self._session_type(
                backend=self._backend,
                encoder_name=self._encoder_name,
                codec=self._config.codec,
                quality=self._config.quality,
                topic=self._source.topic,
                scale=_adaptive_max_dimension(
                    self._config.scale,
                    width=width,
                    height=height,
                    scale_factor=self._scale_factor,
                ),
            )
            self._session = session
        outcome = session.encode_with_fallback(frame, width, height)
        if outcome.failed:
            raise PlaybackError(f"Video encoding failed for {self._source.topic!r}")
        if outcome.swapped and len(self._pending) > 1:
            raise PlaybackError(
                f"Video encoder fallback lost buffered frames for {self._source.topic!r}"
            )
        if outcome.data is None:
            return ()
        return (self._output_for(self._pending.popleft(), outcome.data),)

    def finish(self) -> tuple[PlaybackOutput, ...]:
        session = self._session
        encoder = None if session is None else session.encoder
        packets = [] if encoder is None else encoder.flush_packets()
        outputs = [self._output_for(self._pending.popleft(), packet) for packet in packets]
        if self._pending:
            raise PlaybackError(
                f"Video encoder did not flush every frame for {self._source.topic!r}"
            )
        return tuple(outputs)

    def _output_for(self, meta: _VideoMeta, data: bytes) -> PlaybackOutput:
        encoded = self._encoder(
            self._build_payload(
                codec=self._config.codec,
                stamp_sec=meta.stamp_sec,
                stamp_nanosec=meta.stamp_nanosec,
                frame_id=meta.frame_id,
                data=data,
            )
        )
        return PlaybackOutput(self._output, meta.timestamp_ns, encoded)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()


class _VideoDecompressTransform:
    def __init__(
        self,
        source: PlaybackChannel,
        output: PlaybackChannel,
        config: RosdecompressConfig,
    ) -> None:
        from mcap_codec_support.video import EncoderMode  # noqa: PLC0415
        from mcap_codec_support.video.compression import create_video_decompressor  # noqa: PLC0415
        from mcap_ros2_support_fast.decoder import DecoderFactory  # noqa: PLC0415
        from mcap_ros2_support_fast.writer import ROS2EncoderFactory  # noqa: PLC0415

        self._source = source
        self._output = output
        self._config = config
        source_schema = Schema(
            1, source.schema_name, source.schema_encoding, source.schema_text.encode()
        )
        decoder = DecoderFactory().decoder_for(source.message_encoding, source_schema)
        if decoder is None:
            raise PlaybackError(f"No CDR decoder for {source.topic!r} ({source.schema_name})")
        self._decoder: Callable[[bytes | memoryview], object] = decoder
        output_schema = Schema(
            2, output.schema_name, output.schema_encoding, output.schema_text.encode()
        )
        encoder = ROS2EncoderFactory().encoder_for(output_schema)
        if encoder is None:
            raise PlaybackError(f"No CDR encoder for {output.schema_name!r}")
        self._encoder: Callable[[object], bytes | memoryview] = encoder
        self._decompressor = create_video_decompressor(
            video_format=config.video_format,
            jpeg_quality=config.jpeg_quality,
            mode=EncoderMode(config.backend),
        )
        self._pending: deque[_VideoMeta] = deque()

    def process(self, payload: bytes | memoryview, timestamp_ns: int) -> tuple[PlaybackOutput, ...]:
        decoded = cast("_CompressedVideoMessage", self._decoder(payload))
        self._pending.append(
            _VideoMeta(
                timestamp_ns,
                decoded.timestamp.sec,
                decoded.timestamp.nanosec,
                decoded.frame_id,
            )
        )
        frame = self._decompressor.decompress(bytes(decoded.data), decoded.format)
        if frame is None:
            return ()
        return (self._output_for(self._pending.popleft(), frame),)

    def finish(self) -> tuple[PlaybackOutput, ...]:
        outputs = [
            self._output_for(self._pending.popleft(), frame)
            for frame in self._decompressor.flush()
            if self._pending
        ]
        if self._pending:
            raise PlaybackError(
                f"Video decoder did not flush every frame for {self._source.topic!r}"
            )
        return tuple(outputs)

    def _output_for(self, meta: _VideoMeta, frame: _DecodedFrame) -> PlaybackOutput:
        header = {
            "stamp": {"sec": meta.stamp_sec, "nanosec": meta.stamp_nanosec},
            "frame_id": meta.frame_id,
        }
        if frame.is_jpeg:
            payload: object = {"header": header, "format": "jpeg", "data": frame.data}
        else:
            payload = {
                "header": header,
                "height": frame.height,
                "width": frame.width,
                "encoding": "rgb8",
                "is_bigendian": 0,
                "step": frame.width * 3,
                "data": frame.data,
            }
        return PlaybackOutput(self._output, meta.timestamp_ns, self._encoder(payload))

    def close(self) -> None:
        return


class _PointcloudDecompressTransform:
    def __init__(self, source: PlaybackChannel, output: PlaybackChannel) -> None:
        from mcap_codec_support.pointcloud.factories import (  # noqa: PLC0415
            CompressedPointCloudDecompressFactory,
        )
        from mcap_ros2_support_fast.writer import ROS2EncoderFactory  # noqa: PLC0415

        self._output = output
        source_schema = Schema(
            1, source.schema_name, source.schema_encoding, source.schema_text.encode()
        )
        decoder = CompressedPointCloudDecompressFactory().decoder_for(
            source.message_encoding, source_schema
        )
        if decoder is None:
            raise PlaybackError(f"No point-cloud decoder for {source.schema_name!r}")
        self._decoder: Callable[[bytes | memoryview], object] = decoder
        output_schema = Schema(
            2, output.schema_name, output.schema_encoding, output.schema_text.encode()
        )
        encoder = ROS2EncoderFactory().encoder_for(output_schema)
        if encoder is None:
            raise PlaybackError(f"No CDR encoder for {output.schema_name!r}")
        self._encoder: Callable[[object], bytes | memoryview] = encoder

    def process(self, payload: bytes | memoryview, timestamp_ns: int) -> tuple[PlaybackOutput, ...]:
        return (PlaybackOutput(self._output, timestamp_ns, self._encoder(self._decoder(payload))),)

    def finish(self) -> tuple[PlaybackOutput, ...]:
        return ()

    def close(self) -> None:
        return
