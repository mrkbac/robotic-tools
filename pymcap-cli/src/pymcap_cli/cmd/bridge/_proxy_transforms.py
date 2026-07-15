"""Rule-based live transforms for `pymcap-cli bridge proxy`."""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from robo_ws_bridge.ws_types import ChannelInfo
from small_mcap import Channel, Schema

from pymcap_cli.cmd._pointcloud_cleanup import pointcloud_worker_count
from pymcap_cli.cmd.bridge._proxy_runtime import (
    MESSAGE_ENCODING,
    SCHEMA_ENCODING,
    LiveTransformer,
    TransformResult,
    schema_from_channel,
)
from pymcap_cli.core.processors.message_transform import (
    MessageTransformProcessor,
    TransformOutput,
)

logger = logging.getLogger(__name__)

_RAW_IMAGE_SCHEMAS = frozenset({"sensor_msgs/Image"})
_COMPRESSED_IMAGE_SCHEMAS = frozenset({"sensor_msgs/CompressedImage"})
_IMAGE_SCHEMAS = _RAW_IMAGE_SCHEMAS | _COMPRESSED_IMAGE_SCHEMAS
_POINTCLOUD2_SCHEMAS = frozenset({"sensor_msgs/PointCloud2"})
COMPRESSED_VIDEO_SCHEMA = "foxglove_msgs/msg/CompressedVideo"


def _normalize_schema_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) == 3 and parts[1] in ("msg", "srv", "action"):
        return f"{parts[0]}/{parts[2]}"
    return name


class _StampLike(Protocol):
    sec: int
    nanosec: int


class _HeaderLike(Protocol):
    stamp: _StampLike
    frame_id: str


class _HeaderMessageLike(Protocol):
    header: _HeaderLike


@dataclass(frozen=True, slots=True)
class ImageConfig:
    image_format: Literal["video", "jpeg", "png", "none"]
    codec: str
    quality: int
    encoder: str | None
    backend: Literal["auto", "pyav", "ffmpeg-cli", "gstreamer"]
    scale: int | None
    jpeg_quality: int


@dataclass(frozen=True, slots=True)
class PointCloudConfig:
    enabled: bool
    pc_format: Literal["cloudini", "draco"]
    pc_schema: Literal["auto", "pointcloud2", "foxglove"]
    pc_encoding: Literal["lossy", "lossless", "none"]
    pc_compression: Literal["zstd", "lz4", "none"]
    resolution: float
    draco_compression_level: int
    drop_invalid: bool = True
    sort_field: str | None = None


@dataclass(frozen=True, slots=True)
class ProxyConfig:
    image: ImageConfig
    pointcloud: PointCloudConfig
    transform_queue_size: int
    send_queue_size: int
    throttle_hz: float
    max_message_size: int | None


@dataclass(frozen=True, slots=True)
class _LiveChannel:
    topic: str


@dataclass(frozen=True, slots=True)
class _LiveDecodedMessage:
    decoded_message: object
    channel: _LiveChannel


class LiveTransformRule(Protocol):
    def create_transformer(self, channel: ChannelInfo) -> LiveTransformer | None: ...


@dataclass(frozen=True, slots=True)
class ProcessorRule:
    """Apply an existing MessageTransformProcessor to a live topic/schema."""

    processor_factory: Callable[[], MessageTransformProcessor]
    preprocessor_factories: tuple[Callable[[], MessageTransformProcessor], ...] = ()
    output_schema_name: str | None = None
    output_schema_text: str | None = None
    output_schema_encoding: str | None = None
    output_message_encoding: str | None = None

    def create_transformer(self, channel: ChannelInfo) -> LiveTransformer | None:
        if not _is_ros_cdr_channel(channel):
            return None
        schema = schema_from_channel(channel)
        processor = self.processor_factory()
        preprocessors = tuple(factory() for factory in self.preprocessor_factories)
        small_channel = _small_channel_from_info(channel, schema)
        if not processor.matches(small_channel, schema):
            return None
        return ProcessorTransformer(
            processor=processor,
            preprocessors=preprocessors,
            channel=small_channel,
            schema=schema,
            output_schema_name=self.output_schema_name or schema.name,
            output_schema_text=(
                self.output_schema_text
                if self.output_schema_text is not None
                else channel.get("schema", "")
            ),
            output_schema_encoding=self.output_schema_encoding or schema.encoding,
            output_message_encoding=self.output_message_encoding or channel["encoding"],
        )


class ProcessorTransformer:
    def __init__(
        self,
        *,
        processor: MessageTransformProcessor,
        preprocessors: tuple[MessageTransformProcessor, ...],
        channel: Channel,
        schema: Schema,
        output_schema_name: str,
        output_schema_text: str,
        output_schema_encoding: str,
        output_message_encoding: str,
    ) -> None:
        self.output_schema_name = output_schema_name
        self.output_schema_text = output_schema_text
        self.output_schema_encoding = output_schema_encoding
        self.output_message_encoding = output_message_encoding
        self.worker_count = max(
            1,
            processor.worker_count,
            *(preprocessor.worker_count for preprocessor in preprocessors),
        )
        self._processor = processor
        self._preprocessors = preprocessors
        self._channel = channel
        self._schema = schema

    def transform(self, decoded: object, timestamp_ns: int) -> TransformResult | None:
        del timestamp_ns
        current = decoded
        for preprocessor in self._preprocessors:
            if not preprocessor.matches(self._channel, self._schema):
                continue
            output = _single_output(preprocessor.transform(self._channel, self._schema, current))
            if output is None:
                continue
            _validate_output_schema(output, self._schema.name, self._schema.encoding)
            current = output.data

        outputs = self._processor.transform(self._channel, self._schema, current)
        output = _single_output(outputs)
        if output is None:
            return None
        _validate_output_schema(output, self.output_schema_name, self.output_schema_encoding)
        return TransformResult(payload=cast("dict[str, object]", output.data))

    def close(self) -> None:
        for preprocessor in self._preprocessors:
            for _message in preprocessor.finalize():
                logger.debug("Ignoring finalize output from live processor %s", type(preprocessor))
        for _message in self._processor.finalize():
            logger.debug("Ignoring finalize output from live processor %s", type(self._processor))


@dataclass(frozen=True, slots=True)
class VideoRule:
    config: ImageConfig

    def create_transformer(self, channel: ChannelInfo) -> LiveTransformer | None:
        if self.config.image_format != "video" or not _is_ros_cdr_channel(channel):
            return None
        schema_name = _normalize_schema_name(channel.get("schemaName", ""))
        if schema_name not in _IMAGE_SCHEMAS:
            return None
        return VideoTransformer(self.config, channel["topic"])


class VideoTransformer:
    """Encode one live image topic to CompressedVideo, one frame at a time.

    Wraps the shared :class:`VideoEncoderSession` (the same encoder used by the
    batch ``VideoCompressProcessor``) but drives it synchronously: a live proxy
    wants the latest frame with minimal latency, so there is no lookahead or
    B-frame buffering — each decoded frame is encoded and emitted immediately.
    """

    output_schema_encoding = SCHEMA_ENCODING
    output_message_encoding = MESSAGE_ENCODING
    worker_count = 1

    def __init__(self, config: ImageConfig, topic: str) -> None:
        from mcap_codec_support.video import (  # noqa: PLC0415
            COMPRESSED_VIDEO_SCHEMA,
            FOXGLOVE_COMPRESSED_VIDEO,
            EncoderMode,
        )

        from pymcap_cli.core.processors.video_compress import (  # noqa: PLC0415
            VideoEncoderSession,
            build_compressed_video_payload,
            resolve_video_compression_backend,
        )

        self.output_schema_name = COMPRESSED_VIDEO_SCHEMA
        self.output_schema_text = FOXGLOVE_COMPRESSED_VIDEO
        self._config = config
        self._topic = topic
        self._build_payload = build_compressed_video_payload
        resolved = resolve_video_compression_backend(
            codec=config.codec,
            encoder=config.encoder,
            backend=EncoderMode(config.backend),
        )
        self._session = VideoEncoderSession(
            backend=resolved.backend,
            encoder_name=resolved.encoder_name,
            codec=config.codec,
            quality=config.quality,
            topic=topic,
            scale=config.scale,
        )

    def transform(self, decoded: object, timestamp_ns: int) -> TransformResult | None:
        del timestamp_ns
        schema_name = _normalize_schema_name(_message_type(decoded))
        live = _LiveDecodedMessage(decoded_message=decoded, channel=_LiveChannel(self._topic))
        frame, width, height = self._session.decode_image(live, schema_name)
        outcome = self._session.encode_with_fallback(frame, width, height)
        if outcome.failed or outcome.data is None:
            return None
        header = _message_header(decoded)
        return TransformResult(
            payload=self._build_payload(
                codec=self._config.codec,
                stamp_sec=header.stamp.sec,
                stamp_nanosec=header.stamp.nanosec,
                frame_id=header.frame_id,
                data=outcome.data,
            ),
            is_compressed_video=True,
            is_keyframe=is_video_keyframe(outcome.data, self._config.codec),
        )

    def close(self) -> None:
        self._session.close()


def build_transform_rules(config: ProxyConfig) -> list[LiveTransformRule]:
    rules: list[LiveTransformRule] = [VideoRule(config.image)]

    if config.image.image_format in {"jpeg", "png"}:
        from mcap_codec_support.video import COMPRESSED_IMAGE  # noqa: PLC0415

        from pymcap_cli.core.processors.image_compress import (  # noqa: PLC0415
            ImageCompressProcessor,
        )

        image_format: Literal["jpeg", "png"] = (
            "jpeg" if config.image.image_format == "jpeg" else "png"
        )
        rules.append(
            ProcessorRule(
                processor_factory=lambda: ImageCompressProcessor(
                    image_format=image_format,
                    jpeg_quality=config.image.jpeg_quality,
                    scale=config.image.scale,
                ),
                output_schema_name="sensor_msgs/msg/CompressedImage",
                output_schema_text=COMPRESSED_IMAGE,
            )
        )

    if config.pointcloud.enabled:
        from mcap_codec_support.pointcloud import (  # noqa: PLC0415
            COMPRESSED_POINTCLOUD2,
            COMPRESSED_POINTCLOUD2_SCHEMA,
            FOXGLOVE_COMPRESSED_POINTCLOUD,
            FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
        )

        from pymcap_cli.core.processors.pointcloud_compress import (  # noqa: PLC0415
            PointcloudCompressProcessor,
        )

        preprocessor_factories: tuple[Callable[[], MessageTransformProcessor], ...] = ()
        if config.pointcloud.drop_invalid or config.pointcloud.sort_field is not None:
            from pymcap_cli.core.processors.pointcloud_clean import (  # noqa: PLC0415
                PointcloudCleanProcessor,
            )

            preprocessor_factories = (
                lambda: PointcloudCleanProcessor(
                    drop_invalid=config.pointcloud.drop_invalid,
                    sort_field=config.pointcloud.sort_field,
                ),
            )

        pc_schema = config.pointcloud.pc_schema
        if pc_schema == "auto":
            pc_schema = "foxglove" if config.pointcloud.pc_format == "draco" else "pointcloud2"
        if pc_schema == "foxglove":
            output_schema_name = FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA
            output_schema_text = FOXGLOVE_COMPRESSED_POINTCLOUD
        else:
            output_schema_name = COMPRESSED_POINTCLOUD2_SCHEMA
            output_schema_text = COMPRESSED_POINTCLOUD2
        rules.append(
            ProcessorRule(
                processor_factory=lambda: PointcloudCompressProcessor(
                    pc_format=config.pointcloud.pc_format,
                    pc_schema=config.pointcloud.pc_schema,
                    pc_encoding=config.pointcloud.pc_encoding,
                    pc_compression=config.pointcloud.pc_compression,
                    resolution=config.pointcloud.resolution,
                    draco_compression_level=config.pointcloud.draco_compression_level,
                    workers=pointcloud_worker_count(max_workers=8),
                ),
                preprocessor_factories=preprocessor_factories,
                output_schema_name=output_schema_name,
                output_schema_text=output_schema_text,
            )
        )

    return rules


def create_transformer(
    rules: Iterable[LiveTransformRule], channel: ChannelInfo
) -> LiveTransformer | None:
    for rule in rules:
        transformer = rule.create_transformer(channel)
        if transformer is not None:
            return transformer
    return None


def is_video_keyframe(data: bytes, video_format: str) -> bool:
    fmt = video_format.lower()
    if "265" in fmt or "hevc" in fmt:
        return _has_h265_keyframe(data)
    if "264" in fmt or "avc" in fmt:
        return _has_h264_keyframe(data)
    if "vp9" in fmt:
        return _is_vp9_keyframe(data)
    if "av1" in fmt:
        return _is_av1_keyframe(data)
    return True


def _small_channel_from_info(channel: ChannelInfo, schema: Schema) -> Channel:
    return Channel(
        id=channel["id"],
        schema_id=schema.id,
        topic=channel["topic"],
        message_encoding=channel["encoding"],
        metadata={},
    )


def _is_ros_cdr_channel(channel: ChannelInfo) -> bool:
    return (
        channel["encoding"] == MESSAGE_ENCODING
        and channel.get("schemaEncoding", SCHEMA_ENCODING) == SCHEMA_ENCODING
    )


def _single_output(outputs: Iterable[TransformOutput] | None) -> TransformOutput | None:
    if outputs is None:
        return None
    iterator = iter(outputs)
    try:
        output = next(iterator)
    except StopIteration:
        return None
    try:
        next(iterator)
    except StopIteration:
        return output
    raise RuntimeError("Live processor rules must produce at most one output message")


def _validate_output_schema(
    output: TransformOutput, output_schema_name: str, output_schema_encoding: str
) -> None:
    if output.schema_name != output_schema_name or output.schema_encoding != output_schema_encoding:
        raise RuntimeError(
            "Live processor output schema changed from advertised "
            f"{output_schema_name!r} to {output.schema_name!r}"
        )


def _message_type(message: object) -> str:
    msg_type = vars(type(message)).get("_type", "")
    return msg_type if isinstance(msg_type, str) else ""


def _message_header(message: object) -> _HeaderLike:
    return cast("_HeaderMessageLike", message).header


def _start_code_positions(data: bytes) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    idx = 0
    while idx < len(data) - 3:
        if data[idx : idx + 3] == b"\x00\x00\x01":
            positions.append((idx, idx + 3))
            idx += 3
        elif idx < len(data) - 4 and data[idx : idx + 4] == b"\x00\x00\x00\x01":
            positions.append((idx, idx + 4))
            idx += 4
        else:
            idx += 1
    return positions


def _has_h264_keyframe(data: bytes) -> bool:
    for _start, header_pos in _start_code_positions(data):
        if header_pos >= len(data):
            continue
        nal_type = data[header_pos] & 0x1F
        if nal_type == 5:
            return True
    return False


def _has_h265_keyframe(data: bytes) -> bool:
    for _start, header_pos in _start_code_positions(data):
        if header_pos + 1 >= len(data):
            continue
        nal_type = (data[header_pos] >> 1) & 0x3F
        if 16 <= nal_type <= 21:
            return True
    return False


def _is_vp9_keyframe(data: bytes) -> bool:
    """Read the VP9 uncompressed header: frame_type == 0 (KEY_FRAME).

    Bits are read MSB-first from the first byte: a 2-bit frame_marker (0b10),
    profile_low/high bits, an optional reserved bit for profile 3, then
    show_existing_frame and — when that is 0 — the frame_type bit.
    """
    if not data:
        return False
    byte = data[0]

    def bit(index: int) -> int:
        return (byte >> (7 - index)) & 1

    if (bit(0) << 1 | bit(1)) != 0b10:  # frame_marker
        return False
    profile = (bit(3) << 1) | bit(2)
    pos = 4 + (1 if profile == 3 else 0)  # skip the profile-3 reserved bit
    if bit(pos) == 1:  # show_existing_frame → not a new coded frame
        return False
    return bit(pos + 1) == 0  # frame_type: 0 = KEY_FRAME


def _is_av1_keyframe(data: bytes) -> bool:
    """Detect an AV1 key frame by the presence of a sequence-header OBU.

    libaom/ffmpeg emit an ``OBU_SEQUENCE_HEADER`` (type 1) ahead of every key
    frame and not before inter frames, so scanning the temporal unit's OBU
    headers for one is a robust keyframe signal without decoding the frame.
    """
    idx = 0
    length = len(data)
    while idx < length:
        header = data[idx]
        obu_type = (header >> 3) & 0xF
        has_extension = (header >> 2) & 1
        has_size_field = (header >> 1) & 1
        idx += 1 + has_extension
        if obu_type == 1:  # OBU_SEQUENCE_HEADER
            return True
        if not has_size_field:
            return False  # size-less OBU: cannot skip to the next one
        size, idx = _read_leb128(data, idx)
        if size < 0:
            return False
        idx += size
    return False


def _read_leb128(data: bytes, idx: int) -> tuple[int, int]:
    """Read an unsigned LEB128 value; return ``(value, next_index)`` or ``(-1, idx)``."""
    value = 0
    for shift in range(0, 56, 7):
        if idx >= len(data):
            return -1, idx
        byte = data[idx]
        idx += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, idx
    return -1, idx
