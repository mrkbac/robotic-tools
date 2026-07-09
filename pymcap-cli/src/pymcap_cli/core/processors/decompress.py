"""Decompression processors — the inverse of the compress processors.

- :class:`PointcloudDecompressProcessor` — CompressedPointCloud2 /
  CompressedPointCloud → PointCloud2 (synchronous).
- :class:`VideoDecompressProcessor` — CompressedVideo → Image (raw) or
  CompressedImage (JPEG). Stateful/buffered per channel (a decoder returns a
  frame's image several packets later), so it drives the decompressor directly
  and flushes tails in ``finalize()`` — mirroring VideoCompressProcessor.

Both decode with the specialised decompress decoders (not generic CDR), then
re-encode the recovered ROS message on a new channel/schema.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from mcap_codec_support.pointcloud import (
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    POINTCLOUD2,
)
from mcap_codec_support.pointcloud.factories import CompressedPointCloudDecompressFactory
from mcap_codec_support.video import (
    COMPRESSED_IMAGE,
    COMPRESSED_VIDEO_SCHEMA,
    IMAGE,
    EncoderMode,
)
from mcap_codec_support.video.compression import create_video_decompressor
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from small_mcap import Channel, Message, Schema
from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChannelContext,
    ChunkContext,
    InputContext,
    InputProcessor,
    MessageContext,
    MessageScope,
    MessageWithContext,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from mcap_codec_support._protocols import VideoDecompressorProtocol

_POINTCLOUD2_SCHEMA = "sensor_msgs/msg/PointCloud2"
_IMAGE_SCHEMA = "sensor_msgs/msg/Image"
_COMPRESSED_IMAGE_SCHEMA = "sensor_msgs/msg/CompressedImage"
_COMPRESSED_PC_SCHEMAS = {COMPRESSED_POINTCLOUD2_SCHEMA, FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA}


class _OutputChannelMixin:
    """Shared lazy schema/channel registration + CDR encoding for one output schema."""

    def _init_output(self, schema_name: str, schema_data: bytes) -> None:
        self._out_schema_name = schema_name
        self._out_schema_data = schema_data
        self._encoder_factory = ROS2EncoderFactory()
        self._out_schema_id: int | None = None
        self._out_channels: dict[int, int] = {}  # input channel id -> output channel id
        self._encoder: Callable[[Any], bytes | memoryview] | None = None

    def _output_channel_id(self, context: MessageContext, in_channel: Channel) -> int:
        cached = self._out_channels.get(in_channel.id)
        if cached is not None:
            return cached
        if self._out_schema_id is None:
            self._out_schema_id = context.input.register_schema(
                self._out_schema_name, "ros2msg", self._out_schema_data
            )
        out = context.input.register_channel(
            Channel(
                id=0,
                schema_id=self._out_schema_id,
                topic=in_channel.topic,
                message_encoding="cdr",
                metadata=dict(in_channel.metadata),
            )
        )
        self._out_channels[in_channel.id] = out.id
        return out.id

    def _encode(self, payload: Any) -> bytes | memoryview:
        if self._encoder is None:
            schema = Schema(
                id=0,
                name=self._out_schema_name,
                encoding="ros2msg",
                data=self._out_schema_data,
            )
            encoder = self._encoder_factory.encoder_for(schema)
            if encoder is None:
                raise ValueError(f"no CDR encoder for {self._out_schema_name!r}")
            self._encoder = encoder
        return self._encoder(payload)


class PointcloudDecompressProcessor(InputProcessor, _OutputChannelMixin):
    """CompressedPointCloud2 / CompressedPointCloud → PointCloud2."""

    def __init__(self) -> None:
        self._init_output(_POINTCLOUD2_SCHEMA, POINTCLOUD2.encode())
        self._factory = CompressedPointCloudDecompressFactory()
        # input channel id -> (channel, decode fn)
        self._targets: dict[int, tuple[Channel, Callable[[bytes | memoryview], Any]]] = {}
        self._streams_with_summary: set[int] = set()

    @override
    def prepare_input(self, context: InputContext) -> None:
        if context.summary is not None:
            self._streams_with_summary.add(context.stream_id)

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        if schema is not None and schema.name in _COMPRESSED_PC_SCHEMAS:
            decoder = self._factory.decoder_for(channel.message_encoding, schema)
            if decoder is not None:
                self._targets[channel.id] = (channel, decoder)
        return Action.CONTINUE

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        if context.input.stream_id not in self._streams_with_summary:
            return MessageScope.all()
        return MessageScope.channels(set(self._targets))

    @override
    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        target = self._targets.get(message.channel_id)
        if target is None:
            yield message
            return
        channel, decoder = target
        pc_dict = decoder(message.data)
        out_id = self._output_channel_id(context, channel)

        yield replace(message, channel_id=out_id, data=self._encode(pc_dict))


@dataclass(slots=True)
class _PendingVideo:
    log_time: int
    publish_time: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    stream_id: int
    input_channel_id: int | None


@dataclass(slots=True)
class _VideoChannelState:
    decompressor: VideoDecompressorProtocol
    out_channel_id: int
    pending: deque[_PendingVideo]


class VideoDecompressProcessor(InputProcessor, _OutputChannelMixin):
    """CompressedVideo → Image (raw) or CompressedImage (JPEG)."""

    def __init__(
        self,
        *,
        video_format: str = "compressed",
        jpeg_quality: int = 90,
        backend: EncoderMode = EncoderMode.AUTO,
    ) -> None:
        self._video_format = video_format
        self._jpeg_quality = jpeg_quality
        self._backend = backend
        if video_format == "compressed":
            self._init_output(_COMPRESSED_IMAGE_SCHEMA, COMPRESSED_IMAGE.encode())
        else:
            self._init_output(_IMAGE_SCHEMA, IMAGE.encode())
        self._decoder_factory = DecoderFactory()
        self._targets: dict[int, tuple[Channel, Schema]] = {}
        self._streams_with_summary: set[int] = set()
        self._states: dict[int, _VideoChannelState] = {}

    @override
    def prepare_input(self, context: InputContext) -> None:
        if context.summary is not None:
            self._streams_with_summary.add(context.stream_id)

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        if schema is not None and schema.name == COMPRESSED_VIDEO_SCHEMA:
            self._targets[channel.id] = (channel, schema)
        return Action.CONTINUE

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        if context.input.stream_id not in self._streams_with_summary:
            return MessageScope.all()
        return MessageScope.channels(set(self._targets))

    @override
    def on_message(
        self, context: MessageContext, message: Message
    ) -> Iterable[Message | MessageWithContext]:
        target = self._targets.get(message.channel_id)
        if target is None:
            yield message
            return
        channel, schema = target
        state = self._states.get(channel.id)
        if state is None:
            state = _VideoChannelState(
                decompressor=create_video_decompressor(
                    video_format=self._video_format,
                    jpeg_quality=self._jpeg_quality,
                    mode=self._backend,
                ),
                out_channel_id=self._output_channel_id(context, channel),
                pending=deque(),
            )
            self._states[channel.id] = state

        decoder = self._decoder_factory.decoder_for(channel.message_encoding, schema)
        if decoder is None:
            yield message
            return
        decoded = decoder(message.data)
        state.pending.append(
            _PendingVideo(
                log_time=message.log_time,
                publish_time=message.publish_time,
                stamp_sec=decoded.timestamp.sec,
                stamp_nanosec=decoded.timestamp.nanosec,
                frame_id=decoded.frame_id,
                stream_id=context.input.stream_id,
                input_channel_id=context.input_channel_id,
            )
        )
        frame = state.decompressor.decompress(bytes(decoded.data), decoded.format)
        if frame is not None and state.pending:
            yield self._emit(state, state.pending.popleft(), frame)

    @override
    def finalize(self) -> Iterable[MessageWithContext]:
        for state in self._states.values():
            for frame in state.decompressor.flush():
                if not state.pending:
                    break
                yield self._emit(state, state.pending.popleft(), frame)

    def _emit(
        self, state: _VideoChannelState, meta: _PendingVideo, frame: Any
    ) -> MessageWithContext:
        header = {
            "stamp": {"sec": meta.stamp_sec, "nanosec": meta.stamp_nanosec},
            "frame_id": meta.frame_id,
        }
        if frame.is_jpeg:
            payload: dict[str, Any] = {"header": header, "format": "jpeg", "data": frame.data}
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
        return MessageWithContext(
            message=Message(
                channel_id=state.out_channel_id,
                sequence=0,
                log_time=meta.log_time,
                publish_time=meta.publish_time,
                data=self._encode(payload),
            ),
            stream_id=meta.stream_id,
            input_channel_id=meta.input_channel_id,
        )
