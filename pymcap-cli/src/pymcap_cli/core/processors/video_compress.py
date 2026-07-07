"""Video compression as a pipeline processor (async, NVENC-parallel).

Transcodes raw ``sensor_msgs/Image`` and ``sensor_msgs/CompressedImage`` topics
to ``foxglove_msgs/CompressedVideo`` (H.264/H.265) in-place on the same topic.

Unlike a plain :class:`MessageTransformProcessor`, video encoding is stateful
and *buffered*: a hardware encoder returns a frame's packet several frames
later, so a frame in cannot map to a packet out one-for-one. This processor
therefore drives the encoder directly — one dedicated worker thread per topic
(so independent camera streams encode concurrently, budgeted by
``probe_max_concurrent_encoders`` to stay within the GPU's useful session
count), a per-topic FIFO pairing returned packets with the oldest pending
frame's metadata, and ``finalize()`` to flush the encoder tails at end of
stream. Output messages keep their source frame's ``log_time``; reads are
time-ordered, so per-topic order is all that must hold.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from mcap_codec_support._schemas import normalize_schema_name
from mcap_codec_support.video import (
    COMPRESSED_VIDEO_SCHEMA,
    FOXGLOVE_COMPRESSED_VIDEO,
    IMAGE_SCHEMAS,
    EncoderMode,
    VideoEncoderError,
    calculate_downscale_dimensions,
    create_video_compression_backend,
    get_software_encoder,
    probe_max_concurrent_encoders,
)
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from small_mcap import DecodedMessage, Message, Schema
from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChannelContext,
    ChunkContext,
    InputContext,
    InputProcessor,
    MessageContext,
    MessageScope,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mcap_codec_support._protocols import AnyVideoBackend, VideoEncoderProtocol
    from small_mcap import Channel

logger = logging.getLogger(__name__)

# Per-topic lookahead: frames submitted to a topic's encoder ahead of draining
# its packets. Keeps memory bounded while giving the encoder thread slack to
# run in parallel with the main loop.
_INFLIGHT_PER_TOPIC = 128


@dataclass(slots=True)
class _FrameMeta:
    """Metadata carried from an input frame to its (later) output packet."""

    log_time: int
    publish_time: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str


@dataclass(slots=True)
class _TopicState:
    encoder: VideoEncoderProtocol[Any]
    pool: ThreadPoolExecutor
    out_channel_id: int
    schema_name: str
    # (decode+encode future, the DecodedMessage) — the message is kept so a
    # hardware encoder that dies mid-stream can re-decode and retry the frame
    # on the software fallback.
    futures: deque[tuple[Future[bytes | None], DecodedMessage]]
    pending: deque[_FrameMeta]


class VideoCompressProcessor(InputProcessor):
    """Encode image topics to CompressedVideo, one encoder thread per topic."""

    def __init__(
        self,
        *,
        codec: str = "h264",
        quality: int = 28,
        encoder: str | None = None,
        scale: int | None = None,
        backend: EncoderMode = EncoderMode.AUTO,
    ) -> None:
        self._codec = codec
        self._quality = quality
        self._scale = scale
        self._backend: AnyVideoBackend = create_video_compression_backend(
            backend, codec, do_video=True
        )
        if encoder is not None:
            if not self._backend.test_encoder(encoder):
                raise VideoEncoderError(f"Encoder '{encoder}' not available on this system")
            self._encoder_name = encoder
        else:
            self._encoder_name = self._backend.resolve_encoder(codec)
        self._max_dedicated_pools = probe_max_concurrent_encoders(
            self._backend, self._encoder_name
        )

        self._decoder_factory = DecoderFactory()
        # channel_id -> (channel, schema, schema_name) for matched image channels
        self._targets: dict[int, tuple[Channel, Schema, str]] = {}
        self._streams_with_summary: set[int] = set()
        self._states: dict[str, _TopicState] = {}
        self._overflow_pool: ThreadPoolExecutor | None = None
        self._out_schema_id: int | None = None
        self._video_encoder_fn: Any = None
        # Decode and encode are two pipelined stages, as in standalone
        # roscompress: a shared pool decodes frames (stateless, parallel across
        # topics) while each topic's single encode thread consumes them in
        # order. Collapsing both onto one per-topic thread serializes them and
        # loses the decode/encode overlap.
        decode_workers = min(4, max(2, (os.cpu_count() or 4) - 2))
        self._decode_pool = ThreadPoolExecutor(max_workers=decode_workers)

    # ---------------------------------------------------------------- scoping
    @override
    def prepare_input(self, context: InputContext) -> None:
        if context.summary is not None:
            self._streams_with_summary.add(context.stream_id)

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        if schema is not None:
            name = normalize_schema_name(schema.name)
            if name in IMAGE_SCHEMAS:
                self._targets[channel.id] = (channel, schema, name)
        return Action.CONTINUE

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        if context.input.stream_id not in self._streams_with_summary:
            return MessageScope.all()
        return MessageScope.channels(set(self._targets))

    # ------------------------------------------------------------- per-message
    @override
    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        target = self._targets.get(message.channel_id)
        if target is None:
            yield message
            return
        channel, schema, schema_name = target
        dm = DecodedMessage(schema, channel, message, self._decode)

        state = self._states.get(channel.topic)
        if state is None:
            state = self._start_topic(context, channel, schema_name, dm)
            if state is None:
                # Encoder init failed — pass the raw frame through rather than lose it.
                yield message
                return

        # Cheap CDR decode (cached on dm) for the output timestamp/frame_id.
        decoded = dm.decoded_message
        state.pending.append(
            _FrameMeta(
                log_time=message.log_time,
                publish_time=message.publish_time,
                stamp_sec=decoded.header.stamp.sec,
                stamp_nanosec=decoded.header.stamp.nanosec,
                frame_id=decoded.header.frame_id,
            )
        )
        # Two-stage pipeline: decode on the shared pool (parallel), encode on the
        # topic's single thread (serial, stateful). The encode task blocks on its
        # decode future, but later frames' decodes run ahead on the pool — so
        # decode and encode overlap instead of serializing.
        decode_future = self._decode_pool.submit(self._decode_frame, dm, schema_name)
        encode_future = state.pool.submit(self._encode_frame, state.encoder, decode_future)
        state.futures.append((encode_future, dm))

        # Drain completed packets (bounded) and emit them in order.
        while len(state.futures) > _INFLIGHT_PER_TOPIC:
            emitted = self._drain_one(channel.topic, state)
            if emitted is not None:
                yield emitted

    @override
    def finalize(self) -> Iterable[Message]:
        for topic, state in self._states.items():
            # Drain everything still queued on the encoder thread.
            while state.futures:
                emitted = self._drain_one(topic, state)
                if emitted is not None:
                    yield emitted
            # Flush the encoder's internal buffer; each packet pairs with the
            # oldest still-pending frame, in order.
            try:
                packets = state.encoder.flush_packets()
            except VideoEncoderError:
                logger.exception("Failed to flush encoder for %s", topic)
                packets = []
            for packet in packets:
                if not state.pending:
                    break
                yield self._build_video_message(state, state.pending.popleft(), packet)
            state.pool.shutdown(wait=True)
        if self._overflow_pool is not None:
            self._overflow_pool.shutdown(wait=True)
        self._decode_pool.shutdown(wait=True)

    # ---------------------------------------------------------------- helpers
    def _decode(self, schema: Schema | None, channel: Channel, message: Message) -> Any:
        decoder = self._decoder_factory.decoder_for(channel.message_encoding, schema)
        if decoder is None:
            name = schema.name if schema else None
            raise VideoEncoderError(f"no CDR decoder for schema {name!r}")
        return decoder(message.data)

    def _start_topic(
        self,
        context: MessageContext,
        channel: Channel,
        schema_name: str,
        dm: DecodedMessage,
    ) -> _TopicState | None:
        _frame, width, height = self._backend.decode_image(dm, schema_name)
        pix_fmt = self._backend.get_pix_fmt(channel.topic)
        if self._scale is not None:
            width, height = calculate_downscale_dimensions(width, height, self._scale)
        width -= width % 2
        height -= height % 2
        try:
            encoder = self._backend.create_encoder(
                width,
                height,
                self._encoder_name,
                self._quality,
                input_pix_fmt=pix_fmt,
                scale=(width, height) if pix_fmt is None and self._scale is not None else None,
            )
        except VideoEncoderError:
            logger.exception("Failed to create encoder for %s", channel.topic)
            return None

        if self._out_schema_id is None:
            self._out_schema_id = context.input.register_schema(
                COMPRESSED_VIDEO_SCHEMA, "ros2msg", FOXGLOVE_COMPRESSED_VIDEO.encode()
            )
        out_channel = context.input.register_channel(
            replace(channel, schema_id=self._out_schema_id, metadata=dict(channel.metadata))
        )

        # Only the first few topics get a dedicated encoder thread; the rest
        # share one overflow thread (see probe_max_concurrent_encoders).
        if len(self._states) < self._max_dedicated_pools:
            pool = ThreadPoolExecutor(max_workers=1)
        else:
            if self._overflow_pool is None:
                self._overflow_pool = ThreadPoolExecutor(max_workers=1)
            pool = self._overflow_pool

        logger.info(
            "[green]✓[/green] Converting %s: %dx%d (%s → CompressedVideo)",
            channel.topic,
            width,
            height,
            schema_name,
        )
        state = _TopicState(
            encoder=encoder,
            pool=pool,
            out_channel_id=out_channel.id,
            schema_name=schema_name,
            futures=deque(),
            pending=deque(),
        )
        self._states[channel.topic] = state
        return state

    def _decode_frame(self, dm: DecodedMessage, schema_name: str) -> Any:
        """Decode one image to a frame — runs on the shared decode pool."""
        return self._backend.decode_image(dm, schema_name)[0]

    def _encode_frame(
        self, encoder: VideoEncoderProtocol[Any], decode_future: Future[Any]
    ) -> bytes | None:
        """Encode a decoded frame — runs on the topic's single encode thread.

        Blocks on the decode future, but the decode pool has already run later
        frames' decodes ahead, so this rarely waits.
        """
        return encoder.encode(decode_future.result())

    def _drain_one(self, topic: str, state: _TopicState) -> Message | None:
        """Resolve the oldest in-flight decode+encode; emit a packet if one came out."""
        future, dm = state.futures.popleft()
        try:
            video_data = future.result()
        except VideoEncoderError:
            video_data = self._fallback_encode(topic, state, dm)
        if video_data is None:
            return None
        return self._build_video_message(state, state.pending.popleft(), video_data)

    def _fallback_encode(
        self, topic: str, state: _TopicState, dm: DecodedMessage
    ) -> bytes | None:
        """Swap a failed hardware encoder to software once, then retry the frame."""
        sw = get_software_encoder(self._codec)
        cfg = state.encoder.config
        if cfg.codec_name == sw:
            raise VideoEncoderError(f"software encoder {sw} failed for {topic}")
        logger.warning("Encoder failed for %s, falling back to %s", topic, sw)
        pix_fmt = self._backend.get_pix_fmt(topic)
        state.encoder = self._backend.create_encoder(
            cfg.width,
            cfg.height,
            sw,
            self._quality,
            input_pix_fmt=pix_fmt,
            scale=(cfg.width, cfg.height) if pix_fmt is None and self._scale else None,
        )
        frame: Any = self._backend.decode_image(dm, state.schema_name)[0]
        return state.encoder.encode(frame)

    def _build_video_message(
        self, state: _TopicState, meta: _FrameMeta, video_data: bytes
    ) -> Message:
        payload = {
            "timestamp": {"sec": meta.stamp_sec, "nanosec": meta.stamp_nanosec},
            "frame_id": meta.frame_id,
            "data": video_data,
            "format": self._codec,
        }
        return Message(
            channel_id=state.out_channel_id,
            sequence=0,
            log_time=meta.log_time,
            publish_time=meta.publish_time,
            data=self._encode_video(payload),
        )

    def _encode_video(self, payload: dict[str, Any]) -> bytes | memoryview:
        if self._video_encoder_fn is None:
            factory = ROS2EncoderFactory()
            schema = Schema(
                id=0,
                name=COMPRESSED_VIDEO_SCHEMA,
                encoding="ros2msg",
                data=FOXGLOVE_COMPRESSED_VIDEO.encode(),
            )
            encoder = factory.encoder_for(schema)
            if encoder is None:
                raise VideoEncoderError("no CDR encoder for CompressedVideo")
            self._video_encoder_fn = encoder
        return self._video_encoder_fn(payload)
