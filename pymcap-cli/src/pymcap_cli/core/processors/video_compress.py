"""Video compression as a pipeline processor (async, hardware-accelerated).

Transcodes raw ``sensor_msgs/Image`` and ``sensor_msgs/CompressedImage`` topics
to ``foxglove_msgs/CompressedVideo`` (H.264/H.265) in-place on the same topic.

Unlike a plain :class:`MessageTransformProcessor`, video encoding is stateful
and *buffered*: a hardware encoder returns a frame's packet a couple of frames
later, so a frame in cannot map to a packet out one-for-one. Each image topic is
encoded as one continuous stream by a dedicated single-thread encoder, while a
shared pool decodes frames ahead of it (stateless, parallel) so decode overlaps
encode. Independent camera topics therefore encode concurrently. ``finalize()``
flushes each encoder's tail at end of stream.

Output messages keep their source frame's ``log_time``; reads are time-ordered,
so only per-topic frame order must hold — which it does, since one encoder emits
one topic's packets in frame order (``max_b_frames = 0``).

To compress many time windows of one recording in parallel and merge the
results, run several of these in separate processes over disjoint ``--start``/
``--end`` ranges: each window's stream is self-contained (its first frame is an
IDR), so the merged output stays valid.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from hashlib import sha256
from threading import Event
from typing import TYPE_CHECKING, Any, Protocol

from mcap_codec_support._schemas import normalize_schema_name
from mcap_codec_support.video import (
    COMPRESSED_VIDEO_SCHEMA,
    FOXGLOVE_COMPRESSED_VIDEO,
    IMAGE_SCHEMAS,
    CorruptImageError,
    EncoderMode,
    VideoEncoderError,
    calculate_downscale_dimensions,
    create_video_compression_backend,
    get_software_encoder,
)
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from small_mcap import DecodedMessage, Message, Schema
from typing_extensions import override

from pymcap_cli.core.message_filter import ALL_TOPICS, TopicSelection
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
    from collections.abc import Iterable

    from mcap_codec_support._protocols import (
        AnyVideoBackend,
        DecodableImageMessage,
        VideoEncoderProtocol,
    )
    from small_mcap import Channel

logger = logging.getLogger(__name__)

# Per-topic lookahead: frames submitted to a topic's encoder ahead of draining
# its packets. Bounds decoded frames held in flight while giving the encoder
# thread slack to run in parallel with the main loop.
_DEFAULT_INFLIGHT_PER_TOPIC = 128
_GSTREAMER_INFLIGHT_PER_TOPIC = 8


def _decode_pool_size() -> int:
    """Return the automatic image-decode worker count."""
    return min(8, max(2, (os.cpu_count() or 4) - 2))


def split_decode_workers(
    processor_count: int,
    *,
    worker_budget: int | None = None,
) -> int | None:
    """Per-processor decode-pool size when several compressors run in one chain.

    Each processor owns its pool, so a chain of per-topic compressors would
    otherwise start ``processor_count`` full-size pools and oversubscribe the
    CPU. ``None`` keeps the undivided default for the single-processor case.
    """
    if processor_count < 1:
        raise ValueError("processor_count must be positive")
    if worker_budget is not None and worker_budget < 1:
        raise ValueError("worker_budget must be positive")
    if processor_count == 1:
        return worker_budget
    total_workers = _decode_pool_size() if worker_budget is None else worker_budget
    return max(1, total_workers // processor_count)


@dataclass(slots=True)
class _FrameMeta:
    """Metadata carried from an input frame to its (later) output packet."""

    log_time: int
    publish_time: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    stream_id: int
    input_channel_id: int | None
    topic: str
    sha256: str


@dataclass(frozen=True, slots=True)
class DroppedVideoFrame:
    """An independently rejected image, identified by its original payload."""

    topic: str
    log_time: int
    publish_time: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    reason: str
    sha256: str


@dataclass(slots=True)
class _QueuedFrame:
    meta: _FrameMeta
    future: Future[EncodeOutcome]


@dataclass(slots=True)
class _TopicState:
    session: VideoEncoderSession
    pool: ThreadPoolExecutor
    out_channel_id: int
    schema_name: str
    futures: deque[_QueuedFrame]
    pending: deque[_FrameMeta]


@dataclass(frozen=True, slots=True)
class ResolvedVideoCompressionBackend:
    """Backend plus encoder name selected with roscompress semantics."""

    backend: AnyVideoBackend
    encoder_name: str


class VideoGeometryBackend(Protocol):
    def get_pix_fmt(self, topic: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class VideoEncoderSettings:
    """Encoder input geometry, pixel format, and optional output scaling."""

    width: int
    height: int
    pix_fmt: str | None
    scale_dims: tuple[int, int] | None


def resolve_video_compression_backend(
    *,
    codec: str,
    encoder: str | None,
    backend: EncoderMode,
) -> ResolvedVideoCompressionBackend:
    """Resolve the video backend and encoder exactly like roscompress."""
    if encoder is not None and backend is EncoderMode.AUTO:
        for candidate_mode in (
            EncoderMode.PYAV,
            EncoderMode.FFMPEG_CLI,
            EncoderMode.GSTREAMER,
        ):
            candidate = create_video_compression_backend(candidate_mode, codec, do_video=True)
            if candidate.test_encoder(encoder):
                return ResolvedVideoCompressionBackend(candidate, encoder)
        raise VideoEncoderError(f"Encoder '{encoder}' not available on this system")

    resolved_backend = create_video_compression_backend(backend, codec, do_video=True)
    if encoder is not None:
        if not resolved_backend.test_encoder(encoder):
            raise VideoEncoderError(f"Encoder '{encoder}' not available on this system")
        return ResolvedVideoCompressionBackend(resolved_backend, encoder)
    return ResolvedVideoCompressionBackend(
        resolved_backend, resolved_backend.resolve_encoder(codec)
    )


def resolve_video_encoder_settings(
    *,
    backend: VideoGeometryBackend,
    topic: str,
    width: int,
    height: int,
    scale: int | None,
) -> VideoEncoderSettings:
    """Resolve scaled video dimensions exactly like roscompress."""
    pix_fmt = backend.get_pix_fmt(topic)
    source_width, source_height = width, height
    if scale is not None:
        width, height = calculate_downscale_dimensions(width, height, scale)
    width -= width % 2
    height -= height % 2
    if pix_fmt is not None:
        # rawvideo has no frame headers: its input dimensions must describe
        # the submitted bytes, while resizing belongs to the output filter.
        return VideoEncoderSettings(
            width=source_width,
            height=source_height,
            pix_fmt=pix_fmt,
            scale_dims=(width, height)
            if (width, height) != (source_width, source_height)
            else None,
        )
    return VideoEncoderSettings(
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        scale_dims=(width, height) if scale is not None else None,
    )


def build_compressed_video_payload(
    *,
    codec: str,
    stamp_sec: int,
    stamp_nanosec: int,
    frame_id: str,
    data: bytes,
) -> dict[str, Any]:
    """Build a ``foxglove_msgs/CompressedVideo`` payload dict for CDR encoding."""
    return {
        "timestamp": {"sec": stamp_sec, "nanosec": stamp_nanosec},
        "frame_id": frame_id,
        "data": data,
        "format": codec,
    }


@dataclass(frozen=True, slots=True)
class EncodeOutcome:
    """Result of encoding one frame through a :class:`VideoEncoderSession`.

    ``swapped`` flags that this frame triggered a hardware→software fallback,
    so a buffered/async caller can detect frames lost with the dead encoder.
    ``failed`` means encoding did not accept this frame. Batch compression fails
    on either loss instead of treating an encoder failure as a corrupt image.
    """

    data: bytes | None
    swapped: bool
    failed: bool
    decode_error: CorruptImageError | None = None


class VideoEncoderSession:
    """One topic's stateful video encoder.

    The single encoder implementation shared by the batch
    :class:`VideoCompressProcessor` and the live ``bridge proxy``. It owns
    encoder creation, adaptive geometry (recreating the encoder when a frame's
    resolution or pixel format changes), and the hardware→software fallback
    after an encoder failure. Callers differ only in how they schedule the work:
    the proxy encodes each frame synchronously; the batch processor drives
    ``encode_with_fallback`` from a per-topic encode thread and uses
    ``EncodeOutcome.swapped`` to keep buffered-frame timestamps aligned.

    Encoding is not thread-safe: a single session must be driven from one
    thread at a time. ``abort()`` may be called from the coordinating thread.
    """

    def __init__(
        self,
        *,
        backend: AnyVideoBackend,
        encoder_name: str,
        codec: str,
        quality: int,
        topic: str,
        scale: int | None = None,
        ffmpeg_args: tuple[str, ...] = (),
    ) -> None:
        self._backend = backend
        self._encoder_name = encoder_name
        self._codec = codec
        self._quality = quality
        self._topic = topic
        self._scale = scale
        self._ffmpeg_args = ffmpeg_args
        self._encoder: VideoEncoderProtocol[Any] | None = None
        self._settings: VideoEncoderSettings | None = None
        self._is_aborting = Event()
        self._has_failed = False

    @property
    def backend(self) -> AnyVideoBackend:
        return self._backend

    @property
    def encoder(self) -> VideoEncoderProtocol[Any] | None:
        return self._encoder

    @property
    def codec(self) -> str:
        return self._codec

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def is_aborting(self) -> bool:
        return self._is_aborting.is_set()

    def abort(self) -> None:
        self._is_aborting.set()

    def decode_image(
        self, message: DecodableImageMessage, schema_name: str
    ) -> tuple[Any, int, int]:
        """Decode an image message to ``(frame, width, height)``."""
        return self._backend.decode_image(message, schema_name)

    def ensure_encoder(self, width: int, height: int) -> VideoEncoderProtocol[Any]:
        """Return an encoder sized for ``width``x``height``, (re)creating on change."""
        settings = resolve_video_encoder_settings(
            backend=self._backend,
            topic=self._topic,
            width=width,
            height=height,
            scale=self._scale,
        )
        if self._encoder is not None and settings == self._settings:
            return self._encoder
        recreating = self._encoder is not None
        self.close()
        # Record geometry before creating the encoder so that a creation failure
        # (e.g. a hardware encoder that probes available but cannot open on a
        # GPU-less host) still leaves ``swap_to_software`` able to retry at this
        # geometry — the first frame then falls back instead of passing through.
        self._settings = settings
        self._encoder = self._backend.create_encoder(
            settings.width,
            settings.height,
            self._encoder_name,
            self._quality,
            input_pix_fmt=settings.pix_fmt,
            scale=settings.scale_dims,
            extra_args=self._ffmpeg_args,
        )
        logger.info(
            "%s %s: %dx%d using %s",
            "Re-encoding" if recreating else "Encoding",
            self._topic,
            settings.width,
            settings.height,
            self._encoder_name,
        )
        return self._encoder

    def swap_to_software(self) -> bool:
        """Recreate the encoder on the software codec at the current geometry.

        Returns ``False`` if the encoder already runs the software codec (so a
        caller can tell whether this call is the one that dropped the dead
        hardware encoder). Usable both after a mid-stream encoder failure and
        after a failed initial creation (``ensure_encoder`` records the geometry
        before creating, so the retry has dimensions); raises only if no
        geometry has been resolved yet.
        """
        software = get_software_encoder(self._codec)
        if self._encoder is not None and self._encoder.config.codec_name == software:
            return False
        if self._settings is None:
            raise VideoEncoderError(
                f"cannot fall back to software before an encoder exists for {self._topic}"
            )
        settings = self._settings
        self.close()
        self._encoder_name = software
        self._encoder = self._backend.create_encoder(
            settings.width,
            settings.height,
            software,
            self._quality,
            input_pix_fmt=settings.pix_fmt,
            scale=settings.scale_dims,
            extra_args=self._ffmpeg_args,
        )
        self._settings = settings
        logger.warning("Encoder failed for %s, falling back to %s", self._topic, software)
        return True

    def encode_with_fallback(self, frame: Any, width: int, height: int) -> EncodeOutcome:
        """Encode one frame; stop after both hardware and software have failed."""
        if self._has_failed:
            return EncodeOutcome(data=None, swapped=False, failed=True)
        try:
            encoder = self.ensure_encoder(width, height)
            return EncodeOutcome(data=encoder.encode(frame), swapped=False, failed=False)
        except Exception:  # noqa: BLE001 — any encode/geometry failure retries on software
            if self.is_aborting:
                return EncodeOutcome(data=None, swapped=False, failed=True)
            logger.debug("Hardware encode failed for %s; trying software", self._topic)
        swapped = False
        try:
            swapped = self.swap_to_software()
            encoder = self._encoder
            if encoder is None:  # pragma: no cover - swap_to_software raises instead
                return EncodeOutcome(data=None, swapped=swapped, failed=True)
            return EncodeOutcome(data=encoder.encode(frame), swapped=swapped, failed=False)
        except Exception as exc:  # noqa: BLE001 — encoder failures disable this session
            self._has_failed = True
            self.close()
            logger.error(  # noqa: TRY400 — report the terminal failure once without a traceback
                "Video encoding stopped for %s: %s", self._topic, exc
            )
            return EncodeOutcome(data=None, swapped=swapped, failed=True)

    def close(self) -> None:
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None


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
        ffmpeg_args: tuple[str, ...] = (),
        topics: TopicSelection = ALL_TOPICS,
        decode_workers: int | None = None,
    ) -> None:
        if decode_workers is not None and decode_workers < 1:
            raise ValueError("decode_workers must be positive")
        self._codec = codec
        self._quality = quality
        self._scale = scale
        self._ffmpeg_args = ffmpeg_args
        self._topics = topics
        resolved = resolve_video_compression_backend(
            codec=codec,
            encoder=encoder,
            backend=backend,
        )
        self._backend = resolved.backend
        if ffmpeg_args and resolved.backend.label != "ffmpeg-cli":
            raise VideoEncoderError(
                "extra FFmpeg arguments require the ffmpeg-cli backend; use --backend ffmpeg-cli"
            )
        self._encoder_name = resolved.encoder_name
        self._prefetch_decode = resolved.backend.prefetch_supported
        self._inflight_per_topic = (
            _GSTREAMER_INFLIGHT_PER_TOPIC
            if resolved.backend.label == "gstreamer"
            else _DEFAULT_INFLIGHT_PER_TOPIC
        )

        self._decoder_factory = DecoderFactory()
        # channel_id -> (channel, schema, schema_name) for matched image channels
        self._targets: dict[int, tuple[Channel, Schema, str]] = {}
        self._streams_with_summary: set[int] = set()
        self._states: dict[str, _TopicState] = {}
        self._dropped_frames: list[DroppedVideoFrame] = []
        self._out_schema_id: int | None = None
        self._video_encoder_fn: Any = None
        # Decode and encode are two pipelined stages: a shared pool decodes
        # frames (stateless, parallel across topics) while each topic's single
        # encode thread consumes them in order, so decode and encode overlap.
        self._decode_pool = (
            ThreadPoolExecutor(
                max_workers=decode_workers if decode_workers is not None else _decode_pool_size()
            )
            if self._prefetch_decode
            else None
        )

    @property
    def matched_topics(self) -> frozenset[str]:
        """Image topics this processor selected, known after channels were read."""
        return frozenset(channel.topic for channel, _schema, _name in self._targets.values())

    @property
    def dropped_frames(self) -> tuple[DroppedVideoFrame, ...]:
        """Images rejected before encoding; unexplained encoder losses are errors."""
        return tuple(self._dropped_frames)

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
            if name in IMAGE_SCHEMAS and self._topics.selects(channel.topic):
                self._targets[channel.id] = (channel, schema, name)
        return Action.CONTINUE

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        if context.input.stream_id not in self._streams_with_summary:
            return MessageScope.all()
        return MessageScope.channels(set(self._targets))

    # ------------------------------------------------------------- per-message
    @override
    def on_message(
        self, context: MessageContext, message: Message
    ) -> Iterable[Message | MessageWithContext]:
        target = self._targets.get(message.channel_id)
        if target is None:
            yield message
            return
        channel, schema, schema_name = target
        dm = DecodedMessage(schema, channel, message, self._decode)

        decoded = dm.decoded_message
        meta = _FrameMeta(
            log_time=message.log_time,
            publish_time=message.publish_time,
            stamp_sec=decoded.header.stamp.sec,
            stamp_nanosec=decoded.header.stamp.nanosec,
            frame_id=decoded.header.frame_id,
            stream_id=context.input.stream_id,
            input_channel_id=context.input_channel_id,
            topic=channel.topic,
            sha256=sha256(bytes(decoded.data)).hexdigest(),
        )

        state = self._states.get(channel.topic)
        if state is None:
            try:
                state = self._start_topic(context, channel, schema_name, dm)
            except CorruptImageError as exc:
                self._record_corrupt_frame(meta, exc)
                return
            if state is None:
                # Encoder init failed — pass the raw frame through rather than lose it.
                yield message
                return

        # Two-stage pipeline: decode on the shared pool (parallel), encode on the
        # topic's single thread (serial, stateful). The encode task blocks on its
        # decode future, but later frames' decodes run ahead on the pool — so
        # decode and encode overlap instead of serializing.
        if self._decode_pool is None:
            encode_future = state.pool.submit(
                self._decode_encode_frame, state.session, dm, schema_name
            )
        else:
            decode_future = self._decode_pool.submit(self._decode_frame, dm, schema_name)
            encode_future = state.pool.submit(self._encode_frame, state.session, decode_future)
        state.futures.append(_QueuedFrame(meta, encode_future))

        # Drain completed packets (bounded) and emit them in order.
        while len(state.futures) > self._inflight_per_topic:
            emitted = self._drain_one(state)
            if emitted is not None:
                yield emitted

    @override
    def finalize(self) -> Iterable[MessageWithContext]:
        try:
            for topic, state in self._states.items():
                yield from self._finalize_topic(topic, state)
        except BaseException:
            self.abort()
            raise
        finally:
            for state in self._states.values():
                state.session.close()
                state.pool.shutdown(wait=False)
            if self._decode_pool is not None:
                self._decode_pool.shutdown(wait=False)

    @override
    def abort(self) -> None:
        for state in self._states.values():
            state.session.abort()
            for queued in state.futures:
                queued.future.cancel()
        if self._decode_pool is not None:
            self._decode_pool.shutdown(wait=False, cancel_futures=True)
        for state in self._states.values():
            state.pool.shutdown(wait=True, cancel_futures=True)
            state.session.close()
        if self._decode_pool is not None:
            self._decode_pool.shutdown(wait=True, cancel_futures=True)

    def _finalize_topic(self, topic: str, state: _TopicState) -> Iterable[MessageWithContext]:
        # Drain everything still queued on the encoder thread.
        while state.futures:
            emitted = self._drain_one(state)
            if emitted is not None:
                yield emitted
        # Flush the encoder's internal buffer; each packet pairs with the oldest
        # still-pending frame, in order.
        encoder = state.session.encoder
        packets = encoder.flush_packets() if encoder is not None else []
        for packet in packets:
            if not state.pending:
                raise VideoEncoderError(f"Encoder emitted an unexpected extra frame for {topic}")
            meta = state.pending.popleft()
            yield MessageWithContext(
                message=self._build_video_message(state, meta, packet),
                stream_id=meta.stream_id,
                input_channel_id=meta.input_channel_id,
            )
        if state.pending:
            raise VideoEncoderError(
                f"Encoder did not emit {len(state.pending)} accepted frame(s) for {topic}"
            )

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
        session = VideoEncoderSession(
            backend=self._backend,
            encoder_name=self._encoder_name,
            codec=self._codec,
            quality=self._quality,
            topic=channel.topic,
            scale=self._scale,
            ffmpeg_args=self._ffmpeg_args,
        )
        try:
            session.ensure_encoder(width, height)
        except VideoEncoderError:
            # A hardware encoder can probe available yet fail to open (e.g.
            # h264_nvenc on a GPU-less host). Fall back to software so the topic
            # still transcodes to CompressedVideo instead of passing raw frames.
            logger.warning(
                "Encoder %s unavailable for %s; falling back to software",
                self._encoder_name,
                channel.topic,
            )
            try:
                session.swap_to_software()
            except VideoEncoderError:
                logger.exception("Failed to create encoder for %s", channel.topic)
                return None

        if self._out_schema_id is None:
            self._out_schema_id = context.input.register_schema(
                COMPRESSED_VIDEO_SCHEMA, "ros2msg", FOXGLOVE_COMPRESSED_VIDEO.encode()
            )
        # Deterministic output channel id (derived from the source channel) so
        # the same input compressed in independent time windows yields a
        # byte-compatible channel table that ``merge`` can fast-copy.
        out_channel = context.input.register_channel(
            replace(channel, schema_id=self._out_schema_id, metadata=dict(channel.metadata)),
            channel.id,
        )
        state = _TopicState(
            session=session,
            pool=ThreadPoolExecutor(max_workers=1),
            out_channel_id=out_channel.id,
            schema_name=schema_name,
            futures=deque(),
            pending=deque(),
        )
        self._states[channel.topic] = state
        return state

    def _decode_frame(self, dm: DecodedMessage, schema_name: str) -> tuple[Any, int, int]:
        """Decode one image to ``(frame, width, height)`` — runs on the shared pool."""
        return self._backend.decode_image(dm, schema_name)

    def _encode_frame(
        self, session: VideoEncoderSession, decode_future: Future[tuple[Any, int, int]]
    ) -> EncodeOutcome:
        """Encode a decoded frame — runs on the topic's single encode thread.

        Blocks on the decode future (later frames' decodes have already run
        ahead on the shared pool, so this rarely waits), then hands off to the
        session, which owns the adaptive geometry and software fallback. Because
        this runs on the topic's one encode thread, the session's encoder-state
        mutations stay serialized.
        """
        try:
            frame, width, height = decode_future.result()
        except CorruptImageError as exc:
            return EncodeOutcome(data=None, swapped=False, failed=True, decode_error=exc)
        return session.encode_with_fallback(frame, width, height)

    def _decode_encode_frame(
        self,
        session: VideoEncoderSession,
        dm: DecodedMessage,
        schema_name: str,
    ) -> EncodeOutcome:
        """Decode/parse and encode on the topic thread for subprocess backends."""
        try:
            frame, width, height = self._decode_frame(dm, schema_name)
        except CorruptImageError as exc:
            return EncodeOutcome(data=None, swapped=False, failed=True, decode_error=exc)
        return session.encode_with_fallback(frame, width, height)

    def _drain_one(self, state: _TopicState) -> MessageWithContext | None:
        """Resolve the oldest in-flight encode; emit a packet if one came out.

        Rejected images never enter ``pending``. Metadata is queued only for
        accepted frames, so a corrupt input cannot consume a buffered frame's
        timestamp. A hardware swap losing accepted frames fails closed.
        """
        queued = state.futures.popleft()
        outcome = queued.future.result()
        if outcome.decode_error is not None:
            self._record_corrupt_frame(queued.meta, outcome.decode_error)
            return None
        if outcome.swapped and state.pending:
            raise VideoEncoderError(
                f"Encoder fallback lost {len(state.pending)} accepted frame(s) "
                f"for {queued.meta.topic}"
            )
        if outcome.failed:
            raise VideoEncoderError(f"Failed to encode accepted frame for {queued.meta.topic}")
        state.pending.append(queued.meta)
        if outcome.data is None:
            return None
        meta = state.pending.popleft()
        return MessageWithContext(
            message=self._build_video_message(state, meta, outcome.data),
            stream_id=meta.stream_id,
            input_channel_id=meta.input_channel_id,
        )

    def _record_corrupt_frame(self, meta: _FrameMeta, error: CorruptImageError) -> None:
        dropped = DroppedVideoFrame(
            topic=meta.topic,
            log_time=meta.log_time,
            publish_time=meta.publish_time,
            stamp_sec=meta.stamp_sec,
            stamp_nanosec=meta.stamp_nanosec,
            frame_id=meta.frame_id,
            reason=str(error),
            sha256=meta.sha256,
        )
        self._dropped_frames.append(dropped)
        logger.warning(
            "Dropping corrupt image topic=%s log_time=%d publish_time=%d "
            "stamp=%d.%09d frame_id=%r sha256=%s reason=%s",
            dropped.topic,
            dropped.log_time,
            dropped.publish_time,
            dropped.stamp_sec,
            dropped.stamp_nanosec,
            dropped.frame_id,
            dropped.sha256,
            dropped.reason,
        )

    def _build_video_message(
        self, state: _TopicState, meta: _FrameMeta, video_data: bytes
    ) -> Message:
        payload = build_compressed_video_payload(
            codec=self._codec,
            stamp_sec=meta.stamp_sec,
            stamp_nanosec=meta.stamp_nanosec,
            frame_id=meta.frame_id,
            data=video_data,
        )
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
