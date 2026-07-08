"""Base class for decode → transform → re-encode message processors.

Most existing processors are container-level: they route, drop, or relabel
messages without touching the payload. ``MessageTransformProcessor`` is the
payload-level counterpart — it CDR-decodes a matched message, hands the
decoded ROS object to a subclass hook, and re-encodes whatever the hook
returns. Two shapes fall out of the same base:

- **Transcode** — output schema/topic differs from input (e.g. PointCloud2 →
  CompressedPointCloud2). A new output channel/schema is registered lazily.
- **Value edit** — output schema/topic identical, field values changed. The
  base reuses the input channel so mixed edit/passthrough stays on one
  channel.

Subclasses implement ``matches`` (which channels to transform) and
``transform`` (decoded input → zero or more :class:`TransformOutput`; zero
drops the message). The base handles scoping so chunks with no matched
channel fast-copy verbatim, decode/encode, and channel/schema registration.

Decode cost: there is no cross-processor decode cache. Each processor decodes
the messages it matches, in its own ``on_message``. This is decode-once in the
intended composition — matchers are expected to target **disjoint** channel
sets (e.g. video vs point-cloud vs distinct value-editors), so any given
message matches at most one processor. A transcode also changes the schema, so
downstream transform processors stop matching it. The only double-decode case
is two processors matching the *same* channel with the container left
unchanged (two value-editors on one topic); add a shared per-message decode
cache if that composition ever becomes real.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from small_mcap import Channel, Schema
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
    from collections.abc import Callable, Iterable

    from small_mcap import Message

    _Decoder = Callable[[bytes | memoryview], Any]
    _PendingEntry = tuple[
        "Future[Iterable[TransformOutput] | None]",
        MessageContext,
        Message,
        Channel,
        Schema,
    ]

# How many transform tasks may be submitted ahead of the in-order drain when
# running with a worker pool — bounds memory and gives workers slack to run.
_MAX_INFLIGHT_MULTIPLIER = 4


@dataclass(frozen=True, slots=True)
class TransformOutput:
    """One output message produced from a decoded input message.

    ``data`` is a dict or object encodable against the named schema (the CDR
    encoder reads fields by ``dict.get`` or ``getattr``, so either works).
    """

    topic: str
    schema_name: str
    schema_encoding: str
    schema_data: bytes
    data: Any
    message_encoding: str = "cdr"


_OutChannelKey = tuple[str, str, str, bytes, str]
_EncoderKey = tuple[str, str, bytes]


class MessageTransformProcessor(InputProcessor):
    """Decode matched messages, run a subclass transform, re-encode the result.

    Pass ``workers > 0`` to run the (expensive) decode+``transform`` step on a
    worker pool while the main thread keeps reading, routing, and writing — the
    results are still emitted in input order and any tail is flushed in
    ``finalize()``. **When ``workers > 0``, ``transform`` (and the decode it
    wraps) run on multiple threads concurrently, so a subclass holding
    non-thread-safe state (e.g. a native compressor) must make it per-thread
    (``threading.local``).** With ``workers == 0`` (the default) everything runs
    inline on the main thread — right for cheap edits, where a handoff would
    cost more than the work.
    """

    def __init__(self, *, workers: int = 0) -> None:
        self._decoder_factory = DecoderFactory()
        self._encoder_factory = ROS2EncoderFactory()
        # Matched input channels: id -> (channel, schema, decoder). The decoder
        # is resolved once here (not per message) so worker threads reuse a
        # ready reentrant decode function rather than racing on decoder_for.
        self._targets: dict[int, tuple[Channel, Schema, _Decoder | None]] = {}
        # Streams whose channels were known up front (summary present); only
        # those can be scoped to specific channels. Streamed/unindexed inputs
        # stay pessimistic (decode everything) like TopicAliasProcessor.
        self._streams_with_summary: set[int] = set()
        self._out_channels: dict[_OutChannelKey, int] = {}
        self._encoders: dict[_EncoderKey, Callable[[Any], bytes | memoryview]] = {}

        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
        )
        self._max_inflight = max(workers, 1) * _MAX_INFLIGHT_MULTIPLIER
        # In submission order: (transform future, context, message, channel, schema).
        self._pending: deque[_PendingEntry] = deque()

    # ------------------------------------------------------------------ hooks
    def matches(self, channel: Channel, schema: Schema | None) -> bool:
        """Return True for channels this processor should transform."""
        raise NotImplementedError

    def transform(
        self, channel: Channel, schema: Schema, decoded: Any
    ) -> Iterable[TransformOutput] | None:
        """Turn one decoded input message into output(s).

        - Return ``None`` → pass the original message through unchanged (e.g.
          the transform inspected it but declined, or a codec failed and the
          raw message should survive).
        - Return an empty iterable → drop the message.
        - Return one or more :class:`TransformOutput` → replace / fan out.

        Because ``None`` is meaningful, implement this with ``return`` (a list
        or ``None``), not ``yield``. Must be thread-safe if ``workers > 0``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------- framework
    @override
    def prepare_input(self, context: InputContext) -> None:
        if context.summary is not None:
            self._streams_with_summary.add(context.stream_id)

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        if schema is not None and self.matches(channel, schema):
            decoder = self._decoder_factory.decoder_for(channel.message_encoding, schema)
            self._targets[channel.id] = (channel, schema, decoder)
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
        channel, schema, decoder = target
        if decoder is None:
            # Not CDR/ros2msg after all — pass through rather than corrupt.
            yield message
            return

        if self._pool is None:
            # Inline: decode + transform on the main thread.
            outputs = self._decode_transform(decoder, channel, schema, message)
            yield from self._emit(context, message, channel, schema, outputs)
            return

        # Offload decode + transform; emit results in submission order.
        future = self._pool.submit(self._decode_transform, decoder, channel, schema, message)
        self._pending.append((future, context, message, channel, schema))
        while len(self._pending) > self._max_inflight:
            yield from self._drain_one()

    @override
    def finalize(self) -> Iterable[Message]:
        while self._pending:
            yield from self._drain_one()
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    # --------------------------------------------------------------- helpers
    def _decode_transform(
        self, decoder: _Decoder, channel: Channel, schema: Schema, message: Message
    ) -> Iterable[TransformOutput] | None:
        """Decode + transform one message. Runs on a worker thread when pooled."""
        return self.transform(channel, schema, decoder(message.data))

    def _drain_one(self) -> Iterable[Message]:
        future, context, message, channel, schema = self._pending.popleft()
        yield from self._emit(context, message, channel, schema, future.result())

    def _emit(
        self,
        context: MessageContext,
        message: Message,
        channel: Channel,
        schema: Schema,
        outputs: Iterable[TransformOutput] | None,
    ) -> Iterable[Message]:
        """Resolve output channels + encode on the main thread, emit in order."""
        if outputs is None:
            yield message  # transform declined — keep the original
            return
        for out in outputs:
            out_channel_id = self._resolve_output_channel(context, message, channel, schema, out)
            yield replace(message, channel_id=out_channel_id, data=self._encode(out))

    # --------------------------------------------------------------- helpers
    def _resolve_output_channel(
        self,
        context: MessageContext,
        message: Message,
        in_channel: Channel,
        in_schema: Schema,
        out: TransformOutput,
    ) -> int:
        # Identical container (value edit): reuse the input channel so that
        # edited and passed-through messages on the same topic don't split
        # across two channels.
        in_key = (in_channel.topic, in_schema.name, in_schema.encoding, in_schema.data)
        out_container = (out.topic, out.schema_name, out.schema_encoding, out.schema_data)
        if in_key == out_container:
            return message.channel_id

        key: _OutChannelKey = (
            out.topic,
            out.schema_name,
            out.schema_encoding,
            out.schema_data,
            out.message_encoding,
        )
        cached = self._out_channels.get(key)
        if cached is not None:
            return cached
        schema_id = context.input.register_schema(
            out.schema_name, out.schema_encoding, out.schema_data
        )
        new_channel = context.input.register_channel(
            Channel(
                id=0,  # replaced by register_channel
                schema_id=schema_id,
                topic=out.topic,
                message_encoding=out.message_encoding,
                metadata={},
            ),
            in_channel.id,
        )
        self._out_channels[key] = new_channel.id
        return new_channel.id

    def _encode(self, out: TransformOutput) -> bytes | memoryview:
        key: _EncoderKey = (out.schema_name, out.schema_encoding, out.schema_data)
        encoder = self._encoders.get(key)
        if encoder is None:
            schema = Schema(
                id=0,
                name=out.schema_name,
                encoding=out.schema_encoding,
                data=out.schema_data,
            )
            encoder = self._encoder_factory.encoder_for(schema)
            if encoder is None:
                raise ValueError(f"no CDR encoder for schema {out.schema_name!r}")
            self._encoders[key] = encoder
        return encoder(out.data)
