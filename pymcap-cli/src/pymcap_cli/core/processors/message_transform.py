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

Subclasses implement ``matches`` (which containers to transform) and
``transform`` (decoded input → zero or more :class:`TransformOutput`; zero
drops the message). The base handles scoping so chunks with no matched
channel fast-copy verbatim, decode/encode, and channel/schema registration.

Adjacent ``MessageTransformProcessor`` instances are coalesced by
``build_input_processors`` into :class:`DecodedProcessorChain`. The chain
decodes each matched input message once, keeps intermediate outputs decoded
while they flow through downstream transforms, and only encodes final emitted
messages. A standalone processor still works through this base class.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from types import SimpleNamespace
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
    MessageScopeKind,
    MessageWithContext,
    PipelineContext,
)

if TYPE_CHECKING:
    from small_mcap import Message

    _PendingEntry = tuple[
        "Future[Iterable[TransformOutput] | None]",
        MessageContext,
        Message,
        Channel,
        Schema,
    ]
    _PendingChainEntry = tuple[
        "Future[tuple[_DecodedItem, ...]]",
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


@dataclass(frozen=True, slots=True)
class _DecodedItem:
    topic: str
    schema_name: str
    schema_encoding: str
    schema_data: bytes
    data: Any
    message_encoding: str
    modified: bool


_OutChannelKey = tuple[str, str, str, bytes, str]
_EncoderKey = tuple[str, str, bytes]
_Target = tuple[Channel, Schema, Callable[[bytes | memoryview], Any] | None]


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
        self._workers = workers
        # Matched input channels: id -> (channel, schema, decoder). The decoder
        # is resolved once here (not per message) so worker threads reuse a
        # ready reentrant decode function rather than racing on decoder_for.
        self._targets: dict[int, _Target] = {}
        # Streams whose channels were known up front (summary present); only
        # those can be scoped to specific channels. Streamed/unindexed inputs
        # stay pessimistic (decode everything) like TopicAliasProcessor.
        self._streams_with_summary: set[int] = set()
        self._out_channels: dict[_OutChannelKey, int] = {}
        self._encoders: dict[_EncoderKey, Callable[[Any], bytes | memoryview]] = {}

        self._pool: ThreadPoolExecutor | None = None
        self._max_inflight = max(workers, 1) * _MAX_INFLIGHT_MULTIPLIER
        # In submission order: (transform future, context, message, channel, schema).
        self._pending: deque[_PendingEntry] = deque()

    @property
    def worker_count(self) -> int:
        return self._workers

    def target_for_channel(self, channel_id: int) -> _Target | None:
        return self._targets.get(channel_id)

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
          current message should survive unchanged).
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
    def on_message(
        self, context: MessageContext, message: Message
    ) -> Iterable[Message | MessageWithContext]:
        target = self._targets.get(message.channel_id)
        if target is None:
            yield message
            return
        channel, schema, decoder = target
        if decoder is None:
            # Not CDR/ros2msg after all — pass through rather than corrupt.
            yield message
            return

        if self._workers <= 0:
            # Inline: decode + transform on the main thread.
            outputs = self._decode_transform(decoder, channel, schema, message)
            yield from self._emit(context, message, channel, schema, outputs)
            return

        # Offload decode + transform; emit results in submission order.
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self._workers)
        future = self._pool.submit(self._decode_transform, decoder, channel, schema, message)
        self._pending.append((future, context, message, channel, schema))
        while len(self._pending) > self._max_inflight:
            yield from self._drain_one()

    @override
    def finalize(self) -> Iterable[MessageWithContext]:
        while self._pending:
            yield from self._drain_one()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # --------------------------------------------------------------- helpers
    def _decode_transform(
        self,
        decoder: Callable[[bytes | memoryview], Any],
        channel: Channel,
        schema: Schema,
        message: Message,
    ) -> Iterable[TransformOutput] | None:
        """Decode + transform one message. Runs on a worker thread when pooled."""
        return self.transform(channel, schema, decoder(message.data))

    def _drain_one(self) -> Iterable[MessageWithContext]:
        future, context, message, channel, schema = self._pending.popleft()
        for out in self._emit(context, message, channel, schema, future.result()):
            yield MessageWithContext(
                message=out,
                stream_id=context.input.stream_id,
                input_channel_id=context.input_channel_id,
            )

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


class DecodedProcessorChain(InputProcessor):
    """Run adjacent decoded transforms with one decode and one final encode.

    This is an internal adapter built by ``build_input_processors``. It keeps
    ``MessageTransformProcessor`` subclasses simple while allowing value-edit
    and transcode processors to compose on the decoded payload without
    re-decoding an already-decoded message.
    """

    def __init__(self, processors: list[MessageTransformProcessor]) -> None:
        if not processors:
            raise ValueError("DecodedProcessorChain requires at least one processor")
        self._processors = tuple(processors)
        self._worker_count = max(processor.worker_count for processor in processors)
        self._pool: ThreadPoolExecutor | None = None
        self._max_inflight = max(self._worker_count, 1) * _MAX_INFLIGHT_MULTIPLIER
        self._pending: deque[_PendingChainEntry] = deque()
        self._out_channels: dict[_OutChannelKey, int] = {}
        self._encoder_factory = ROS2EncoderFactory()
        self._encoders: dict[_EncoderKey, Callable[[Any], bytes | memoryview]] = {}

    @override
    def initialize(self, context: PipelineContext) -> None:
        for processor in self._processors:
            processor.initialize(context)

    @override
    def prepare_input(self, context: InputContext) -> None:
        for processor in self._processors:
            processor.prepare_input(context)

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        action = Action.CONTINUE
        for processor in self._processors:
            action |= processor.on_channel(context, channel, schema)
        return action

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        channels: set[int] = set()
        for processor in self._processors:
            scope = processor.message_scope(context)
            if scope.kind is MessageScopeKind.ALL:
                return MessageScope.all()
            if scope.kind is MessageScopeKind.CHANNELS:
                channels.update(scope.channel_ids)
        return MessageScope.channels(channels)

    @override
    def on_message(
        self, context: MessageContext, message: Message
    ) -> Iterable[Message | MessageWithContext]:
        target = self._target_for(message.channel_id)
        if target is None:
            yield message
            return
        channel, schema, decoder = target
        if decoder is None:
            yield message
            return

        if self._worker_count <= 0:
            decoded = self._process_decoded(channel, schema, decoder, message)
            yield from self._emit_decoded(context, message, channel, schema, decoded)
            return

        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self._worker_count)
        future = self._pool.submit(self._process_decoded, channel, schema, decoder, message)
        self._pending.append((future, context, message, channel, schema))
        while len(self._pending) > self._max_inflight:
            yield from self._drain_one()

    @override
    def finalize(self) -> Iterable[Message | MessageWithContext]:
        while self._pending:
            yield from self._drain_one()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for processor in self._processors:
            yield from processor.finalize()

    def _target_for(self, channel_id: int) -> _Target | None:
        for processor in self._processors:
            target = processor.target_for_channel(channel_id)
            if target is not None and target[2] is not None:
                return target
        for processor in self._processors:
            target = processor.target_for_channel(channel_id)
            if target is not None:
                return target
        return None

    def _process_decoded(
        self,
        channel: Channel,
        schema: Schema,
        decoder: Callable[[bytes | memoryview], Any],
        message: Message,
    ) -> tuple[_DecodedItem, ...]:
        items = (
            _DecodedItem(
                topic=channel.topic,
                schema_name=schema.name,
                schema_encoding=schema.encoding,
                schema_data=schema.data,
                message_encoding=channel.message_encoding,
                data=decoder(message.data),
                modified=False,
            ),
        )
        for processor in self._processors:
            next_items: list[_DecodedItem] = []
            for item in items:
                item_channel, item_schema = self._virtual_container(channel, schema, item)
                if not processor.matches(item_channel, item_schema):
                    next_items.append(item)
                    continue
                outputs = processor.transform(item_channel, item_schema, item.data)
                if outputs is None:
                    next_items.append(item)
                    continue
                next_items.extend(
                    _DecodedItem(
                        topic=output.topic,
                        schema_name=output.schema_name,
                        schema_encoding=output.schema_encoding,
                        schema_data=output.schema_data,
                        message_encoding=output.message_encoding,
                        data=_decoded_view(output.data),
                        modified=True,
                    )
                    for output in outputs
                )
            items = tuple(next_items)
            if not items:
                break
        return items

    def _virtual_container(
        self, input_channel: Channel, input_schema: Schema, item: _DecodedItem
    ) -> tuple[Channel, Schema]:
        if _same_container(input_channel, input_schema, item):
            return input_channel, input_schema
        schema = Schema(
            id=0,
            name=item.schema_name,
            encoding=item.schema_encoding,
            data=item.schema_data,
        )
        channel = Channel(
            id=0,
            schema_id=0,
            topic=item.topic,
            message_encoding=item.message_encoding,
            metadata={},
        )
        return channel, schema

    def _drain_one(self) -> Iterable[MessageWithContext]:
        future, context, message, channel, schema = self._pending.popleft()
        for out in self._emit_decoded(context, message, channel, schema, future.result()):
            yield MessageWithContext(
                message=out,
                stream_id=context.input.stream_id,
                input_channel_id=context.input_channel_id,
            )

    def _emit_decoded(
        self,
        context: MessageContext,
        message: Message,
        input_channel: Channel,
        input_schema: Schema,
        items: tuple[_DecodedItem, ...],
    ) -> Iterable[Message]:
        for item in items:
            if not item.modified and _same_container(input_channel, input_schema, item):
                yield message
                continue
            out_channel_id = self._resolve_output_channel(
                context, message, input_channel, input_schema, item
            )
            yield replace(message, channel_id=out_channel_id, data=self._encode(item))

    def _resolve_output_channel(
        self,
        context: MessageContext,
        message: Message,
        in_channel: Channel,
        in_schema: Schema,
        item: _DecodedItem,
    ) -> int:
        if _same_container(in_channel, in_schema, item):
            return message.channel_id

        key: _OutChannelKey = (
            item.topic,
            item.schema_name,
            item.schema_encoding,
            item.schema_data,
            item.message_encoding,
        )
        cached = self._out_channels.get(key)
        if cached is not None:
            return cached
        schema_id = context.input.register_schema(
            item.schema_name, item.schema_encoding, item.schema_data
        )
        new_channel = context.input.register_channel(
            Channel(
                id=0,
                schema_id=schema_id,
                topic=item.topic,
                message_encoding=item.message_encoding,
                metadata={},
            ),
            in_channel.id,
        )
        self._out_channels[key] = new_channel.id
        return new_channel.id

    def _encode(self, item: _DecodedItem) -> bytes | memoryview:
        key: _EncoderKey = (item.schema_name, item.schema_encoding, item.schema_data)
        encoder = self._encoders.get(key)
        if encoder is None:
            schema = Schema(
                id=0,
                name=item.schema_name,
                encoding=item.schema_encoding,
                data=item.schema_data,
            )
            encoder = self._encoder_factory.encoder_for(schema)
            if encoder is None:
                raise ValueError(f"no CDR encoder for schema {item.schema_name!r}")
            self._encoders[key] = encoder
        return encoder(item.data)


def _same_container(channel: Channel, schema: Schema, item: _DecodedItem) -> bool:
    return (
        channel.topic == item.topic
        and schema.name == item.schema_name
        and schema.encoding == item.schema_encoding
        and schema.data == item.schema_data
        and channel.message_encoding == item.message_encoding
    )


def _decoded_view(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{str(key): _decoded_view(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_decoded_view(item) for item in value]
    return value
