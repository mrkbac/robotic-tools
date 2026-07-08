# ruff: noqa: ARG002
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag, auto
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    from small_mcap import (
        Attachment,
        Channel,
        Chunk,
        ChunkIndex,
        CompressionType,
        LazyChunk,
        Message,
        MessageIndex,
        Metadata,
        Schema,
        Statistics,
        Summary,
    )

OutputKey = int | str | tuple[int | str, ...]
RouteKey = int | str


@dataclass(frozen=True, slots=True)
class OutputSegmentInfo:
    key: OutputKey
    start_time: int
    end_time: int


@dataclass(frozen=True, slots=True)
class InputContext:
    stream_id: int
    summary: Summary | None
    statistics: Statistics | None
    chunk_indexes: tuple[ChunkIndex, ...] | None
    remap_channel: Callable[[Channel], Channel]
    remap_message: Callable[[Message], Message]
    # Register a new output-only Channel (e.g. for topic aliasing). The
    # returned Channel has a freshly-assigned id and is added to the
    # writer's channel registry; the processor is responsible for emitting
    # messages on the new id (typically by yielding additional messages from
    # ``on_message``).
    # Register a new output-only Channel: ``register_channel(channel,
    # source_channel_id=None)``. Pass ``source_channel_id`` (the input channel a
    # transcode derives from) to get a deterministic output id that is stable
    # across independent runs of the same input — so parallel time-window outputs
    # share a channel table and ``merge`` can fast-copy them.
    register_channel: Callable[..., Channel]
    # Register a new output-only Schema and return its assigned id, for
    # transcode processors whose output schema (e.g. CompressedVideo) has no
    # input counterpart. Deduped by (name, encoding, data). Pair the returned
    # id with a ``register_channel`` call to emit messages on it.
    register_schema: Callable[[str, str, bytes], int]


@dataclass(frozen=True, slots=True)
class PipelineContext:
    inputs: tuple[InputContext, ...]
    output_segments: tuple[OutputSegmentInfo, ...]


@dataclass(frozen=True, slots=True)
class ChannelContext:
    input: InputContext
    input_channel_id: int


@dataclass(frozen=True, slots=True)
class MessageContext:
    input: InputContext
    input_channel_id: int | None


@dataclass(frozen=True, slots=True)
class ChunkContext:
    input: InputContext
    message_indexes: tuple[MessageIndex, ...] | None
    # Chunk log-time bounds when known (None for streamed/unindexed inputs).
    # Lets a time-window processor report that a wholly-inside chunk needs no
    # per-message work, so the dispatcher can fast-path its messages.
    chunk_start_time: int | None = None
    chunk_end_time: int | None = None


@dataclass(frozen=True, slots=True)
class SegmentContext:
    key: OutputKey
    start_time: int
    observed_message: Message | None = None


class _SplitRequiredSentinel:
    __slots__ = ()


SPLIT_REQUIRED: Final = _SplitRequiredSentinel()


class Action(IntFlag):
    """Record-level action for input processors."""

    CONTINUE = 0
    SKIP = auto()


class ChunkDecision(Enum):
    """Chunk-level processing decision.

    Processors typically return CONTINUE, SKIP, or DECODE from ``on_chunk()``.
    DECODE_VERIFY is also accepted: it asks the dispatcher to decompress the
    chunk, check that any embedded Schema/Channel records still match the
    writer's view, and fast-copy if clean — used by processors that mutate
    Channel/Schema records (e.g. ``TopicRewriteProcessor``) where the chunk
    body may or may not contain a stale embedded copy. RECOMPRESS is
    produced exclusively by the dispatcher based on compression-target
    state and should not be returned by processors.
    """

    CONTINUE = auto()
    SKIP = auto()
    DECODE = auto()
    # Chunk data must be re-compressed (different target compression) but no
    # per-message work is needed — just decompress + re-compress the chunk's
    # data bytes. Avoids parsing/re-emitting every record. Internal only.
    RECOMPRESS = auto()
    # Stream had an id remap, but everything else looks clean. Decode the
    # chunk, verify its in-chunk Schema/Channel records match the writer's
    # view, and fast-copy if so; otherwise fall through to DECODE.
    # Internal only.
    DECODE_VERIFY = auto()


class MessageScopeKind(Enum):
    ALL = auto()
    NONE = auto()
    CHANNELS = auto()


@dataclass(frozen=True, slots=True)
class MessageScope:
    kind: MessageScopeKind
    channel_ids: frozenset[int] = frozenset()

    @classmethod
    def all(cls) -> MessageScope:
        return cls(MessageScopeKind.ALL)

    @classmethod
    def none(cls) -> MessageScope:
        return cls(MessageScopeKind.NONE)

    @classmethod
    def channels(cls, channel_ids: set[int] | frozenset[int]) -> MessageScope:
        return cls(MessageScopeKind.CHANNELS, frozenset(channel_ids))


def chunk_decision_for_message_scope(scope: MessageScope, context: ChunkContext) -> ChunkDecision:
    if scope.kind is MessageScopeKind.NONE:
        return ChunkDecision.CONTINUE
    if scope.kind is MessageScopeKind.ALL:
        return ChunkDecision.DECODE

    indexes = context.message_indexes
    if indexes is None:
        return ChunkDecision.DECODE
    if any(idx.channel_id in scope.channel_ids for idx in indexes):
        return ChunkDecision.DECODE
    return ChunkDecision.CONTINUE


class InputProcessor:
    """Input-side processor.

    Input processors observe, transform, drop, or fan out input records before
    output routing happens. ``on_message`` yields zero or more messages; every
    yielded message continues at the next processor in the chain.
    """

    def initialize(self, context: PipelineContext) -> None:
        """Called before processing with all resources known to the pipeline."""

    def prepare_input(self, context: InputContext) -> None:
        """Called once per input stream after summaries/channels are known."""

    def message_scope(self, context: ChunkContext) -> MessageScope:
        """Messages that must pass through ``on_message`` for correctness.

        The default is conservative: decode every chunk so custom processors
        that only override ``on_message`` behave correctly. Processors that do
        not inspect messages should return ``MessageScope.none()``; processors
        keyed to known channels should return ``MessageScope.channels(ids)``.
        """
        return MessageScope.all()

    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        """Default: derive the chunk decision from ``message_scope``."""
        return chunk_decision_for_message_scope(self.message_scope(context), context)

    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        return Action.CONTINUE

    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        """Yield 0+ messages to forward downstream.

        - Yield nothing → drop the message.
        - Yield one message → pass through or replace the message.
        - Yield multiple messages → fan out. Every yielded message enters the
          chain at the next processor; this processor is not re-invoked for
          messages it produced.
        """
        yield message

    def on_metadata(self, context: InputContext, metadata: Metadata) -> Action:
        return Action.CONTINUE

    def on_attachment(self, context: InputContext, attachment: Attachment) -> Action:
        return Action.CONTINUE

    def on_segment_open(self, context: SegmentContext) -> Iterable[tuple[int, Message]]:
        return ()

    def finalize(self) -> Iterable[Message]:
        """Flush any output buffered past the end of the input stream.

        Called once, after every input record has been consumed, for each
        unique processor in the chain. A processor that holds messages back
        during ``on_message`` — e.g. an async encoder still draining frames
        from a background thread, or one that buffers to reorder — emits the
        remainder here. Yielded messages are treated as fully-formed output
        records: they are routed and written directly, not fed back through
        the processor chain. Default: emit nothing.
        """
        return ()


class OutputRouter:
    """Output-side router.

    Routers decide which output segment(s) a surviving message or fast-copied
    chunk should be written to. Yielding multiple route keys duplicates the
    record across those segments.
    """

    def initialize(self, context: PipelineContext) -> None:
        """Called before processing with all resources known to the pipeline."""

    def on_channel(self, context: ChannelContext, channel: Channel, schema: Schema | None) -> None:
        """Observe channels discovered without summaries."""

    def message_scope(self, context: ChunkContext) -> MessageScope:
        """Messages that must route through ``route_message`` for correctness.

        The default is conservative: custom routers that only implement
        ``route_message`` decode chunks. Routers with complete ``route_chunk``
        handling should return ``MessageScope.none()`` or override
        ``on_chunk`` directly for richer boundary logic.
        """
        return MessageScope.all()

    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        return chunk_decision_for_message_scope(self.message_scope(context), context)

    def route_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> Iterable[RouteKey] | _SplitRequiredSentinel:
        return ()

    def route_message(self, context: MessageContext, message: Message) -> Iterable[RouteKey]:
        return ()

    def output_segments(self) -> tuple[OutputSegmentInfo, ...] | None:
        """Return statically-known output segments, or None for dynamic routing."""
        return None


class OutputProcessor:
    """Output-side processor.

    Output processors run on the write side, deciding how surviving messages
    are grouped into chunks within each segment. ``chunk_group_key`` returns a
    hashable identifier; channels yielding equal keys share one chunk group,
    distinct keys get distinct chunk groups. Returning ``None`` opts out (the
    processor has no opinion).

    When multiple output processors are configured, their keys are composed
    into a tuple; an all-``None`` tuple collapses to a single shared group.
    """

    def initialize(self, context: PipelineContext) -> None:
        """Called before processing with all resources known to the pipeline."""

    def chunk_group_key(
        self,
        segment_key: OutputKey,
        channel: Channel,
        schema: Schema | None,
    ) -> Hashable | None:
        """Return the chunk-group identifier for ``channel`` in ``segment_key``."""
        return None

    def chunk_compression(
        self,
        segment_key: OutputKey,
        channel: Channel,
        schema: Schema | None,
    ) -> CompressionType | None:
        """Return a compression override for the group ``channel`` joins.

        Checked once, when a new chunk group is created for a channel's
        composite key. Return ``None`` to defer to the run's default
        compression; the first non-``None`` answer across the configured
        processor chain wins. Lets a grouper mark its own groups as, e.g.,
        already-compressed payloads (video, point clouds) that gain nothing
        from an extra compression pass.
        """
        return None
