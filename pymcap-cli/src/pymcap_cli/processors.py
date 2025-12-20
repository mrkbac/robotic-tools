from dataclasses import dataclass
from enum import Enum, IntFlag, auto

from small_mcap import (
    Attachment,
    Channel,
    Chunk,
    LazyChunk,
    Message,
    MessageIndex,
    Metadata,
    Schema,
)

from pymcap_cli.utils import MAX_INT64, compile_topic_patterns


class Action(IntFlag):
    """Result actions - combinable flags."""

    CONTINUE = 0  # Continue processing
    SKIP = auto()  # Skip this item (don't write/register)
    STOP = auto()  # Stop pipeline for this item


class ChunkDecision(Enum):
    """Chunk-level processing decision."""

    CONTINUE = auto()  # No opinion, let other processors decide
    SKIP = auto()  # Skip chunk entirely (all content filtered out)
    DECODE = auto()  # Must decode to filter at message level


@dataclass(slots=True, frozen=True)
class Context:
    """Shared context passed to processors."""

    stream_id: int


class Processor:
    """Base processor - override methods as needed."""

    def on_chunk(
        self,
        _ctx: Context,
        _chunk: Chunk | LazyChunk,
        _indexes: list[MessageIndex],
    ) -> ChunkDecision:
        """Decide how to handle a chunk. Called before message-level processing.

        Receives LazyChunk for efficiency - only metadata is read until needed.
        """
        return ChunkDecision.CONTINUE

    def on_channel(
        self,
        _ctx: Context,
        _channel: Channel,
        _schema: Schema | None,
    ) -> Action:
        """Handle channel registration stage."""
        return Action.CONTINUE

    def on_message(self, _ctx: Context, _message: Message) -> Action:
        """Handle message record."""
        return Action.CONTINUE

    def on_metadata(self, _ctx: Context, _metadata: Metadata) -> Action:
        """Handle metadata record."""
        return Action.CONTINUE

    def on_attachment(self, _ctx: Context, _attachment: Attachment) -> Action:
        """Handle attachment record."""
        return Action.CONTINUE


class TopicFilterProcessor(Processor):
    """Filter channels by topic regex patterns.

    Compiles patterns internally. Uses search() for flexible matching.
    """

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self.include = compile_topic_patterns(include or [])
        self.exclude = compile_topic_patterns(exclude or [])

    def on_channel(
        self,
        _ctx: Context,
        channel: Channel,
        _schema: Schema | None,
    ) -> Action:
        # Include filter: if patterns exist, topic must match at least one
        if self.include:
            if any(p.search(channel.topic) for p in self.include):
                return Action.CONTINUE
            return Action.SKIP

        # Exclude filter: skip if topic matches any pattern
        if self.exclude and any(p.search(channel.topic) for p in self.exclude):
            return Action.SKIP

        return Action.CONTINUE


class MetadataFilterProcessor(Processor):
    """Filter metadata records (include or exclude all)."""

    def __init__(self, include: bool = True) -> None:
        self.include = include

    def on_metadata(self, _ctx: Context, _metadata: Metadata) -> Action:
        return Action.CONTINUE if self.include else Action.SKIP


class AttachmentFilterProcessor(Processor):
    """Filter attachment records (include or exclude all)."""

    def __init__(self, include: bool = True) -> None:
        self.include = include

    def on_attachment(self, _ctx: Context, _attachment: Attachment) -> Action:
        return Action.CONTINUE if self.include else Action.SKIP


class AlwaysDecodeProcessor(Processor):
    """Forces all chunks to be decoded."""

    def on_chunk(
        self,
        _ctx: Context,
        _chunk: Chunk | LazyChunk,
        _indexes: list[MessageIndex],
    ) -> ChunkDecision:
        return ChunkDecision.DECODE


class TimeFilterProcessor(Processor):
    """Filter messages and attachments by time range.

    Pre-computes bounds with defaults for fast comparisons.
    Implements on_chunk for chunk-level skip/decode decisions.
    """

    def __init__(self, start_ns: int | None = None, end_ns: int | None = None) -> None:
        # Pre-compute bounds with defaults for fast path
        self.start = start_ns if start_ns is not None else 0
        self.end = end_ns if end_ns is not None else MAX_INT64
        # Store whether bounds were explicitly set
        self._has_start = start_ns is not None
        self._has_end = end_ns is not None

    def on_chunk(
        self,
        _ctx: Context,
        chunk: Chunk | LazyChunk,
        _indexes: list[MessageIndex],
    ) -> ChunkDecision:
        # Chunk entirely outside time range - skip it
        if self._has_start and chunk.message_end_time < self.start:
            return ChunkDecision.SKIP
        if self._has_end and chunk.message_start_time >= self.end:
            return ChunkDecision.SKIP

        # Chunk partially overlaps - must decode to filter per-message
        if self._has_start and chunk.message_start_time < self.start:
            return ChunkDecision.DECODE
        if self._has_end and chunk.message_end_time >= self.end:
            return ChunkDecision.DECODE

        # Chunk entirely within range - no filtering needed
        return ChunkDecision.CONTINUE

    def on_message(self, _ctx: Context, message: Message) -> Action:
        if self.start <= message.log_time < self.end:
            return Action.CONTINUE
        return Action.SKIP

    def on_attachment(self, _ctx: Context, attachment: Attachment) -> Action:
        if self.start <= attachment.log_time < self.end:
            return Action.CONTINUE
        return Action.SKIP
