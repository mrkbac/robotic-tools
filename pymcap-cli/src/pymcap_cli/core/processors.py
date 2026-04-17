# ruff: noqa: ARG002
from __future__ import annotations

import bisect
from enum import Enum, IntFlag, auto
from typing import TYPE_CHECKING

from pymcap_cli.utils import MAX_INT64, compile_topic_patterns

if TYPE_CHECKING:
    from collections.abc import Callable

    from small_mcap import (
        Attachment,
        Channel,
        Chunk,
        LazyChunk,
        Message,
        MessageIndex,
        Metadata,
        Schema,
        Summary,
    )

# Sentinel: chunk spans multiple output segments, needs DECODE
SPLIT_REQUIRED: object = object()


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


class Processor:
    """Base processor - override methods as needed.

    Processors handle filtering (on_chunk, on_channel, on_message, etc.)
    and optionally output routing (route_chunk, route_message) for
    multi-output splitting.
    """

    def initialize(self, summaries: list[Summary | None]) -> None:
        """Called before processing with summaries from all input files.

        Override to precompute boundaries or other state. Summaries may be
        None for broken/truncated files (recovery mode). Use the helper
        ``global_time_range`` to extract the time range across summaries.
        """

    def on_chunk(
        self,
        chunk: Chunk | LazyChunk,
        indexes: list[MessageIndex],
    ) -> ChunkDecision:
        """Decide how to handle a chunk. Called before message-level processing."""
        return ChunkDecision.CONTINUE

    def on_channel(self, channel: Channel, schema: Schema | None) -> Action:
        return Action.CONTINUE

    def on_message(self, message: Message) -> Action:
        return Action.CONTINUE

    def on_metadata(self, metadata: Metadata) -> Action:
        return Action.CONTINUE

    def on_attachment(self, attachment: Attachment) -> Action:
        return Action.CONTINUE

    def route_chunk(self, chunk: Chunk | LazyChunk) -> int | str | object | None:
        """Return output key for entire chunk, SPLIT_REQUIRED if it spans
        multiple outputs, or None if this processor has no routing opinion."""
        return None

    def route_message(self, message: Message) -> int | str | None:
        """Return output key for a message, or None for default output."""
        return None

    def output_keys(self) -> list[int | str] | None:
        """Return all possible output keys for pre-creating files, or None if dynamic."""
        return None


def global_time_range(summaries: list[Summary | None]) -> tuple[int, int] | None:
    """Extract the global time range across all summaries.

    Returns (start_ns, end_ns) or None if no statistics are available.
    """
    start: int | None = None
    end: int | None = None
    for summary in summaries:
        if summary and summary.statistics:
            s = summary.statistics
            if start is None or s.message_start_time < start:
                start = s.message_start_time
            if end is None or s.message_end_time > end:
                end = s.message_end_time
    if start is not None and end is not None:
        return (start, end)
    return None


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

    def on_channel(self, channel: Channel, schema: Schema | None) -> Action:
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

    def on_metadata(self, metadata: Metadata) -> Action:
        return Action.CONTINUE if self.include else Action.SKIP


class AttachmentFilterProcessor(Processor):
    """Filter attachment records (include or exclude all)."""

    def __init__(self, include: bool = True) -> None:
        self.include = include

    def on_attachment(self, attachment: Attachment) -> Action:
        return Action.CONTINUE if self.include else Action.SKIP


class AlwaysDecodeProcessor(Processor):
    """Forces all chunks to be decoded."""

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        return ChunkDecision.DECODE


class TimeFilterProcessor(Processor):
    """Filter messages and attachments by time range.

    Pre-computes bounds with defaults for fast comparisons.
    Implements on_chunk for chunk-level skip/decode decisions.
    """

    def __init__(self, start_ns: int | None = None, end_ns: int | None = None) -> None:
        if start_ns is not None and end_ns is not None and start_ns >= end_ns:
            raise ValueError(f"start_ns ({start_ns}) must be less than end_ns ({end_ns})")
        # Pre-compute bounds with defaults for fast path
        self.start = start_ns if start_ns is not None else 0
        self.end = end_ns if end_ns is not None else MAX_INT64
        # Store whether bounds were explicitly set
        self._has_start = start_ns is not None
        self._has_end = end_ns is not None

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
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

    def on_message(self, message: Message) -> Action:
        if self.start <= message.log_time < self.end:
            return Action.CONTINUE
        return Action.SKIP

    def on_attachment(self, attachment: Attachment) -> Action:
        if self.start <= attachment.log_time < self.end:
            return Action.CONTINUE
        return Action.SKIP


class DurationSplitProcessor(Processor):
    """Split output every N nanoseconds.

    Preserves the COPY fast-path for chunks that fall entirely within one segment.
    Only chunks spanning a segment boundary are decoded.
    """

    def __init__(self, duration_ns: int) -> None:
        if duration_ns <= 0:
            raise ValueError(f"duration_ns must be positive, got {duration_ns}")
        self.duration_ns = duration_ns
        self.boundaries: list[int] = []
        self.n_segments: int = 0

    def initialize(self, summaries: list[Summary | None]) -> None:
        time_range = global_time_range(summaries)
        if time_range is None:
            return
        global_start_ns, global_end_ns = time_range
        self.boundaries = []
        t = global_start_ns
        while t <= global_end_ns:
            self.boundaries.append(t)
            t += self.duration_ns
        # Ensure we have at least the end boundary
        if not self.boundaries or self.boundaries[-1] <= global_end_ns:
            self.boundaries.append(global_end_ns + 1)
        self.n_segments = len(self.boundaries) - 1

    def _segment_index(self, timestamp_ns: int) -> int:
        idx = bisect.bisect_right(self.boundaries, timestamp_ns) - 1
        return max(0, min(idx, self.n_segments - 1))

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        if not self.boundaries:
            return ChunkDecision.CONTINUE
        start_seg = self._segment_index(chunk.message_start_time)
        end_seg = self._segment_index(chunk.message_end_time)
        if start_seg != end_seg:
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def route_chunk(self, chunk: Chunk | LazyChunk) -> int | object:
        if not self.boundaries:
            return 0
        start_seg = self._segment_index(chunk.message_start_time)
        end_seg = self._segment_index(chunk.message_end_time)
        if start_seg != end_seg:
            return SPLIT_REQUIRED
        return start_seg

    def route_message(self, message: Message) -> int:
        return self._segment_index(message.log_time)

    def output_keys(self) -> list[int | str] | None:
        return list(range(self.n_segments))


class TimestampSplitProcessor(Processor):
    """Split output at specific nanosecond timestamps.

    Given split points [t1, t2], creates 3 segments:
    [global_start, t1), [t1, t2), [t2, global_end].

    Preserves the COPY fast-path for non-boundary chunks.
    """

    def __init__(self, split_points: list[int]) -> None:
        if not split_points:
            raise ValueError("split_points must not be empty")
        self.split_points = sorted(split_points)
        self.boundaries: list[int] = []
        self.n_segments: int = 0

    def initialize(self, summaries: list[Summary | None]) -> None:
        time_range = global_time_range(summaries)
        if time_range is None:
            return
        global_start_ns, global_end_ns = time_range
        # Filter split points within the global range
        valid = [t for t in self.split_points if global_start_ns < t <= global_end_ns]
        self.boundaries = [global_start_ns, *valid, global_end_ns + 1]
        self.n_segments = len(self.boundaries) - 1

    def _segment_index(self, timestamp_ns: int) -> int:
        idx = bisect.bisect_right(self.boundaries, timestamp_ns) - 1
        return max(0, min(idx, self.n_segments - 1))

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        if not self.boundaries:
            return ChunkDecision.CONTINUE
        start_seg = self._segment_index(chunk.message_start_time)
        end_seg = self._segment_index(chunk.message_end_time)
        if start_seg != end_seg:
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def route_chunk(self, chunk: Chunk | LazyChunk) -> int | object:
        if not self.boundaries:
            return 0
        start_seg = self._segment_index(chunk.message_start_time)
        end_seg = self._segment_index(chunk.message_end_time)
        if start_seg != end_seg:
            return SPLIT_REQUIRED
        return start_seg

    def route_message(self, message: Message) -> int:
        return self._segment_index(message.log_time)

    def output_keys(self) -> list[int | str] | None:
        return list(range(self.n_segments))


class ExpressionSplitProcessor(Processor):
    """Split output based on an arbitrary callable.

    The callable receives a Message and a dict of channels, and returns
    a segment key (int or str). Always forces chunk decoding since the
    expression must evaluate each message.

    Output keys are discovered dynamically (lazy segment creation).
    """

    def __init__(
        self,
        fn: Callable[[Message, dict[int, Channel]], int | str],
        channels: dict[int, Channel] | None = None,
    ) -> None:
        self.fn = fn
        self.channels: dict[int, Channel] = channels or {}

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        return ChunkDecision.DECODE

    def route_chunk(self, chunk: Chunk | LazyChunk) -> object:
        return SPLIT_REQUIRED

    def route_message(self, message: Message) -> int | str:
        return self.fn(message, self.channels)

    def output_keys(self) -> None:
        return None
