# ruff: noqa: ARG002
"""Split output every N nanoseconds, anchored to the first message seen.

Streaming-anchor design — no summary required. The anchor is set lazily to
the `message_start_time` of the first chunk (or `log_time` of the first
message in DECODE paths) the processor observes; segments are emitted
dynamically. Works identically on indexed and unindexed inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import (
    SPLIT_REQUIRED,
    ChunkContext,
    ChunkDecision,
    MessageContext,
    OutputRouter,
    _SplitRequiredSentinel,
)

if TYPE_CHECKING:
    from small_mcap import Chunk, LazyChunk, Message


class DurationSplitProcessor(OutputRouter):
    """Split output every ``duration_ns`` nanoseconds from the first message."""

    def __init__(self, duration_ns: int) -> None:
        if duration_ns <= 0:
            msg = f"duration_ns must be positive, got {duration_ns}"
            raise ValueError(msg)
        self.duration_ns = duration_ns
        # Anchor is the earliest message_start_time observed; segments grow
        # dynamically from there. ``None`` until the first chunk/message
        # arrives.
        self._anchor_ns: int | None = None

    def _segment_index(self, timestamp_ns: int) -> int:
        if self._anchor_ns is None:
            self._anchor_ns = timestamp_ns
        return max(0, (timestamp_ns - self._anchor_ns) // self.duration_ns)

    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        start_seg = self._segment_index(chunk.message_start_time)
        end_seg = self._segment_index(chunk.message_end_time)
        if start_seg != end_seg:
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def route_chunk(
        self, context: ChunkContext, chunk: Chunk | LazyChunk
    ) -> tuple[int, ...] | _SplitRequiredSentinel:
        start_seg = self._segment_index(chunk.message_start_time)
        end_seg = self._segment_index(chunk.message_end_time)
        if start_seg != end_seg:
            return SPLIT_REQUIRED
        return (start_seg,)

    def route_message(self, context: MessageContext, message: Message) -> tuple[int, ...]:
        return (self._segment_index(message.log_time),)
