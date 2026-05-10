# ruff: noqa: ARG002
from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import (
    SPLIT_REQUIRED,
    ChunkDecision,
    Processor,
    _SplitRequiredSentinel,
)

if TYPE_CHECKING:
    from small_mcap import Chunk, LazyChunk, Message, MessageIndex


class BoundarySplitProcessor(Processor):
    """Shared chunk/message routing for processors backed by time boundaries."""

    def __init__(self) -> None:
        self.boundaries: list[int] = []
        self.n_segments: int = 0

    def _segment_index(self, timestamp_ns: int) -> int:
        index = bisect.bisect_right(self.boundaries, timestamp_ns) - 1
        return max(0, min(index, self.n_segments - 1))

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        if not self.boundaries:
            return ChunkDecision.CONTINUE
        start_segment = self._segment_index(chunk.message_start_time)
        end_segment = self._segment_index(chunk.message_end_time)
        if start_segment != end_segment:
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def route_chunk(self, chunk: Chunk | LazyChunk) -> int | _SplitRequiredSentinel:
        if not self.boundaries:
            return 0
        start_segment = self._segment_index(chunk.message_start_time)
        end_segment = self._segment_index(chunk.message_end_time)
        if start_segment != end_segment:
            return SPLIT_REQUIRED
        return start_segment

    def route_message(self, message: Message) -> int:
        return self._segment_index(message.log_time)

    def output_keys(self) -> list[int | str] | None:
        return list(range(self.n_segments))
