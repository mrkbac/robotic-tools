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
from pymcap_cli.core.processors.utils import global_time_range

if TYPE_CHECKING:
    from small_mcap import Chunk, LazyChunk, Message, MessageIndex, Summary


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
        timestamp = global_start_ns
        while timestamp <= global_end_ns:
            self.boundaries.append(timestamp)
            timestamp += self.duration_ns
        if not self.boundaries or self.boundaries[-1] <= global_end_ns:
            self.boundaries.append(global_end_ns + 1)
        self.n_segments = len(self.boundaries) - 1

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
