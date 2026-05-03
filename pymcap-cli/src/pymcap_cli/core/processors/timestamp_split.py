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
        valid = [point for point in self.split_points if global_start_ns < point <= global_end_ns]
        self.boundaries = [global_start_ns, *valid, global_end_ns + 1]
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
