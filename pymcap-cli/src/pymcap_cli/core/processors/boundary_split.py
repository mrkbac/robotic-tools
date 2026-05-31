from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

from typing_extensions import override

from pymcap_cli.core.processors.base import (
    SPLIT_REQUIRED,
    ChunkContext,
    ChunkDecision,
    MessageContext,
    OutputRouter,
    OutputSegmentInfo,
    _SplitRequiredSentinel,
)

if TYPE_CHECKING:
    from small_mcap import Chunk, LazyChunk, Message


class BoundarySplitProcessor(OutputRouter):
    """Shared chunk/message routing for processors backed by time boundaries."""

    def __init__(self) -> None:
        self.boundaries: list[int] = []

    @property
    def n_segments(self) -> int:
        return max(0, len(self.boundaries) - 1)

    def _segment_index(self, timestamp_ns: int) -> int:
        index = bisect.bisect_right(self.boundaries, timestamp_ns) - 1
        return max(0, min(index, self.n_segments - 1))

    @override
    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        if not self.boundaries:
            return ChunkDecision.CONTINUE
        start_segment = self._segment_index(chunk.message_start_time)
        end_segment = self._segment_index(chunk.message_end_time)
        if start_segment != end_segment:
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    @override
    def route_chunk(
        self, context: ChunkContext, chunk: Chunk | LazyChunk
    ) -> tuple[int, ...] | _SplitRequiredSentinel:
        if not self.boundaries:
            return (0,)
        start_segment = self._segment_index(chunk.message_start_time)
        end_segment = self._segment_index(chunk.message_end_time)
        if start_segment != end_segment:
            return SPLIT_REQUIRED
        return (start_segment,)

    @override
    def route_message(self, context: MessageContext, message: Message) -> tuple[int, ...]:
        return (self._segment_index(message.log_time),)

    @override
    def output_segments(self) -> tuple[OutputSegmentInfo, ...] | None:
        return tuple(
            OutputSegmentInfo(
                key=i,
                start_time=self.boundaries[i],
                end_time=self.boundaries[i + 1],
            )
            for i in range(self.n_segments)
        )
