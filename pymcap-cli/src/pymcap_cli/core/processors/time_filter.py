# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import Action, ChunkDecision, Processor
from pymcap_cli.utils import MAX_INT64

if TYPE_CHECKING:
    from small_mcap import Attachment, Chunk, LazyChunk, Message, MessageIndex


class TimeFilterProcessor(Processor):
    """Filter messages and attachments by time range.

    Pre-computes bounds with defaults for fast comparisons.
    Implements on_chunk for chunk-level skip/decode decisions.
    """

    def __init__(self, start_ns: int | None = None, end_ns: int | None = None) -> None:
        if start_ns is not None and end_ns is not None and start_ns >= end_ns:
            raise ValueError(f"start_ns ({start_ns}) must be less than end_ns ({end_ns})")
        self.start = start_ns if start_ns is not None else 0
        self.end = end_ns if end_ns is not None else MAX_INT64
        self._has_start = start_ns is not None
        self._has_end = end_ns is not None

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        if self._has_start and chunk.message_end_time < self.start:
            return ChunkDecision.SKIP
        if self._has_end and chunk.message_start_time >= self.end:
            return ChunkDecision.SKIP
        if self._has_start and chunk.message_start_time < self.start:
            return ChunkDecision.DECODE
        if self._has_end and chunk.message_end_time >= self.end:
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def on_message(self, message: Message) -> Action:
        if self.start <= message.log_time < self.end:
            return Action.CONTINUE
        return Action.SKIP

    def on_attachment(self, attachment: Attachment) -> Action:
        if self.start <= attachment.log_time < self.end:
            return Action.CONTINUE
        return Action.SKIP
