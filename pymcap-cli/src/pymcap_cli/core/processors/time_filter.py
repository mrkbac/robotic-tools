# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import Action, ChunkDecision, Processor

if TYPE_CHECKING:
    from small_mcap import Attachment, Chunk, LazyChunk, Message, MessageIndex


class TimeFilterProcessor(Processor):
    """Filter messages and attachments by time range.

    ``None`` bounds mean "open-ended"; comparisons skip them.
    Implements on_chunk for chunk-level skip/decode decisions.
    """

    def __init__(self, start_ns: int | None = None, end_ns: int | None = None) -> None:
        if start_ns is not None and end_ns is not None and start_ns >= end_ns:
            raise ValueError(f"start_ns ({start_ns}) must be less than end_ns ({end_ns})")
        self.start_ns = start_ns
        self.end_ns = end_ns

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        if self.start_ns is not None and chunk.message_end_time < self.start_ns:
            return ChunkDecision.SKIP
        if self.end_ns is not None and chunk.message_start_time >= self.end_ns:
            return ChunkDecision.SKIP
        if self.start_ns is not None and chunk.message_start_time < self.start_ns:
            return ChunkDecision.DECODE
        if self.end_ns is not None and chunk.message_end_time >= self.end_ns:
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def on_message(self, message: Message) -> Action:
        if self.start_ns is not None and message.log_time < self.start_ns:
            return Action.SKIP
        if self.end_ns is not None and message.log_time >= self.end_ns:
            return Action.SKIP
        return Action.CONTINUE

    def on_attachment(self, attachment: Attachment) -> Action:
        if self.start_ns is not None and attachment.log_time < self.start_ns:
            return Action.SKIP
        if self.end_ns is not None and attachment.log_time >= self.end_ns:
            return Action.SKIP
        return Action.CONTINUE
