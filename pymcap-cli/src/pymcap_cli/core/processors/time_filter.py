# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import Action, ChunkDecision, Processor
from pymcap_cli.utils import RelativeTime

if TYPE_CHECKING:
    from small_mcap import Attachment, Chunk, LazyChunk, Message, MessageIndex, Summary


class TimeFilterProcessor(Processor):
    """Filter messages and attachments by time range.

    ``None`` bounds mean "open-ended"; comparisons skip them. Bounds may also
    be ``RelativeTime`` instances (e.g. ``@5s``, ``end-30s``); those are
    resolved against summary statistics in :py:meth:`initialize`.

    When ``invert`` is True the in-window decision is flipped: messages
    INSIDE [start, end] are SKIPped, messages outside CONTINUE.
    """

    def __init__(
        self,
        start_ns: int | RelativeTime | None = None,
        end_ns: int | RelativeTime | None = None,
        *,
        invert: bool = False,
    ) -> None:
        if isinstance(start_ns, int) and isinstance(end_ns, int) and start_ns >= end_ns:
            raise ValueError(f"start_ns ({start_ns}) must be less than end_ns ({end_ns})")
        self._start: int | RelativeTime | None = start_ns
        self._end: int | RelativeTime | None = end_ns
        self._invert = invert
        self.start_ns: int | None = start_ns if isinstance(start_ns, int) else None
        self.end_ns: int | None = end_ns if isinstance(end_ns, int) else None

    def initialize(self, summaries: list[Summary | None]) -> None:
        """Resolve any RelativeTime bounds against the input summaries."""
        if not (isinstance(self._start, RelativeTime) or isinstance(self._end, RelativeTime)):
            return
        file_start: int | None = None
        file_end: int | None = None
        for summary in summaries:
            if summary is None or summary.statistics is None:
                continue
            stats = summary.statistics
            if file_start is None or stats.message_start_time < file_start:
                file_start = stats.message_start_time
            file_end = (
                stats.message_end_time
                if file_end is None
                else max(file_end, stats.message_end_time)
            )
        if file_start is None or file_end is None:
            raise ValueError("Relative time bounds require MCAP summary statistics")
        if isinstance(self._start, RelativeTime):
            self.start_ns = self._start.resolve(file_start, file_end)
        if isinstance(self._end, RelativeTime):
            self.end_ns = self._end.resolve(file_start, file_end)
        if self.start_ns is not None and self.end_ns is not None and self.start_ns >= self.end_ns:
            raise ValueError(
                f"resolved start_ns ({self.start_ns}) must be less than "
                f"resolved end_ns ({self.end_ns})"
            )

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        if self._invert:
            # Inverted window: any chunk fully inside is skipped, fully outside
            # passes through. Spanning chunks must DECODE to filter per-message.
            if (
                self.start_ns is not None
                and self.end_ns is None
                and chunk.message_start_time >= self.start_ns
            ):
                return ChunkDecision.SKIP
            if (
                self.start_ns is None
                and self.end_ns is not None
                and chunk.message_end_time < self.end_ns
            ):
                return ChunkDecision.SKIP
            if (
                self.start_ns is not None
                and self.end_ns is not None
                and chunk.message_start_time >= self.start_ns
                and chunk.message_end_time < self.end_ns
            ):
                return ChunkDecision.SKIP
            if self.start_ns is not None and chunk.message_end_time < self.start_ns:
                return ChunkDecision.CONTINUE
            if self.end_ns is not None and chunk.message_start_time >= self.end_ns:
                return ChunkDecision.CONTINUE
            return ChunkDecision.DECODE

        if self.start_ns is not None and chunk.message_end_time < self.start_ns:
            return ChunkDecision.SKIP
        if self.end_ns is not None and chunk.message_start_time >= self.end_ns:
            return ChunkDecision.SKIP
        if self.start_ns is not None and chunk.message_start_time < self.start_ns:
            return ChunkDecision.DECODE
        if self.end_ns is not None and chunk.message_end_time >= self.end_ns:
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def _in_window(self, log_time: int) -> bool:
        if self.start_ns is not None and log_time < self.start_ns:
            return False
        return not (self.end_ns is not None and log_time >= self.end_ns)

    def on_message(self, message: Message) -> Action:
        inside = self._in_window(message.log_time)
        keep = inside ^ self._invert  # invert flips inside↔outside
        return Action.CONTINUE if keep else Action.SKIP

    def on_attachment(self, attachment: Attachment) -> Action:
        inside = self._in_window(attachment.log_time)
        keep = inside ^ self._invert
        return Action.CONTINUE if keep else Action.SKIP
