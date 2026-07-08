from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChunkContext,
    ChunkDecision,
    InputContext,
    InputProcessor,
    MessageContext,
    MessageScope,
    PipelineContext,
)
from pymcap_cli.utils import RelativeTime

if TYPE_CHECKING:
    from collections.abc import Iterable

    from small_mcap import Attachment, Chunk, LazyChunk, Message


class TimeFilterProcessor(InputProcessor):
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

    @override
    def initialize(self, context: PipelineContext) -> None:
        """Resolve any RelativeTime bounds against the input summaries."""
        if not (isinstance(self._start, RelativeTime) or isinstance(self._end, RelativeTime)):
            return
        file_start: int | None = None
        file_end: int | None = None
        for input_context in context.inputs:
            if input_context.statistics is None:
                continue
            stats = input_context.statistics
            if file_start is None or stats.message_start_time < file_start:
                file_start = stats.message_start_time
            file_end = (
                stats.message_end_time
                if file_end is None
                else max(file_end, stats.message_end_time)
            )
        if file_start is None or file_end is None:
            msg = (
                "Relative time bounds (e.g. start+5s, end-1m) need the input's "
                "summary statistics to resolve. Either pass absolute --start / "
                "--end values, or run `pymcap-cli recover-inplace <file>` to "
                "rebuild the missing summary section."
            )
            raise ValueError(msg)
        if isinstance(self._start, RelativeTime):
            self.start_ns = self._start.resolve(file_start, file_end)
        if isinstance(self._end, RelativeTime):
            self.end_ns = self._end.resolve(file_start, file_end)
        if self.start_ns is not None and self.end_ns is not None and self.start_ns >= self.end_ns:
            raise ValueError(
                f"resolved start_ns ({self.start_ns}) must be less than "
                f"resolved end_ns ({self.end_ns})"
            )

    @override
    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
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

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        """A chunk wholly inside the (non-inverted) window drops nothing, so its
        messages need no per-message pass; only boundary-spanning chunks do.

        Interior chunks still get decoded when *another* processor transcodes a
        channel in them — reporting NONE here lets the dispatcher fast-path this
        filter for the untouched (telemetry) messages riding along.
        """
        start, end = context.chunk_start_time, context.chunk_end_time
        if self._invert or start is None or end is None:
            return MessageScope.all()
        inside = (self.start_ns is None or start >= self.start_ns) and (
            self.end_ns is None or end < self.end_ns
        )
        return MessageScope.none() if inside else MessageScope.all()

    def _in_window(self, log_time: int) -> bool:
        if self.start_ns is not None and log_time < self.start_ns:
            return False
        return not (self.end_ns is not None and log_time >= self.end_ns)

    @override
    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        inside = self._in_window(message.log_time)
        keep = inside ^ self._invert  # invert flips inside↔outside
        if keep:
            yield message

    @override
    def on_attachment(self, context: InputContext, attachment: Attachment) -> Action:
        inside = self._in_window(attachment.log_time)
        keep = inside ^ self._invert
        return Action.CONTINUE if keep else Action.SKIP
