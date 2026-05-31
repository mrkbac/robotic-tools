"""Split output every N bytes of uncompressed message payload."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from pymcap_cli.core.processors.base import ChunkContext, MessageContext, MessageScope, OutputRouter

if TYPE_CHECKING:
    from small_mcap import Chunk, LazyChunk, Message


# Rough MCAP message record overhead per message (opcode + length + header
# fields). Used to budget the per-message decode path against `--max-size`.
_MESSAGE_RECORD_OVERHEAD_BYTES = 31


class SizeSplitProcessor(OutputRouter):
    """Split output when accumulated payload exceeds ``max_size_bytes``.

    Whole chunks are kept together — each chunk fast-copies into the current
    segment, or starts a new one when adding it would overflow. The boundary
    is therefore approximate: each segment may overshoot ``max_size_bytes``
    by up to one input chunk's worth, and the output file size also depends
    on the writer's compression ratio.

    Segment count is dynamic — segments are created lazily as the writer
    crosses the budget.

    Per-message routing fires when other processors force chunk decoding;
    the same overflow rule applies, using ``len(data) + record-overhead`` as
    the per-message size estimate.

    The chunk decision is made in ``route_chunk`` rather than ``on_chunk``
    because the dispatcher pre-classifies chunks ahead of routing — at
    classification time the segment state hasn't seen the prior chunk yet.
    """

    def __init__(self, max_size_bytes: int) -> None:
        if max_size_bytes <= 0:
            msg = f"max_size_bytes must be positive, got {max_size_bytes}"
            raise ValueError(msg)
        self.max_size_bytes = max_size_bytes
        self._current_segment = 0
        self._bytes_in_current = 0

    def _advance(self, n_bytes: int) -> int:
        if self._bytes_in_current > 0 and self._bytes_in_current + n_bytes > self.max_size_bytes:
            self._current_segment += 1
            self._bytes_in_current = n_bytes
        else:
            self._bytes_in_current += n_bytes
        return self._current_segment

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        _ = context
        return MessageScope.none()

    @override
    def route_chunk(self, context: ChunkContext, chunk: Chunk | LazyChunk) -> tuple[int, ...]:
        _ = context
        return (self._advance(chunk.uncompressed_size),)

    @override
    def route_message(self, context: MessageContext, message: Message) -> tuple[int, ...]:
        _ = context
        return (self._advance(len(message.data) + _MESSAGE_RECORD_OVERHEAD_BYTES),)
