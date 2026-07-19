from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from pymcap_cli.core.processors.base import (
    ChunkContext,
    ChunkDecision,
    InputProcessor,
    MessageContext,
    MessageHeader,
    MessageHeaderDecision,
    MessageScope,
)

if TYPE_CHECKING:
    from small_mcap import Chunk, LazyChunk


class AlwaysDecodeProcessor(InputProcessor):
    """Forces all chunks to be decoded."""

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        # This processor only forces chunk decoding (via on_chunk); it never inspects
        # or transforms messages, so no message needs to traverse the processor chain.
        # Declaring an empty scope lets every decoded message skip the per-message chain
        # machinery and route straight to the writer.
        return MessageScope.none()

    @override
    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        return ChunkDecision.DECODE

    @override
    def on_message_header(
        self, context: MessageContext, header: MessageHeader
    ) -> MessageHeaderDecision:
        return MessageHeaderDecision.CONTINUE
