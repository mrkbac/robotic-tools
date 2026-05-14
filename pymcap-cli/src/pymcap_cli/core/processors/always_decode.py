# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import ChunkContext, ChunkDecision, InputProcessor

if TYPE_CHECKING:
    from small_mcap import Chunk, LazyChunk


class AlwaysDecodeProcessor(InputProcessor):
    """Forces all chunks to be decoded."""

    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        return ChunkDecision.DECODE
