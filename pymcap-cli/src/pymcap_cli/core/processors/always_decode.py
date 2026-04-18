# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import ChunkDecision, Processor

if TYPE_CHECKING:
    from small_mcap import Chunk, LazyChunk, MessageIndex


class AlwaysDecodeProcessor(Processor):
    """Forces all chunks to be decoded."""

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        return ChunkDecision.DECODE
