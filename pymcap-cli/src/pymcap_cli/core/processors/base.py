# ruff: noqa: ARG002
from __future__ import annotations

from enum import Enum, IntFlag, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from small_mcap import (
        Attachment,
        Channel,
        Chunk,
        LazyChunk,
        Message,
        MessageIndex,
        Metadata,
        Schema,
        Summary,
    )

SPLIT_REQUIRED: object = object()


class Action(IntFlag):
    """Result actions - combinable flags."""

    CONTINUE = 0
    SKIP = auto()
    STOP = auto()


class ChunkDecision(Enum):
    """Chunk-level processing decision."""

    CONTINUE = auto()
    SKIP = auto()
    DECODE = auto()


class Processor:
    """Base processor."""

    def initialize(self, summaries: list[Summary | None]) -> None:
        """Called before processing with summaries from all input files."""

    def on_chunk(
        self,
        chunk: Chunk | LazyChunk,
        indexes: list[MessageIndex],
    ) -> ChunkDecision:
        return ChunkDecision.CONTINUE

    def on_channel(self, channel: Channel, schema: Schema | None) -> Action:
        return Action.CONTINUE

    def on_message(self, message: Message) -> Action:
        return Action.CONTINUE

    def on_metadata(self, metadata: Metadata) -> Action:
        return Action.CONTINUE

    def on_attachment(self, attachment: Attachment) -> Action:
        return Action.CONTINUE

    def route_chunk(self, chunk: Chunk | LazyChunk) -> int | str | object | None:
        return None

    def route_message(self, message: Message) -> int | str | None:
        return None

    def output_keys(self) -> list[int | str] | None:
        return None
