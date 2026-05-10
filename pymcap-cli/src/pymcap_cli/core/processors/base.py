# ruff: noqa: ARG002
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag, auto
from typing import IO, TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

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

OutputKey = int | str | tuple[int | str, ...]


@dataclass(frozen=True, slots=True)
class OutputSegmentInfo:
    key: OutputKey
    start_time: int
    end_time: int


@dataclass(frozen=True, slots=True)
class ProcessorInputContext:
    stream_id: int
    stream: IO[bytes]
    summary: Summary | None
    output_segments: tuple[OutputSegmentInfo, ...]
    remap_channel: Callable[[Channel], Channel]
    remap_message: Callable[[Message], Message]


class _SplitRequiredSentinel:
    __slots__ = ()


SPLIT_REQUIRED: Final = _SplitRequiredSentinel()


class Action(IntFlag):
    """Result actions - combinable flags.

    ``KEEP`` is a positive vote: when any processor returns ``KEEP`` for a
    channel or message, ``SKIP`` from other processors is ignored for that
    record.
    """

    CONTINUE = 0
    SKIP = auto()
    STOP = auto()
    KEEP = auto()


class ChunkDecision(Enum):
    """Chunk-level processing decision."""

    CONTINUE = auto()
    SKIP = auto()
    DECODE = auto()
    # Chunk data must be re-compressed (different target compression) but no
    # per-message work is needed — just decompress + re-compress the chunk's
    # data bytes. Avoids parsing/re-emitting every record.
    RECOMPRESS = auto()


class Processor:
    """Base processor."""

    def initialize(self, summaries: list[Summary | None]) -> None:
        """Called before processing with summaries from all input files."""

    def prepare_input(self, context: ProcessorInputContext) -> None:
        """Called once per input stream after summaries/channels are known."""

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

    def route_chunk(self, chunk: Chunk | LazyChunk) -> int | str | _SplitRequiredSentinel | None:
        return None

    def route_message(self, message: Message) -> int | str | None:
        return None

    def also_route_to(self, message: Message) -> Iterable[OutputKey]:
        """Yield additional output keys to write this message into.

        ``route_message`` returns the *primary* destination; this hook lets a
        processor duplicate the message into one or more *extra* segments
        (used by ``split`` trailing-context to copy target-topic messages
        into the segment that just ended).
        """
        return ()

    def output_keys(self) -> list[int | str] | None:
        return None

    def on_segment_open(self, key: OutputKey) -> Iterable[tuple[int, Message]]:
        """Yield ``(channel_id, message)`` pairs to inject into a freshly-opened
        output segment before any normal writes.

        Called lazily the first time a segment is actually written to.
        """
        return ()
