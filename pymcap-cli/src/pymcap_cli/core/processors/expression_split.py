# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import SPLIT_REQUIRED, ChunkDecision, Processor

if TYPE_CHECKING:
    from collections.abc import Callable

    from small_mcap import Channel, Chunk, LazyChunk, Message, MessageIndex


class ExpressionSplitProcessor(Processor):
    """Split output based on an arbitrary callable.

    The callable receives a Message and a dict of channels, and returns
    a segment key (int or str). Always forces chunk decoding since the
    expression must evaluate each message.

    Output keys are discovered dynamically (lazy segment creation).
    """

    def __init__(
        self,
        fn: Callable[[Message, dict[int, Channel]], int | str],
        channels: dict[int, Channel] | None = None,
    ) -> None:
        self.fn = fn
        self.channels: dict[int, Channel] = channels or {}

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        return ChunkDecision.DECODE

    def route_chunk(self, chunk: Chunk | LazyChunk) -> object:
        return SPLIT_REQUIRED

    def route_message(self, message: Message) -> int | str:
        return self.fn(message, self.channels)

    def output_keys(self) -> None:
        return None
