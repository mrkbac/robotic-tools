from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChunkContext,
    InputContext,
    InputProcessor,
    MessageContext,
    MessageHeader,
    MessageHeaderDecision,
    MessageScope,
)

if TYPE_CHECKING:
    from small_mcap import Metadata


class MetadataFilterProcessor(InputProcessor):
    """Filter metadata records."""

    def __init__(self, include: bool = True) -> None:
        self.include = include

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        return MessageScope.none()

    @override
    def on_message_header(
        self, context: MessageContext, header: MessageHeader
    ) -> MessageHeaderDecision:
        return MessageHeaderDecision.CONTINUE

    @override
    def on_metadata(self, context: InputContext, metadata: Metadata) -> Action:
        return Action.CONTINUE if self.include else Action.SKIP
