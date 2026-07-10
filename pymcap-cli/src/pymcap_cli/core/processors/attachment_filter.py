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
    from small_mcap import Attachment


class AttachmentFilterProcessor(InputProcessor):
    """Filter attachment records."""

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
    def on_attachment(self, context: InputContext, attachment: Attachment) -> Action:
        return Action.CONTINUE if self.include else Action.SKIP
