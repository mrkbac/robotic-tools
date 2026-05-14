# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import (
    Action,
    ChunkContext,
    InputContext,
    InputProcessor,
    MessageScope,
)

if TYPE_CHECKING:
    from small_mcap import Attachment


class AttachmentFilterProcessor(InputProcessor):
    """Filter attachment records."""

    def __init__(self, include: bool = True) -> None:
        self.include = include

    def message_scope(self, context: ChunkContext) -> MessageScope:
        return MessageScope.none()

    def on_attachment(self, context: InputContext, attachment: Attachment) -> Action:
        return Action.CONTINUE if self.include else Action.SKIP
