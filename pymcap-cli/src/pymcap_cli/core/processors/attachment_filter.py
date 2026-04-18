# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import Action, Processor

if TYPE_CHECKING:
    from small_mcap import Attachment


class AttachmentFilterProcessor(Processor):
    """Filter attachment records."""

    def __init__(self, include: bool = True) -> None:
        self.include = include

    def on_attachment(self, attachment: Attachment) -> Action:
        return Action.CONTINUE if self.include else Action.SKIP
