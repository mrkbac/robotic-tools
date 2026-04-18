# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import Action, Processor

if TYPE_CHECKING:
    from small_mcap import Metadata


class MetadataFilterProcessor(Processor):
    """Filter metadata records."""

    def __init__(self, include: bool = True) -> None:
        self.include = include

    def on_metadata(self, metadata: Metadata) -> Action:
        return Action.CONTINUE if self.include else Action.SKIP
