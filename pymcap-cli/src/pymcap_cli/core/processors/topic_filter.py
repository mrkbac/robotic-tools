# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import Action, Processor
from pymcap_cli.utils import compile_topic_patterns

if TYPE_CHECKING:
    from small_mcap import Channel, Schema


class TopicFilterProcessor(Processor):
    """Filter channels by topic regex patterns.

    Compiles patterns internally. Uses search() for flexible matching.
    """

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self.include = compile_topic_patterns(include or [])
        self.exclude = compile_topic_patterns(exclude or [])

    def on_channel(self, channel: Channel, schema: Schema | None) -> Action:
        if self.include:
            if any(pattern.search(channel.topic) for pattern in self.include):
                return Action.CONTINUE
            return Action.SKIP

        if self.exclude and any(pattern.search(channel.topic) for pattern in self.exclude):
            return Action.SKIP

        return Action.CONTINUE
