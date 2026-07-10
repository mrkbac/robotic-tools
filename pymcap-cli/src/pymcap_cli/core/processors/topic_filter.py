from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChannelContext,
    ChunkContext,
    InputProcessor,
    MessageContext,
    MessageHeader,
    MessageHeaderDecision,
    MessageScope,
)
from pymcap_cli.utils import compile_topic_patterns

if TYPE_CHECKING:
    from small_mcap import Channel, Schema


class TopicFilterProcessor(InputProcessor):
    """Filter channels by topic regex patterns.

    Compiles patterns internally. Uses search() for flexible matching.

    When ``invert`` is True, the include/exclude decision is flipped: a
    matching include topic becomes SKIP, and a matching exclude topic
    becomes CONTINUE. This is what users expect from ``--invert-topics``.
    """

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        *,
        invert: bool = False,
    ) -> None:
        self.include = compile_topic_patterns(include or [])
        self.exclude = compile_topic_patterns(exclude or [])
        self._invert = invert

    def _decide(self, topic: str) -> Action:
        if self.include:
            if any(pattern.search(topic) for pattern in self.include):
                return Action.CONTINUE
            return Action.SKIP

        if self.exclude and any(pattern.search(topic) for pattern in self.exclude):
            return Action.SKIP

        return Action.CONTINUE

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        return MessageScope.none()

    @override
    def on_message_header(
        self, context: MessageContext, header: MessageHeader
    ) -> MessageHeaderDecision:
        return MessageHeaderDecision.CONTINUE

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        decision = self._decide(channel.topic)
        if self._invert:
            return Action.SKIP if decision == Action.CONTINUE else Action.CONTINUE
        return decision
