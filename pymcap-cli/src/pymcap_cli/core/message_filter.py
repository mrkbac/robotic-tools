"""Shared topic and time filtering for MCAP message readers."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pymcap_cli.utils import RelativeTime, parse_timestamp_args

if TYPE_CHECKING:
    from collections.abc import Callable

    from small_mcap import Channel, Schema, Summary


@dataclass(frozen=True, slots=True)
class ResolvedMessageFilterOptions:
    """Message filters with concrete integer time bounds."""

    start_time_ns: int
    end_time_ns: int
    early_bail: bool


@dataclass(frozen=True, slots=True)
class MessageFilterOptions:
    """Canonical message-selection options shared by CLI commands.

    Topic selectors are regular expressions evaluated with ``fullmatch``.
    Ordinary ROS topic names therefore match exactly without added anchors.
    """

    topics: tuple[str, ...] = ()
    exclude_topics: tuple[str, ...] = ()
    start_time: int | RelativeTime | None = None
    end_time: int | RelativeTime | None = None
    early_bail: bool = False

    @classmethod
    def from_args(
        cls,
        *,
        topic: list[str] | None = None,
        exclude_topic: list[str] | None = None,
        start: str = "",
        end: str = "",
        early_bail: bool = False,
    ) -> MessageFilterOptions:
        start_time = parse_timestamp_args(start, 0, 0, allow_relative=True)
        end_time = parse_timestamp_args(end, 0, 0, allow_relative=True)
        options = cls(
            topics=tuple(topic or ()),
            exclude_topics=tuple(exclude_topic or ()),
            start_time=start_time,
            end_time=end_time,
            early_bail=early_bail,
        )
        options._compile_selectors(options.topics)
        options._compile_selectors(options.exclude_topics)
        options._validate_unresolved()
        return options

    @property
    def has_positive_topics(self) -> bool:
        return bool(self.topics)

    @property
    def has_topic_filters(self) -> bool:
        return bool(self.has_positive_topics or self.exclude_topics)

    @property
    def has_time_filters(self) -> bool:
        return bool(self.start_time is not None or self.end_time is not None or self.early_bail)

    @staticmethod
    def _compile_selectors(selectors: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
        try:
            return tuple(re.compile(selector, re.IGNORECASE) for selector in selectors)
        except re.error as exc:
            raise ValueError(f"Invalid topic regex: {exc}") from exc

    def include_patterns(self) -> list[str]:
        """Return equivalent regex patterns for ``TopicFilterProcessor``."""
        return [rf"\A(?:{selector})\Z" for selector in self.topics]

    def exclude_patterns(self) -> list[str]:
        """Return equivalent exclusion regexes for ``TopicFilterProcessor``."""
        return [rf"\A(?:{selector})\Z" for selector in self.exclude_topics]

    def _validate_unresolved(self) -> None:
        if self.early_bail and self.end_time is None:
            raise ValueError("--early-bail requires --end")
        if (
            isinstance(self.start_time, int)
            and isinstance(self.end_time, int)
            and self.start_time >= self.end_time
        ):
            raise ValueError(
                f"start time ({self.start_time}) must be less than end time ({self.end_time})"
            )

    def create_channel_predicate(
        self,
        base_predicate: Callable[[Channel, Schema | None], bool] | None = None,
    ) -> Callable[[Channel, Schema | None], bool]:
        include_regex = self._compile_selectors(self.topics)
        exclude_regex = self._compile_selectors(self.exclude_topics)
        has_positive = self.has_positive_topics

        def should_include(channel: Channel, schema: Schema | None) -> bool:
            if base_predicate is not None and not base_predicate(channel, schema):
                return False

            topic = channel.topic
            positive = not has_positive or any(
                pattern.fullmatch(topic) for pattern in include_regex
            )
            excluded = any(pattern.fullmatch(topic) for pattern in exclude_regex)
            return positive and not excluded

        return should_include

    def resolve(self, summary: Summary | None) -> ResolvedMessageFilterOptions:
        start_time = self.start_time
        end_time = self.end_time
        if isinstance(start_time, RelativeTime) or isinstance(end_time, RelativeTime):
            statistics = summary.statistics if summary is not None else None
            if statistics is None:
                raise ValueError(
                    "Relative --start/--end values require MCAP summary statistics. "
                    "Use absolute timestamps or rebuild the file summary."
                )
            if isinstance(start_time, RelativeTime):
                start_time = start_time.resolve(
                    statistics.message_start_time,
                    statistics.message_end_time,
                )
            if isinstance(end_time, RelativeTime):
                end_time = end_time.resolve(
                    statistics.message_start_time,
                    statistics.message_end_time,
                )

        start_time_ns = 0 if start_time is None else start_time
        end_time_ns = sys.maxsize if end_time is None else end_time
        if start_time_ns >= end_time_ns:
            raise ValueError(
                f"resolved start time ({start_time_ns}) must be less than end time ({end_time_ns})"
            )
        return ResolvedMessageFilterOptions(
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            early_bail=self.early_bail,
        )
