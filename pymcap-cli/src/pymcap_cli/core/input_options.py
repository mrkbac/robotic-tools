"""Pure-data input filtering options for ``McapProcessor``.

This module deliberately imports **none** of the processor implementations.
``InputOptions`` is a plain dataclass; the processor chain it implies is
constructed elsewhere (see :func:`build_input_processors` in
``pymcap_cli.core.mcap_processor``). That keeps callers — CLI commands,
tests, third-party scripts — free to construct ``InputOptions`` without
pulling the entire processor subtree into their import graph.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymcap_cli.utils import parse_timestamp_args

if TYPE_CHECKING:
    from pymcap_cli.core.message_filter import MessageFilterOptions
    from pymcap_cli.core.processors.base import InputProcessor
    from pymcap_cli.utils import RelativeTime


@dataclass
class InputOptions:
    """Input file filtering options.

    Stores explicit fields for proper merging. The corresponding processor
    chain is built by ``McapProcessor`` via ``build_input_processors`` —
    this dataclass intentionally does **not** import any processor
    implementations so callers can build options without pulling the chain.
    """

    always_decode_chunk: bool
    start_time_ns: int | RelativeTime | None
    end_time_ns: int | RelativeTime | None
    include_topics: list[str]
    exclude_topics: list[str]
    include_metadata: bool
    include_attachments: bool
    latch_topics: list[str] = field(default_factory=list)
    latch_from_metadata: bool = False
    invert_topics: bool = False
    invert_time: bool = False
    is_early_bail_enabled: bool = False
    extra_processors: list[InputProcessor] = field(default_factory=list)

    @classmethod
    def from_args(
        cls,
        always_decode_chunk: bool = False,
        # Raw CLI args for time (accept any combination)
        start: str = "",
        start_nsecs: int = 0,
        start_secs: int = 0,
        end: str = "",
        end_nsecs: int = 0,
        end_secs: int = 0,
        # Raw CLI args for topics (regex strings, not compiled)
        include_topic_regex: list[str] | None = None,
        exclude_topic_regex: list[str] | None = None,
        # Topic globs (shell-style patterns, converted to regex)
        include_topic_glob: list[str] | None = None,
        exclude_topic_glob: list[str] | None = None,
        # Content filtering
        include_metadata: bool = True,
        include_attachments: bool = True,
        # Latching
        latch_topics: list[str] | None = None,
        latch_from_metadata: bool = False,
        # Inversion
        invert_topics: bool = False,
        invert_time: bool = False,
        is_early_bail_enabled: bool = False,
        # Caller-supplied processors appended after the standard filter chain.
        extra_processors: list[InputProcessor] | None = None,
    ) -> InputOptions:
        include_topics = list(include_topic_regex or [])
        include_topics.extend(r"\A" + fnmatch.translate(glob) for glob in include_topic_glob or [])
        exclude_topics = list(exclude_topic_regex or [])
        exclude_topics.extend(r"\A" + fnmatch.translate(glob) for glob in exclude_topic_glob or [])

        return cls(
            always_decode_chunk=always_decode_chunk,
            start_time_ns=parse_timestamp_args(start, start_secs, start_nsecs, allow_relative=True),
            end_time_ns=parse_timestamp_args(end, end_secs, end_nsecs, allow_relative=True),
            include_topics=include_topics,
            exclude_topics=exclude_topics,
            include_metadata=include_metadata,
            include_attachments=include_attachments,
            latch_topics=latch_topics or [],
            latch_from_metadata=latch_from_metadata,
            invert_topics=invert_topics,
            invert_time=invert_time,
            is_early_bail_enabled=is_early_bail_enabled,
            extra_processors=list(extra_processors) if extra_processors else [],
        )

    @classmethod
    def from_message_filter(
        cls,
        message_filter: MessageFilterOptions,
        *,
        always_decode_chunk: bool = False,
        include_metadata: bool = True,
        include_attachments: bool = True,
        latch_topics: list[str] | None = None,
        latch_from_metadata: bool = False,
        invert_topics: bool = False,
        invert_time: bool = False,
        extra_processors: list[InputProcessor] | None = None,
    ) -> InputOptions:
        """Adapt the canonical CLI filter to the processor input pipeline."""
        return cls(
            always_decode_chunk=always_decode_chunk,
            start_time_ns=message_filter.start_time,
            end_time_ns=message_filter.end_time,
            include_topics=message_filter.include_patterns(),
            exclude_topics=message_filter.exclude_patterns(),
            include_metadata=include_metadata,
            include_attachments=include_attachments,
            latch_topics=latch_topics or [],
            latch_from_metadata=latch_from_metadata,
            invert_topics=invert_topics,
            invert_time=invert_time,
            is_early_bail_enabled=message_filter.early_bail,
            extra_processors=list(extra_processors) if extra_processors else [],
        )

    def __or__(self, other: InputOptions) -> InputOptions:
        """Merge options - other (per-file) overrides self (global) for non-default values."""
        return InputOptions(
            always_decode_chunk=self.always_decode_chunk or other.always_decode_chunk,
            start_time_ns=other.start_time_ns
            if other.start_time_ns is not None
            else self.start_time_ns,
            end_time_ns=other.end_time_ns if other.end_time_ns is not None else self.end_time_ns,
            include_topics=other.include_topics or self.include_topics,
            exclude_topics=other.exclude_topics or self.exclude_topics,
            include_metadata=self.include_metadata and other.include_metadata,
            include_attachments=self.include_attachments and other.include_attachments,
            latch_topics=other.latch_topics or self.latch_topics,
            latch_from_metadata=self.latch_from_metadata or other.latch_from_metadata,
            invert_topics=self.invert_topics or other.invert_topics,
            invert_time=self.invert_time or other.invert_time,
            is_early_bail_enabled=(self.is_early_bail_enabled or other.is_early_bail_enabled),
            extra_processors=[*self.extra_processors, *other.extra_processors],
        )
