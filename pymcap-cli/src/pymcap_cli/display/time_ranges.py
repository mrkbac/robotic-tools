"""Shared display helpers for timestamp range summaries."""

from __future__ import annotations

from typing import Protocol

from pymcap_cli.utils import format_ts_short


class TimestampRangeSegmentLike(Protocol):
    @property
    def start_time(self) -> int: ...

    @property
    def end_time(self) -> int: ...

    @property
    def message_count(self) -> int: ...


class TimestampRangeSummaryLike(Protocol):
    @property
    def ranges(self) -> tuple[TimestampRangeSegmentLike, ...]: ...

    @property
    def hidden_messages(self) -> int: ...


def format_count(value: int, singular: str, plural: str | None = None) -> str:
    unit = singular if value == 1 else plural or f"{singular}s"
    return f"{value:,} {unit}"


def format_optional_time_window(start_time: int | None, end_time: int | None) -> str:
    if start_time is None or end_time is None:
        return "N/A"
    if start_time == end_time:
        return format_ts_short(start_time)
    return f"{format_ts_short(start_time)} - {format_ts_short(end_time)}"


def format_range_summary(summary: TimestampRangeSummaryLike) -> str:
    if not summary.ranges and summary.hidden_messages == 0:
        return "-"

    parts: list[str] = []
    for segment in summary.ranges:
        count_label = format_count(segment.message_count, "msg")
        if segment.start_time == segment.end_time:
            parts.append(f"{format_ts_short(segment.start_time)} ({count_label})")
        else:
            parts.append(
                f"{format_ts_short(segment.start_time)} - "
                f"{format_ts_short(segment.end_time)} ({count_label})"
            )

    if summary.hidden_messages > 0:
        parts.append(f"+{format_count(summary.hidden_messages, 'msg')}")

    return ", ".join(parts)
