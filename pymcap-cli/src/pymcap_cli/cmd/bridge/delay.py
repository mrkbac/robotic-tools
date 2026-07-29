"""`pymcap-cli bridge delay` — measure bridge clock offset and ROS header age."""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from cyclopts import Group as CycloptsGroup
from cyclopts import Parameter
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from pymcap_cli.cmd._arg_constraints import constraint_group, requires
from pymcap_cli.cmd._cli_options import (
    TOPIC_FILTERING_GROUP,
    BridgeTarget,
    ConnectTimeoutOption,
    JsonOutputOption,
    SampleDurationOption,
    TopicOption,
)
from pymcap_cli.cmd.bridge._shared import (
    BridgeFetchError,
    console,
    to_ws_url,
)
from pymcap_cli.cmd.bridge._topic_monitor import (
    BridgeMonitorSession,
    ChannelDelayStats,
    RunningTimeStats,
    TimeReference,
)
from pymcap_cli.constants import NS_TO_SEC
from pymcap_cli.core.message_filter import MessageFilterOptions
from pymcap_cli.display.display_utils import _format_parts_with_colors
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)

FILTER_GROUP = CycloptsGroup("Filtering")

# --against only affects header-age mode, which is only entered when topics are supplied.
_AGAINST_CONSTRAINT = constraint_group(requires("--against", "--topic"))

DelayReference = TimeReference
RunningDelayStats = RunningTimeStats


@dataclass(frozen=True)
class DelayReport:
    url: str
    duration: float
    against: DelayReference
    wants_header_age: bool
    time_offset: RunningDelayStats
    channels: tuple[ChannelDelayStats, ...]

    @property
    def total_messages(self) -> int:
        return sum(stats.clock_offset.count for stats in self.channels)


async def _collect_delay_async(
    url: str,
    *,
    message_filter: MessageFilterOptions,
    against: DelayReference,
    duration: float,
    connect_timeout: float,
    now_ns: Callable[[], int] = time.time_ns,
) -> DelayReport:
    wants_header_age = message_filter.has_positive_topics
    session = BridgeMonitorSession(
        url,
        message_filter=message_filter,
        window_seconds=duration,
        connect_timeout=connect_timeout,
        subscribe_messages=wants_header_age,
        collect_time=True,
        require_time_capability=not wants_header_age,
        decode_header_stamps=wants_header_age,
        header_reference=against,
        wall_time_ns=now_ns,
    )
    await session.run(
        interval_seconds=duration,
        duration_seconds=duration,
        on_snapshot=lambda _snapshot: None,
    )
    return DelayReport(
        url=url,
        duration=duration,
        against=against,
        wants_header_age=wants_header_age,
        time_offset=session.time_offset,
        channels=session.delay_channels,
    )


def _ms(value_ns: float) -> float:
    return value_ns / 1_000_000


def _format_duration(value_ns: float) -> str:
    sign = "+" if value_ns >= 0 else "-"
    magnitude = abs(value_ns)
    if magnitude < 1_000:
        return f"{sign}{magnitude:.0f}ns"
    if magnitude < 1_000_000:
        return f"{sign}{magnitude / 1_000:.1f}us"
    if magnitude < NS_TO_SEC:
        return f"{sign}{magnitude / 1_000_000:.1f}ms"
    if magnitude < 60 * NS_TO_SEC:
        return f"{sign}{magnitude / NS_TO_SEC:.2f}s"
    minutes = int(magnitude // (60 * NS_TO_SEC))
    seconds = (magnitude - minutes * 60 * NS_TO_SEC) / NS_TO_SEC
    return f"{sign}{minutes}m {seconds:04.1f}s"


def _format_latest(stats: RunningDelayStats) -> str:
    if stats.count == 0:
        return "-"
    return _format_duration(stats.latest_ns)


def _format_mean(stats: RunningDelayStats) -> str:
    if stats.count == 0:
        return "-"
    return _format_duration(stats.mean_ns)


def _format_range(stats: RunningDelayStats) -> str:
    if stats.count == 0:
        return "-"
    if stats.min_ns == stats.max_ns:
        return _format_duration(stats.min_ns)
    return f"{_format_duration(stats.min_ns)}..{_format_duration(stats.max_ns)}"


def _format_issues(stats: ChannelDelayStats) -> str:
    missing = stats.missing_header_stamp + stats.undecodable_messages
    if missing and stats.decode_errors:
        return "both"
    if missing:
        return "stamp"
    if stats.decode_errors:
        return "decode"
    return "-"


def _delay_stats_to_dict(stats: RunningDelayStats) -> dict[str, float | int] | None:
    if stats.count == 0:
        return None
    return {
        "count": stats.count,
        "latest_ms": _ms(stats.latest_ns),
        "min_ms": _ms(stats.min_ns),
        "mean_ms": _ms(stats.mean_ns),
        "max_ms": _ms(stats.max_ns),
        "stddev_ms": _ms(stats.stddev_ns),
    }


def _delay_to_dict(report: DelayReport) -> dict[str, object]:
    return {
        "url": report.url,
        "duration_seconds": report.duration,
        "against": report.against.value,
        "mode": "header_age" if report.wants_header_age else "bridge_time",
        "time_offset": _delay_stats_to_dict(report.time_offset),
        "total_messages": report.total_messages,
        "channels": [
            {
                "id": stats.channel["id"],
                "topic": stats.channel["topic"],
                "schema": stats.channel.get("schemaName", ""),
                "encoding": stats.channel["encoding"],
                "payload_bytes": stats.payload_bytes,
                "clock_offset": _delay_stats_to_dict(stats.clock_offset),
                "header_age": _delay_stats_to_dict(stats.header_age),
                "undecodable_messages": stats.undecodable_messages,
                "decode_errors": stats.decode_errors,
                "missing_header_stamp": stats.missing_header_stamp,
            }
            for stats in report.channels
        ],
    }


def _build_summary(report: DelayReport) -> Table:
    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold blue")
    summary.add_column()
    summary.add_row("Bridge:", f"[green]{report.url}[/]")
    summary.add_row("Duration:", f"[cyan]{report.duration:.1f}s[/]")
    if report.wants_header_age:
        summary.add_row("Mode:", f"message age vs [yellow]{report.against.value}[/] time")
        summary.add_row("Messages:", f"[green]{report.total_messages:,}[/]")
        summary.add_row("Channels:", f"[green]{len(report.channels):,}[/]")
    else:
        summary.add_row("Mode:", "bridge time offset")
        summary.add_row("Time samples:", f"[green]{report.time_offset.count:,}[/]")
    return summary


def _build_time_table(report: DelayReport) -> Table:
    table = Table(title="Bridge Time", title_justify="left", title_style="bold cyan")
    table.add_column("N", justify="right", no_wrap=True)
    table.add_column("Offset last", justify="right", no_wrap=True)
    table.add_column("Offset avg", justify="right", no_wrap=True)
    table.add_column("Offset range", justify="right", no_wrap=True)
    table.add_row(
        str(report.time_offset.count),
        _format_latest(report.time_offset),
        _format_mean(report.time_offset),
        _format_range(report.time_offset),
    )
    return table


def _build_delay_table(report: DelayReport) -> Table:
    table = Table(title="Bridge Delay", title_justify="left", title_style="bold cyan")
    table.add_column("Topic", no_wrap=True, overflow="ellipsis", max_width=44)
    table.add_column("B last", justify="right", no_wrap=True)
    table.add_column("B avg", justify="right", no_wrap=True)
    if report.wants_header_age:
        table.add_column("H last", justify="right", no_wrap=True)
        table.add_column("H avg", justify="right", no_wrap=True)
        table.add_column("Err", justify="right", no_wrap=True)

    for stats in report.channels:
        row = [
            _format_parts_with_colors(stats.channel["topic"]),
            _format_latest(stats.clock_offset),
            _format_mean(stats.clock_offset),
        ]
        if report.wants_header_age:
            row.extend(
                [
                    _format_latest(stats.header_age),
                    _format_mean(stats.header_age),
                    _format_issues(stats),
                ]
            )
        table.add_row(*row)
    return table


def _build_display(report: DelayReport) -> RenderableType:
    parts: list[RenderableType] = [_build_summary(report), Text("")]
    if not report.wants_header_age:
        parts.append(_build_time_table(report))
        if report.time_offset.count == 0:
            parts.extend(
                [
                    Text(""),
                    Text(
                        "No bridge time updates received during the sample window.",
                        style="yellow",
                    ),
                ]
            )
        return Group(*parts)

    if not report.channels:
        parts.append(Text("No matching channels advertised.", style="yellow"))
        return Group(*parts)
    parts.append(_build_delay_table(report))
    parts.append(
        Text(
            "B = local receive - bridge timestamp. H = selected reference - header.stamp.",
            style="dim",
        )
    )
    if report.total_messages == 0:
        parts.extend(
            [Text(""), Text("No messages received during the sample window.", style="yellow")]
        )
    return Group(*parts)


def delay(
    target: BridgeTarget,
    *,
    topic: Annotated[
        TopicOption, Parameter(group=[TOPIC_FILTERING_GROUP, _AGAINST_CONSTRAINT])
    ] = None,
    against: Annotated[
        DelayReference,
        Parameter(
            name=["--against"],
            group=[FILTER_GROUP, _AGAINST_CONSTRAINT],
            help="Reference time for header.stamp age when topics are supplied.",
        ),
    ] = DelayReference.LOCAL,
    duration: SampleDurationOption = 5.0,
    json_output: JsonOutputOption = False,
    connect_timeout: ConnectTimeoutOption = 5.0,
) -> int:
    """Measure live bridge clock offset and optional ROS `header.stamp` message age.

    With no topics, requires the bridge ``time`` capability, listens for time frames,
    and measures ``local_receive_time - bridge_time`` without subscribing to message channels.
    With topic regexes, also decodes matching messages and measures
    ``reference_time - header.stamp`` for messages that carry a ROS-style header.

    Examples
    --------
    ```
    pymcap-cli bridge delay robot:8765
    pymcap-cli bridge delay robot:8765 -t /camera/image_raw
    pymcap-cli bridge delay robot:8765 -t '/imu/.*' --against bridge
    pymcap-cli bridge delay robot:8765 --duration 10 --json
    ```
    """
    if duration <= 0:
        ERR.print("[red]Error:[/] --duration must be positive")
        return 1

    try:
        message_filter = MessageFilterOptions.from_args(topic=topic)
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    url = to_ws_url(target)
    try:
        report = asyncio.run(
            _collect_delay_async(
                url,
                message_filter=message_filter,
                against=against,
                duration=duration,
                connect_timeout=connect_timeout,
            )
        )
    except BridgeFetchError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Failed to connect to {url}: {exc}")
        return 1
    except KeyboardInterrupt:
        return 0

    if json_output:
        print(json.dumps(_delay_to_dict(report), separators=(",", ":")))  # noqa: T201
    else:
        console.print(_build_display(report))
    return 0
