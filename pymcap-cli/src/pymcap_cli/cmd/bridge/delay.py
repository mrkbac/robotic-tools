"""`pymcap-cli bridge delay` — measure bridge clock offset and ROS header age."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import Group as CycloptsGroup
from cyclopts import Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from robo_ws_bridge import WebSocketBridgeClient
from robo_ws_bridge.ws_types import ServerCapabilities
from small_mcap import JSONDecoderFactory

from pymcap_cli.cmd.bridge._shared import (
    CONNECTION_GROUP,
    DISPLAY_GROUP,
    BridgeFetchError,
    BridgeTarget,
    ChannelSubscriptionManager,
    channel_to_schema,
    console,
    to_ws_url,
)
from pymcap_cli.constants import NS_TO_SEC
from pymcap_cli.display.display_utils import _format_parts_with_colors
from pymcap_cli.log_setup import ERR
from pymcap_cli.types.to_plain import to_plain

if TYPE_CHECKING:
    from collections.abc import Callable

    from robo_ws_bridge.ws_types import ChannelInfo

logger = logging.getLogger(__name__)

FILTER_GROUP = CycloptsGroup("Filtering")


class DelayReference(str, Enum):
    LOCAL = "local"
    BRIDGE = "bridge"


@dataclass
class RunningDelayStats:
    count: int = 0
    latest_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0
    _mean_ns: float = 0.0
    _m2_ns: float = 0.0

    def add(self, value_ns: int) -> None:
        self.latest_ns = value_ns
        if self.count == 0:
            self.min_ns = value_ns
            self.max_ns = value_ns
        else:
            self.min_ns = min(self.min_ns, value_ns)
            self.max_ns = max(self.max_ns, value_ns)

        self.count += 1
        delta = value_ns - self._mean_ns
        self._mean_ns += delta / self.count
        self._m2_ns += delta * (value_ns - self._mean_ns)

    @property
    def mean_ns(self) -> float:
        return self._mean_ns

    @property
    def stddev_ns(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self._m2_ns / self.count)


@dataclass
class ChannelDelayStats:
    channel: ChannelInfo
    clock_offset: RunningDelayStats = field(default_factory=RunningDelayStats)
    header_age: RunningDelayStats = field(default_factory=RunningDelayStats)
    payload_bytes: int = 0
    undecodable_messages: int = 0
    decode_errors: int = 0
    missing_header_stamp: int = 0


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


def _compile_topic_patterns(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


def _header_stamp_ns(decoded_message: object) -> int | None:
    plain = to_plain(decoded_message)
    if not isinstance(plain, dict):
        return None
    header = plain.get("header")
    if not isinstance(header, dict):
        return None
    stamp = header.get("stamp")
    if isinstance(stamp, bool):
        return None
    if isinstance(stamp, int):
        return stamp
    if not isinstance(stamp, dict):
        return None
    sec = stamp.get("sec")
    nanosec = stamp.get("nanosec")
    if isinstance(sec, bool) or isinstance(nanosec, bool):
        return None
    if isinstance(sec, int) and isinstance(nanosec, int):
        return sec * NS_TO_SEC + nanosec
    return None


async def _collect_delay_async(
    url: str,
    *,
    topic_patterns: tuple[re.Pattern[str], ...],
    against: DelayReference,
    duration: float,
    connect_timeout: float,
    now_ns: Callable[[], int] = time.time_ns,
) -> DelayReport:
    wants_header_age = bool(topic_patterns)
    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
    server_info_event = asyncio.Event()
    client.on_server_info(lambda *_: server_info_event.set())
    time_offset = RunningDelayStats()

    factories = [JSONDecoderFactory(), DecoderFactory()]
    decoders: dict[int, Callable[[bytes | memoryview], Any] | None] = {}
    stats_by_channel: dict[int, ChannelDelayStats] = {}

    def _topic_matches(topic: str) -> bool:
        return not topic_patterns or any(pattern.search(topic) for pattern in topic_patterns)

    def _stats_for(channel: ChannelInfo) -> ChannelDelayStats:
        channel_id = channel["id"]
        stats = stats_by_channel.get(channel_id)
        if stats is None:
            stats = ChannelDelayStats(channel=channel)
            stats_by_channel[channel_id] = stats
        return stats

    def _decoder_for(channel: ChannelInfo) -> Callable[[bytes | memoryview], Any] | None:
        channel_id = channel["id"]
        if channel_id in decoders:
            return decoders[channel_id]
        schema = channel_to_schema(channel)
        decoder: Callable[[bytes | memoryview], Any] | None = None
        for factory in factories:
            try:
                decoder = factory.decoder_for(channel["encoding"], schema)
            except Exception:
                logger.exception(
                    f"Decoder construction failed for {channel['topic']} "
                    f"(schema={schema.name!r}, encoding={channel['encoding']!r})"
                )
                decoder = None
            if decoder is not None:
                break
        decoders[channel_id] = decoder
        return decoder

    def _should_subscribe(channel: ChannelInfo) -> bool:
        if not _topic_matches(channel["topic"]):
            return False
        _stats_for(channel)
        return True

    def _on_message(channel: ChannelInfo, bridge_timestamp_ns: int, payload: bytes) -> None:
        local_receive_ns = now_ns()
        stats = _stats_for(channel)
        stats.payload_bytes += len(payload)
        stats.clock_offset.add(local_receive_ns - bridge_timestamp_ns)

        if not wants_header_age:
            return

        decoder = _decoder_for(channel)
        if decoder is None:
            stats.undecodable_messages += 1
            return
        try:
            decoded = decoder(payload)
        except Exception:
            stats.decode_errors += 1
            logger.exception(f"Failed to decode message on {channel['topic']}")
            return

        stamp_ns = _header_stamp_ns(decoded)
        if stamp_ns is None:
            stats.missing_header_stamp += 1
            return

        reference_ns = local_receive_ns if against is DelayReference.LOCAL else bridge_timestamp_ns
        stats.header_age.add(reference_ns - stamp_ns)

    def _on_time_update(server_time_ns: int) -> None:
        time_offset.add(now_ns() - server_time_ns)

    subscriber: ChannelSubscriptionManager | None = None
    if wants_header_age:
        subscriber = ChannelSubscriptionManager(client, _should_subscribe)
        subscriber.install()
    client.on_time_update(_on_time_update)
    client.on_message(_on_message)

    await client.connect()
    try:
        try:
            await asyncio.wait_for(server_info_event.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError as exc:
            raise BridgeFetchError(
                f"Timed out after {connect_timeout:.1f}s waiting for serverInfo from {url}"
            ) from exc
        server_info = client.server_info
        if server_info is None:
            raise BridgeFetchError(f"No serverInfo received from {url}")
        if (
            not wants_header_age
            and ServerCapabilities.TIME.value not in server_info["capabilities"]
        ):
            raise BridgeFetchError(
                f"{url} does not advertise the '{ServerCapabilities.TIME.value}' capability. "
                "Pass topic regexes to measure delay from message timestamps instead."
            )

        if subscriber is not None:
            await subscriber.subscribe_existing()
        await asyncio.sleep(duration)

        channels = tuple(
            sorted(
                stats_by_channel.values(),
                key=lambda stats: (stats.channel["topic"], stats.channel["id"]),
            )
        )
        return DelayReport(
            url=url,
            duration=duration,
            against=against,
            wants_header_age=wants_header_age,
            time_offset=time_offset,
            channels=channels,
        )
    finally:
        await client.disconnect()


def _ms(value_ns: float) -> float:
    return value_ns / 1_000_000


def _format_ms(value_ns: float) -> str:
    return f"{_ms(value_ns):+,.3f}"


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


def _format_latest_mean(stats: RunningDelayStats) -> str:
    if stats.count == 0:
        return "-"
    return f"{_format_ms(stats.latest_ns)} / {_format_ms(stats.mean_ns)}"


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
    topics: Annotated[
        list[str],
        Parameter(
            group=FILTER_GROUP,
            help=(
                "Topic regex(es) to decode for header.stamp age. "
                "With no topics, requires bridge time capability and sends no topic subscriptions."
            ),
        ),
    ] = [],  # noqa: B006
    *,
    topic_options: Annotated[
        list[str],
        Parameter(
            name=["-t", "--topics"],
            group=FILTER_GROUP,
            help="Additional topic regex(es) to decode for header.stamp age.",
        ),
    ] = [],  # noqa: B006
    against: Annotated[
        DelayReference,
        Parameter(
            name=["--against"],
            group=FILTER_GROUP,
            help="Reference time for header.stamp age when topics are supplied.",
        ),
    ] = DelayReference.LOCAL,
    duration: Annotated[
        float,
        Parameter(
            name=["-d", "--duration"],
            group=DISPLAY_GROUP,
            help="Seconds to sample before printing the report.",
        ),
    ] = 5.0,
    json_output: Annotated[
        bool,
        Parameter(name=["--json"], group=DISPLAY_GROUP, help="Print JSON instead of a table."),
    ] = False,
    connect_timeout: Annotated[
        float,
        Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
    ] = 5.0,
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
    pymcap-cli bridge delay robot:8765 /camera/image_raw
    pymcap-cli bridge delay robot:8765 -t '^/imu' --against bridge
    pymcap-cli bridge delay robot:8765 --duration 10 --json
    ```
    """
    if duration <= 0:
        ERR.print("[red]Error:[/] --duration must be positive")
        return 1

    topic_filters = [*topics, *topic_options]
    try:
        topic_patterns = _compile_topic_patterns(topic_filters)
    except re.error:
        logger.exception("Invalid topic regex")
        return 1

    url = to_ws_url(target)
    try:
        report = asyncio.run(
            _collect_delay_async(
                url,
                topic_patterns=topic_patterns,
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
