"""Shared rolling measurements for live bridge topic traffic."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from robo_ws_bridge import WebSocketBridgeClient
from robo_ws_bridge.ws_types import ServerCapabilities
from small_mcap import JSONDecoderFactory

from pymcap_cli.cmd._arg_constraints import MutuallyExclusive, at_least_one, constraint_group
from pymcap_cli.cmd.bridge._shared import (
    BridgeFetchError,
    ChannelDecoderCache,
    ChannelSubscriptionManager,
    console,
    to_ws_url,
)
from pymcap_cli.constants import NS_TO_SEC
from pymcap_cli.core.message_filter import MessageFilterOptions
from pymcap_cli.display.display_utils import _format_parts_with_colors
from pymcap_cli.log_setup import ERR
from pymcap_cli.types.to_plain import to_plain
from pymcap_cli.utils import bytes_to_human

if TYPE_CHECKING:
    from collections.abc import Callable

    from robo_ws_bridge.ws_types import ChannelInfo

TOPIC_SELECTION_CONSTRAINT = constraint_group(at_least_one, MutuallyExclusive())
logger = logging.getLogger(__name__)


class TopicMonitorView(str, Enum):
    HZ = "hz"
    BW = "bw"
    STATS = "stats"


class TimeReference(str, Enum):
    LOCAL = "local"
    BRIDGE = "bridge"


@dataclass(slots=True)
class RunningTimeStats:
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


@dataclass(slots=True)
class ChannelDelayStats:
    channel: ChannelInfo
    clock_offset: RunningTimeStats = field(default_factory=RunningTimeStats)
    header_age: RunningTimeStats = field(default_factory=RunningTimeStats)
    payload_bytes: int = 0
    undecodable_messages: int = 0
    decode_errors: int = 0
    missing_header_stamp: int = 0


@dataclass(frozen=True, slots=True)
class _TopicEvent:
    arrival_ns: int
    payload_size: int
    bridge_age_ns: int


@dataclass(frozen=True, slots=True)
class _TimedValue:
    observed_at_ns: int
    value_ns: int


@dataclass(slots=True)
class _TopicState:
    topic: str
    observed_since_ns: int
    events: deque[_TopicEvent] = field(default_factory=deque)
    total_messages: int = 0
    last_arrival_ns: int | None = None


@dataclass(frozen=True, slots=True)
class TopicMetrics:
    topic: str
    message_count: int
    total_messages: int
    hz: float | None
    payload_bytes_per_second: float | None
    message_size_mean: float | None
    message_size_min: int | None
    message_size_max: int | None
    period_mean_ns: float | None
    period_min_ns: int | None
    period_max_ns: int | None
    period_stddev_ns: float | None
    bridge_age_mean_ns: float | None
    bridge_age_min_ns: int | None
    bridge_age_max_ns: int | None
    last_age_ns: int | None


@dataclass(frozen=True, slots=True)
class TopicMonitorSnapshot:
    sampled_at_ns: int
    window_seconds: float
    topics: tuple[TopicMetrics, ...]
    bridge_clock_offset_mean_ns: float | None = None


class TopicMonitor:
    """Aggregate arrival, payload-size, and bridge-timestamp observations by topic."""

    def __init__(self, *, window_seconds: float) -> None:
        self.window_ns = int(window_seconds * NS_TO_SEC)
        self._states: dict[str, _TopicState] = {}

    @property
    def window_seconds(self) -> float:
        return self.window_ns / NS_TO_SEC

    def register_channel(self, channel: ChannelInfo, *, now_ns: int) -> None:
        topic = channel["topic"]
        if topic not in self._states:
            self._states[topic] = _TopicState(topic=topic, observed_since_ns=now_ns)

    def observe(
        self,
        channel: ChannelInfo,
        bridge_timestamp_ns: int,
        payload_size: int,
        *,
        arrival_ns: int,
        wall_time_ns: int,
    ) -> None:
        self.register_channel(channel, now_ns=arrival_ns)
        state = self._states[channel["topic"]]
        state.events.append(
            _TopicEvent(
                arrival_ns=arrival_ns,
                payload_size=payload_size,
                bridge_age_ns=wall_time_ns - bridge_timestamp_ns,
            )
        )
        state.total_messages += 1
        state.last_arrival_ns = arrival_ns
        self._prune(state, arrival_ns)

    def snapshot(self, *, now_ns: int) -> TopicMonitorSnapshot:
        topics = tuple(
            self._metrics_for(state, now_ns)
            for state in sorted(self._states.values(), key=lambda item: item.topic)
        )
        return TopicMonitorSnapshot(
            sampled_at_ns=now_ns,
            window_seconds=self.window_seconds,
            topics=topics,
        )

    def _prune(self, state: _TopicState, now_ns: int) -> None:
        cutoff_ns = now_ns - self.window_ns
        while state.events and state.events[0].arrival_ns <= cutoff_ns:
            state.events.popleft()

    def _metrics_for(self, state: _TopicState, now_ns: int) -> TopicMetrics:
        self._prune(state, now_ns)
        events = tuple(state.events)
        observation_ns = min(self.window_ns, max(0, now_ns - state.observed_since_ns))
        observation_seconds = observation_ns / NS_TO_SEC
        hz = len(events) / observation_seconds if observation_seconds > 0 else None
        payload_bytes_per_second = (
            sum(event.payload_size for event in events) / observation_seconds
            if observation_seconds > 0
            else None
        )

        sizes = [event.payload_size for event in events]
        periods = [
            current.arrival_ns - previous.arrival_ns for previous, current in pairwise(events)
        ]
        bridge_ages = [event.bridge_age_ns for event in events]

        return TopicMetrics(
            topic=state.topic,
            message_count=len(events),
            total_messages=state.total_messages,
            hz=hz,
            payload_bytes_per_second=payload_bytes_per_second,
            message_size_mean=sum(sizes) / len(sizes) if sizes else None,
            message_size_min=min(sizes) if sizes else None,
            message_size_max=max(sizes) if sizes else None,
            period_mean_ns=sum(periods) / len(periods) if periods else None,
            period_min_ns=min(periods) if periods else None,
            period_max_ns=max(periods) if periods else None,
            period_stddev_ns=_stddev(periods),
            bridge_age_mean_ns=sum(bridge_ages) / len(bridge_ages) if bridge_ages else None,
            bridge_age_min_ns=min(bridge_ages) if bridge_ages else None,
            bridge_age_max_ns=max(bridge_ages) if bridge_ages else None,
            last_age_ns=(
                max(0, now_ns - state.last_arrival_ns)
                if state.last_arrival_ns is not None
                else None
            ),
        )


def _stddev(values: list[int]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


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


class BridgeMonitorSession:
    """Own one bridge connection and the metric capabilities enabled for its view."""

    def __init__(
        self,
        url: str,
        *,
        message_filter: MessageFilterOptions,
        window_seconds: float,
        connect_timeout: float,
        subscribe_messages: bool = True,
        collect_time: bool = True,
        require_time_capability: bool = False,
        decode_header_stamps: bool = False,
        header_reference: TimeReference = TimeReference.LOCAL,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.url = url
        self.message_filter = message_filter
        self.connect_timeout = connect_timeout
        self.subscribe_messages = subscribe_messages
        self.collect_time = collect_time
        self.require_time_capability = require_time_capability
        self.decode_header_stamps = decode_header_stamps
        self.header_reference = header_reference
        self.monotonic_ns = monotonic_ns
        self.wall_time_ns = wall_time_ns

        self.topic_monitor = TopicMonitor(window_seconds=window_seconds)
        self.time_offset = RunningTimeStats()
        self._time_offset_window: deque[_TimedValue] = deque()
        self._delay_stats_by_channel: dict[int, ChannelDelayStats] = {}
        self._client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
        self._server_info_event = asyncio.Event()
        self._client.on_server_info(lambda *_: self._server_info_event.set())

        self._decoder_cache: ChannelDecoderCache | None = None
        if decode_header_stamps:
            self._decoder_cache = ChannelDecoderCache([JSONDecoderFactory(), DecoderFactory()])

        self._subscriber: ChannelSubscriptionManager | None = None
        if subscribe_messages:
            self._subscriber = ChannelSubscriptionManager(
                self._client,
                self._should_subscribe,
            )
            self._subscriber.install()
            self._client.on_message(self._on_message)
        if collect_time:
            self._client.on_time_update(self._on_time_update)

    @property
    def delay_channels(self) -> tuple[ChannelDelayStats, ...]:
        return tuple(
            sorted(
                self._delay_stats_by_channel.values(),
                key=lambda stats: (stats.channel["topic"], stats.channel["id"]),
            )
        )

    def _stats_for(self, channel: ChannelInfo) -> ChannelDelayStats:
        channel_id = channel["id"]
        stats = self._delay_stats_by_channel.get(channel_id)
        if stats is None:
            stats = ChannelDelayStats(channel=channel)
            self._delay_stats_by_channel[channel_id] = stats
        return stats

    def _should_subscribe(self, channel: ChannelInfo) -> bool:
        if not self.message_filter.matches_topic(channel["topic"]):
            return False
        self.topic_monitor.register_channel(channel, now_ns=self.monotonic_ns())
        self._stats_for(channel)
        return True

    def _on_message(
        self,
        channel: ChannelInfo,
        bridge_timestamp_ns: int,
        payload: bytes,
    ) -> None:
        arrival_ns = self.monotonic_ns()
        local_receive_ns = self.wall_time_ns()
        self.topic_monitor.observe(
            channel,
            bridge_timestamp_ns,
            len(payload),
            arrival_ns=arrival_ns,
            wall_time_ns=local_receive_ns,
        )

        stats = self._stats_for(channel)
        stats.payload_bytes += len(payload)
        stats.clock_offset.add(local_receive_ns - bridge_timestamp_ns)

        decoder_cache = self._decoder_cache
        if decoder_cache is None:
            return
        decoder = decoder_cache.decoder_for(channel)
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

        reference_ns = (
            local_receive_ns
            if self.header_reference is TimeReference.LOCAL
            else bridge_timestamp_ns
        )
        stats.header_age.add(reference_ns - stamp_ns)

    def _on_time_update(self, server_time_ns: int) -> None:
        observed_at_ns = self.monotonic_ns()
        offset_ns = self.wall_time_ns() - server_time_ns
        self.time_offset.add(offset_ns)
        self._time_offset_window.append(
            _TimedValue(observed_at_ns=observed_at_ns, value_ns=offset_ns)
        )

    def snapshot(self, *, now_ns: int | None = None) -> TopicMonitorSnapshot:
        sampled_at_ns = self.monotonic_ns() if now_ns is None else now_ns
        snapshot = self.topic_monitor.snapshot(now_ns=sampled_at_ns)
        cutoff_ns = sampled_at_ns - self.topic_monitor.window_ns
        while self._time_offset_window and self._time_offset_window[0].observed_at_ns <= cutoff_ns:
            self._time_offset_window.popleft()
        offset_mean = (
            sum(sample.value_ns for sample in self._time_offset_window)
            / len(self._time_offset_window)
            if self._time_offset_window
            else None
        )
        return replace(snapshot, bridge_clock_offset_mean_ns=offset_mean)

    async def run(
        self,
        *,
        interval_seconds: float,
        duration_seconds: float | None,
        on_snapshot: Callable[[TopicMonitorSnapshot], None],
    ) -> TopicMonitorSnapshot:
        await self._client.connect()
        try:
            try:
                await asyncio.wait_for(
                    self._server_info_event.wait(),
                    timeout=self.connect_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise BridgeFetchError(
                    f"Timed out after {self.connect_timeout:.1f}s "
                    f"waiting for serverInfo from {self.url}"
                ) from exc

            server_info = self._client.server_info
            if server_info is None:
                raise BridgeFetchError(f"No serverInfo received from {self.url}")
            if (
                self.require_time_capability
                and ServerCapabilities.TIME.value not in server_info["capabilities"]
            ):
                raise BridgeFetchError(
                    f"{self.url} does not advertise the "
                    f"'{ServerCapabilities.TIME.value}' capability. "
                    "Pass topic regexes to measure delay from message timestamps instead."
                )

            if self._subscriber is not None:
                await self._subscriber.subscribe_existing()

            started_ns = self.monotonic_ns()
            last_snapshot = self.snapshot(now_ns=started_ns)
            while True:
                sleep_seconds = interval_seconds
                if duration_seconds is not None:
                    now_ns = self.monotonic_ns()
                    remaining_seconds = duration_seconds - (now_ns - started_ns) / NS_TO_SEC
                    if remaining_seconds <= 0:
                        return last_snapshot
                    sleep_seconds = min(sleep_seconds, remaining_seconds)

                await asyncio.sleep(sleep_seconds)
                now_ns = self.monotonic_ns()
                last_snapshot = self.snapshot(now_ns=now_ns)
                on_snapshot(last_snapshot)
                if (
                    duration_seconds is not None
                    and (now_ns - started_ns) / NS_TO_SEC >= duration_seconds
                ):
                    return last_snapshot
        finally:
            await self._client.disconnect()


async def _collect_topic_metrics_async(
    url: str,
    *,
    message_filter: MessageFilterOptions,
    window_seconds: float,
    interval_seconds: float,
    duration_seconds: float | None,
    connect_timeout: float,
    on_snapshot: Callable[[TopicMonitorSnapshot], None],
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    wall_time_ns: Callable[[], int] = time.time_ns,
) -> TopicMonitorSnapshot:
    session = BridgeMonitorSession(
        url,
        message_filter=message_filter,
        window_seconds=window_seconds,
        connect_timeout=connect_timeout,
        monotonic_ns=monotonic_ns,
        wall_time_ns=wall_time_ns,
    )
    return await session.run(
        interval_seconds=interval_seconds,
        duration_seconds=duration_seconds,
        on_snapshot=on_snapshot,
    )


def _format_rate(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_duration_ns(value_ns: float | None) -> str:
    if value_ns is None:
        return "-"
    magnitude = abs(value_ns)
    sign = "-" if value_ns < 0 else ""
    if magnitude < 1_000:
        return f"{sign}{magnitude:.0f}ns"
    if magnitude < 1_000_000:
        return f"{sign}{magnitude / 1_000:.1f}us"
    if magnitude < NS_TO_SEC:
        return f"{sign}{magnitude / 1_000_000:.1f}ms"
    return f"{sign}{magnitude / NS_TO_SEC:.2f}s"


def _format_bytes(value: float | None) -> str:
    return "-" if value is None else bytes_to_human(value)


def _format_last_age(value_ns: int | None) -> str:
    return "never" if value_ns is None else _format_duration_ns(value_ns)


def _topic_column_width(snapshot: TopicMonitorSnapshot) -> int:
    widest_topic = max((len(topic.topic) for topic in snapshot.topics), default=5)
    return min(max(widest_topic, 5), 52)


def _add_stream_column(
    table: Table,
    header: str,
    *,
    width: int,
    justify: Literal["left", "center", "right", "full"] = "right",
) -> None:
    table.add_column(
        Text(header, justify="center"),
        width=width,
        justify=justify,
        no_wrap=True,
        overflow="ellipsis",
    )


def _stream_table(snapshot: TopicMonitorSnapshot, *, show_header: bool) -> Table:
    table = Table(
        box=None,
        show_header=show_header,
        padding=(0, 1),
        collapse_padding=True,
    )
    _add_stream_column(table, "Time", width=8, justify="left")
    _add_stream_column(
        table,
        "Topic",
        width=_topic_column_width(snapshot),
        justify="left",
    )
    return table


def _build_hz_table(
    snapshot: TopicMonitorSnapshot,
    *,
    show_header: bool = True,
) -> Table:
    table = _stream_table(snapshot, show_header=show_header)
    _add_stream_column(table, "Hz", width=7)
    _add_stream_column(table, "Period avg", width=10)
    _add_stream_column(table, "Min", width=10)
    _add_stream_column(table, "Max", width=10)
    _add_stream_column(table, "Stddev", width=10)
    _add_stream_column(table, "N", width=7)
    _add_stream_column(table, "Last", width=10)
    sampled_at = time.strftime("%H:%M:%S")
    for topic in snapshot.topics:
        table.add_row(
            sampled_at,
            _format_parts_with_colors(topic.topic),
            _format_rate(topic.hz),
            _format_duration_ns(topic.period_mean_ns),
            _format_duration_ns(topic.period_min_ns),
            _format_duration_ns(topic.period_max_ns),
            _format_duration_ns(topic.period_stddev_ns),
            str(topic.message_count),
            _format_last_age(topic.last_age_ns),
        )
    return table


def _build_bw_table(
    snapshot: TopicMonitorSnapshot,
    *,
    show_header: bool = True,
) -> Table:
    table = _stream_table(snapshot, show_header=show_header)
    _add_stream_column(table, "Payload/s", width=12)
    _add_stream_column(table, "Size avg", width=10)
    _add_stream_column(table, "Min", width=10)
    _add_stream_column(table, "Max", width=10)
    _add_stream_column(table, "N", width=7)
    _add_stream_column(table, "Last", width=10)
    sampled_at = time.strftime("%H:%M:%S")
    for topic in snapshot.topics:
        bandwidth = (
            f"{_format_bytes(topic.payload_bytes_per_second)}/s"
            if topic.payload_bytes_per_second is not None
            else "-"
        )
        table.add_row(
            sampled_at,
            _format_parts_with_colors(topic.topic),
            bandwidth,
            _format_bytes(topic.message_size_mean),
            _format_bytes(topic.message_size_min),
            _format_bytes(topic.message_size_max),
            str(topic.message_count),
            _format_last_age(topic.last_age_ns),
        )
    return table


def _build_stats_table(
    snapshot: TopicMonitorSnapshot,
    *,
    show_header: bool = True,
) -> Table:
    table = _stream_table(snapshot, show_header=show_header)
    _add_stream_column(table, "Hz", width=7)
    _add_stream_column(table, "Payload/s", width=12)
    _add_stream_column(table, "Size avg", width=10)
    _add_stream_column(table, "Delay", width=10)
    _add_stream_column(table, "N", width=7)
    _add_stream_column(table, "Last", width=10)
    sampled_at = time.strftime("%H:%M:%S")
    for topic in snapshot.topics:
        bandwidth = (
            f"{_format_bytes(topic.payload_bytes_per_second)}/s"
            if topic.payload_bytes_per_second is not None
            else "-"
        )
        message_delay_ns = (
            topic.bridge_age_mean_ns - snapshot.bridge_clock_offset_mean_ns
            if topic.bridge_age_mean_ns is not None
            and snapshot.bridge_clock_offset_mean_ns is not None
            else None
        )
        table.add_row(
            sampled_at,
            _format_parts_with_colors(topic.topic),
            _format_rate(topic.hz),
            bandwidth,
            _format_bytes(topic.message_size_mean),
            _format_duration_ns(message_delay_ns),
            str(topic.message_count),
            _format_last_age(topic.last_age_ns),
        )
    return table


def build_topic_monitor_display(
    snapshot: TopicMonitorSnapshot,
    *,
    view: TopicMonitorView,
    url: str,
    show_context: bool = True,
    show_header: bool = True,
) -> RenderableType:
    if view is TopicMonitorView.HZ:
        title = "Topic frequency"
        table = _build_hz_table(snapshot, show_header=show_header)
    elif view is TopicMonitorView.BW:
        title = "Topic payload bandwidth"
        table = _build_bw_table(snapshot, show_header=show_header)
    else:
        title = "Topic statistics"
        table = _build_stats_table(snapshot, show_header=show_header)

    status = Text.from_markup(
        f"[bold cyan]{title}[/] [dim]| Bridge: {url}"
        f" | rolling window: {snapshot.window_seconds:g}s"
        " | messages stream continuously | Ctrl+C to stop[/]"
    )
    if not snapshot.topics:
        no_topics = Text(
            f"{time.strftime('%H:%M:%S')}  No matching channels advertised.",
            style="yellow",
        )
        return Group(status, no_topics) if show_context else no_topics
    if view is TopicMonitorView.STATS:
        if show_context:
            return Group(
                status,
                table,
                Text(
                    "Delay = local receive - bridge message timestamp - measured clock offset.",
                    style="dim",
                ),
            )
        return table
    return Group(status, table) if show_context else table


def topic_monitor_snapshot_to_dict(
    snapshot: TopicMonitorSnapshot,
    *,
    url: str,
    view: TopicMonitorView,
) -> dict[str, object]:
    return {
        "url": url,
        "view": view.value,
        "sampled_at_monotonic_ns": snapshot.sampled_at_ns,
        "window_seconds": snapshot.window_seconds,
        "bridge_clock_offset_ms": _ns_to_ms(snapshot.bridge_clock_offset_mean_ns),
        "topics": [
            {
                "topic": topic.topic,
                "message_count": topic.message_count,
                "total_messages": topic.total_messages,
                "hz": topic.hz,
                "payload_bytes_per_second": topic.payload_bytes_per_second,
                "message_size_bytes": {
                    "mean": topic.message_size_mean,
                    "min": topic.message_size_min,
                    "max": topic.message_size_max,
                },
                "period_ms": {
                    "mean": _ns_to_ms(topic.period_mean_ns),
                    "min": _ns_to_ms(topic.period_min_ns),
                    "max": _ns_to_ms(topic.period_max_ns),
                    "stddev": _ns_to_ms(topic.period_stddev_ns),
                },
                "bridge_age_ms": {
                    "mean": _ns_to_ms(topic.bridge_age_mean_ns),
                    "min": _ns_to_ms(topic.bridge_age_min_ns),
                    "max": _ns_to_ms(topic.bridge_age_max_ns),
                },
                "message_delay_ms": (
                    _ns_to_ms(topic.bridge_age_mean_ns - snapshot.bridge_clock_offset_mean_ns)
                    if topic.bridge_age_mean_ns is not None
                    and snapshot.bridge_clock_offset_mean_ns is not None
                    else None
                ),
                "last_age_ms": _ns_to_ms(topic.last_age_ns),
            }
            for topic in snapshot.topics
        ],
    }


def _ns_to_ms(value_ns: float | None) -> float | None:
    return None if value_ns is None else value_ns / 1_000_000


def run_topic_monitor(
    target: str,
    *,
    topic: list[str] | None,
    all_topics: bool,
    exclude_topic: list[str] | None,
    window: float,
    interval: float,
    duration: float | None,
    json_output: bool,
    connect_timeout: float,
    view: TopicMonitorView,
) -> int:
    if window <= 0:
        ERR.print("[red]Error:[/] --window must be positive")
        return 1
    if interval <= 0:
        ERR.print("[red]Error:[/] --interval must be positive")
        return 1
    if duration is not None and duration <= 0:
        ERR.print("[red]Error:[/] --duration must be positive")
        return 1

    try:
        message_filter = MessageFilterOptions.from_args(
            topic=None if all_topics else topic,
            exclude_topic=exclude_topic,
        )
    except ValueError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1

    url = to_ws_url(target)
    has_printed_context = False
    has_printed_header = False

    def emit(snapshot: TopicMonitorSnapshot) -> None:
        nonlocal has_printed_context, has_printed_header
        if json_output:
            print(  # noqa: T201
                json.dumps(
                    topic_monitor_snapshot_to_dict(snapshot, url=url, view=view),
                    separators=(",", ":"),
                )
            )
        else:
            console.print(
                build_topic_monitor_display(
                    snapshot,
                    view=view,
                    url=url,
                    show_context=not has_printed_context,
                    show_header=not has_printed_header and bool(snapshot.topics),
                )
            )
        has_printed_context = True
        has_printed_header = has_printed_header or bool(snapshot.topics)

    try:
        asyncio.run(
            _collect_topic_metrics_async(
                url,
                message_filter=message_filter,
                window_seconds=window,
                interval_seconds=interval,
                duration_seconds=duration,
                connect_timeout=connect_timeout,
                on_snapshot=emit,
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
    return 0
