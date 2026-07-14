"""Shared MCAP playback engine for bridge client and server transports."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rich.console import Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text
from small_mcap import Channel, Message, Schema, Statistics, Summary, get_summary, read_message

from pymcap_cli.cmd.bridge._shared import console
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.rosbag2_layout import expand_bag_paths
from pymcap_cli.utils import bytes_to_human

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymcap_cli.core.message_filter import (
        MessageFilterOptions,
        ResolvedMessageFilterOptions,
    )

_SETTLE_SECONDS = 1.0


class PlaybackError(RuntimeError):
    """Playback configuration, input, or transport failure."""


@dataclass(frozen=True, slots=True)
class PlaybackChannel:
    topic: str
    message_encoding: str
    schema_name: str
    schema_encoding: str
    schema_text: str


@dataclass(frozen=True, slots=True)
class PreparedPlayback:
    files: tuple[str, ...]
    channels: tuple[PlaybackChannel, ...]
    message_filter: MessageFilterOptions
    resolved_filter: ResolvedMessageFilterOptions
    recording_start_ns: int
    recording_end_ns: int


@dataclass(slots=True)
class PlaybackStats:
    state: str = "Preparing"
    loop_number: int = 1
    messages: int = 0
    payload_bytes: int = 0
    playhead_ns: int = 0
    current_lag: float = 0.0
    max_lag: float = 0.0


@dataclass(frozen=True, slots=True)
class PlaybackClock:
    record_origin_ns: int
    wall_origin: float
    speed: float
    recording_end_ns: int

    def deadline(self, timestamp_ns: int) -> float:
        elapsed = (timestamp_ns - self.record_origin_ns) / 1_000_000_000 / self.speed
        return self.wall_origin + elapsed

    def current_time_ns(self, now: float | None = None) -> int:
        wall_now = time.monotonic() if now is None else now
        elapsed_ns = int((wall_now - self.wall_origin) * self.speed * 1_000_000_000)
        return min(self.record_origin_ns + max(0, elapsed_ns), self.recording_end_ns)


class PlaybackSink(Protocol):
    async def start(self, channels: tuple[PlaybackChannel, ...]) -> None: ...

    async def wait_until_ready(self) -> None: ...

    async def publish(
        self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
    ) -> None: ...

    async def timeline_started(self, clock: PlaybackClock) -> None: ...

    async def timeline_finished(self, timestamp_ns: int) -> None: ...

    async def close(self) -> None: ...

    def status_rows(self) -> tuple[tuple[str, str], ...]: ...


def _playback_channel(schema: Schema | None, channel: Channel) -> PlaybackChannel:
    try:
        schema_text = "" if schema is None else schema.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PlaybackError(
            f"Schema for topic {channel.topic!r} is not UTF-8 and cannot be advertised"
        ) from exc
    return PlaybackChannel(
        topic=channel.topic,
        message_encoding=channel.message_encoding,
        schema_name="" if schema is None else schema.name,
        schema_encoding="" if schema is None else schema.encoding,
        schema_text=schema_text,
    )


def _input_description(
    path: str,
    message_filter: MessageFilterOptions,
) -> tuple[int | None, int | None, list[PlaybackChannel]]:
    with open_input(path) as (stream, _size):
        summary = get_summary(stream)
        if summary is not None and summary.statistics is not None and summary.channels:
            stats = summary.statistics
            channels = [
                _playback_channel(summary.schemas.get(channel.schema_id), channel)
                for channel in summary.channels.values()
                if message_filter.matches_topic(channel.topic)
            ]
            if stats.message_count == 0:
                return None, None, channels
            return stats.message_start_time, stats.message_end_time, channels

        stream.seek(0)
        start_ns: int | None = None
        end_ns: int | None = None
        channels_by_id: dict[int, PlaybackChannel] = {}
        for schema, channel, message in read_message(stream):
            if message_filter.matches_topic(channel.topic):
                channels_by_id.setdefault(channel.id, _playback_channel(schema, channel))
            start_ns = message.log_time if start_ns is None else min(start_ns, message.log_time)
            end_ns = message.log_time if end_ns is None else max(end_ns, message.log_time)
        return start_ns, end_ns, list(channels_by_id.values())


def prepare_playback(files: list[str], message_filter: MessageFilterOptions) -> PreparedPlayback:
    expanded = expand_bag_paths(files)
    if not expanded:
        raise PlaybackError("At least one MCAP input is required")

    starts: list[int] = []
    ends: list[int] = []
    by_topic: dict[str, PlaybackChannel] = {}
    for path in expanded:
        start_ns, end_ns, channels = _input_description(path, message_filter)
        if start_ns is not None and end_ns is not None:
            starts.append(start_ns)
            ends.append(end_ns)
        for channel in channels:
            previous = by_topic.get(channel.topic)
            if previous is not None and previous != channel:
                raise PlaybackError(
                    f"Topic {channel.topic!r} has incompatible channel definitions across inputs"
                )
            by_topic[channel.topic] = channel

    if not starts:
        raise PlaybackError("Inputs contain no messages")
    if not by_topic:
        raise PlaybackError("Topic filters selected no channels")

    recording_start_ns = min(starts)
    recording_end_ns = max(ends)
    aggregate = Summary(
        statistics=Statistics(
            message_count=1,
            schema_count=0,
            channel_count=0,
            attachment_count=0,
            metadata_count=0,
            chunk_count=0,
            message_start_time=recording_start_ns,
            message_end_time=recording_end_ns,
            channel_message_counts={},
        )
    )
    try:
        resolved = message_filter.resolve(aggregate)
    except ValueError as exc:
        raise PlaybackError(str(exc)) from exc

    prepared = PreparedPlayback(
        files=tuple(expanded),
        channels=tuple(sorted(by_topic.values(), key=lambda item: item.topic)),
        message_filter=message_filter,
        resolved_filter=resolved,
        recording_start_ns=recording_start_ns,
        recording_end_ns=recording_end_ns,
    )
    with open_playback_messages(prepared) as messages:
        try:
            next(messages)
        except StopIteration as exc:
            raise PlaybackError("Filters selected no messages") from exc
    return prepared


@contextmanager
def open_playback_messages(
    prepared: PreparedPlayback,
) -> Iterator[Iterator[tuple[Schema | None, Channel, Message]]]:
    with ExitStack() as stack:
        streams = [stack.enter_context(open_input(path))[0] for path in prepared.files]
        resolved = prepared.resolved_filter
        end_time_ns = 2**63 - 1 if resolved.early_bail else resolved.end_time_ns
        messages = iter(
            read_message(
                streams,
                should_include=prepared.message_filter.create_channel_predicate(),
                start_time_ns=resolved.start_time_ns,
                end_time_ns=end_time_ns,
            )
        )
        yield messages


def _build_status(
    prepared: PreparedPlayback,
    sink: PlaybackSink,
    stats: PlaybackStats,
    speed: float,
) -> RenderableType:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold blue")
    table.add_column()
    for label, value in sink.status_rows():
        table.add_row(f"{label}:", value)
    table.add_row("State:", stats.state)
    table.add_row("Inputs:", str(len(prepared.files)))
    table.add_row("Topics:", str(len(prepared.channels)))
    table.add_row("Speed:", f"{speed:g}x")
    table.add_row("Loop:", str(stats.loop_number))
    table.add_row("Messages:", f"{stats.messages:,}")
    table.add_row("Payload:", bytes_to_human(stats.payload_bytes))
    table.add_row("Lag:", f"{stats.current_lag * 1000:.1f} ms (max {stats.max_lag * 1000:.1f} ms)")
    return Group(table, Text("Ctrl+C to stop", style="dim"))


async def run_playback(
    prepared: PreparedPlayback,
    sink: PlaybackSink,
    *,
    speed: float,
    loop: bool,
    show_status: bool,
) -> PlaybackStats:
    if not math.isfinite(speed) or speed <= 0:
        raise PlaybackError("--speed must be finite and positive")

    stats = PlaybackStats()
    done = asyncio.Event()
    live: Live | None = None
    status_task: asyncio.Task[None] | None = None
    channel_cache: dict[tuple[int, int], PlaybackChannel] = {}

    async def refresh_status() -> None:
        assert live is not None
        while not done.is_set():
            live.update(_build_status(prepared, sink, stats, speed))
            try:
                await asyncio.wait_for(done.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

    try:
        await sink.start(prepared.channels)
        stats.state = "Waiting"
        await sink.wait_until_ready()
        if show_status:
            live = Live(console=console, refresh_per_second=4)
            live.start()
            status_task = asyncio.create_task(refresh_status())

        while True:
            stats.state = "Playing"
            first_time_ns: int | None = None
            last_time_ns: int | None = None
            clock: PlaybackClock | None = None
            with open_playback_messages(prepared) as messages:
                for schema, channel, message in messages:
                    if (
                        prepared.resolved_filter.early_bail
                        and message.log_time >= prepared.resolved_filter.end_time_ns
                    ):
                        break
                    if first_time_ns is None:
                        first_time_ns = message.log_time
                        clock = PlaybackClock(
                            record_origin_ns=first_time_ns,
                            wall_origin=time.monotonic(),
                            speed=speed,
                            recording_end_ns=prepared.recording_end_ns,
                        )
                        await sink.timeline_started(clock)
                    assert clock is not None
                    deadline = clock.deadline(message.log_time)
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    lag = max(0.0, time.monotonic() - deadline)
                    stats.current_lag = lag
                    stats.max_lag = max(stats.max_lag, lag)
                    cache_key = (0 if schema is None else schema.id, channel.id)
                    playback_channel = channel_cache.get(cache_key)
                    if playback_channel is None:
                        playback_channel = _playback_channel(schema, channel)
                        channel_cache[cache_key] = playback_channel
                    await sink.publish(playback_channel, message.log_time, message.data)
                    stats.messages += 1
                    stats.payload_bytes += len(message.data)
                    stats.playhead_ns = message.log_time
                    last_time_ns = message.log_time

            if first_time_ns is None or last_time_ns is None:
                raise PlaybackError("Filters selected no messages")
            await sink.timeline_finished(last_time_ns)
            if not loop:
                break
            if last_time_ns == first_time_ns:
                raise PlaybackError("Cannot loop a zero-duration selection")
            stats.loop_number += 1

        stats.state = "Finished"
        return stats
    finally:
        done.set()
        if status_task is not None:
            await status_task
        if live is not None:
            live.update(_build_status(prepared, sink, stats, speed))
            live.stop()
        await sink.close()
