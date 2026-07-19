"""Shared MCAP playback engine for bridge client and server transports."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

from rich.console import Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text
from small_mcap import Channel, Message, Schema, Statistics, Summary, get_summary, read_message

from pymcap_cli.cmd.bridge._shared import console
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.rosbag2_layout import expand_bag_paths
from pymcap_cli.rihs01 import compute_rihs01
from pymcap_cli.utils import bytes_to_human

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymcap_cli.core.message_filter import (
        MessageFilterOptions,
        ResolvedMessageFilterOptions,
    )

_SETTLE_SECONDS = 1.0
_MAX_PLAYBACK_LAG_SECONDS = 0.1
_STRESSED_FRAME_INTERVAL_SECONDS = 0.25
# Schemas whose messages stand alone: dropping a late or superseded frame is
# safe because the next message fully replaces it. Video streams are absent on
# purpose — every encoded packet must reach the decoder.
_FRAME_SCHEMAS = frozenset(
    {
        "sensor_msgs/Image",
        "sensor_msgs/msg/Image",
        "sensor_msgs/CompressedImage",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/PointCloud2",
        "sensor_msgs/msg/PointCloud2",
        "sensor_msgs/LaserScan",
        "sensor_msgs/msg/LaserScan",
        "point_cloud_interfaces/msg/CompressedPointCloud2",
        "foxglove_msgs/msg/CompressedPointCloud",
        "foxglove.RawImage",
        "foxglove.CompressedImage",
        "foxglove.PointCloud",
    }
)
PlaybackControllerState: TypeAlias = Literal["Paused", "Playing", "Stopped"]
PlaybackState: TypeAlias = Literal[
    "Preparing",
    "Waiting",
    "Playing",
    "Paused",
    "Stopped",
    "Finished",
    "Error",
]


class PlaybackError(RuntimeError):
    """Playback configuration, input, or transport failure."""


def is_frame_channel(channel: PlaybackChannel) -> bool:
    """True when each message on the channel stands alone and may be dropped."""
    return channel.schema_name in _FRAME_SCHEMAS


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
    state: PlaybackState = "Preparing"
    loop_number: int = 1
    messages: int = 0
    dropped_messages: int = 0
    payload_bytes: int = 0
    playhead_ns: int = 0
    current_lag: float = 0.0
    max_lag: float = 0.0


@dataclass(slots=True)
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

    def delay(self, duration: float) -> None:
        self.wall_origin += duration

    def set_speed(self, speed: float, now: float | None = None) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        wall_now = time.monotonic() if now is None else now
        self.record_origin_ns = self.current_time_ns(wall_now)
        self.wall_origin = wall_now
        self.speed = speed


class _PlaybackStoppedError(Exception):
    """Internal signal used to end controlled playback without an error."""


class PlaybackController:
    """Pause, resume, and stop one playback task."""

    def __init__(
        self,
        *,
        start_paused: bool = False,
        is_looping: bool | None = None,
        speed: float | None = None,
    ) -> None:
        self._can_play = asyncio.Event()
        self._state: PlaybackControllerState = "Paused" if start_paused else "Playing"
        self._is_looping = is_looping
        self._speed = speed
        self._pause_started = time.monotonic() if start_paused else None
        self._pending_clock_delay = 0.0
        if not start_paused:
            self._can_play.set()

    @property
    def state(self) -> PlaybackControllerState:
        return self._state

    @property
    def is_looping(self) -> bool | None:
        return self._is_looping

    def set_looping(self, is_looping: bool) -> None:
        self._is_looping = is_looping

    @property
    def speed(self) -> float | None:
        return self._speed

    def set_speed(self, speed: float) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        self._speed = speed

    def pause(self) -> None:
        if self._state != "Playing":
            return
        self._state = "Paused"
        self._pause_started = time.monotonic()
        self._can_play.clear()

    def play(self) -> None:
        if self._state != "Paused":
            return
        assert self._pause_started is not None
        self._pending_clock_delay += time.monotonic() - self._pause_started
        self._pause_started = None
        self._state = "Playing"
        self._can_play.set()

    def stop(self) -> None:
        if self._state == "Stopped":
            return
        self._state = "Stopped"
        self._can_play.set()

    async def wait_until_playing(self) -> float:
        """Wait until playback may run and return clock delay accumulated while paused."""
        await self._can_play.wait()
        if self._state == "Stopped":
            raise _PlaybackStoppedError
        delay = self._pending_clock_delay
        self._pending_clock_delay = 0.0
        return delay


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

    def is_channel_active(self, channel: PlaybackChannel) -> bool: ...

    def is_channel_congested(self, channel: PlaybackChannel) -> bool: ...

    async def wait_until_active(self) -> float: ...


@dataclass(frozen=True, slots=True)
class PlaybackOutput:
    channel: PlaybackChannel
    timestamp_ns: int
    payload: bytes | memoryview


class PlaybackTransformSession(Protocol):
    async def observe_congestion(
        self,
        channel: PlaybackChannel,
        *,
        is_congested: bool,
        now: float,
    ) -> None: ...

    async def transform(
        self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
    ) -> tuple[PlaybackOutput, ...]: ...

    def should_drop_frame(self, channel: PlaybackChannel, *, now: float) -> bool: ...

    async def finish(self) -> tuple[PlaybackOutput, ...]: ...

    async def restart(self) -> None: ...

    async def deactivate(self, channel: PlaybackChannel) -> None: ...

    async def close(self) -> None: ...


class PlaybackTransformPlan(Protocol):
    mode: str
    channels: tuple[PlaybackChannel, ...]

    def create_session(self) -> PlaybackTransformSession: ...

    def output_channel(self, source: PlaybackChannel) -> PlaybackChannel: ...


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


def _channel_definitions_are_compatible(
    left: PlaybackChannel,
    right: PlaybackChannel,
) -> bool:
    if (
        left.topic != right.topic
        or left.message_encoding != right.message_encoding
        or left.schema_encoding != right.schema_encoding
    ):
        return False
    if left.schema_name == right.schema_name and left.schema_text == right.schema_text:
        return True
    if left.schema_encoding != "ros2msg":
        return False
    try:
        left_hash = compute_rihs01(left.schema_name, left.schema_text.encode())
        right_hash = compute_rihs01(right.schema_name, right.schema_text.encode())
    except ValueError:
        return False
    return left_hash == right_hash


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
    source_by_topic: dict[str, str] = {}
    for path in expanded:
        start_ns, end_ns, channels = _input_description(path, message_filter)
        if start_ns is not None and end_ns is not None:
            starts.append(start_ns)
            ends.append(end_ns)
        for channel in channels:
            previous = by_topic.get(channel.topic)
            if previous is not None and not _channel_definitions_are_compatible(
                previous,
                channel,
            ):
                raise PlaybackError(
                    f"Topic {channel.topic!r} has incompatible channel definitions "
                    f"in {source_by_topic[channel.topic]!r} and {path!r}"
                )
            by_topic[channel.topic] = channel
            source_by_topic[channel.topic] = path

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
    *,
    start_time_ns: int | None = None,
) -> Iterator[Iterator[tuple[Schema | None, Channel, Message]]]:
    with ExitStack() as stack:
        streams = [stack.enter_context(open_input(path))[0] for path in prepared.files]
        resolved = prepared.resolved_filter
        end_time_ns = 2**63 - 1 if resolved.early_bail else resolved.end_time_ns
        effective_start_ns = (
            resolved.start_time_ns
            if start_time_ns is None
            else max(resolved.start_time_ns, start_time_ns)
        )
        messages = iter(
            read_message(
                streams,
                should_include=prepared.message_filter.create_channel_predicate(),
                start_time_ns=effective_start_ns,
                end_time_ns=end_time_ns,
            )
        )
        yield messages


def _build_status(
    prepared: PreparedPlayback,
    sink: PlaybackSink,
    stats: PlaybackStats,
    speed: float,
    transform_mode: str,
) -> RenderableType:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold blue")
    table.add_column()
    for label, value in sink.status_rows():
        table.add_row(f"{label}:", value)
    table.add_row("State:", stats.state)
    table.add_row("Inputs:", str(len(prepared.files)))
    table.add_row("Topics:", str(len(prepared.channels)))
    table.add_row("Transform:", transform_mode)
    table.add_row("Speed:", f"{speed:g}x")
    table.add_row("Loop:", str(stats.loop_number))
    table.add_row("Messages:", f"{stats.messages:,}")
    table.add_row("Dropped:", f"{stats.dropped_messages:,}")
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
    transform_plan: PlaybackTransformPlan | None = None,
    controller: PlaybackController | None = None,
    stats: PlaybackStats | None = None,
    start_time_ns: int | None = None,
) -> PlaybackStats:
    if not math.isfinite(speed) or speed <= 0:
        raise PlaybackError("--speed must be finite and positive")

    stats = PlaybackStats() if stats is None else stats
    done = asyncio.Event()
    live: Live | None = None
    status_task: asyncio.Task[None] | None = None
    transform_session: PlaybackTransformSession | None = None
    channel_cache: dict[tuple[int, int], PlaybackChannel] = {}
    last_frame_sent: dict[PlaybackChannel, float] = {}
    selected_channels = {channel.topic: channel for channel in prepared.channels}
    transform_mode = "none" if transform_plan is None else transform_plan.mode

    async def refresh_status() -> None:
        assert live is not None
        while not done.is_set():
            live.update(_build_status(prepared, sink, stats, speed, transform_mode))
            try:
                await asyncio.wait_for(done.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

    async def wait_for_deadline(clock: PlaybackClock, timestamp_ns: int) -> tuple[float, float]:
        while True:
            if controller is not None:
                clock.delay(await controller.wait_until_playing())
                if controller.speed is not None and controller.speed != clock.speed:
                    clock.set_speed(controller.speed)
            clock.delay(await sink.wait_until_active())
            deadline = clock.deadline(timestamp_ns)
            delay = deadline - time.monotonic()
            if delay <= 0:
                return deadline, -delay
            await asyncio.sleep(min(delay, 0.1))

    def record_drop(timestamp_ns: int, lag: float) -> None:
        stats.current_lag = lag
        stats.max_lag = max(stats.max_lag, lag)
        stats.dropped_messages += 1
        stats.playhead_ns = timestamp_ns

    try:
        output_channels = prepared.channels if transform_plan is None else transform_plan.channels
        await sink.start(output_channels)
        stats.state = "Waiting"
        await sink.wait_until_ready()
        if controller is not None:
            await controller.wait_until_playing()
        if show_status:
            live = Live(console=console, refresh_per_second=4)
            live.start()
            status_task = asyncio.create_task(refresh_status())
        if transform_plan is not None:
            transform_session = transform_plan.create_session()

        while True:
            stats.state = "Playing"
            first_time_ns: int | None = None
            last_input_time_ns: int | None = None
            last_time_ns: int | None = None
            clock: PlaybackClock | None = None

            async def publish_output(
                output: PlaybackOutput,
                playback_clock: PlaybackClock,
                *,
                deadline_cap_ns: int | None = None,
            ) -> None:
                nonlocal last_time_ns
                deadline_timestamp_ns = (
                    output.timestamp_ns
                    if deadline_cap_ns is None
                    else min(output.timestamp_ns, deadline_cap_ns)
                )
                deadline, _ = await wait_for_deadline(playback_clock, deadline_timestamp_ns)
                await sink.publish(output.channel, output.timestamp_ns, output.payload)
                lag = max(0.0, time.monotonic() - deadline)
                stats.current_lag = lag
                stats.max_lag = max(stats.max_lag, lag)
                stats.messages += 1
                stats.payload_bytes += len(output.payload)
                stats.playhead_ns = deadline_timestamp_ns
                last_time_ns = (
                    output.timestamp_ns
                    if last_time_ns is None
                    else max(last_time_ns, output.timestamp_ns)
                )

            with open_playback_messages(
                prepared,
                start_time_ns=start_time_ns,
            ) as messages:
                while True:
                    activity_delay = await sink.wait_until_active()
                    if clock is not None:
                        clock.delay(activity_delay)
                    try:
                        schema, channel, message = next(messages)
                    except StopIteration:
                        break
                    if (
                        prepared.resolved_filter.early_bail
                        and message.log_time >= prepared.resolved_filter.end_time_ns
                    ):
                        break
                    if first_time_ns is None:
                        first_time_ns = message.log_time
                        playback_speed = (
                            speed
                            if controller is None or controller.speed is None
                            else controller.speed
                        )
                        clock = PlaybackClock(
                            record_origin_ns=first_time_ns,
                            wall_origin=time.monotonic(),
                            speed=playback_speed,
                            recording_end_ns=prepared.recording_end_ns,
                        )
                        await sink.timeline_started(clock)
                    assert clock is not None
                    last_input_time_ns = message.log_time
                    clock.delay(await sink.wait_until_active())
                    cache_key = (0 if schema is None else schema.id, channel.id)
                    playback_channel = channel_cache.get(cache_key)
                    if playback_channel is None:
                        playback_channel = _playback_channel(schema, channel)
                        channel_cache[cache_key] = playback_channel
                    selected_channel = selected_channels.get(playback_channel.topic)
                    if selected_channel is None or not _channel_definitions_are_compatible(
                        playback_channel,
                        selected_channel,
                    ):
                        raise PlaybackError(
                            f"Topic {playback_channel.topic!r} changed to an "
                            "incompatible channel definition during playback"
                        )
                    playback_channel = selected_channel
                    advertised_channel = (
                        playback_channel
                        if transform_plan is None
                        else transform_plan.output_channel(playback_channel)
                    )
                    if not sink.is_channel_active(advertised_channel):
                        if transform_session is not None:
                            await transform_session.deactivate(playback_channel)
                        continue
                    deadline, lag = await wait_for_deadline(clock, message.log_time)
                    # wait_for_deadline returned at wall time deadline + lag.
                    processing_started = deadline + lag
                    if is_frame_channel(playback_channel):
                        now = processing_started
                        last_sent = last_frame_sent.get(advertised_channel, -math.inf)
                        is_congested = sink.is_channel_congested(advertised_channel)
                        if transform_session is not None:
                            await transform_session.observe_congestion(
                                playback_channel,
                                is_congested=is_congested,
                                now=now,
                            )
                        # When behind, keep every frame channel alive at a
                        # reduced rate without sending frames that are
                        # already older than the stressed interval.
                        is_too_stale = lag > _STRESSED_FRAME_INTERVAL_SECONDS
                        is_rate_limited = (
                            lag > _MAX_PLAYBACK_LAG_SECONDS
                            and now - last_sent < _STRESSED_FRAME_INTERVAL_SECONDS
                        )
                        should_drop = is_too_stale or is_rate_limited or is_congested
                        if (
                            not should_drop
                            and transform_session is not None
                            and transform_session.should_drop_frame(
                                playback_channel,
                                now=now,
                            )
                        ):
                            should_drop = True
                        if should_drop:
                            record_drop(message.log_time, lag)
                            continue
                        last_frame_sent[advertised_channel] = now
                    if transform_session is None:
                        outputs = (
                            PlaybackOutput(
                                playback_channel,
                                message.log_time,
                                message.data,
                            ),
                        )
                    else:
                        outputs = await transform_session.transform(
                            playback_channel,
                            message.log_time,
                            message.data,
                        )
                    for output in outputs:
                        await publish_output(
                            output,
                            clock,
                            deadline_cap_ns=message.log_time,
                        )

            if first_time_ns is None or last_input_time_ns is None:
                raise PlaybackError("Filters selected no messages")
            is_looping = (
                loop
                if controller is None or controller.is_looping is None
                else controller.is_looping
            )
            if transform_session is not None and not is_looping:
                for output in await transform_session.finish():
                    assert clock is not None
                    await publish_output(output, clock)
            await sink.timeline_finished(
                last_input_time_ns if last_time_ns is None else last_time_ns
            )
            if not is_looping:
                break
            if last_input_time_ns == first_time_ns:
                raise PlaybackError("Cannot loop a zero-duration selection")
            if transform_session is not None:
                await transform_session.restart()
            stats.loop_number += 1

    except _PlaybackStoppedError:
        stats.state = "Stopped"
        return stats
    else:
        stats.state = "Finished"
        return stats
    finally:
        done.set()
        if status_task is not None:
            await status_task
        if live is not None:
            live.update(_build_status(prepared, sink, stats, speed, transform_mode))
            live.stop()
        try:
            if transform_session is not None:
                await transform_session.close()
        finally:
            await sink.close()
