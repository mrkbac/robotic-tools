"""Shared and transport integration tests for MCAP bridge playback."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import time
from typing import TYPE_CHECKING

import pytest
from mcap_codec_support.pointcloud import POINTCLOUD2
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cmd.bridge import _playback, _playback_transforms
from pymcap_cli.cmd.bridge._adaptive import AdaptiveVideoRung
from pymcap_cli.cmd.bridge._playback import (
    PlaybackChannel,
    PlaybackClock,
    PlaybackController,
    PlaybackError,
    PlaybackOutput,
    open_playback_messages,
    prepare_playback,
    run_playback,
)
from pymcap_cli.cmd.bridge._playback_transforms import (
    RoscompressConfig,
    RosdecompressConfig,
    create_playback_transform_plan,
    resolve_playback_transform_config,
)
from pymcap_cli.cmd.bridge.play import BridgeClientPlaybackSink
from pymcap_cli.cmd.bridge.serve import BridgeServerPlaybackSink
from pymcap_cli.core.message_filter import MessageFilterOptions
from pymcap_cli.core.processors.image_compress import ImageCompressProcessor
from robo_ws_bridge import ConnectionGraph, WebSocketBridgeEndpoint, WebSocketBridgeServer
from robo_ws_bridge.ws_types import BinaryOpCodes
from small_mcap import McapWriter, Schema
from websockets.asyncio.client import connect

if TYPE_CHECKING:
    from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_mcap(
    path: Path,
    topic: str,
    messages: list[tuple[int, bytes]],
    *,
    schema_name: str = "example/Raw",
    schema_data: bytes = b"bytes data",
    schema_encoding: str = "text",
    message_encoding: str = "raw",
) -> None:
    with path.open("wb") as stream:
        writer = McapWriter(stream)
        writer.start()
        writer.add_schema(1, schema_name, schema_encoding, schema_data)
        writer.add_channel(1, topic, message_encoding, 1)
        for timestamp_ns, payload in messages:
            writer.add_message(1, timestamp_ns, payload, publish_time=timestamp_ns)
        writer.finish()


def _write_pointcloud_mcap(
    path: Path,
    points: list[tuple[float, float, float]] | None = None,
) -> None:
    fields = [
        {"name": "x", "offset": 0, "datatype": 7, "count": 1},
        {"name": "y", "offset": 4, "datatype": 7, "count": 1},
        {"name": "z", "offset": 8, "datatype": 7, "count": 1},
    ]
    if points is None:
        points = [(1.0, 2.0, 3.0), (0.0, 0.0, 0.0)]
    data = b"".join(struct.pack("<fff", *point) for point in points)
    with path.open("wb") as stream:
        writer = McapWriter(stream, encoder_factory=ROS2EncoderFactory())
        writer.start(profile="ros2")
        writer.add_schema(1, "sensor_msgs/msg/PointCloud2", "ros2msg", POINTCLOUD2.encode())
        writer.add_channel(1, "/points", "cdr", 1)
        writer.add_message_encode(
            1,
            100,
            {
                "header": {"stamp": {"sec": 0, "nanosec": 100}, "frame_id": "lidar"},
                "height": 1,
                "width": len(points),
                "fields": fields,
                "is_bigendian": False,
                "point_step": 12,
                "row_step": 12 * len(points),
                "data": data,
                "is_dense": True,
            },
            publish_time=100,
        )
        writer.finish()


def test_prepare_playback_merges_files_chronologically(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first, "/first", [(1, b"a"), (3, b"c")])
    _write_mcap(second, "/second", [(2, b"b"), (4, b"d")])

    prepared = prepare_playback([str(first), str(second)], MessageFilterOptions.from_args())
    with open_playback_messages(prepared) as messages:
        merged = [
            (channel.topic, message.log_time, bytes(message.data))
            for _, channel, message in messages
        ]

    assert merged == [
        ("/first", 1, b"a"),
        ("/second", 2, b"b"),
        ("/first", 3, b"c"),
        ("/second", 4, b"d"),
    ]


def test_prepare_playback_applies_shared_topic_and_time_filters(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first, "/keep", [(1, b"a"), (3, b"c")])
    _write_mcap(second, "/drop", [(2, b"b"), (4, b"d")])
    prepared = prepare_playback(
        [str(first), str(second)],
        MessageFilterOptions.from_args(topic=["/keep"], start="2", end="4"),
    )
    with open_playback_messages(prepared) as messages:
        merged = [(channel.topic, message.log_time) for _, channel, message in messages]
    assert merged == [("/keep", 3)]


def test_prepare_playback_resolves_relative_time_globally(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first, "/first", [(100, b"a"), (200, b"b")])
    _write_mcap(second, "/second", [(300, b"c"), (400, b"d")])
    prepared = prepare_playback(
        [str(first), str(second)],
        MessageFilterOptions.from_args(start="+150ns", end="-50ns"),
    )
    with open_playback_messages(prepared) as messages:
        times = [message.log_time for _, _, message in messages]
    assert times == [300]


def test_prepare_playback_ignores_invalid_schema_on_excluded_topic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mcap"
    good = tmp_path / "good.mcap"
    _write_mcap(bad, "/bad", [(1, b"a")], schema_data=b"\xff")
    _write_mcap(good, "/good", [(2, b"b")])
    prepared = prepare_playback(
        [str(bad), str(good)], MessageFilterOptions.from_args(topic=["/good"])
    )
    assert [channel.topic for channel in prepared.channels] == ["/good"]


def test_prepare_playback_rejects_incompatible_same_topic(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first, "/same", [(1, b"a")], schema_name="example/One")
    _write_mcap(second, "/same", [(2, b"b")], schema_name="example/Two")
    with pytest.raises(PlaybackError, match="incompatible"):
        prepare_playback([str(first), str(second)], MessageFilterOptions.from_args())


def test_prepare_playback_accepts_structurally_equal_ros2_schemas_and_uses_last(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    first_schema = b"int32 value # old comment\n"
    second_schema = b"# newer documentation\nint32 value\n"
    for path, timestamp, schema_data in (
        (first, 1, first_schema),
        (second, 2, second_schema),
    ):
        _write_mcap(
            path,
            "/same",
            [(timestamp, b"payload")],
            schema_name="example_msgs/msg/Value",
            schema_data=schema_data,
            schema_encoding="ros2msg",
            message_encoding="cdr",
        )

    prepared = prepare_playback(
        [str(first), str(second)],
        MessageFilterOptions.from_args(),
    )

    assert prepared.channels[0].schema_text == second_schema.decode()
    with open_playback_messages(prepared) as messages:
        assert [message.log_time for _, _, message in messages] == [1, 2]
    sink = _TransformedCollectingSink()
    asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=1_000_000,
            loop=False,
            show_status=False,
        )
    )
    assert [timestamp for _, timestamp, _ in sink.messages] == [1, 2]
    assert {channel.schema_text for channel, _, _ in sink.messages} == {second_schema.decode()}


class _CollectingSink:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int, bytes]] = []
        self.was_closed = False

    async def start(self, _channels: tuple[PlaybackChannel, ...]) -> None:
        return

    async def wait_until_ready(self) -> None:
        return

    async def publish(
        self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
    ) -> None:
        self.messages.append((channel.topic, timestamp_ns, bytes(payload)))

    async def timeline_started(self, _clock: PlaybackClock) -> None:
        return

    async def timeline_finished(self, _timestamp_ns: int) -> None:
        return

    async def close(self) -> None:
        self.was_closed = True

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        return ()

    def is_channel_active(self, _channel: PlaybackChannel) -> bool:
        return True

    def is_channel_congested(self, _channel: PlaybackChannel) -> bool:
        return False

    async def wait_until_active(self) -> float:
        return 0.0


class _TransformedCollectingSink:
    def __init__(self, *, is_active: bool = True) -> None:
        self.channels: tuple[PlaybackChannel, ...] = ()
        self.messages: list[tuple[PlaybackChannel, int, bytes]] = []
        self.is_active = is_active

    async def start(self, channels: tuple[PlaybackChannel, ...]) -> None:
        self.channels = channels

    async def wait_until_ready(self) -> None:
        return

    async def publish(
        self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
    ) -> None:
        self.messages.append((channel, timestamp_ns, bytes(payload)))

    async def timeline_started(self, _clock: PlaybackClock) -> None:
        return

    async def timeline_finished(self, _timestamp_ns: int) -> None:
        return

    async def close(self) -> None:
        return

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        return ()

    def is_channel_active(self, _channel: PlaybackChannel) -> bool:
        return self.is_active

    def is_channel_congested(self, _channel: PlaybackChannel) -> bool:
        return False

    async def wait_until_active(self) -> float:
        return 0.0


def test_run_playback_preserves_merged_order_and_closes(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first, "/first", [(1, b"a"), (3, b"c")])
    _write_mcap(second, "/second", [(2, b"b")])
    prepared = prepare_playback([str(first), str(second)], MessageFilterOptions.from_args())
    sink = _CollectingSink()
    stats = asyncio.run(
        run_playback(prepared, sink, speed=1_000_000, loop=False, show_status=False)
    )
    assert sink.messages == [
        ("/first", 1, b"a"),
        ("/second", 2, b"b"),
        ("/first", 3, b"c"),
    ]
    assert sink.was_closed
    assert stats.messages == 3


def test_run_playback_can_start_from_seek_time(tmp_path: Path) -> None:
    path = tmp_path / "seek.mcap"
    _write_mcap(path, "/seek", [(1, b"a"), (2, b"b"), (3, b"c")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    sink = _CollectingSink()

    asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=1_000_000,
            loop=False,
            show_status=False,
            start_time_ns=2,
        )
    )

    assert sink.messages == [
        ("/seek", 2, b"b"),
        ("/seek", 3, b"c"),
    ]


def test_run_playback_waits_while_controller_is_paused(tmp_path: Path) -> None:
    path = tmp_path / "paused.mcap"
    _write_mcap(path, "/paused", [(1, b"payload")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    sink = _CollectingSink()
    controller = PlaybackController(start_paused=True)

    async def run() -> None:
        task = asyncio.create_task(
            run_playback(
                prepared,
                sink,
                speed=1_000_000,
                loop=False,
                show_status=False,
                controller=controller,
            )
        )
        await asyncio.sleep(0.02)
        assert sink.messages == []
        controller.play()
        await task

    asyncio.run(run())
    assert sink.messages == [("/paused", 1, b"payload")]


def test_run_playback_stop_returns_stopped_without_messages(tmp_path: Path) -> None:
    path = tmp_path / "stopped.mcap"
    _write_mcap(path, "/stopped", [(1, b"payload")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    sink = _CollectingSink()
    controller = PlaybackController(start_paused=True)
    controller.stop()

    stats = asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=1,
            loop=False,
            show_status=False,
            controller=controller,
        )
    )

    assert stats.state == "Stopped"
    assert sink.messages == []


def test_run_playback_controller_can_disable_loop_after_current_pass(
    tmp_path: Path,
) -> None:
    path = tmp_path / "loop.mcap"
    _write_mcap(path, "/loop", [(1, b"first"), (2, b"second")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    controller = PlaybackController(is_looping=True)

    class _LoopDisablingSink(_CollectingSink):
        async def publish(
            self,
            channel: PlaybackChannel,
            timestamp_ns: int,
            payload: bytes | memoryview,
        ) -> None:
            await super().publish(channel, timestamp_ns, payload)
            if len(self.messages) == 2:
                controller.set_looping(False)

    sink = _LoopDisablingSink()
    stats = asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=1_000_000,
            loop=False,
            show_status=False,
            controller=controller,
        )
    )

    assert [payload for _, _, payload in sink.messages] == [b"first", b"second"]
    assert stats.loop_number == 1


def test_run_playback_reuses_transform_session_across_loop_passes(tmp_path: Path) -> None:
    path = tmp_path / "loop-transform.mcap"
    _write_mcap(path, "/loop", [(1, b"first"), (2, b"second")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    controller = PlaybackController(is_looping=True)
    events: list[str] = []

    class _Session:
        async def observe_congestion(
            self,
            _channel: PlaybackChannel,
            *,
            is_congested: bool,  # noqa: ARG002
            now: float,  # noqa: ARG002
        ) -> None:
            return

        async def transform(
            self,
            channel: PlaybackChannel,
            timestamp_ns: int,
            payload: bytes | memoryview,
        ) -> tuple[PlaybackOutput, ...]:
            return (PlaybackOutput(channel, timestamp_ns, payload),)

        async def finish(self) -> tuple[PlaybackOutput, ...]:
            events.append("finish")
            return ()

        async def restart(self) -> None:
            events.append("restart")

        async def deactivate(self, _channel: PlaybackChannel) -> None:
            return

        async def close(self) -> None:
            events.append("close")

    class _Plan:
        mode = "test"
        channels = prepared.channels

        def create_session(self) -> _Session:
            events.append("create")
            return _Session()

        def output_channel(self, source: PlaybackChannel) -> PlaybackChannel:
            return source

    class _Sink(_CollectingSink):
        async def timeline_started(self, _clock: PlaybackClock) -> None:
            events.append("timeline-start")

        async def timeline_finished(self, _timestamp_ns: int) -> None:
            events.append("timeline-finish")

        async def publish(
            self,
            channel: PlaybackChannel,
            timestamp_ns: int,
            payload: bytes | memoryview,
        ) -> None:
            await super().publish(channel, timestamp_ns, payload)
            if len(self.messages) == 4:
                controller.set_looping(False)

    sink = _Sink()
    stats = asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=1_000_000,
            loop=False,
            show_status=False,
            controller=controller,
            transform_plan=_Plan(),
        )
    )

    assert stats.loop_number == 2
    assert events == [
        "create",
        "timeline-start",
        "timeline-finish",
        "restart",
        "timeline-start",
        "finish",
        "timeline-finish",
        "close",
    ]


def test_run_playback_controller_changes_speed_without_restarting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "speed.mcap"
    _write_mcap(path, "/speed", [(1, b"first"), (1_000_000_001, b"second")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    controller = PlaybackController(speed=1)

    class _SpeedChangingSink(_CollectingSink):
        async def publish(
            self,
            channel: PlaybackChannel,
            timestamp_ns: int,
            payload: bytes | memoryview,
        ) -> None:
            await super().publish(channel, timestamp_ns, payload)
            if len(self.messages) == 1:
                controller.set_speed(1_000)

    sink = _SpeedChangingSink()
    started = time.monotonic()
    asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=1,
            loop=False,
            show_status=False,
            controller=controller,
        )
    )

    assert time.monotonic() - started < 0.2
    assert [payload for _, _, payload in sink.messages] == [b"first", b"second"]


def test_run_playback_drops_stale_messages_before_slow_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "slow-transform.mcap"
    _write_mcap(
        path,
        "/camera",
        [
            (1, b"first"),
            (1_000_000_001, b"second"),
            (2_000_000_001, b"third"),
        ],
        schema_name="sensor_msgs/msg/CompressedImage",
    )
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    now = [0.0]
    transformed: list[int] = []
    monkeypatch.setattr(_playback.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(_playback, "_MAX_PLAYBACK_LAG_SECONDS", 0.005, raising=False)

    class _SlowTransformSession:
        async def observe_congestion(
            self,
            _channel: PlaybackChannel,
            *,
            is_congested: bool,  # noqa: ARG002
            now: float,  # noqa: ARG002
        ) -> None:
            return

        async def transform(
            self,
            channel: PlaybackChannel,
            timestamp_ns: int,
            payload: bytes | memoryview,
        ) -> tuple[PlaybackOutput, ...]:
            transformed.append(timestamp_ns)
            now[0] += 0.02
            return (PlaybackOutput(channel, timestamp_ns, payload),)

        def should_drop_frame(self, _channel: PlaybackChannel, *, now: float) -> bool:  # noqa: ARG002
            return False

        async def finish(self) -> tuple[PlaybackOutput, ...]:
            return ()

        async def restart(self) -> None:
            return

        async def deactivate(self, _channel: PlaybackChannel) -> None:
            return

        async def close(self) -> None:
            return

    class _SlowTransformPlan:
        mode = "test"
        channels = prepared.channels

        def create_session(self) -> _SlowTransformSession:
            return _SlowTransformSession()

        def output_channel(self, source: PlaybackChannel) -> PlaybackChannel:
            return source

    sink = _CollectingSink()
    stats = asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=500,
            loop=False,
            show_status=False,
            transform_plan=_SlowTransformPlan(),
        )
    )

    assert transformed == [1]
    assert sink.messages == [("/camera", 1, b"first")]
    assert stats.dropped_messages == 2


def test_run_playback_speed_reduction_skips_high_speed_backlog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "speed-recovery.mcap"
    timestamps = [1 + offset * 1_000_000_000 for offset in range(12)]
    _write_mcap(
        path,
        "/camera",
        [(timestamp_ns, str(offset).encode()) for offset, timestamp_ns in enumerate(timestamps)],
        schema_name="sensor_msgs/msg/CompressedImage",
    )
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    controller = PlaybackController(speed=500)
    now = [0.0]
    monkeypatch.setattr(_playback.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(_playback, "_MAX_PLAYBACK_LAG_SECONDS", 0.005, raising=False)

    class _SpeedReducingSink(_CollectingSink):
        async def publish(
            self,
            channel: PlaybackChannel,
            timestamp_ns: int,
            payload: bytes | memoryview,
        ) -> None:
            await super().publish(channel, timestamp_ns, payload)
            if len(self.messages) == 1:
                now[0] = 0.02
                controller.set_speed(1)
            elif timestamp_ns == timestamps[10]:
                controller.stop()

    sink = _SpeedReducingSink()
    stats = asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=500,
            loop=False,
            show_status=False,
            controller=controller,
        )
    )

    assert [timestamp_ns for _, timestamp_ns, _ in sink.messages] == [
        timestamps[0],
        timestamps[10],
    ]
    assert stats.dropped_messages == 9
    assert stats.state == "Stopped"


def test_transform_config_uses_standalone_defaults_and_rejects_wrong_mode() -> None:
    common = {
        "adaptive_quality": None,
        "image_format": None,
        "codec": None,
        "quality": None,
        "encoder": None,
        "backend": None,
        "scale": None,
        "jpeg_quality": None,
        "video": None,
        "video_format": None,
        "pointcloud": None,
        "resolution": None,
        "pc_format": None,
        "pc_schema": None,
        "pc_encoding": None,
        "pc_compression": None,
        "draco_compression_level": None,
        "pointcloud_drop_invalid": None,
        "pointcloud_sort_field": None,
    }

    compress = resolve_playback_transform_config(preset="compress", **common)
    assert compress == RoscompressConfig()
    decompress = resolve_playback_transform_config(preset="decompress", **common)
    assert decompress == RosdecompressConfig()

    with pytest.raises(ValueError, match="requires --preset"):
        resolve_playback_transform_config(preset=None, **(common | {"codec": "h265"}))
    with pytest.raises(ValueError, match="requires --preset decompress"):
        resolve_playback_transform_config(preset="compress", **(common | {"video_format": "raw"}))


def test_transform_config_adaptive_quality_requires_roscompress_video() -> None:
    common = {
        "adaptive_quality": True,
        "image_format": None,
        "codec": None,
        "quality": None,
        "encoder": None,
        "backend": None,
        "scale": None,
        "jpeg_quality": None,
        "video": None,
        "video_format": None,
        "pointcloud": None,
        "resolution": None,
        "pc_format": None,
        "pc_schema": None,
        "pc_encoding": None,
        "pc_compression": None,
        "draco_compression_level": None,
        "pointcloud_drop_invalid": None,
        "pointcloud_sort_field": None,
    }

    config = resolve_playback_transform_config(preset="compress", **common)
    assert config == RoscompressConfig(adaptive_quality=True)

    with pytest.raises(ValueError, match="requires --image-format video"):
        resolve_playback_transform_config(
            preset="compress",
            **(common | {"image_format": "jpeg"}),
        )
    with pytest.raises(ValueError, match="requires --preset compress"):
        resolve_playback_transform_config(preset=None, **common)


def test_roscompress_video_has_general_wall_clock_fps_cap(image_small_mcap: Path) -> None:
    prepared = prepare_playback([str(image_small_mcap)], MessageFilterOptions.from_args())
    plan = create_playback_transform_plan(
        RoscompressConfig(pointcloud=False),
        prepared.channels,
    )
    assert plan is not None
    session = plan.create_session()
    channel = prepared.channels[0]

    try:
        assert not session.should_drop_frame(channel, now=0.0)
        assert session.should_drop_frame(channel, now=0.01)
        assert not session.should_drop_frame(channel, now=0.034)
    finally:
        asyncio.run(session.close())


def test_apply_preset_maps_image_format_and_scale() -> None:
    apply_preset = _playback_transforms.apply_preset
    assert apply_preset(None, image_format=None, scale=None) == (None, None)
    assert apply_preset("compress", image_format=None, scale=None) == ("video", None)
    assert apply_preset("decompress", image_format=None, scale=None) == (None, None)
    assert apply_preset("fast", image_format=None, scale=None) == ("video", 960)
    assert apply_preset("low", image_format=None, scale=None) == ("video", 480)


def test_apply_preset_lets_explicit_flags_override_defaults() -> None:
    apply_preset = _playback_transforms.apply_preset
    assert apply_preset("compress", image_format="jpeg", scale=None) == ("jpeg", None)
    assert apply_preset("fast", image_format=None, scale=720) == ("video", 720)


def test_adaptive_quality_session_restarts_encoder_at_new_rung() -> None:
    channel = PlaybackChannel("/camera", "raw", "example/Raw", "text", "bytes")
    created: list[int] = []
    closed: list[int] = []

    class _Transform:
        def __init__(self, rung: int) -> None:
            self._rung = rung
            created.append(rung)

        def process(
            self,
            _payload: bytes | memoryview,
            timestamp_ns: int,
        ) -> tuple[PlaybackOutput, ...]:
            return (PlaybackOutput(channel, timestamp_ns, bytes([self._rung])),)

        def finish(self) -> tuple[PlaybackOutput, ...]:
            return ()

        def close(self) -> None:
            closed.append(self._rung)

    spec = _playback_transforms._ChannelTransformSpec(
        source=channel,
        output=channel,
        factories=tuple(lambda rung=rung: _Transform(rung) for rung in range(3)),
        video_rungs=tuple(AdaptiveVideoRung(quality, 1.0) for quality in (28, 34, 40)),
    )
    session = _playback_transforms._JitPlaybackTransformSession((spec,))

    async def run() -> tuple[PlaybackOutput, ...]:
        first = await session.transform(channel, 1, b"first")
        for now, is_congested in (
            (0.0, False),
            (0.5, True),
            (1.0, True),
            (1.5, False),
            (2.0, True),
        ):
            await session.observe_congestion(
                channel,
                is_congested=is_congested,
                now=now,
            )
        second = await session.transform(channel, 2, b"second")
        await session.close()
        return first + second

    outputs = asyncio.run(run())

    assert [bytes(output.payload) for output in outputs] == [b"\x00", b"\x01"]
    assert created == [0, 1]
    assert closed == [0, 1]


def test_adaptive_video_frame_rate_rung_drops_frames_before_transform() -> None:
    channel = PlaybackChannel("/camera", "raw", "example/Raw", "text", "bytes")
    rung = AdaptiveVideoRung
    spec = _playback_transforms._ChannelTransformSpec(
        source=channel,
        output=channel,
        factories=(
            lambda: _playback_transforms._PassthroughTransform(channel),
            lambda: _playback_transforms._PassthroughTransform(channel),
        ),
        video_rungs=(rung(46, 0.375, None), rung(46, 0.375, 5.0)),
    )
    session = _playback_transforms._JitPlaybackTransformSession((spec,))

    async def enter_frame_rate_rung() -> None:
        await session.observe_congestion(channel, is_congested=True, now=0.0)
        await session.observe_congestion(channel, is_congested=True, now=2.0)

    asyncio.run(enter_frame_rate_rung())

    assert not session.should_drop_frame(channel, now=2.0)
    assert session.should_drop_frame(channel, now=2.1)
    assert not session.should_drop_frame(channel, now=2.2)


def test_adaptive_video_frame_rate_change_reuses_encoder() -> None:
    channel = PlaybackChannel("/camera", "raw", "example/Raw", "text", "bytes")
    created = 0
    closed = 0

    class _Transform:
        def __init__(self) -> None:
            nonlocal created
            created += 1

        def process(
            self,
            _payload: bytes | memoryview,
            timestamp_ns: int,
        ) -> tuple[PlaybackOutput, ...]:
            return (PlaybackOutput(channel, timestamp_ns, b"frame"),)

        def finish(self) -> tuple[PlaybackOutput, ...]:
            return ()

        def close(self) -> None:
            nonlocal closed
            closed += 1

    rung = AdaptiveVideoRung
    spec = _playback_transforms._ChannelTransformSpec(
        source=channel,
        output=channel,
        factories=(_Transform, _Transform),
        video_rungs=(rung(28, 1.0, 30.0), rung(28, 1.0, 10.0)),
    )
    session = _playback_transforms._JitPlaybackTransformSession((spec,))

    async def run() -> None:
        await session.transform(channel, 1, b"first")
        await session.observe_congestion(channel, is_congested=True, now=0.0)
        await session.observe_congestion(channel, is_congested=True, now=2.0)
        await session.transform(channel, 2, b"second")
        await session.close()

    asyncio.run(run())

    assert created == 1
    assert closed == 1


def test_run_playback_roscompress_jpeg_is_jit_and_lossless(image_small_mcap: Path) -> None:
    prepared = prepare_playback([str(image_small_mcap)], MessageFilterOptions.from_args())
    plan = create_playback_transform_plan(
        RoscompressConfig(image_format="jpeg", pointcloud=False),
        prepared.channels,
    )
    assert plan is not None
    sink = _TransformedCollectingSink()

    stats = asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=1_000_000,
            loop=False,
            show_status=False,
            transform_plan=plan,
        )
    )

    assert sink.channels[0].schema_name == "sensor_msgs/msg/CompressedImage"
    assert stats.messages == len(sink.messages) > 0
    output_channel = sink.channels[0]
    schema = Schema(
        id=1,
        name=output_channel.schema_name,
        encoding=output_channel.schema_encoding,
        data=output_channel.schema_text.encode(),
    )
    decoder = DecoderFactory().decoder_for(output_channel.message_encoding, schema)
    assert decoder is not None
    decoded = [decoder(payload) for _, _, payload in sink.messages]
    assert all(message.format == "jpeg" for message in decoded)
    assert all(bytes(message.data).startswith(b"\xff\xd8") for message in decoded)
    assert [timestamp for _, timestamp, _ in sink.messages] == sorted(
        timestamp for _, timestamp, _ in sink.messages
    )


def test_jit_transform_skips_inactive_channels_before_decoding(
    image_small_mcap: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_playback([str(image_small_mcap)], MessageFilterOptions.from_args())
    plan = create_playback_transform_plan(
        RoscompressConfig(image_format="jpeg", pointcloud=False),
        prepared.channels,
    )
    assert plan is not None

    def fail_if_called(*_args) -> None:
        raise AssertionError("inactive topic was transformed")

    monkeypatch.setattr(ImageCompressProcessor, "transform", fail_if_called)
    sink = _TransformedCollectingSink(is_active=False)
    stats = asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=1_000_000,
            loop=False,
            show_status=False,
            transform_plan=plan,
        )
    )

    assert stats.messages == 0
    assert sink.messages == []


def test_jit_video_compress_then_decompress_preserves_every_frame(
    image_small_mcap: Path,
) -> None:
    pytest.importorskip("av")
    prepared = prepare_playback([str(image_small_mcap)], MessageFilterOptions.from_args())
    compress_plan = create_playback_transform_plan(
        RoscompressConfig(
            image_format="video",
            encoder="libx264",
            backend="pyav",
            pointcloud=False,
        ),
        prepared.channels,
    )
    assert compress_plan is not None
    compressed_sink = _TransformedCollectingSink()
    asyncio.run(
        run_playback(
            prepared,
            compressed_sink,
            speed=1_000_000,
            loop=False,
            show_status=False,
            transform_plan=compress_plan,
        )
    )

    assert compressed_sink.messages
    assert compressed_sink.channels[0].schema_name == "foxglove_msgs/msg/CompressedVideo"
    decompress_plan = create_playback_transform_plan(
        RosdecompressConfig(backend="pyav", pointcloud=False),
        compressed_sink.channels,
    )
    assert decompress_plan is not None

    async def decompress() -> list[tuple[int, bytes]]:
        session = decompress_plan.create_session()
        outputs = []
        try:
            for channel, timestamp_ns, payload in compressed_sink.messages:
                outputs.extend(await session.transform(channel, timestamp_ns, payload))
            outputs.extend(await session.finish())
        finally:
            await session.close()
        return [(output.timestamp_ns, bytes(output.payload)) for output in outputs]

    decompressed = asyncio.run(decompress())
    assert len(decompressed) == len(compressed_sink.messages)
    assert [timestamp for timestamp, _ in decompressed] == [
        timestamp for _, timestamp, _ in compressed_sink.messages
    ]
    assert decompress_plan.channels[0].schema_name == "sensor_msgs/msg/CompressedImage"


def test_adaptive_video_encoder_restart_produces_decodable_segment(
    image_small_mcap: Path,
) -> None:
    pytest.importorskip("av")
    prepared = prepare_playback([str(image_small_mcap)], MessageFilterOptions.from_args())
    compress_plan = create_playback_transform_plan(
        RoscompressConfig(
            image_format="video",
            encoder="libx264",
            backend="pyav",
            pointcloud=False,
            adaptive_quality=True,
        ),
        prepared.channels,
    )
    assert compress_plan is not None
    with open_playback_messages(prepared) as messages:
        inputs = [(message.log_time, bytes(message.data)) for _, _, message in messages]
    assert len(inputs) >= 2

    async def compress_after_restart() -> tuple[PlaybackOutput, ...]:
        session = compress_plan.create_session()
        channel = prepared.channels[0]
        try:
            await session.transform(channel, *inputs[0])
            for now, is_congested in (
                (0.0, False),
                (0.5, True),
                (1.0, True),
                (1.5, False),
                (2.0, True),
            ):
                await session.observe_congestion(
                    channel,
                    is_congested=is_congested,
                    now=now,
                )
            outputs = []
            for timestamp_ns, payload in inputs[1:]:
                outputs.extend(await session.transform(channel, timestamp_ns, payload))
            outputs.extend(await session.finish())
            return tuple(outputs)
        finally:
            await session.close()

    restarted_segment = asyncio.run(compress_after_restart())
    assert restarted_segment
    decompress_plan = create_playback_transform_plan(
        RosdecompressConfig(backend="pyav", pointcloud=False),
        compress_plan.channels,
    )
    assert decompress_plan is not None

    async def decompress_segment() -> tuple[PlaybackOutput, ...]:
        session = decompress_plan.create_session()
        outputs = []
        try:
            for output in restarted_segment:
                outputs.extend(
                    await session.transform(
                        output.channel,
                        output.timestamp_ns,
                        output.payload,
                    )
                )
            outputs.extend(await session.finish())
            return tuple(outputs)
        finally:
            await session.close()

    decoded = asyncio.run(decompress_segment())
    assert len(decoded) == len(restarted_segment)


def test_adaptive_video_resolution_switch_is_decodable_in_one_stream(
    image_small_mcap: Path,
) -> None:
    av = pytest.importorskip("av")
    prepared = prepare_playback([str(image_small_mcap)], MessageFilterOptions.from_args())
    compress_plan = create_playback_transform_plan(
        RoscompressConfig(
            image_format="video",
            encoder="libx264",
            backend="pyav",
            pointcloud=False,
            adaptive_quality=True,
        ),
        prepared.channels,
    )
    assert compress_plan is not None
    with open_playback_messages(prepared) as messages:
        inputs = [(message.log_time, bytes(message.data)) for _, _, message in messages]

    async def compress_at_first_resolution_rung() -> tuple[PlaybackOutput, ...]:
        session = compress_plan.create_session()
        channel = prepared.channels[0]
        try:
            outputs: list[PlaybackOutput] = []
            for timestamp_ns, payload in inputs[:2]:
                outputs.extend(await session.transform(channel, timestamp_ns, payload))
            for now in (0.0, 2.0, 5.0, 8.0, 11.0, 14.0):
                await session.observe_congestion(
                    channel,
                    is_congested=True,
                    now=now,
                )
            for timestamp_ns, payload in inputs[2:]:
                outputs.extend(await session.transform(channel, timestamp_ns, payload))
            outputs.extend(await session.finish())
            return tuple(outputs)
        finally:
            await session.close()

    scaled_segment = asyncio.run(compress_at_first_resolution_rung())
    assert scaled_segment
    output_channel = compress_plan.channels[0]
    schema = Schema(
        id=1,
        name=output_channel.schema_name,
        encoding=output_channel.schema_encoding,
        data=output_channel.schema_text.encode(),
    )
    decode_cdr = DecoderFactory().decoder_for(output_channel.message_encoding, schema)
    assert decode_cdr is not None
    decoder = av.CodecContext.create("h264", "r")
    decoded_sizes = {
        (frame.width, frame.height)
        for output in scaled_segment
        for frame in decoder.decode(av.Packet(bytes(decode_cdr(output.payload).data)))
    }
    assert decoded_sizes == {(160, 120), (120, 90)}


def test_video_session_restart_starts_a_self_contained_segment(
    image_small_mcap: Path,
) -> None:
    # At a loop boundary run_playback calls session.restart(); the video encoder
    # must then open a fresh GOP whose first packet is a keyframe, so the
    # post-restart frames decode on their own without the pre-restart stream.
    # Otherwise Foxglove shows a red cross after every loop.
    pytest.importorskip("av")
    prepared = prepare_playback([str(image_small_mcap)], MessageFilterOptions.from_args())
    compress_plan = create_playback_transform_plan(
        RoscompressConfig(
            image_format="video",
            encoder="libx264",
            backend="pyav",
            pointcloud=False,
        ),
        prepared.channels,
    )
    assert compress_plan is not None
    with open_playback_messages(prepared) as messages:
        inputs = [(message.log_time, bytes(message.data)) for _, _, message in messages]
    assert len(inputs) >= 2

    async def compress_second_pass() -> tuple[PlaybackOutput, ...]:
        session = compress_plan.create_session()
        channel = prepared.channels[0]
        try:
            for timestamp_ns, payload in inputs:
                await session.transform(channel, timestamp_ns, payload)
            # Discard the first pass' flush; restart mirrors the loop boundary.
            await session.restart()
            outputs: list[PlaybackOutput] = []
            for timestamp_ns, payload in inputs:
                outputs.extend(await session.transform(channel, timestamp_ns, payload))
            outputs.extend(await session.finish())
            return tuple(outputs)
        finally:
            await session.close()

    second_pass = asyncio.run(compress_second_pass())
    assert second_pass

    decompress_plan = create_playback_transform_plan(
        RosdecompressConfig(backend="pyav", pointcloud=False),
        compress_plan.channels,
    )
    assert decompress_plan is not None

    async def decompress_second_pass() -> tuple[PlaybackOutput, ...]:
        session = decompress_plan.create_session()
        outputs: list[PlaybackOutput] = []
        try:
            for output in second_pass:
                outputs.extend(
                    await session.transform(
                        output.channel,
                        output.timestamp_ns,
                        output.payload,
                    )
                )
            outputs.extend(await session.finish())
            return tuple(outputs)
        finally:
            await session.close()

    decoded = asyncio.run(decompress_second_pass())
    # Every frame of the isolated post-restart segment decodes: the segment is
    # self-contained, i.e. it began with a keyframe.
    assert len(decoded) == len(inputs)


def test_jit_pointcloud_compress_then_decompress_preserves_message(tmp_path: Path) -> None:
    path = tmp_path / "pointcloud.mcap"
    _write_pointcloud_mcap(path)
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    compress_plan = create_playback_transform_plan(
        RoscompressConfig(image_format="none"), prepared.channels
    )
    assert compress_plan is not None
    compressed_sink = _TransformedCollectingSink()
    asyncio.run(
        run_playback(
            prepared,
            compressed_sink,
            speed=1_000_000,
            loop=False,
            show_status=False,
            transform_plan=compress_plan,
        )
    )

    assert len(compressed_sink.messages) == 1
    assert "CompressedPointCloud" in compressed_sink.channels[0].schema_name
    decompress_plan = create_playback_transform_plan(
        RosdecompressConfig(video=False), compressed_sink.channels
    )
    assert decompress_plan is not None

    async def decompress():
        session = decompress_plan.create_session()
        channel, timestamp_ns, payload = compressed_sink.messages[0]
        try:
            outputs = list(await session.transform(channel, timestamp_ns, payload))
            outputs.extend(await session.finish())
            return outputs
        finally:
            await session.close()

    outputs = asyncio.run(decompress())
    assert len(outputs) == 1
    output_channel = decompress_plan.channels[0]
    assert output_channel.schema_name == "sensor_msgs/msg/PointCloud2"
    schema = Schema(
        1,
        output_channel.schema_name,
        output_channel.schema_encoding,
        output_channel.schema_text.encode(),
    )
    decoder = DecoderFactory().decoder_for(output_channel.message_encoding, schema)
    assert decoder is not None
    decoded = decoder(outputs[0].payload)
    assert decoded.width == 1
    assert outputs[0].timestamp_ns == 100


def test_playback_clock_uses_absolute_speed_scaled_deadlines() -> None:
    clock = PlaybackClock(
        record_origin_ns=1_000_000_000,
        wall_origin=10.0,
        speed=2.0,
        recording_end_ns=5_000_000_000,
    )
    assert clock.deadline(3_000_000_000) == 11.0
    assert clock.current_time_ns(now=10.5) == 2_000_000_000
    clock.delay(2.0)
    assert clock.deadline(3_000_000_000) == 13.0
    assert clock.current_time_ns(now=12.5) == 2_000_000_000
    clock.set_speed(4.0, now=12.5)
    assert clock.current_time_ns(now=12.5) == 2_000_000_000
    assert clock.deadline(3_000_000_000) == 12.75


def test_bridge_client_sink_publishes_to_existing_server(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "input.mcap"
    _write_mcap(path, "/raw", [(1, b"payload")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    port = _free_port()
    received: list[bytes] = []

    async def run() -> None:
        server = WebSocketBridgeServer(
            host="127.0.0.1",
            port=port,
            capabilities=["clientPublish"],
            supported_encodings=["raw"],
        )

        async def on_message(_state, payload: bytes) -> None:
            received.append(payload[5:])

        server.register_binary_handler(BinaryOpCodes.CLIENT_MESSAGE_DATA, on_message)
        await server.start()
        try:
            await run_playback(
                prepared,
                BridgeClientPlaybackSink(f"ws://127.0.0.1:{port}", connect_timeout=2),
                speed=1,
                loop=False,
                show_status=False,
            )
            await asyncio.sleep(0.05)
        finally:
            await server.stop()

    monkeypatch.setattr("pymcap_cli.cmd.bridge.play._SETTLE_SECONDS", 0.0)
    asyncio.run(run())
    assert received == [b"payload"]


def test_bridge_client_only_subscribed_tracks_dynamic_graph() -> None:
    sink = BridgeClientPlaybackSink("ws://127.0.0.1:1", connect_timeout=1, only_subscribed=True)
    channel = PlaybackChannel(
        topic="/camera",
        message_encoding="cdr",
        schema_name="sensor_msgs/msg/Image",
        schema_encoding="ros2msg",
        schema_text="uint8[] data",
    )
    sink._selected_topics = {channel.topic}
    sink._on_connection_graph_update(
        ConnectionGraph(
            published_topics=(),
            subscribed_topics=(({"name": "/camera", "subscriberIds": ["viewer"]}),),
            advertised_services=(),
        )
    )
    assert sink.is_channel_active(channel)

    sink._on_connection_graph_update(
        ConnectionGraph(
            published_topics=(),
            subscribed_topics=(({"name": "/camera", "subscriberIds": []}),),
            advertised_services=(),
        )
    )
    assert not sink.is_channel_active(channel)

    async def wait_for_consumer() -> float:
        waiting = asyncio.create_task(sink.wait_until_active())
        await asyncio.sleep(0.01)
        assert not waiting.done()
        sink._on_connection_graph_update(
            ConnectionGraph(
                published_topics=(),
                subscribed_topics=(({"name": "/camera", "subscriberIds": ["viewer-2"]}),),
                advertised_services=(),
            )
        )
        return await waiting

    assert asyncio.run(wait_for_consumer()) > 0


def test_bridge_client_only_subscribed_requires_connection_graph(tmp_path: Path) -> None:
    path = tmp_path / "input.mcap"
    _write_mcap(path, "/raw", [(1, b"payload")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    port = _free_port()

    async def run() -> None:
        server = WebSocketBridgeServer(
            host="127.0.0.1",
            port=port,
            capabilities=["clientPublish"],
            supported_encodings=["raw"],
        )
        sink = BridgeClientPlaybackSink(
            f"ws://127.0.0.1:{port}", connect_timeout=2, only_subscribed=True
        )
        await server.start()
        try:
            with pytest.raises(PlaybackError, match="connectionGraph"):
                await sink.start(prepared.channels)
        finally:
            await sink.close()
            await server.stop()

    asyncio.run(run())


def test_bridge_client_sink_publishes_jit_transformed_payload(
    image_small_mcap: Path,
    monkeypatch,
) -> None:
    prepared = prepare_playback([str(image_small_mcap)], MessageFilterOptions.from_args())
    plan = create_playback_transform_plan(
        RoscompressConfig(image_format="jpeg", pointcloud=False),
        prepared.channels,
    )
    assert plan is not None
    port = _free_port()
    advertised: list[dict] = []
    received: list[bytes] = []

    async def run() -> None:
        server = WebSocketBridgeServer(
            host="127.0.0.1",
            port=port,
            capabilities=["clientPublish", "connectionGraph"],
            supported_encodings=["cdr"],
        )

        async def on_advertise(_state, message: dict) -> None:
            advertised.extend(message["channels"])

        async def on_message(_state, payload: bytes) -> None:
            received.append(payload[5:])

        async def on_graph_subscription(state, _message: dict) -> None:
            await state.websocket.send(
                json.dumps(
                    {
                        "op": "connectionGraphUpdate",
                        "subscribedTopics": [
                            {"name": plan.channels[0].topic, "subscriberIds": ["viewer-1"]}
                        ],
                    }
                )
            )

        server.register_json_handler("advertise", on_advertise)
        server.register_json_handler("subscribeConnectionGraph", on_graph_subscription)
        server.register_binary_handler(BinaryOpCodes.CLIENT_MESSAGE_DATA, on_message)
        await server.start()
        try:
            await run_playback(
                prepared,
                BridgeClientPlaybackSink(
                    f"ws://127.0.0.1:{port}",
                    connect_timeout=2,
                    only_subscribed=True,
                ),
                speed=1_000_000,
                loop=False,
                show_status=False,
                transform_plan=plan,
            )
            await asyncio.sleep(0.05)
        finally:
            await server.stop()

    monkeypatch.setattr("pymcap_cli.cmd.bridge.play._SETTLE_SECONDS", 0.0)
    asyncio.run(run())
    assert advertised[0]["schemaName"] == "sensor_msgs/msg/CompressedImage"
    assert len(received) > 0

    channel = plan.channels[0]
    schema = Schema(1, channel.schema_name, channel.schema_encoding, channel.schema_text.encode())
    decoder = DecoderFactory().decoder_for(channel.message_encoding, schema)
    assert decoder is not None
    assert decoder(received[0]).format == "jpeg"


def test_bridge_server_sink_sends_recorded_timestamp_and_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "input.mcap"
    _write_mcap(path, "/raw", [(123, b"payload")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    port = _free_port()

    async def run() -> tuple[int, bytes, list[int]]:
        sink = BridgeServerPlaybackSink("127.0.0.1", port)
        task = asyncio.create_task(
            run_playback(
                prepared,
                sink,
                speed=1,
                loop=False,
                show_status=False,
            )
        )
        await sink.started.wait()
        async with connect(
            f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"]
        ) as websocket:
            server_info = json.loads(await websocket.recv())
            advertise = json.loads(await websocket.recv())
            assert "time" in server_info["capabilities"]
            channel_id = advertise["channels"][0]["id"]
            await websocket.send(
                json.dumps(
                    {
                        "op": "subscribe",
                        "subscriptions": [{"id": 7, "channelId": channel_id}],
                    }
                )
            )
            message_timestamp = -1
            payload = b""
            times: list[int] = []
            while message_timestamp < 0:
                frame = await websocket.recv()
                assert isinstance(frame, bytes)
                if frame[0] == int(BinaryOpCodes.TIME):
                    times.append(struct.unpack_from("<Q", frame, 1)[0])
                elif frame[0] == int(BinaryOpCodes.MESSAGE_DATA):
                    message_timestamp = struct.unpack_from("<Q", frame, 5)[0]
                    payload = frame[13:]
            await task
            return message_timestamp, payload, times

    monkeypatch.setattr("pymcap_cli.cmd.bridge.serve._SETTLE_SECONDS", 0.0)
    timestamp, payload, times = asyncio.run(run())
    assert timestamp == 123
    assert payload == b"payload"
    assert 123 in times


def test_bridge_server_sink_stops_clock_without_subscriptions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "input.mcap"
    _write_mcap(path, "/raw", [(1, b"first"), (1_000_000_001, b"second")])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    port = _free_port()

    async def run() -> None:
        sink = BridgeServerPlaybackSink("127.0.0.1", port)
        became_inactive = asyncio.Event()

        def activity_changed(is_active: bool) -> None:
            if not is_active:
                became_inactive.set()

        sink.on_activity_change(activity_changed)
        task = asyncio.create_task(
            run_playback(
                prepared,
                sink,
                speed=1,
                loop=True,
                show_status=False,
            )
        )
        await sink.started.wait()
        try:
            async with connect(
                f"ws://127.0.0.1:{port}",
                subprotocols=["foxglove.websocket.v1"],
            ) as websocket:
                await websocket.recv()
                advertise = json.loads(await websocket.recv())
                channel_id = advertise["channels"][0]["id"]
                await websocket.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "subscriptions": [{"id": 7, "channelId": channel_id}],
                        }
                    )
                )
                while True:
                    frame = await websocket.recv()
                    if isinstance(frame, bytes) and frame[0] == int(BinaryOpCodes.MESSAGE_DATA):
                        break

                await websocket.send(json.dumps({"op": "unsubscribe", "subscriptionIds": [7]}))
                await asyncio.wait_for(became_inactive.wait(), timeout=1)

                while True:
                    try:
                        await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    except asyncio.TimeoutError:
                        break
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(websocket.recv(), timeout=0.08)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    monkeypatch.setattr("pymcap_cli.cmd.bridge.serve._SETTLE_SECONDS", 0.0)
    asyncio.run(run())


def test_bridge_server_sink_advertises_and_sends_jit_transform(
    image_small_mcap: Path,
    monkeypatch,
) -> None:
    prepared = prepare_playback([str(image_small_mcap)], MessageFilterOptions.from_args())
    plan = create_playback_transform_plan(
        RoscompressConfig(image_format="jpeg", pointcloud=False),
        prepared.channels,
    )
    assert plan is not None
    port = _free_port()

    async def run() -> tuple[dict, bytes]:
        sink = BridgeServerPlaybackSink("127.0.0.1", port)
        task = asyncio.create_task(
            run_playback(
                prepared,
                sink,
                speed=1_000_000,
                loop=False,
                show_status=False,
                transform_plan=plan,
            )
        )
        await sink.started.wait()
        async with connect(
            f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"]
        ) as websocket:
            await websocket.recv()
            advertise = json.loads(await websocket.recv())
            channel_id = advertise["channels"][0]["id"]
            await websocket.send(
                json.dumps(
                    {
                        "op": "subscribe",
                        "subscriptions": [{"id": 7, "channelId": channel_id}],
                    }
                )
            )
            payload = b""
            while not payload:
                frame = await websocket.recv()
                if isinstance(frame, bytes) and frame[0] == int(BinaryOpCodes.MESSAGE_DATA):
                    payload = frame[13:]
            await task
            return advertise, payload

    monkeypatch.setattr("pymcap_cli.cmd.bridge.serve._SETTLE_SECONDS", 0.0)
    advertise, payload = asyncio.run(run())
    channel_info = advertise["channels"][0]
    assert channel_info["schemaName"] == "sensor_msgs/msg/CompressedImage"
    schema = Schema(
        1,
        channel_info["schemaName"],
        channel_info["schemaEncoding"],
        channel_info["schema"].encode(),
    )
    decoder = DecoderFactory().decoder_for(channel_info["encoding"], schema)
    assert decoder is not None
    assert decoder(payload).format == "jpeg"


class _CongestibleSink(_CollectingSink):
    def __init__(self, congested_topics: set[str]) -> None:
        super().__init__()
        self.congested_topics = congested_topics

    def is_channel_congested(self, channel: PlaybackChannel) -> bool:
        return channel.topic in self.congested_topics


def test_is_frame_channel_marks_standalone_sensor_frames() -> None:
    def channel(schema_name: str) -> PlaybackChannel:
        return PlaybackChannel(
            topic="/x",
            message_encoding="cdr",
            schema_name=schema_name,
            schema_encoding="ros2msg",
            schema_text="",
        )

    assert _playback.is_frame_channel(channel("sensor_msgs/msg/CompressedImage"))
    assert _playback.is_frame_channel(channel("sensor_msgs/msg/PointCloud2"))
    assert _playback.is_frame_channel(channel("point_cloud_interfaces/msg/CompressedPointCloud2"))
    assert not _playback.is_frame_channel(channel("foxglove_msgs/msg/CompressedVideo"))
    assert not _playback.is_frame_channel(channel("tf2_msgs/msg/TFMessage"))


def test_run_playback_skips_congested_frame_channels_but_keeps_small_topics(
    tmp_path: Path,
) -> None:
    camera = tmp_path / "camera.mcap"
    control = tmp_path / "control.mcap"
    _write_mcap(
        camera,
        "/camera",
        [(1, b"f1"), (2, b"f2"), (3, b"f3")],
        schema_name="sensor_msgs/msg/CompressedImage",
    )
    _write_mcap(control, "/tf", [(1, b"t1"), (2, b"t2"), (3, b"t3")])
    prepared = prepare_playback([str(camera), str(control)], MessageFilterOptions.from_args())
    sink = _CongestibleSink({"/camera"})
    stats = asyncio.run(
        run_playback(prepared, sink, speed=1_000_000, loop=False, show_status=False)
    )
    topics = [topic for topic, _, _ in sink.messages]
    assert topics == ["/tf", "/tf", "/tf"]
    assert stats.dropped_messages == 3


def test_run_playback_does_not_drop_reliable_channels_when_late(tmp_path: Path) -> None:
    recording = tmp_path / "tf.mcap"
    _write_mcap(recording, "/tf", [(i * 1_000_000, b"x") for i in range(1, 6)])
    prepared = prepare_playback([str(recording)], MessageFilterOptions.from_args())

    class _SlowSink(_CollectingSink):
        async def publish(
            self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
        ) -> None:
            if not self.messages:
                await asyncio.sleep(0.3)
            await super().publish(channel, timestamp_ns, payload)

    sink = _SlowSink()
    stats = asyncio.run(run_playback(prepared, sink, speed=1.0, loop=False, show_status=False))
    assert [timestamp for _, timestamp, _ in sink.messages] == [i * 1_000_000 for i in range(1, 6)]
    assert stats.dropped_messages == 0


def test_run_playback_repeated_frame_stalls_drop_to_preserve_freshness(tmp_path: Path) -> None:
    recording = tmp_path / "camera.mcap"
    _write_mcap(
        recording,
        "/camera",
        [(i * 40_000_000, b"x") for i in range(8)],
        schema_name="sensor_msgs/msg/CompressedImage",
    )
    prepared = prepare_playback([str(recording)], MessageFilterOptions.from_args())

    class _ConsistentlySlowSink(_CollectingSink):
        async def publish(
            self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
        ) -> None:
            await super().publish(channel, timestamp_ns, payload)
            await asyncio.sleep(0.3)

    sink = _ConsistentlySlowSink()
    stats = asyncio.run(run_playback(prepared, sink, speed=1.0, loop=False, show_status=False))

    assert len(sink.messages) < 5
    assert stats.dropped_messages > 3


def test_run_playback_stressed_frame_channels_drop_frames_older_than_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "stressed.mcap"
    _write_mcap(
        path,
        "/camera",
        [(1, b"a"), (2, b"b"), (3, b"c")],
        schema_name="sensor_msgs/msg/CompressedImage",
    )
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    now = [0.0]
    monkeypatch.setattr(_playback.time, "monotonic", lambda: now[0])

    class _LagInducingSink(_CollectingSink):
        async def publish(
            self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
        ) -> None:
            await super().publish(channel, timestamp_ns, payload)
            if len(self.messages) == 1:
                now[0] += 0.3  # the first publish leaves the loop 0.3s behind

    sink = _LagInducingSink()
    stats = asyncio.run(
        run_playback(prepared, sink, speed=1_000_000, loop=False, show_status=False)
    )
    # First frame is on time; the remaining frames are already older than the
    # stressed interval and must not be sent merely to preserve cadence.
    assert [payload for _, _, payload in sink.messages] == [b"a"]
    assert stats.dropped_messages == 2


def test_jit_pointcloud_compress_handles_clouds_needing_no_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "radar.mcap"
    # All points valid (like radar detections): the cleanup processor reports
    # "no change", which must not count as a transform failure.
    _write_pointcloud_mcap(path, points=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    prepared = prepare_playback([str(path)], MessageFilterOptions.from_args())
    plan = create_playback_transform_plan(RoscompressConfig(image_format="none"), prepared.channels)
    assert plan is not None
    sink = _TransformedCollectingSink()
    asyncio.run(
        run_playback(
            prepared,
            sink,
            speed=1_000_000,
            loop=False,
            show_status=False,
            transform_plan=plan,
        )
    )
    assert len(sink.messages) == 1
    assert "CompressedPointCloud" in sink.messages[0][0].schema_name


def test_bridge_server_sink_status_rows_report_network_drops() -> None:
    endpoint = WebSocketBridgeEndpoint()
    sink = BridgeServerPlaybackSink("127.0.0.1", 0, endpoint=endpoint, url="/ws")
    rows = dict(sink.status_rows())
    assert rows["Dropped (network)"] == "0"
