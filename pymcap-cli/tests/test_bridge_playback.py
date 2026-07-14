"""Shared and transport integration tests for MCAP bridge playback."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.bridge._playback import (
    PlaybackChannel,
    PlaybackClock,
    PlaybackError,
    open_playback_messages,
    prepare_playback,
    run_playback,
)
from pymcap_cli.cmd.bridge.play import BridgeClientPlaybackSink
from pymcap_cli.cmd.bridge.serve import BridgeServerPlaybackSink
from pymcap_cli.core.message_filter import MessageFilterOptions
from robo_ws_bridge import WebSocketBridgeServer
from robo_ws_bridge.ws_types import BinaryOpCodes
from small_mcap import McapWriter
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
) -> None:
    with path.open("wb") as stream:
        writer = McapWriter(stream)
        writer.start()
        writer.add_schema(1, schema_name, "text", schema_data)
        writer.add_channel(1, topic, "raw", 1)
        for timestamp_ns, payload in messages:
            writer.add_message(1, timestamp_ns, payload, publish_time=timestamp_ns)
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


def test_playback_clock_uses_absolute_speed_scaled_deadlines() -> None:
    clock = PlaybackClock(
        record_origin_ns=1_000_000_000,
        wall_origin=10.0,
        speed=2.0,
        recording_end_ns=5_000_000_000,
    )
    assert clock.deadline(3_000_000_000) == 11.0
    assert clock.current_time_ns(now=10.5) == 2_000_000_000


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
