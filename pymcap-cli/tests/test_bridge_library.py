"""Recording-library tests for ``pymcap-cli bridge serve``."""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import urllib.error
import urllib.request
from contextlib import closing, suppress
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import pymcap_cli.cmd.bridge._playback as _playback
import pytest
from pymcap_cli.cmd.bridge._library import (
    _APP_JS,
    _INDEX_HTML,
    _STYLE_CSS,
    RecordingEntry,
    RecordingLibrary,
    RecordingLibraryServer,
    RecordingSession,
    _session_transform_config,
)
from pymcap_cli.cmd.bridge._playback import (
    PlaybackClock,
    PlaybackController,
    PlaybackSink,
    PlaybackStats,
    PlaybackTransformPlan,
    PreparedPlayback,
)
from pymcap_cli.cmd.bridge._playback_transforms import RoscompressConfig, RosdecompressConfig
from pymcap_cli.cmd.bridge.serve import _library_root
from pymcap_cli.core.message_filter import MessageFilterOptions
from robo_ws_bridge import (
    PlaybackCommand,
    PlaybackControlRequest,
    PlaybackState,
    PlaybackStatus,
    StatusLevel,
)
from robo_ws_bridge.ws_types import BinaryOpCodes, ServerInfoMessage
from small_mcap import McapFile, McapWriter
from websockets.asyncio.client import ClientConnection, connect

if TYPE_CHECKING:
    from pathlib import Path


def _write_mcap(path: Path, topic: str, timestamp_ns: int, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = McapWriter(stream)
        writer.start()
        writer.add_schema(1, "example/Raw", "text", b"bytes data")
        writer.add_channel(1, topic, "raw", 1)
        writer.add_message(1, timestamp_ns, payload, publish_time=timestamp_ns)
        writer.finish()


def _write_mcap_messages(
    path: Path,
    topic: str,
    messages: list[tuple[int, bytes]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = McapWriter(stream)
        writer.start()
        writer.add_schema(1, "example/Raw", "text", b"bytes data")
        writer.add_channel(1, topic, "raw", 1)
        for timestamp_ns, payload in messages:
            writer.add_message(1, timestamp_ns, payload, publish_time=timestamp_ns)
        writer.finish()


def _playback_request(
    command: PlaybackCommand,
    *,
    speed: float,
    seek_time: int | None,
    request_id: str,
) -> bytes:
    encoded_request_id = request_id.encode()
    return (
        struct.pack(
            "<BBfBQI",
            int(BinaryOpCodes.PLAYBACK_CONTROL_REQUEST),
            int(command),
            speed,
            int(seek_time is not None),
            seek_time or 0,
            len(encoded_request_id),
        )
        + encoded_request_id
    )


def _playback_state(frame: bytes) -> tuple[PlaybackStatus, int, float, bool, str]:
    opcode, status, current_time, speed, did_seek, request_id_length = struct.unpack_from(
        "<BBQfBI", frame
    )
    assert opcode == int(BinaryOpCodes.PLAYBACK_STATE)
    request_id = frame[struct.calcsize("<BBQfBI") :].decode()
    assert len(request_id.encode()) == request_id_length
    return PlaybackStatus(status), current_time, speed, bool(did_seek), request_id


async def _receive_playback_state(
    websocket: ClientConnection,
    request_id: str,
) -> tuple[PlaybackStatus, int, float, bool, str]:
    while True:
        frame = await websocket.recv()
        if isinstance(frame, bytes) and frame[0] == int(BinaryOpCodes.PLAYBACK_STATE):
            state = _playback_state(frame)
            if state[-1] == request_id:
                return state


def test_library_websocket_accepts_foxglove_playback_control(
    tmp_path: Path,
) -> None:
    start_time = 1_000_000_001
    end_time = 3_000_000_001
    _write_mcap(tmp_path / "first.mcap", "/first", start_time, b"first")
    _write_mcap(tmp_path / "second.mcap", "/second", end_time, b"second")
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=False,
    )

    async def run() -> tuple[
        ServerInfoMessage,
        tuple[PlaybackStatus, int, float, bool, str],
        tuple[PlaybackStatus, int, float, bool, str],
        tuple[PlaybackStatus, int, float, bool, str],
        tuple[PlaybackStatus, int, float, bool, str],
    ]:
        await server.start()
        try:
            query = urlencode([("file", "first.mcap"), ("file", "second.mcap")])
            async with connect(
                f"ws://127.0.0.1:{server.port}/ws?{query}",
                subprotocols=["foxglove.sdk.v1"],
            ) as websocket:
                server_info: ServerInfoMessage = json.loads(await websocket.recv())
                await websocket.recv()  # channel advertisement
                initial_frame = await websocket.recv()
                assert isinstance(initial_frame, bytes)
                started_state = await _receive_playback_state(websocket, "")

                await websocket.send(
                    _playback_request(
                        PlaybackCommand.PAUSE,
                        speed=0,
                        seek_time=None,
                        request_id="pause-1",
                    )
                )
                paused_state = await _receive_playback_state(websocket, "pause-1")

                seek_time = start_time + 1_000_000_000
                await websocket.send(
                    _playback_request(
                        PlaybackCommand.PLAY,
                        speed=2.0,
                        seek_time=seek_time,
                        request_id="seek-2",
                    )
                )
                playing_state = await _receive_playback_state(websocket, "seek-2")
                return (
                    server_info,
                    _playback_state(initial_frame),
                    started_state,
                    paused_state,
                    playing_state,
                )
        finally:
            await server.stop()

    server_info, initial, started, paused, playing = asyncio.run(run())
    assert server_info["capabilities"] == ["time", "playbackControl"]
    assert server_info["dataStartTime"] == {"sec": 1, "nsec": 1}
    assert server_info["dataEndTime"] == {"sec": 3, "nsec": 1}
    assert initial == (PlaybackStatus.BUFFERING, start_time, 1.0, False, "")
    assert started[0] is PlaybackStatus.PLAYING
    assert start_time <= started[1] <= end_time
    assert started[2:] == (1.0, False, "")
    assert paused[0] is PlaybackStatus.PAUSED
    assert start_time <= paused[1] <= end_time
    assert paused[2:] == (1.0, False, "pause-1")
    assert playing == (
        PlaybackStatus.BUFFERING,
        start_time + 1_000_000_000,
        2.0,
        True,
        "seek-2",
    )


def test_library_websocket_isolates_playback_for_the_same_recording(
    tmp_path: Path,
) -> None:
    _write_mcap(tmp_path / "recording.mcap", "/topic", 1, b"message")
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=True,
    )

    async def run() -> None:
        await server.start()
        query = urlencode([("file", "recording.mcap")])
        url = f"ws://127.0.0.1:{server.port}/ws?{query}"
        first = await connect(url, subprotocols=["foxglove.sdk.v1"])
        second = await connect(url, subprotocols=["foxglove.sdk.v1"])
        try:
            first_info: ServerInfoMessage = json.loads(await first.recv())
            await first.recv()
            second_info: ServerInfoMessage = json.loads(await second.recv())
            await second.recv()
            assert first_info["sessionId"] != second_info["sessionId"]
            await first.send(
                _playback_request(
                    PlaybackCommand.PAUSE,
                    speed=0.5,
                    seek_time=None,
                    request_id="first-pause",
                )
            )
            first_state = await asyncio.wait_for(
                _receive_playback_state(first, "first-pause"),
                timeout=1,
            )
            assert first_state[0] is PlaybackStatus.PAUSED
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    _receive_playback_state(second, "first-pause"),
                    timeout=0.1,
                )
        finally:
            await first.close()
            await second.close()
            await server.stop()

    asyncio.run(run())


def test_recording_library_discovers_nested_mcap_files(tmp_path: Path) -> None:
    _write_mcap(tmp_path / "first.mcap", "/first", 1, b"first")
    _write_mcap(tmp_path / "nested" / "second.mcap", "/second", 2, b"second")
    (tmp_path / "ignore.txt").write_text("not an MCAP")

    recordings = RecordingLibrary(tmp_path).recordings()

    assert [recording.path for recording in recordings] == [
        "first.mcap",
        "nested/second.mcap",
    ]
    assert all(recording.size_bytes > 0 for recording in recordings)


def test_recording_library_from_paths_exposes_only_selected_files(tmp_path: Path) -> None:
    selected = tmp_path / "selected.mcap"
    sibling = tmp_path / "sibling.mcap"
    _write_mcap(selected, "/selected", 1, b"selected")
    _write_mcap(sibling, "/sibling", 2, b"sibling")

    library = RecordingLibrary.from_paths((selected,))

    assert library.recordings() == (
        RecordingEntry(path="selected.mcap", size_bytes=selected.stat().st_size),
    )
    assert library.resolve(["selected.mcap"]) == (selected,)
    with pytest.raises(ValueError, match="not available"):
        library.resolve(["sibling.mcap"])


def test_recording_library_groups_rosbag_splits_as_one_entry(tmp_path: Path) -> None:
    first = tmp_path / "bag1" / "bag1_0.mcap"
    second = tmp_path / "bag1" / "bag1_1.mcap"
    solo = tmp_path / "solo.mcap"
    _write_mcap(first, "/first", 1, b"first")
    _write_mcap(second, "/second", 2, b"second")
    _write_mcap(solo, "/solo", 3, b"solo")

    recordings = RecordingLibrary(tmp_path).recordings()

    assert [recording.path for recording in recordings] == ["bag1", "solo.mcap"]
    assert recordings[0].size_bytes == first.stat().st_size + second.stat().st_size


def test_library_ui_always_uses_foxglove_desktop_deep_link() -> None:
    assert 'new URL("foxglove://open")' in _APP_JS
    assert "app.foxglove.dev" not in _APP_JS
    assert "openIn" not in _APP_JS


def test_library_ui_is_only_a_recording_launcher() -> None:
    assert '<a id="open"' in _INDEX_HTML
    assert 'id="play-pause"' not in _INDEX_HTML
    assert 'id="speed"' not in _INDEX_HTML
    assert 'id="loop"' not in _INDEX_HTML
    assert "active-sessions" not in _INDEX_HTML
    assert "controllerUrl" not in _APP_JS
    assert "/api/control" not in _APP_JS
    assert "/api/session" not in _APP_JS
    assert "window.location.assign" not in _APP_JS
    assert "setInterval" not in _APP_JS
    assert "path.href = websocketUrl([recording.path]).toString()" in _APP_JS
    assert "open.href = websocketUrl(files).toString()" in _APP_JS
    assert 'path.addEventListener("click", openFoxglove)' in _APP_JS
    assert 'open.addEventListener("click", openFoxglove)' in _APP_JS
    assert "window.location.href = foxgloveUrl(event.currentTarget.href).toString()" in _APP_JS
    assert ".playback-control" not in _STYLE_CSS


def test_library_ui_builds_multi_recording_foxglove_links() -> None:
    assert 'for (const file of files) params.append("file", file)' in _APP_JS
    assert 'const firstName = files[0].split("/").at(-1)' in _APP_JS
    assert (
        "const label = files.length === 1 ? firstName : `${firstName} +${files.length - 1}`"
        in _APP_JS
    )
    assert "new URL(`/ws/${encodeURIComponent(label)}`, window.location.href)" in _APP_JS
    assert "document.querySelectorAll" in _APP_JS
    assert 'input[name="recording"]:checked' in _APP_JS


def test_library_websocket_accepts_display_name_path(tmp_path: Path) -> None:
    _write_mcap(tmp_path / "my recording.mcap", "/camera", 1, b"frame")
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=False,
    )

    async def connect_with_name() -> None:
        await server.start()
        try:
            async with connect(
                f"ws://127.0.0.1:{server.port}/ws/my%20recording.mcap?file=my+recording.mcap",
                subprotocols=["foxglove.sdk.v1"],
            ) as websocket:
                server_info: ServerInfoMessage = json.loads(await websocket.recv())
                assert server_info["name"] == "pymcap-cli: my recording.mcap"
        finally:
            await server.stop()

    asyncio.run(connect_with_name())


@pytest.mark.parametrize(
    "preset",
    ["none", "compress", "decompress", "fast", "low"],
)
def test_recording_library_resolves_session_preset(tmp_path: Path, preset: str) -> None:
    recording = tmp_path / "recording.mcap"
    _write_mcap(recording, "/camera", 1, b"frame")
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=False,
    )

    request = server._resolve_query(f"file=recording.mcap&preset={preset}")

    assert request.files == (recording,)
    assert request.preset == preset


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("file=recording.mcap&preset=unknown", "preset"),
        ("file=recording.mcap&preset=none&preset=fast", "at most once"),
        ("file=recording.mcap&codec=h265", "Unknown"),
    ],
)
def test_recording_library_rejects_invalid_session_options(
    tmp_path: Path,
    query: str,
    message: str,
) -> None:
    _write_mcap(tmp_path / "recording.mcap", "/camera", 1, b"frame")
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=False,
    )

    with pytest.raises(ValueError, match=message):
        server._resolve_query(query)


def test_session_preset_overrides_server_transform_config() -> None:
    server_default = RoscompressConfig(codec="h265")

    assert _session_transform_config(server_default, None) is server_default
    assert _session_transform_config(server_default, "none") is None
    assert _session_transform_config(server_default, "decompress") == RosdecompressConfig()
    assert _session_transform_config(server_default, "compress") == RoscompressConfig(
        adaptive_quality=True
    )
    assert _session_transform_config(server_default, "fast") == RoscompressConfig(
        adaptive_quality=True,
        scale=960,
    )
    assert _session_transform_config(server_default, "low") == RoscompressConfig(
        adaptive_quality=True,
        scale=480,
    )


@pytest.mark.parametrize("path", ["/control", "/control.js", "/api/control", "/api/session"])
def test_library_server_does_not_serve_legacy_control_routes(tmp_path: Path, path: str) -> None:
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=False,
    )

    async def request() -> int:
        await server.start()
        try:

            def get_status() -> int:
                with pytest.raises(urllib.error.HTTPError) as exc_info:
                    urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}")
                return exc_info.value.code

            return await asyncio.to_thread(get_status)
        finally:
            await server.stop()

    assert asyncio.run(request()) == 404


def test_recording_library_resolves_order_and_rejects_unsafe_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "nested" / "second.mcap"
    _write_mcap(first, "/first", 1, b"first")
    _write_mcap(second, "/second", 2, b"second")
    library = RecordingLibrary(tmp_path)

    assert library.resolve(["nested/second.mcap", "first.mcap"]) == (second, first)

    with pytest.raises(ValueError, match="duplicate"):
        library.resolve(["first.mcap", "first.mcap"])
    with pytest.raises(ValueError, match="outside"):
        library.resolve(["../outside.mcap"])
    with pytest.raises(ValueError, match="MCAP"):
        library.resolve(["missing.txt"])


def test_recording_library_resolves_bag_directory_and_rejects_plain_directory(
    tmp_path: Path,
) -> None:
    bag = tmp_path / "bag"
    plain = tmp_path / "plain"
    _write_mcap(bag / "bag_0.mcap", "/first", 1, b"first")
    plain.mkdir()
    library = RecordingLibrary(tmp_path)

    assert library.resolve(["bag"]) == (bag,)
    with pytest.raises(ValueError, match="rosbag2"):
        library.resolve(["plain"])


def test_library_root_preserves_rosbag2_directory_behavior(tmp_path: Path) -> None:
    library = tmp_path / "recordings"
    library.mkdir()
    _write_mcap(library / "first.mcap", "/first", 1, b"first")
    bag = tmp_path / "bag"
    _write_mcap(bag / "bag_0.mcap", "/first", 1, b"first")

    assert _library_root([str(library)]) == library
    assert _library_root([str(bag)]) is None
    assert _library_root([str(library / "first.mcap")]) is None
    assert _library_root([str(library), str(bag)]) is None


def test_library_server_lists_recordings_and_merges_query_files(
    tmp_path: Path,
) -> None:
    _write_mcap(tmp_path / "first.mcap", "/first", 1, b"first")
    _write_mcap(tmp_path / "second.mcap", "/second", 2, b"second")

    async def get_json(url: str) -> dict[str, object]:
        def request() -> dict[str, object]:
            with closing(urllib.request.urlopen(url)) as response:  # noqa: S310
                assert response.status == 200
                return json.loads(response.read())

        return await asyncio.to_thread(request)

    async def run() -> tuple[list[str], list[tuple[int, bytes]]]:
        server = RecordingLibraryServer(
            RecordingLibrary(tmp_path),
            host="127.0.0.1",
            port=0,
            message_filter=MessageFilterOptions.from_args(),
            transform_config=None,
            speed=1_000_000,
            loop=False,
        )
        await server.start()
        try:
            listing = await get_json(f"http://127.0.0.1:{server.port}/api/recordings")
            paths = [item["path"] for item in listing["recordings"]]

            query = urlencode([("file", "first.mcap"), ("file", "second.mcap")])
            url = f"ws://127.0.0.1:{server.port}/ws?{query}"
            async with connect(url, subprotocols=["foxglove.websocket.v1"]) as websocket:
                await websocket.recv()
                advertisement = json.loads(await websocket.recv())
                await websocket.recv()  # initial playback state
                topics_by_id = {
                    channel["id"]: channel["topic"] for channel in advertisement["channels"]
                }
                await websocket.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "subscriptions": [
                                {"id": channel_id, "channelId": channel_id}
                                for channel_id in topics_by_id
                            ],
                        }
                    )
                )
                received: list[tuple[int, bytes]] = []
                while len(received) < 2:
                    frame = await asyncio.wait_for(websocket.recv(), timeout=2)
                    if isinstance(frame, bytes) and frame[0] == int(BinaryOpCodes.MESSAGE_DATA):
                        timestamp = struct.unpack_from("<Q", frame, 5)[0]
                        received.append((timestamp, frame[13:]))

            return paths, received
        finally:
            await server.stop()

    paths, received = asyncio.run(run())
    assert paths == ["first.mcap", "second.mcap"]
    assert received == [(1, b"first"), (2, b"second")]


def test_library_server_streams_grouped_bag_splits(tmp_path: Path) -> None:
    _write_mcap(tmp_path / "bag" / "bag_0.mcap", "/first", 2, b"second")
    _write_mcap(tmp_path / "bag" / "bag_1.mcap", "/second", 1, b"first")

    async def run() -> tuple[list[str], list[tuple[int, bytes]]]:
        server = RecordingLibraryServer(
            RecordingLibrary(tmp_path),
            host="127.0.0.1",
            port=0,
            message_filter=MessageFilterOptions.from_args(),
            transform_config=None,
            speed=1_000_000,
            loop=False,
        )
        await server.start()
        try:

            def get_listing() -> dict[str, object]:
                with closing(
                    urllib.request.urlopen(f"http://127.0.0.1:{server.port}/api/recordings")
                ) as response:
                    return json.loads(response.read())

            listing = await asyncio.to_thread(get_listing)
            paths = [entry["path"] for entry in listing["recordings"]]
            query = urlencode([("file", "bag")])
            async with connect(
                f"ws://127.0.0.1:{server.port}/ws?{query}",
                subprotocols=["foxglove.websocket.v1"],
            ) as websocket:
                await websocket.recv()
                advertisement = json.loads(await websocket.recv())
                await websocket.recv()  # initial playback state
                await websocket.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "subscriptions": [
                                {"id": channel["id"], "channelId": channel["id"]}
                                for channel in advertisement["channels"]
                            ],
                        }
                    )
                )
                messages: list[tuple[int, bytes]] = []
                while len(messages) < 2:
                    frame = await asyncio.wait_for(websocket.recv(), timeout=2)
                    if isinstance(frame, bytes) and frame[0] == int(BinaryOpCodes.MESSAGE_DATA):
                        messages.append((struct.unpack_from("<Q", frame, 5)[0], frame[13:]))
            return paths, messages
        finally:
            await server.stop()

    paths, messages = asyncio.run(run())
    assert paths == ["bag"]
    assert messages == [(1, b"first"), (2, b"second")]


def test_library_server_removes_session_when_listener_disconnects(
    tmp_path: Path,
) -> None:
    _write_mcap(tmp_path / "first.mcap", "/first", 1, b"first")
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=False,
    )

    async def run() -> None:
        await server.start()
        try:
            query = urlencode([("file", "first.mcap")])
            url = f"ws://127.0.0.1:{server.port}/ws?{query}"
            async with connect(url, subprotocols=["foxglove.websocket.v1"]) as websocket:
                await websocket.recv()
                advertisement = json.loads(await websocket.recv())
                await websocket.recv()  # initial playback state
                channel_id = advertisement["channels"][0]["id"]
                await websocket.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "subscriptions": [{"id": 1, "channelId": channel_id}],
                        }
                    )
                )
                while True:
                    frame = await websocket.recv()
                    if isinstance(frame, bytes) and frame[0] == int(BinaryOpCodes.MESSAGE_DATA):
                        break
                assert server.manager.active_session_count == 1

            for _ in range(100):
                if server.manager.active_session_count == 0:
                    break
                await asyncio.sleep(0.01)
            assert server.manager.active_session_count == 0
        finally:
            await server.stop()

    asyncio.run(run())


def test_library_server_logs_non_websocket_clients_without_traceback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_mcap(tmp_path / "first.mcap", "/first", 1, b"first")
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=False,
    )

    async def request() -> None:
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.write(b"GET / HTTP/1.0\r\nHost: example\r\n\r\n")
            await writer.drain()
            with suppress(ConnectionError):
                await reader.read()
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
        finally:
            await server.stop()

    with caplog.at_level(logging.WARNING, logger="websockets.server"):
        asyncio.run(request())

    records = [record for record in caplog.records if record.name == "websockets.server"]
    assert records, "expected the rejected handshake to be logged"
    assert all(record.levelno < logging.ERROR for record in records)
    assert all(record.exc_info is None for record in records)
    assert any("handshake" in record.getMessage() for record in records)


def test_recording_session_close_retrieves_keyboard_interrupt(tmp_path: Path) -> None:
    _write_mcap(tmp_path / "first.mcap", "/first", 1, b"first")
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=False,
    )

    results: dict[str, bool | BaseException | None] = {}

    async def raise_keyboard_interrupt() -> None:
        raise KeyboardInterrupt

    async def run() -> None:
        try:
            files = server.library.resolve(["first.mcap"])
            session = await server.manager.create(files)
            await session._cancel_task()

            task: asyncio.Task[None] = asyncio.ensure_future(raise_keyboard_interrupt())
            while not task.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.sleep(0)
            session._task = task

            await server.manager.remove(session)
            results["closed"] = True
            results["done"] = task.done()
            results["exception"] = task.exception()
        finally:
            await server.stop()

    with suppress(KeyboardInterrupt):
        asyncio.run(run())

    assert results.get("closed") is True
    assert results.get("done") is True
    assert isinstance(results.get("exception"), KeyboardInterrupt)


def test_recording_session_seek_clears_pending_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mcap(tmp_path / "rec.mcap", "/x", 1, b"a")

    async def run() -> int:
        session = await RecordingSession.create(
            (tmp_path / "rec.mcap",),
            message_filter=MessageFilterOptions.from_args(),
            transform_config=None,
            speed=1_000_000,
            loop=False,
        )
        calls = 0
        original = session.endpoint.clear_pending_frames

        def counting_clear() -> None:
            nonlocal calls
            calls += 1
            original()

        monkeypatch.setattr(session.endpoint, "clear_pending_frames", counting_clear)
        try:
            await session.handle_playback_control(
                PlaybackControlRequest(
                    playback_command=PlaybackCommand.PLAY,
                    playback_speed=1_000_000,
                    seek_time=session.timeline_start_ns,
                    request_id="seek",
                )
            )
        finally:
            await session.close()
        return calls

    assert asyncio.run(run()) >= 1


def test_recording_session_reports_failure_and_clears_it_after_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mcap(tmp_path / "rec.mcap", "/x", 1, b"a")

    should_fail = True
    dropped_messages = 3

    async def playback_runner(
        _prepared: PreparedPlayback,
        _sink: PlaybackSink,
        *,
        speed: float,
        loop: bool,
        show_status: bool,
        transform_plan: PlaybackTransformPlan | None = None,
        controller: PlaybackController | None = None,
        stats: PlaybackStats | None = None,
        start_time_ns: int | None = None,
        recordings: tuple[McapFile, ...] | None = None,
    ) -> PlaybackStats:
        del speed, loop, show_status, transform_plan, controller, start_time_ns, recordings
        if should_fail:
            raise RuntimeError("decoder failed")
        assert stats is not None
        stats.dropped_messages = dropped_messages
        stats.state = "Finished"
        return stats

    async def run() -> None:
        nonlocal dropped_messages, should_fail
        session = await RecordingSession.create(
            (tmp_path / "rec.mcap",),
            message_filter=MessageFilterOptions.from_args(),
            transform_config=None,
            speed=1,
            loop=False,
        )
        await session._cancel_task()
        statuses: list[tuple[StatusLevel, str, str | None]] = []
        removals: list[tuple[str, ...]] = []
        states: list[PlaybackState] = []

        async def record_status(
            level: StatusLevel,
            message: str,
            *,
            status_id: str | None = None,
        ) -> None:
            statuses.append((level, message, status_id))

        async def record_removal(status_ids: tuple[str, ...]) -> None:
            removals.append(status_ids)

        monkeypatch.setattr(session.endpoint, "send_status", record_status)
        monkeypatch.setattr(session.endpoint, "remove_status", record_removal)
        monkeypatch.setattr(session.endpoint, "broadcast_playback_state", states.append)

        try:
            session._playback_runner = playback_runner
            session.play()
            await session.wait()

            assert session.error == "decoder failed"
            assert session.stats.state == "Error"
            assert session.controller.state == "Paused"
            assert statuses == [(StatusLevel.ERROR, "Playback failed: decoder failed", "playback")]
            assert states[-1].status is PlaybackStatus.PAUSED

            should_fail = False
            session.play()
            await session.wait()

            assert session.error is None
            assert removals == [("playback",)]
            assert statuses[-1] == (
                StatusLevel.WARNING,
                "Playback dropped 3 frames to stay synchronized",
                "playback-degraded",
            )
            assert states[-1].status is PlaybackStatus.ENDED

            dropped_messages = 0
            session.play()
            await session.wait()

            assert removals == [("playback",), ("playback-degraded",)]
            assert states[-1].status is PlaybackStatus.ENDED
        finally:
            await session.close()

    asyncio.run(run())


def test_recording_session_owns_and_closes_open_recordings(tmp_path: Path) -> None:
    _write_mcap(tmp_path / "rec.mcap", "/x", 1, b"a")

    async def run() -> tuple[McapFile, ...]:
        session = await RecordingSession.create(
            (tmp_path / "rec.mcap",),
            message_filter=MessageFilterOptions.from_args(),
            transform_config=None,
            speed=1,
            loop=False,
        )
        recordings = session._recordings
        assert recordings is not None
        await session.close()
        return recordings

    recordings = asyncio.run(run())
    for recording in recordings:
        with pytest.raises(RuntimeError, match="closed"):
            recording.read_message()


def test_recording_session_closes_partial_recording_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first, "/first", 1, b"a")
    _write_mcap(second, "/second", 2, b"b")
    real_open = _playback.McapFile.open
    opened = real_open(first)
    calls = 0

    def fail_second_open(_path: Path) -> McapFile:
        nonlocal calls
        calls += 1
        if calls == 1:
            return opened
        raise OSError("open failed")

    monkeypatch.setattr(_playback.McapFile, "open", fail_second_open)

    async def create() -> None:
        await RecordingSession.create(
            (first, second),
            message_filter=MessageFilterOptions.from_args(),
            transform_config=None,
            speed=1,
            loop=False,
        )

    with pytest.raises(OSError, match="open failed"):
        asyncio.run(create())
    with pytest.raises(RuntimeError, match="closed"):
        opened.read_message()


@pytest.mark.parametrize(
    ("command", "speed", "expected_status", "subscribe_before_seek"),
    [
        (PlaybackCommand.PAUSE, 0.0, PlaybackStatus.PAUSED, True),
        (PlaybackCommand.PLAY, 1.0, PlaybackStatus.BUFFERING, True),
        (PlaybackCommand.PAUSE, 0.0, PlaybackStatus.PAUSED, False),
    ],
)
def test_library_seek_immediately_publishes_latest_message(
    tmp_path: Path,
    command: PlaybackCommand,
    speed: float,
    expected_status: PlaybackStatus,
    subscribe_before_seek: bool,
) -> None:
    _write_mcap_messages(
        tmp_path / "rec.mcap",
        "/x",
        [(10, b"a"), (20, b"b"), (30, b"c")],
    )
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=False,
    )

    async def run() -> tuple[PlaybackStatus, int, bytes]:
        await server.start()
        try:
            async with connect(
                f"ws://127.0.0.1:{server.port}/ws?file=rec.mcap",
                subprotocols=["foxglove.sdk.v1"],
            ) as websocket:
                await websocket.recv()
                advertisement = json.loads(await websocket.recv())
                initial_state = await websocket.recv()
                assert isinstance(initial_state, bytes)
                channel_id = advertisement["channels"][0]["id"]
                subscription = json.dumps(
                    {
                        "op": "subscribe",
                        "subscriptions": [{"id": 7, "channelId": channel_id}],
                    }
                )
                if subscribe_before_seek:
                    await websocket.send(subscription)
                await websocket.send(
                    _playback_request(
                        command,
                        speed=speed,
                        seek_time=25,
                        request_id="paused-seek",
                    )
                )
                state = await _receive_playback_state(websocket, "paused-seek")
                if not subscribe_before_seek:
                    await websocket.send(subscription)
                while True:
                    frame = await asyncio.wait_for(websocket.recv(), timeout=1)
                    if isinstance(frame, bytes) and frame[0] == int(BinaryOpCodes.MESSAGE_DATA):
                        timestamp_ns = struct.unpack_from("<Q", frame, 5)[0]
                        return state[0], timestamp_ns, frame[13:]
        finally:
            await server.stop()

    status, timestamp_ns, payload = asyncio.run(run())
    assert status is expected_status
    assert timestamp_ns == 20
    assert payload == b"b"


def test_recording_session_clamps_foxglove_seek_to_timeline_end(tmp_path: Path) -> None:
    _write_mcap(tmp_path / "rec.mcap", "/x", 1, b"a")

    async def run() -> PlaybackState:
        session = await RecordingSession.create(
            (tmp_path / "rec.mcap",),
            message_filter=MessageFilterOptions.from_args(),
            transform_config=None,
            speed=1,
            loop=False,
        )
        try:
            return await session.handle_playback_control(
                PlaybackControlRequest(
                    playback_command=PlaybackCommand.PAUSE,
                    playback_speed=0,
                    seek_time=session.timeline_end_ns + 10_000_000_000,
                    request_id="seek-past-end",
                )
            )
        finally:
            await session.close()

    state = asyncio.run(run())
    assert state.status is PlaybackStatus.PAUSED
    assert state.current_time == 1
    assert state.did_seek is True


def test_recording_session_playback_state_uses_active_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mcap(tmp_path / "rec.mcap", "/x", 10_000_000_000, b"a")
    monkeypatch.setattr(_playback.time, "monotonic", lambda: 10.0)

    async def run() -> PlaybackState:
        session = await RecordingSession.create(
            (tmp_path / "rec.mcap",),
            message_filter=MessageFilterOptions.from_args(),
            transform_config=None,
            speed=1,
            loop=False,
        )
        session.stats.playhead_ns = 1_000_000_000
        session.sink._clock = PlaybackClock(
            record_origin_ns=2_000_000_000,
            wall_origin=9.0,
            speed=1,
            recording_end_ns=10_000_000_000,
        )
        try:
            return session._foxglove_playback_state(did_seek=False)
        finally:
            await session.close()

    assert asyncio.run(run()).current_time == 3_000_000_000
