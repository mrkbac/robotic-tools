"""Recording-library tests for ``pymcap-cli bridge serve DIRECTORY``."""

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

import pytest
from pymcap_cli.cmd.bridge._library import (
    _APP_JS,
    _INDEX_HTML,
    _STYLE_CSS,
    RecordingLibrary,
    RecordingLibraryServer,
    RecordingSession,
)
from pymcap_cli.cmd.bridge.serve import _library_root
from pymcap_cli.core.message_filter import MessageFilterOptions
from robo_ws_bridge import PlaybackCommand, PlaybackControlRequest, PlaybackStatus
from robo_ws_bridge.ws_types import BinaryOpCodes, ServerInfoMessage
from small_mcap import McapWriter
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_time = 1_000_000_001
    end_time = 3_000_000_001
    _write_mcap(tmp_path / "first.mcap", "/first", start_time, b"first")
    _write_mcap(tmp_path / "second.mcap", "/second", end_time, b"second")
    monkeypatch.setattr("pymcap_cli.cmd.bridge.serve._SETTLE_SECONDS", 0.0)
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

                await websocket.send(
                    _playback_request(
                        PlaybackCommand.PAUSE,
                        speed=0.5,
                        seek_time=None,
                        request_id="pause-1",
                    )
                )
                paused_frame = await websocket.recv()
                assert isinstance(paused_frame, bytes)

                seek_time = start_time + 1_000_000_000
                await websocket.send(
                    _playback_request(
                        PlaybackCommand.PLAY,
                        speed=2.0,
                        seek_time=seek_time,
                        request_id="seek-2",
                    )
                )
                playing_frame = await websocket.recv()
                assert isinstance(playing_frame, bytes)
                return (
                    server_info,
                    _playback_state(paused_frame),
                    _playback_state(playing_frame),
                )
        finally:
            await server.stop()

    server_info, paused, playing = asyncio.run(run())
    assert server_info["capabilities"] == ["time", "playbackControl"]
    assert server_info["dataStartTime"] == {"sec": 1, "nsec": 1}
    assert server_info["dataEndTime"] == {"sec": 3, "nsec": 1}
    assert paused == (PlaybackStatus.PAUSED, start_time, 0.5, False, "pause-1")
    assert playing == (
        PlaybackStatus.PLAYING,
        start_time + 1_000_000_000,
        2.0,
        True,
        "seek-2",
    )


def test_library_websocket_isolates_playback_for_the_same_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mcap(tmp_path / "recording.mcap", "/topic", 1, b"message")
    monkeypatch.setattr("pymcap_cli.cmd.bridge.serve._SETTLE_SECONDS", 0.0)
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
    assert "window.location.href =" not in _APP_JS
    assert "window.location.assign" not in _APP_JS
    assert "setInterval" not in _APP_JS
    assert "path.href = foxgloveUrl([recording.path]).toString()" in _APP_JS
    assert "open.href = foxgloveUrl(files).toString()" in _APP_JS
    assert ".playback-control" not in _STYLE_CSS


def test_library_ui_builds_multi_recording_foxglove_links() -> None:
    assert 'for (const file of files) params.append("file", file)' in _APP_JS
    assert "document.querySelectorAll" in _APP_JS
    assert 'input[name="recording"]:checked' in _APP_JS


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mcap(tmp_path / "first.mcap", "/first", 1, b"first")
    _write_mcap(tmp_path / "second.mcap", "/second", 2, b"second")
    monkeypatch.setattr("pymcap_cli.cmd.bridge.serve._SETTLE_SECONDS", 0.0)

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


def test_library_server_streams_grouped_bag_splits(tmp_path: Path, monkeypatch) -> None:
    _write_mcap(tmp_path / "bag" / "bag_0.mcap", "/first", 2, b"second")
    _write_mcap(tmp_path / "bag" / "bag_1.mcap", "/second", 1, b"first")
    monkeypatch.setattr("pymcap_cli.cmd.bridge.serve._SETTLE_SECONDS", 0.0)

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mcap(tmp_path / "first.mcap", "/first", 1, b"first")
    monkeypatch.setattr("pymcap_cli.cmd.bridge.serve._SETTLE_SECONDS", 0.0)
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
