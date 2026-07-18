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
    _CONTROL_HTML,
    _CONTROL_JS,
    _INDEX_HTML,
    _STYLE_CSS,
    RecordingLibrary,
    RecordingLibraryServer,
    RecordingSession,
)
from pymcap_cli.cmd.bridge.serve import _library_root
from pymcap_cli.core.message_filter import MessageFilterOptions
from robo_ws_bridge.ws_types import BinaryOpCodes
from small_mcap import McapWriter
from websockets.asyncio.client import connect

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
    for script in (_APP_JS, _CONTROL_JS):
        assert 'new URL("foxglove://open")' in script
        assert "app.foxglove.dev" not in script
        assert "openIn" not in script


def test_library_ui_navigates_to_control_view_without_popup() -> None:
    assert "window.open" not in _APP_JS
    assert "popup" not in _APP_JS
    assert "controllerUrl(files).toString()" in _APP_JS
    assert 'new URL("/control"' in _APP_JS
    assert "const foxgloveUrl" in _APP_JS


def test_control_ui_has_speed_presets_and_stable_play_pause_width() -> None:
    for preset in ("1", "5", "10"):
        assert f'data-speed="{preset}"' in _CONTROL_HTML
    assert ">1x<" in _CONTROL_HTML
    assert ">5x<" in _CONTROL_HTML
    assert ">10x<" in _CONTROL_HTML
    assert 'class="speed-preset"' in _CONTROL_HTML
    assert "speed-preset" in _CONTROL_JS
    assert 'control("speed", {speed:' in _CONTROL_JS
    assert "#play-pause { min-width:" in _STYLE_CSS


def test_library_ui_uses_one_play_pause_control_and_speed_selector() -> None:
    for page in (_INDEX_HTML, _CONTROL_HTML):
        assert 'id="play-pause"' in page
        assert 'id="speed"' in page
        assert 'id="play"' not in page
        assert 'id="pause"' not in page
        assert 'id="stop"' not in page
    for script in (_APP_JS, _CONTROL_JS):
        assert 'control("toggle")' in script
        assert 'control("speed"' in script
        assert 'control("stop")' not in script
        assert "droppedMessages" in script
        assert "document.activeElement !== speed" in script
        assert "speed.oninput" in script


def test_library_server_serves_controller_for_selected_files(tmp_path: Path) -> None:
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

    async def request() -> tuple[str, bool]:
        await server.start()
        try:
            url = f"http://127.0.0.1:{server.port}/control?" + urlencode([("file", "first.mcap")])

            def get_page() -> str:
                with closing(urllib.request.urlopen(url)) as response:  # noqa: S310
                    return response.read().decode()

            page = await asyncio.to_thread(get_page)
            files = server.library.resolve(["first.mcap"])
            return page, server.manager.get(files) is None
        finally:
            await server.stop()

    page, is_idle = asyncio.run(request())
    assert "<title>MCAP playback</title>" in page
    assert 'id="seek"' in page
    assert 'id="loop"' in page
    assert 'id="speed"' in page
    assert "checked" in page
    assert 'src="/control.js"' in page
    assert is_idle


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

    async def run() -> tuple[list[str], list[tuple[int, bytes]], str, int]:
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

            status = await get_json(f"http://127.0.0.1:{server.port}/api/session?{query}")
            return paths, received, str(status["state"]), int(status["droppedMessages"])
        finally:
            await server.stop()

    paths, received, state, dropped_messages = asyncio.run(run())
    assert paths == ["first.mcap", "second.mcap"]
    assert received == [(1, b"first"), (2, b"second")]
    assert state == "finished"
    assert dropped_messages == 0


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


def test_library_server_control_requires_a_valid_action(tmp_path: Path) -> None:
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

    async def request() -> tuple[int, str]:
        await server.start()
        try:
            url = f"http://127.0.0.1:{server.port}/api/control?" + urlencode(
                [("file", "first.mcap"), ("action", "explode")]
            )

            def get_error() -> tuple[int, str]:
                with pytest.raises(urllib.error.HTTPError) as exc_info:
                    urllib.request.urlopen(url)  # noqa: S310
                return exc_info.value.code, exc_info.value.read().decode()

            return await asyncio.to_thread(get_error)
        finally:
            await server.stop()

    status, body = asyncio.run(request())
    assert status == 400
    assert "toggle, seek, loop, or speed" in body


def test_library_server_controls_shared_session(tmp_path: Path) -> None:
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

    async def get_json(url: str) -> dict[str, object]:
        def request() -> dict[str, object]:
            with closing(urllib.request.urlopen(url)) as response:  # noqa: S310
                return json.loads(response.read())

        return await asyncio.to_thread(request)

    async def run() -> tuple[str, str, str]:
        await server.start()
        try:
            query = urlencode([("file", "first.mcap")])
            base = f"http://127.0.0.1:{server.port}/api/control?{query}"
            started = await get_json(f"{base}&action=toggle")
            paused = await get_json(f"{base}&action=toggle")
            resumed = await get_json(f"{base}&action=toggle")
            return (
                str(started["state"]),
                str(paused["state"]),
                str(resumed["state"]),
            )
        finally:
            await server.stop()

    states = asyncio.run(run())
    assert states[0] in {"preparing", "waiting"}
    assert states[1] == "paused"
    assert states[2] in {"preparing", "waiting"}


def test_library_server_changes_speed_for_shared_session(tmp_path: Path) -> None:
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

    async def get_json(url: str) -> dict[str, object]:
        def request() -> dict[str, object]:
            with closing(urllib.request.urlopen(url)) as response:  # noqa: S310
                return json.loads(response.read())

        return await asyncio.to_thread(request)

    async def run() -> tuple[float, float]:
        await server.start()
        try:
            query = urlencode([("file", "first.mcap")])
            base = f"http://127.0.0.1:{server.port}/api/control?{query}"
            changed = await get_json(f"{base}&action=speed&speed=2")
            status = await get_json(f"http://127.0.0.1:{server.port}/api/session?{query}")
            return float(changed["speed"]), float(status["speed"])
        finally:
            await server.stop()

    assert asyncio.run(run()) == (2.0, 2.0)


def test_library_server_rejects_invalid_speed(tmp_path: Path) -> None:
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

    async def request() -> tuple[int, str, bool]:
        await server.start()
        try:
            query = urlencode([("file", "first.mcap"), ("action", "speed"), ("speed", "0")])
            url = f"http://127.0.0.1:{server.port}/api/control?{query}"

            def get_error() -> tuple[int, str]:
                with pytest.raises(urllib.error.HTTPError) as exc_info:
                    urllib.request.urlopen(url)  # noqa: S310
                return exc_info.value.code, exc_info.value.read().decode()

            status, body = await asyncio.to_thread(get_error)
            files = server.library.resolve(["first.mcap"])
            return status, body, server.manager.get(files) is None
        finally:
            await server.stop()

    status, body, is_idle = asyncio.run(request())
    assert status == 400
    assert "finite and positive" in body
    assert is_idle


def test_library_server_seeks_shared_session_by_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mcap(tmp_path / "first.mcap", "/first", 1, b"first")
    _write_mcap(tmp_path / "second.mcap", "/second", 1_000_000_001, b"second")
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

    async def get_json(url: str) -> dict[str, object]:
        def request() -> dict[str, object]:
            with closing(urllib.request.urlopen(url)) as response:  # noqa: S310
                return json.loads(response.read())

        return await asyncio.to_thread(request)

    async def run() -> tuple[dict[str, object], bytes]:
        await server.start()
        try:
            query = urlencode([("file", "first.mcap"), ("file", "second.mcap")])
            url = f"http://127.0.0.1:{server.port}/api/control?{query}&action=seek&offset=0.5"
            status = await get_json(url)
            websocket_url = f"ws://127.0.0.1:{server.port}/ws?{query}"
            async with connect(
                websocket_url,
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
                while True:
                    frame = await asyncio.wait_for(websocket.recv(), timeout=2)
                    if isinstance(frame, bytes) and frame[0] == int(BinaryOpCodes.MESSAGE_DATA):
                        return status, frame[13:]
        finally:
            await server.stop()

    status, payload = asyncio.run(run())
    assert status["positionSeconds"] == pytest.approx(0.5)
    assert status["durationSeconds"] == pytest.approx(1.0)
    assert payload == b"second"


def test_library_server_changes_loop_state_for_shared_session(tmp_path: Path) -> None:
    _write_mcap(tmp_path / "first.mcap", "/first", 1, b"first")
    server = RecordingLibraryServer(
        RecordingLibrary(tmp_path),
        host="127.0.0.1",
        port=0,
        message_filter=MessageFilterOptions.from_args(),
        transform_config=None,
        speed=1,
        loop=True,
    )

    async def get_json(url: str) -> dict[str, object]:
        def request() -> dict[str, object]:
            with closing(urllib.request.urlopen(url)) as response:  # noqa: S310
                return json.loads(response.read())

        return await asyncio.to_thread(request)

    async def run() -> tuple[bool, bool]:
        await server.start()
        try:
            query = urlencode([("file", "first.mcap")])
            base = f"http://127.0.0.1:{server.port}/api/control?{query}"
            disabled = await get_json(f"{base}&action=loop&enabled=false")
            enabled = await get_json(f"{base}&action=loop&enabled=true")
            return bool(disabled["loop"]), bool(enabled["loop"])
        finally:
            await server.stop()

    assert asyncio.run(run()) == (False, True)


def test_library_server_removes_session_after_last_listener_disconnects(
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
        session_idle_timeout=0.01,
    )

    async def run() -> None:
        await server.start()
        try:
            query = urlencode([("file", "first.mcap")])
            files = server.library.resolve(["first.mcap"])
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
                assert server.manager.get(files) is not None

            for _ in range(100):
                if server.manager.get(files) is None:
                    break
                await asyncio.sleep(0.01)
            assert server.manager.get(files) is None
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


def test_library_ui_counts_network_drops_in_status_line() -> None:
    for script in (_APP_JS, _CONTROL_JS):
        assert "droppedFrames" in script


def test_library_server_lists_active_sessions(tmp_path: Path) -> None:
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

    async def get_json(url: str) -> dict[str, object]:
        def request() -> dict[str, object]:
            with closing(urllib.request.urlopen(url)) as response:  # noqa: S310
                return json.loads(response.read())

        return await asyncio.to_thread(request)

    async def run() -> tuple[list[object], list[object]]:
        await server.start()
        try:
            base = f"http://127.0.0.1:{server.port}"
            empty = await get_json(f"{base}/api/sessions")
            query = urlencode([("file", "first.mcap")])
            await get_json(f"{base}/api/control?{query}&action=toggle")
            populated = await get_json(f"{base}/api/sessions")
            return list(empty["sessions"]), list(populated["sessions"])
        finally:
            await server.stop()

    empty_sessions, populated_sessions = asyncio.run(run())
    assert empty_sessions == []
    assert len(populated_sessions) == 1
    session = populated_sessions[0]
    assert isinstance(session, dict)
    assert session["files"] == ["first.mcap"]
    assert "state" in session


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

    results: dict[str, object] = {}

    async def raise_keyboard_interrupt() -> None:
        raise KeyboardInterrupt

    async def run() -> None:
        try:
            files = server.library.resolve(["first.mcap"])
            session = await server.manager.get_or_create(files)
            await session._cancel_task()

            task: asyncio.Task[None] = asyncio.ensure_future(raise_keyboard_interrupt())
            while not task.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.sleep(0)
            session._task = task

            await session.close()
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


def test_library_session_status_reports_dropped_frames(tmp_path: Path) -> None:
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

    async def request() -> dict[str, object]:
        await server.start()
        try:
            query = urlencode([("file", "first.mcap")])
            url = f"http://127.0.0.1:{server.port}/api/session?{query}"

            def get_json() -> dict[str, object]:
                with closing(urllib.request.urlopen(url)) as response:  # noqa: S310
                    return json.loads(response.read())

            return await asyncio.to_thread(get_json)
        finally:
            await server.stop()

    status = asyncio.run(request())
    assert status["droppedFrames"] == 0


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
            await session.seek(0.0)
        finally:
            await session.close()
        return calls

    assert asyncio.run(run()) >= 1
