"""End-to-end test for `pymcap-cli bridge` against a real WebSocketBridgeServer."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.bridge_cmd import fetch_bridge_info, inspect
from robo_ws_bridge import WebSocketBridgeServer
from robo_ws_bridge.server import Channel

if TYPE_CHECKING:
    from collections.abc import Iterator


def _pick_free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _ServerThread:
    """Run a `WebSocketBridgeServer` on a background thread with its own event loop."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        server = WebSocketBridgeServer(
            host="127.0.0.1",
            port=self.port,
            name="test-bridge",
            capabilities=["clientPublish", "time"],
            supported_encodings=["cdr", "json"],
            metadata={"source": "test"},
        )
        server.register_channel(
            Channel(
                id=1,
                topic="/foo",
                encoding="cdr",
                schema_name="std_msgs/String",
                schema="",
                schema_encoding="ros2msg",
            )
        )
        server.register_channel(
            Channel(
                id=2,
                topic="/bar",
                encoding="cdr",
                schema_name="std_msgs/Int32",
                schema="",
                schema_encoding="ros2msg",
            )
        )
        await server.start()
        self._stop = asyncio.Event()
        self._ready.set()
        try:
            await self._stop.wait()
        finally:
            await server.stop()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("bridge server did not become ready in time")

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5.0)


@pytest.fixture
def bridge_server() -> Iterator[_ServerThread]:
    server = _ServerThread(_pick_free_port())
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.e2e
def test_fetch_bridge_info_against_real_server(bridge_server: _ServerThread) -> None:
    info = fetch_bridge_info(
        f"ws://127.0.0.1:{bridge_server.port}",
        connect_timeout=5.0,
        discover_seconds=1.0,
    )
    assert info.server_info["name"] == "test-bridge"
    assert "clientPublish" in info.server_info["capabilities"]
    assert info.server_info.get("supportedEncodings") == ["cdr", "json"]
    assert info.server_info.get("metadata") == {"source": "test"}
    topics = sorted(c["topic"] for c in info.channels)
    assert topics == ["/bar", "/foo"]


@pytest.mark.e2e
def test_bridge_cmd_json_output(
    bridge_server: _ServerThread, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = inspect(
        target=f"ws://127.0.0.1:{bridge_server.port}",
        json_output=True,
        connect_timeout=5.0,
        discover_seconds=1.0,
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert payload["url"] == f"ws://127.0.0.1:{bridge_server.port}"
    assert payload["server"]["name"] == "test-bridge"
    assert payload["server"]["supportedEncodings"] == ["cdr", "json"]
    topics = sorted(c["topic"] for c in payload["channels"])
    assert topics == ["/bar", "/foo"]


@pytest.mark.e2e
def test_bridge_cmd_rich_output_does_not_crash(bridge_server: _ServerThread) -> None:
    rc = inspect(
        target=f"127.0.0.1:{bridge_server.port}",
        connect_timeout=5.0,
        discover_seconds=1.0,
    )
    assert rc == 0
