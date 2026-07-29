"""End-to-end test for `pymcap-cli bridge` against a real WebSocketBridgeServer."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.bridge._shared import fetch_bridge_info
from pymcap_cli.cmd.bridge.check import check as bridge_check
from pymcap_cli.cmd.bridge.hz import hz
from pymcap_cli.cmd.bridge.info import info
from pymcap_cli.cmd.bridge.stats import stats
from robo_ws_bridge import WebSocketBridgeServer
from robo_ws_bridge.server import Channel

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _pick_free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _ServerThread:
    """Run a `WebSocketBridgeServer` on a background thread with its own event loop."""

    def __init__(self, port: int, *, publish_messages: bool = False) -> None:
        self.port = port
        self.publish_messages = publish_messages
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
        publisher: asyncio.Task[None] | None = None
        if self.publish_messages:

            async def publish_periodically() -> None:
                while True:
                    server_time_ns = time.time_ns() - 10_000_000
                    await server.publish_time(server_time_ns)
                    await server.publish_message(1, b"x" * 100, timestamp_ns=server_time_ns)
                    await asyncio.sleep(0.02)

            publisher = asyncio.create_task(publish_periodically())
        try:
            await self._stop.wait()
        finally:
            if publisher is not None:
                publisher.cancel()
                await asyncio.gather(publisher, return_exceptions=True)
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


@pytest.fixture
def publishing_bridge_server() -> Iterator[_ServerThread]:
    server = _ServerThread(_pick_free_port(), publish_messages=True)
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
    rc = info(
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
def test_bridge_stats_streams_json_snapshots_from_real_server(
    publishing_bridge_server: _ServerThread,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = stats(
        target=f"ws://127.0.0.1:{publishing_bridge_server.port}",
        all_topics=True,
        duration=0.2,
        interval=0.1,
        window=1.0,
        json_output=True,
    )

    assert rc == 0
    snapshots = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(snapshots) == 2
    final_topics = {topic["topic"]: topic for topic in snapshots[-1]["topics"]}
    assert final_topics["/foo"]["hz"] > 0
    assert final_topics["/foo"]["payload_bytes_per_second"] > 0
    assert final_topics["/foo"]["message_size_bytes"]["mean"] == 100
    assert snapshots[-1]["bridge_clock_offset_ms"] is not None
    assert final_topics["/foo"]["message_delay_ms"] is not None
    assert abs(final_topics["/foo"]["message_delay_ms"]) < 100
    assert final_topics["/bar"]["hz"] == 0


@pytest.mark.e2e
def test_bridge_hz_prints_each_snapshot_instead_of_redrawing(
    publishing_bridge_server: _ServerThread,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = hz(
        target=f"ws://127.0.0.1:{publishing_bridge_server.port}",
        all_topics=True,
        duration=0.2,
        interval=0.1,
        window=1.0,
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert output.count("Topic frequency") == 1
    assert output.count("Time") == 1
    assert output.count("/foo") == 2
    assert output.count("/bar") == 2


@pytest.mark.e2e
def test_bridge_check_against_real_server(
    bridge_server: _ServerThread,
    tmp_path: Path,
) -> None:
    spec = tmp_path / "recording.yaml"
    spec.write_text(
        """\
version: 1
topics:
  sample:
    topic: /foo
    schema:
      name: std_msgs/String
      encoding: ros2msg
    message_encoding: cdr
"""
    )

    result = bridge_check(
        target=f"127.0.0.1:{bridge_server.port}",
        spec=spec,
        duration=0.01,
        connect_timeout=5.0,
        discover_seconds=0.05,
    )

    assert result == 0


@pytest.mark.e2e
def test_bridge_check_requires_graph_only_for_live_constraints(
    bridge_server: _ServerThread,
    tmp_path: Path,
) -> None:
    spec = tmp_path / "recording.yaml"
    spec.write_text(
        """\
version: 1
topics:
  sample:
    topic: /foo
    live:
      publishers:
        min: 1
"""
    )

    result = bridge_check(
        target=f"127.0.0.1:{bridge_server.port}",
        spec=spec,
        duration=0,
        connect_timeout=5.0,
        discover_seconds=0.05,
    )

    assert result == 1


@pytest.mark.e2e
def test_bridge_cmd_rich_output_does_not_crash(bridge_server: _ServerThread) -> None:
    rc = info(
        target=f"127.0.0.1:{bridge_server.port}",
        connect_timeout=5.0,
        discover_seconds=1.0,
    )
    assert rc == 0
