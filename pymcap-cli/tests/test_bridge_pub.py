"""Tests for `pymcap-cli bridge pub`."""

from __future__ import annotations

import asyncio
import socket
import struct

from mcap_ros2_support_fast.decoder import DecoderFactory
from pymcap_cli.cmd.bridge.pub import _pub_async, pub
from robo_ws_bridge.server import Channel as ServerChannel
from robo_ws_bridge.server import WebSocketBridgeServer
from robo_ws_bridge.ws_types import BinaryOpCodes
from small_mcap import Schema

_BOOL_SCHEMA = Schema(id=1, name="std_msgs/Bool", encoding="ros2msg", data=b"bool data")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pub_async_publishes_encoded_message() -> None:
    port = _free_port()
    decode = DecoderFactory().decoder_for("cdr", _BOOL_SCHEMA)
    assert decode is not None
    received: list[dict] = []
    got_message = asyncio.Event()

    async def run() -> int:
        server = WebSocketBridgeServer(
            host="127.0.0.1",
            port=port,
            name="pub-bridge",
            capabilities=["clientPublish"],
            supported_encodings=["cdr"],
        )
        server.register_channel(
            ServerChannel(
                id=1,
                topic="/flag",
                encoding="cdr",
                schema_name="std_msgs/Bool",
                schema="bool data",
                schema_encoding="ros2msg",
            )
        )

        async def on_client_message(_state, payload: bytes) -> None:
            _opcode, _channel_id = struct.unpack_from("<BI", payload, 0)
            message = decode(payload[5:])
            received.append({"data": message.data})
            got_message.set()

        server.register_binary_handler(BinaryOpCodes.CLIENT_MESSAGE_DATA, on_client_message)
        await server.start()
        try:
            count = await _pub_async(
                f"ws://127.0.0.1:{port}",
                "/flag",
                {"data": True},
                count=1,
                rate=0.0,
                connect_timeout=5.0,
                discover_seconds=2.0,
            )
            await asyncio.wait_for(got_message.wait(), timeout=5.0)
            return count
        finally:
            await server.stop()

    count = asyncio.run(run())
    assert count == 1
    assert received == [{"data": True}]


def test_pub_returns_1_when_bridge_unreachable() -> None:
    port = _free_port()
    rc = pub(
        target=f"ws://127.0.0.1:{port}", topic="/x", fields=["data:=true"], connect_timeout=0.3
    )
    assert rc == 1


def test_pub_returns_1_on_bad_count() -> None:
    rc = pub(target="ws://127.0.0.1:1", topic="/x", count=0, connect_timeout=0.3)
    assert rc == 1
