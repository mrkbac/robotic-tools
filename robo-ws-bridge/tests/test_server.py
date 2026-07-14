"""Tests for Foxglove WebSocket server publishing helpers."""

import asyncio
import json
import socket
import struct

from robo_ws_bridge import WebSocketBridgeServer
from robo_ws_bridge.ws_types import BinaryOpCodes
from websockets.asyncio.client import connect


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_publish_time_broadcasts_time_frame() -> None:
    port = _free_port()

    async def run() -> bytes:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port, capabilities=["time"])
        await server.start()
        try:
            async with connect(
                f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"]
            ) as websocket:
                server_info = json.loads(await websocket.recv())
                assert server_info["capabilities"] == ["time"]
                await server.publish_time(123456789)
                frame = await websocket.recv()
                assert isinstance(frame, bytes)
                return frame
        finally:
            await server.stop()

    frame = asyncio.run(run())
    assert frame == struct.pack("<BQ", int(BinaryOpCodes.TIME), 123456789)
