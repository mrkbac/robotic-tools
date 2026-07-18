"""Tests for Foxglove WebSocket server publishing helpers."""

import asyncio
import json
import socket
import struct

from robo_ws_bridge import WebSocketBridgeEndpoint, WebSocketBridgeServer
from robo_ws_bridge.server import Channel
from robo_ws_bridge.ws_types import BinaryOpCodes
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve


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


def test_endpoint_can_share_listener_and_isolates_channels_by_path() -> None:
    port = _free_port()

    async def run() -> tuple[str, str]:
        first = WebSocketBridgeEndpoint(name="first")
        first.register_channel(Channel(1, "/first", "raw", "bytes", ""))
        second = WebSocketBridgeEndpoint(name="second")
        second.register_channel(Channel(1, "/second", "raw", "bytes", ""))

        async def route(websocket) -> None:
            assert websocket.request is not None
            endpoint = first if websocket.request.path == "/first" else second
            await endpoint.handle_connection(websocket)

        server = await serve(
            route,
            "127.0.0.1",
            port,
            subprotocols=["foxglove.websocket.v1"],
        )
        try:
            async with connect(
                f"ws://127.0.0.1:{port}/first",
                subprotocols=["foxglove.websocket.v1"],
            ) as websocket:
                await websocket.recv()
                advertised = json.loads(await websocket.recv())
                first_topic = advertised["channels"][0]["topic"]
            async with connect(
                f"ws://127.0.0.1:{port}/second",
                subprotocols=["foxglove.websocket.v1"],
            ) as websocket:
                await websocket.recv()
                advertised = json.loads(await websocket.recv())
                second_topic = advertised["channels"][0]["topic"]
            return first_topic, second_topic
        finally:
            server.close()
            await server.wait_closed()

    assert asyncio.run(run()) == ("/first", "/second")
