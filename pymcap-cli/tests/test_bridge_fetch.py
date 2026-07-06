"""Tests for `pymcap-cli bridge fetch`."""

from __future__ import annotations

import asyncio
import socket
import struct

import pytest
from pymcap_cli.cmd.bridge import fetch as fetch_module
from pymcap_cli.cmd.bridge.fetch import _fetch_async, fetch
from robo_ws_bridge import FetchAssetError
from robo_ws_bridge.server import WebSocketBridgeServer
from robo_ws_bridge.ws_types import BinaryOpCodes


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _asset_response_frame(request_id: int, *, status: int, error: str, data: bytes) -> bytes:
    error_bytes = error.encode("utf-8")
    return (
        struct.pack(
            "<BIBI", int(BinaryOpCodes.FETCH_ASSET_RESPONSE), request_id, status, len(error_bytes)
        )
        + error_bytes
        + data
    )


def _make_server(port: int, *, status: int, error: str, data: bytes) -> WebSocketBridgeServer:
    server = WebSocketBridgeServer(
        host="127.0.0.1", port=port, name="asset-bridge", capabilities=["assets"]
    )

    async def on_fetch(state, message: dict) -> None:
        frame = _asset_response_frame(message["requestId"], status=status, error=error, data=data)
        await state.websocket.send(frame)

    server.register_json_handler("fetchAsset", on_fetch)
    return server


def test_fetch_async_returns_asset_bytes() -> None:
    port = _free_port()
    payload = b"<robot><link/></robot>"

    async def run() -> bytes:
        server = _make_server(port, status=0, error="", data=payload)
        await server.start()
        try:
            return await _fetch_async(
                f"ws://127.0.0.1:{port}",
                "package://robot/urdf/robot.urdf",
                connect_timeout=5.0,
                call_timeout=5.0,
            )
        finally:
            await server.stop()

    assert asyncio.run(run()) == payload


def test_fetch_async_raises_on_failure_status() -> None:
    port = _free_port()

    async def run() -> bytes:
        server = _make_server(port, status=1, error="asset not found", data=b"")
        await server.start()
        try:
            return await _fetch_async(
                f"ws://127.0.0.1:{port}", "package://missing", connect_timeout=5.0, call_timeout=5.0
            )
        finally:
            await server.stop()

    with pytest.raises(FetchAssetError, match="asset not found"):
        asyncio.run(run())


def test_fetch_writes_to_output_file(tmp_path, monkeypatch) -> None:
    payload = b"binary-asset-bytes"
    out = tmp_path / "asset.bin"

    async def fake_fetch_async(_url, _uri, *, connect_timeout, call_timeout) -> bytes:  # noqa: ARG001
        return payload

    monkeypatch.setattr(fetch_module, "_fetch_async", fake_fetch_async)
    rc = fetch(target="ws://127.0.0.1:8765", uri="package://robot/x.bin", output=out)

    assert rc == 0
    assert out.read_bytes() == payload


def test_fetch_returns_1_when_bridge_unreachable(tmp_path) -> None:
    port = _free_port()
    rc = fetch(
        target=f"ws://127.0.0.1:{port}",
        uri="package://x",
        output=tmp_path / "x.bin",
        connect_timeout=0.3,
    )
    assert rc == 1
