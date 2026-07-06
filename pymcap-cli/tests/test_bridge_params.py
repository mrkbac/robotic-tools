"""Tests for `pymcap-cli bridge params`."""

from __future__ import annotations

import asyncio
import json
import socket

from pymcap_cli.cmd.bridge.params import _params_async, params
from pymcap_cli.display.param_render import build_parameters_table
from rich.console import Console, RenderableType
from robo_ws_bridge.server import WebSocketBridgeServer


def _render(renderable: RenderableType) -> str:
    console = Console(record=True, width=120, color_system=None)
    console.print(renderable)
    return console.export_text()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_server(port: int) -> WebSocketBridgeServer:
    store: dict[str, object] = {"/max_speed": 1.0, "/use_sim_time": True}
    server = WebSocketBridgeServer(
        host="127.0.0.1",
        port=port,
        name="param-bridge",
        capabilities=["parameters"],
    )

    async def on_get(state, message: dict) -> None:
        names = message["parameterNames"] or list(store)
        params_out = [{"name": n, "value": store[n]} for n in names if n in store]
        await state.websocket.send(
            json.dumps({"op": "parameterValues", "parameters": params_out, "id": message["id"]})
        )

    async def on_set(state, message: dict) -> None:
        for param in message["parameters"]:
            store[param["name"]] = param["value"]
        await state.websocket.send(
            json.dumps(
                {"op": "parameterValues", "parameters": message["parameters"], "id": message["id"]}
            )
        )

    server.register_json_handler("getParameters", on_get)
    server.register_json_handler("setParameters", on_set)
    return server


def test_params_async_gets_named_parameter() -> None:
    port = _free_port()

    async def run() -> list:
        server = _make_server(port)
        await server.start()
        try:
            return await _params_async(
                f"ws://127.0.0.1:{port}", ["/max_speed"], [], connect_timeout=5.0, call_timeout=5.0
            )
        finally:
            await server.stop()

    result = asyncio.run(run())
    assert result == [{"name": "/max_speed", "value": 1.0}]


def test_params_async_sets_parameter() -> None:
    port = _free_port()

    async def run() -> list:
        server = _make_server(port)
        await server.start()
        try:
            return await _params_async(
                f"ws://127.0.0.1:{port}",
                [],
                [{"name": "/max_speed", "value": 2.5}],
                connect_timeout=5.0,
                call_timeout=5.0,
            )
        finally:
            await server.stop()

    result = asyncio.run(run())
    assert result == [{"name": "/max_speed", "value": 2.5}]


def test_build_parameters_table_renders_names_and_values() -> None:
    output = _render(build_parameters_table([{"name": "/max_speed", "value": 2.5}]))
    assert "/max_speed" in output
    assert "2.5" in output


def test_params_returns_1_when_bridge_unreachable() -> None:
    port = _free_port()
    rc = params(target=f"ws://127.0.0.1:{port}", connect_timeout=0.3)
    assert rc == 1
