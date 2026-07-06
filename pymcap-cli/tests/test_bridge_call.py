"""Tests for `pymcap-cli bridge call`."""

from __future__ import annotations

import asyncio
import json
import socket
import struct

import pytest
from mcap_ros2_support_fast import ROS2EncoderFactory
from pymcap_cli.cmd.bridge._codec import FieldSyntaxError, parse_field_args
from pymcap_cli.cmd.bridge.call import _call_service_async, call
from pymcap_cli.display.service_render import build_service_response_table
from rich.console import Console, RenderableType
from robo_ws_bridge import WebSocketBridgeServer
from robo_ws_bridge.ws_types import BinaryOpCodes, ServiceInfo
from small_mcap import Schema

_SET_BOOL: ServiceInfo = {
    "id": 5,
    "name": "/set_bool",
    "type": "std_srvs/SetBool",
    "request": {
        "encoding": "cdr",
        "schemaName": "std_srvs/SetBool_Request",
        "schemaEncoding": "ros2msg",
        "schema": "bool data",
    },
    "response": {
        "encoding": "cdr",
        "schemaName": "std_srvs/SetBool_Response",
        "schemaEncoding": "ros2msg",
        "schema": "bool success\nstring message",
    },
}


def _render(renderable: RenderableType) -> str:
    console = Console(record=True, width=120, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_parse_field_args_parses_bool_int_and_list() -> None:
    assert parse_field_args(["data:=true", "n:=3", "xs:=[1, 2]"]) == {
        "data": True,
        "n": 3,
        "xs": [1, 2],
    }


def test_parse_field_args_falls_back_to_string() -> None:
    assert parse_field_args(["name:=hello"]) == {"name": "hello"}


def test_parse_field_args_accepts_nested_json_object() -> None:
    assert parse_field_args(['pose:={"x": 1.0}']) == {"pose": {"x": 1.0}}


def test_parse_field_args_empty_returns_empty_dict() -> None:
    assert parse_field_args([]) == {}


@pytest.mark.parametrize("token", ["data", ":=true", "no-separator"])
def test_parse_field_args_rejects_malformed(token: str) -> None:
    with pytest.raises(FieldSyntaxError):
        parse_field_args([token])


def test_build_service_response_table_renders_fields() -> None:
    output = _render(build_service_response_table(_SET_BOOL, {"success": True, "message": "ok"}))
    assert "/set_bool" in output
    assert "std_srvs/SetBool" in output
    assert "success" in output
    assert "True" in output
    assert "'ok'" in output


def test_build_service_response_table_flattens_nested() -> None:
    output = _render(build_service_response_table(_SET_BOOL, {"pose": {"x": 1.5}}))
    assert "pose.x" in output
    assert "1.5" in output


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_call_service_async_round_trips_set_bool() -> None:
    port = _free_port()
    response_schema = Schema(
        id=99,
        name="std_srvs/SetBool_Response",
        encoding="ros2msg",
        data=b"bool success\nstring message",
    )
    encode_response = ROS2EncoderFactory().encoder_for(response_schema)
    assert encode_response is not None

    async def run() -> tuple[ServiceInfo, dict]:
        server = WebSocketBridgeServer(
            host="127.0.0.1",
            port=port,
            name="svc-bridge",
            capabilities=["services"],
            supported_encodings=["cdr"],
        )

        async def advertise(state) -> None:
            await state.websocket.send(
                json.dumps({"op": "advertiseServices", "services": [_SET_BOOL]})
            )

        async def handle_request(state, payload: bytes) -> None:
            _service_id, call_id = struct.unpack_from("<II", payload, 1)
            enc_len = struct.unpack_from("<I", payload, 9)[0]
            encoding = payload[13 : 13 + enc_len].decode("ascii")
            reply = bytes(encode_response({"success": True, "message": "ok"}))
            encoding_bytes = encoding.encode("ascii")
            frame = (
                struct.pack(
                    "<BIII",
                    int(BinaryOpCodes.SERVICE_CALL_RESPONSE),
                    _service_id,
                    call_id,
                    len(encoding_bytes),
                )
                + encoding_bytes
                + reply
            )
            await state.websocket.send(frame)

        server.on_connect(advertise)
        server.register_binary_handler(BinaryOpCodes.SERVICE_CALL_REQUEST, handle_request)
        await server.start()
        try:
            return await _call_service_async(
                f"ws://127.0.0.1:{port}",
                "/set_bool",
                {"data": True},
                connect_timeout=5.0,
                discover_seconds=2.0,
                call_timeout=5.0,
            )
        finally:
            await server.stop()

    service, response = asyncio.run(run())
    assert service["name"] == "/set_bool"
    assert response == {"success": True, "message": "ok"}


def test_call_returns_1_when_bridge_unreachable() -> None:
    port = _free_port()  # nothing is listening on this port
    rc = call(target=f"ws://127.0.0.1:{port}", service="/x", connect_timeout=0.3)
    assert rc == 1


def test_call_returns_1_on_malformed_field() -> None:
    rc = call(target="ws://127.0.0.1:1", service="/x", fields=["bad-token"], connect_timeout=0.3)
    assert rc == 1
