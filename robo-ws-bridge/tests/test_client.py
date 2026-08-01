"""Unit tests for the Foxglove WebSocket bridge client."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
from typing import TYPE_CHECKING, cast

import pytest
from robo_ws_bridge import (
    ConnectionGraph,
    ServiceCallError,
    ServiceCallResponse,
    WebSocketBridgeClient,
    WebSocketBridgeServer,
)
from robo_ws_bridge.ws_types import BinaryOpCodes, ConnectionStatus
from websockets.asyncio.server import serve

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


def test_connection_graph_applies_updates_and_removals() -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    seen: list[ConnectionGraph] = []
    client.on_connection_graph_update(seen.append)

    asyncio.run(
        client._handle_connection_graph_update(
            {
                "op": "connectionGraphUpdate",
                "publishedTopics": [{"name": "/foo", "publisherIds": ["pub-1"]}],
                "subscribedTopics": [{"name": "/foo", "subscriberIds": ["sub-1"]}],
                "advertisedServices": [{"name": "/svc", "providerIds": ["srv-1"]}],
            }
        )
    )

    graph = client.connection_graph
    assert graph.published_topics == ({"name": "/foo", "publisherIds": ["pub-1"]},)
    assert graph.subscribed_topics == ({"name": "/foo", "subscriberIds": ["sub-1"]},)
    assert graph.advertised_services == ({"name": "/svc", "providerIds": ["srv-1"]},)
    assert seen == [graph]

    asyncio.run(
        client._handle_connection_graph_update(
            {
                "op": "connectionGraphUpdate",
                "removedTopics": ["/foo"],
                "removedServices": ["/svc"],
            }
        )
    )

    graph = client.connection_graph
    assert graph.published_topics == ()
    assert graph.subscribed_topics == ()
    assert graph.advertised_services == ()


def test_connection_graph_subscription_is_persistent_intent() -> None:
    client = WebSocketBridgeClient("ws://example:8765")

    asyncio.run(client.subscribe_connection_graph())
    assert client._wants_connection_graph is True

    asyncio.run(
        client._handle_connection_graph_update(
            {
                "op": "connectionGraphUpdate",
                "publishedTopics": [{"name": "/foo", "publisherIds": ["pub-1"]}],
            }
        )
    )
    asyncio.run(client.unsubscribe_connection_graph())

    assert client._wants_connection_graph is False
    assert client.connection_graph == ConnectionGraph((), (), ())


def test_advertise_services_populates_services_map() -> None:
    client = WebSocketBridgeClient("ws://example:8765")

    asyncio.run(
        client._handle_advertise_services(
            {
                "op": "advertiseServices",
                "services": [
                    {
                        "id": 1,
                        "name": "/reset",
                        "type": "std_srvs/Empty",
                    },
                    {
                        "id": 2,
                        "name": "/set_bool",
                        "type": "std_srvs/SetBool",
                        "request": {
                            "encoding": "ros2",
                            "schemaName": "std_srvs/SetBool_Request",
                            "schemaEncoding": "ros2msg",
                            "schema": "bool data",
                        },
                    },
                ],
            }
        )
    )

    services = client.services
    assert set(services) == {1, 2}
    assert services[1]["name"] == "/reset"
    assert services[2]["type"] == "std_srvs/SetBool"


def test_unadvertise_services_removes_from_map() -> None:
    client = WebSocketBridgeClient("ws://example:8765")

    asyncio.run(
        client._handle_advertise_services(
            {
                "op": "advertiseServices",
                "services": [
                    {"id": 1, "name": "/a", "type": "std_srvs/Empty"},
                    {"id": 2, "name": "/b", "type": "std_srvs/Empty"},
                ],
            }
        )
    )

    asyncio.run(
        client._handle_unadvertise_services({"op": "unadvertiseServices", "serviceIds": [1, 99]})
    )

    assert set(client.services) == {2}


def test_remove_status_notifies_handlers() -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    removed: list[list[str]] = []
    client.on_remove_status(removed.append)

    asyncio.run(client._handle_remove_status({"op": "removeStatus", "statusIds": ["boot"]}))

    assert removed == [["boot"]]


def _service_call_response_frame(
    service_id: int, call_id: int, encoding: str, payload: bytes
) -> bytes:
    encoding_bytes = encoding.encode("ascii")
    return (
        struct.pack(
            "<BIII",
            int(BinaryOpCodes.SERVICE_CALL_RESPONSE),
            service_id,
            call_id,
            len(encoding_bytes),
        )
        + encoding_bytes
        + payload
    )


def test_handle_service_call_response_resolves_pending_future() -> None:
    client = WebSocketBridgeClient("ws://example:8765")

    async def scenario() -> ServiceCallResponse:
        future: asyncio.Future[ServiceCallResponse] = asyncio.get_running_loop().create_future()
        client._pending_calls[7] = future
        client._handle_service_call_response(
            _service_call_response_frame(2, 7, "cdr", b"\x01\x02\x03")
        )
        return await future

    result = asyncio.run(scenario())
    assert result == ServiceCallResponse(
        service_id=2, call_id=7, encoding="cdr", payload=b"\x01\x02\x03"
    )


def test_handle_service_call_failure_sets_exception() -> None:
    client = WebSocketBridgeClient("ws://example:8765")

    async def scenario() -> None:
        future: asyncio.Future[ServiceCallResponse] = asyncio.get_running_loop().create_future()
        client._pending_calls[3] = future
        await client._handle_service_call_failure(
            {"op": "serviceCallFailure", "serviceId": 1, "callId": 3, "message": "boom"}
        )
        await future

    with pytest.raises(ServiceCallError, match="boom"):
        asyncio.run(scenario())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_call_service_round_trips_through_a_real_server() -> None:
    port = _free_port()
    seen: dict[str, object] = {}

    async def run() -> ServiceCallResponse:
        server = WebSocketBridgeServer(
            host="127.0.0.1",
            port=port,
            name="svc-bridge",
            capabilities=["services"],
            supported_encodings=["cdr"],
        )

        async def advertise(state) -> None:
            await state.websocket.send(
                json.dumps(
                    {
                        "op": "advertiseServices",
                        "services": [{"id": 5, "name": "/echo", "type": "demo/Echo"}],
                    }
                )
            )

        async def handle_request(state, payload: bytes) -> None:
            service_id, call_id = struct.unpack_from("<II", payload, 1)
            enc_len = struct.unpack_from("<I", payload, 9)[0]
            encoding = payload[13 : 13 + enc_len].decode("ascii")
            request_payload = payload[13 + enc_len :]
            seen.update(service_id=service_id, encoding=encoding, request=bytes(request_payload))
            await state.websocket.send(
                _service_call_response_frame(service_id, call_id, encoding, b"pong")
            )

        server.on_connect(advertise)
        server.register_binary_handler(BinaryOpCodes.SERVICE_CALL_REQUEST, handle_request)
        await server.start()

        client = WebSocketBridgeClient(f"ws://127.0.0.1:{port}", min_retry_delay=0.2)
        ready = asyncio.Event()
        client.on_server_info(lambda *_: ready.set())
        await client.connect()
        try:
            await asyncio.wait_for(ready.wait(), timeout=5.0)
            # Wait for the service advertisement to arrive.
            for _ in range(50):
                if 5 in client.services:
                    break
                await asyncio.sleep(0.05)
            return await client.call_service(5, b"ping", encoding="cdr", timeout=5.0)
        finally:
            await client.disconnect()
            await server.stop()

    result = asyncio.run(run())
    assert result == ServiceCallResponse(service_id=5, call_id=1, encoding="cdr", payload=b"pong")
    assert seen == {"service_id": 5, "encoding": "cdr", "request": b"ping"}


def test_client_does_not_reconnect_after_policy_violation() -> None:
    port = _free_port()
    connection_count = 0

    async def run() -> None:
        nonlocal connection_count

        async def reject(websocket) -> None:
            nonlocal connection_count
            connection_count += 1
            await websocket.close(code=1008, reason="select a recording")

        server = await serve(reject, "127.0.0.1", port)
        client = WebSocketBridgeClient(
            f"ws://127.0.0.1:{port}",
            min_retry_delay=0.01,
            max_retry_delay=0.01,
        )
        await client.connect()
        try:
            assert client._receiver_task is not None
            await asyncio.wait_for(client._receiver_task, timeout=1)
            assert client.get_connection_status() is ConnectionStatus.DISCONNECTED
            assert client._connection_task is not None
            await asyncio.wait_for(client._connection_task, timeout=2)
        finally:
            await client.disconnect()
            server.close()
            await server.wait_closed()

    asyncio.run(run())
    assert connection_count == 1


class RecordingWebSocket:
    def __init__(self) -> None:
        self.sent: list[str | bytes] = []

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)


def test_disconnected_operations_reject_network_requests() -> None:
    client = WebSocketBridgeClient("ws://example:8765")

    async def run() -> None:
        with pytest.raises(RuntimeError, match="Cannot advertise"):
            await client.advertise("/topic", encoding="cdr", schema_name="example/Message")
        with pytest.raises(RuntimeError, match="Cannot publish"):
            await client.publish(1, b"payload")
        with pytest.raises(RuntimeError, match="Cannot get parameters"):
            await client.get_parameters()
        with pytest.raises(RuntimeError, match="Cannot set parameters"):
            await client.set_parameters([])
        with pytest.raises(RuntimeError, match="Cannot fetch asset"):
            await client.fetch_asset("package://example/mesh.dae")
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.subscribe_to_channel(10, 20)
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.unsubscribe_from_channel(10)

    asyncio.run(run())


def test_client_publish_and_subscription_frames() -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    websocket = RecordingWebSocket()
    client._websocket = cast("ClientConnection", websocket)

    async def run() -> int:
        channel_id = await client.advertise(
            "/topic",
            encoding="cdr",
            schema_name="example/Message",
            schema="int32 value",
            schema_encoding="ros2msg",
        )
        await client.publish(channel_id, b"payload")
        await client.subscribe_to_channel(7, 42)
        await client.unsubscribe_from_channel(7)
        await client.unadvertise(channel_id)
        return channel_id

    channel_id = asyncio.run(run())

    assert channel_id == 1
    assert json.loads(cast("str", websocket.sent[0])) == {
        "op": "advertise",
        "channels": [
            {
                "id": 1,
                "topic": "/topic",
                "encoding": "cdr",
                "schemaName": "example/Message",
                "schema": "int32 value",
                "schemaEncoding": "ros2msg",
            }
        ],
    }
    assert websocket.sent[1] == struct.pack("<BI", int(BinaryOpCodes.CLIENT_MESSAGE_DATA), 1) + (
        b"payload"
    )
    assert json.loads(cast("str", websocket.sent[2])) == {
        "op": "subscribe",
        "subscriptions": [{"id": 7, "channelId": 42}],
    }
    assert json.loads(cast("str", websocket.sent[3])) == {
        "op": "unsubscribe",
        "subscriptionIds": [7],
    }
    assert json.loads(cast("str", websocket.sent[4])) == {
        "op": "unadvertise",
        "channelIds": [1],
    }
    assert client._active_subscriptions == set()
    assert client._subscription_to_channel == {}
    assert client._channel_to_subscription == {}


def test_parameter_and_asset_requests_correlate_responses() -> None:
    client = WebSocketBridgeClient("ws://example:8765")

    class RespondingWebSocket(RecordingWebSocket):
        async def send(self, message: str | bytes) -> None:
            await super().send(message)
            request = json.loads(cast("str", message))
            if request["op"] in {"getParameters", "setParameters"}:
                client._handle_parameter_values(
                    {
                        "op": "parameterValues",
                        "id": request["id"],
                        "parameters": [{"name": "speed", "value": 2.0}],
                    }
                )
            elif request["op"] == "fetchAsset":
                response = (
                    struct.pack(
                        "<BIBI",
                        int(BinaryOpCodes.FETCH_ASSET_RESPONSE),
                        request["requestId"],
                        0,
                        0,
                    )
                    + b"mesh"
                )
                client._handle_fetch_asset_response(response)

    websocket = RespondingWebSocket()
    client._websocket = cast("ClientConnection", websocket)

    async def run() -> tuple[list[dict[str, str | float]], list[dict[str, str | float]], bytes]:
        fetched = await client.get_parameters(["speed"])
        updated = await client.set_parameters([{"name": "speed", "value": 2.0}])
        asset = await client.fetch_asset("package://example/mesh.dae")
        return fetched, updated, asset

    fetched, updated, asset = asyncio.run(run())

    assert fetched == [{"name": "speed", "value": 2.0}]
    assert updated == [{"name": "speed", "value": 2.0}]
    assert asset == b"mesh"
    assert client._pending_param_requests == {}
    assert client._pending_asset_requests == {}
