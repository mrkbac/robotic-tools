"""Unit tests for the Foxglove WebSocket bridge client."""

from __future__ import annotations

import asyncio
import json
import socket
import struct

import pytest
from robo_ws_bridge import (
    ConnectionGraph,
    ServiceCallError,
    ServiceCallResponse,
    WebSocketBridgeClient,
    WebSocketBridgeServer,
)
from robo_ws_bridge.ws_types import BinaryOpCodes


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
