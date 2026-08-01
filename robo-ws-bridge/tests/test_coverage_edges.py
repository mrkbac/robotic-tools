"""Focused tests for the less common client and endpoint protocol paths."""

from __future__ import annotations

import asyncio
import json
import struct
from typing import cast
from unittest.mock import AsyncMock

import pytest
from robo_ws_bridge import (
    PlaybackCommand,
    PlaybackState,
    PlaybackStatus,
    WebSocketBridgeClient,
    WebSocketBridgeEndpoint,
    WebSocketBridgeServer,
)
from robo_ws_bridge.server import Channel, ConnectionOutbox, ConnectionState
from robo_ws_bridge.ws_types import BinaryOpCodes, ConnectionStatus, JsonOpCodes
from websockets.exceptions import ConnectionClosed


class FakeWebSocket:
    """Small websocket-shaped object for direct endpoint and client tests."""

    def __init__(self, incoming: list[object] | None = None) -> None:
        self.sent: list[str | bytes] = []
        self.closed = False
        self.close_args: list[tuple[object, ...]] = []
        self.close_kwargs: list[dict[str, object]] = []
        self._incoming = iter(incoming or [])
        self.subprotocol: str | None = None

    async def send(self, payload: str | bytes) -> None:
        self.sent.append(payload)

    async def close(self, *args: object, **kwargs: object) -> None:
        self.closed = True
        self.close_args.append(args)
        self.close_kwargs.append(kwargs)

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> object:
        try:
            item = next(self._incoming)
        except StopIteration as error:
            raise StopAsyncIteration from error
        if isinstance(item, BaseException):
            raise item
        return item


def _client_socket(client: WebSocketBridgeClient, websocket: FakeWebSocket) -> None:
    client._websocket = cast("object", websocket)


def _message_frame(subscription_id: int, timestamp_ns: int, payload: bytes) -> bytes:
    return (
        bytes([int(BinaryOpCodes.MESSAGE_DATA)])
        + struct.pack("<IQ", subscription_id, timestamp_ns)
        + payload
    )


def _fetch_asset_frame(
    request_id: int, status: int, error: bytes = b"", payload: bytes = b""
) -> bytes:
    return (
        struct.pack(
            "<BIBI",
            int(BinaryOpCodes.FETCH_ASSET_RESPONSE),
            request_id,
            status,
            len(error),
        )
        + error
        + payload
    )


def test_client_callbacks_properties_and_connection_graph_frames() -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    websocket = FakeWebSocket()
    _client_socket(client, websocket)
    events: list[str] = []

    async def connected() -> None:
        events.append("connected")

    def broken() -> None:
        events.append("broken")
        raise RuntimeError("handler failure")

    client.on_connect(connected)
    client.on_disconnect(lambda: events.append("disconnected"))
    client.on_reconnecting(lambda: events.append("reconnecting"))
    client.on_server_info(lambda *_: events.append("server-info"))
    client.on_status(lambda *_: events.append("status"))
    client.on_remove_status(lambda *_: events.append("remove-status"))
    client.on_advertised_channel(lambda *_: events.append("advertised"))
    client.on_channel_unadvertised(lambda *_: events.append("unadvertised"))
    client.on_message(lambda *_: events.append("message"))
    client.on_time_update(lambda *_: events.append("time"))
    client.on_connection_graph_update(lambda *_: events.append("graph"))
    client.on_connect(broken)

    async def run() -> None:
        await client._set_connection_status(ConnectionStatus.CONNECTED)
        await client._set_connection_status(ConnectionStatus.RECONNECTING)
        await client._set_connection_status(ConnectionStatus.DISCONNECTED)
        await client._handle_status({"op": "status", "level": 1, "message": "ready"})
        await client.subscribe_connection_graph()
        await client.unsubscribe_connection_graph()

    asyncio.run(run())

    assert client.channels == {}
    assert client.services == {}
    assert client.server_info is None
    client._running = True
    assert client.is_connected
    assert events == ["connected", "broken", "reconnecting", "disconnected", "status"]
    assert json.loads(cast("str", websocket.sent[0])) == {
        "op": JsonOpCodes.SUBSCRIBE_CONNECTION_GRAPH.value
    }
    assert json.loads(cast("str", websocket.sent[1])) == {
        "op": JsonOpCodes.UNSUBSCRIBE_CONNECTION_GRAPH.value
    }
    asyncio.run(client.connect())


def test_client_connect_disconnect_and_deferred_graph_paths() -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    client._running = True
    client._active_subscriptions.add(7)
    websocket = FakeWebSocket()

    async def failing_send(_payload: str | bytes) -> None:
        raise OSError("socket closed")

    websocket.send = failing_send
    client._websocket = cast("object", websocket)

    async def run() -> None:
        await client.disconnect()
        await client.disconnect()

    asyncio.run(run())
    assert websocket.closed
    assert client._websocket is None

    deferred = WebSocketBridgeClient("ws://example:8765")
    asyncio.run(deferred.subscribe("/deferred"))
    asyncio.run(deferred.unsubscribe("/unknown"))
    asyncio.run(deferred.subscribe_connection_graph())
    assert deferred._wants_connection_graph
    asyncio.run(deferred.unsubscribe_connection_graph())
    assert not deferred._wants_connection_graph


def test_client_subscription_restore_and_cleanup_edges() -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    websocket = FakeWebSocket()
    _client_socket(client, websocket)
    client._running = True
    client._advertised_channels[4] = {
        "id": 4,
        "topic": "/camera/image",
        "encoding": "cdr",
        "schemaName": "sensor_msgs/Image",
        "schema": "data",
    }

    async def run() -> None:
        await client.subscribe("/missing")
        await client.subscribe("/camera/image")
        await client.subscribe("/camera/image")
        await client.unsubscribe("/missing")
        await client.unsubscribe("/camera/image")
        await client._subscribe_to_channel(9)
        client._websocket = None
        await client._subscribe_to_channel(10)
        await client._unsubscribe_from_channel(10, "/camera/image")

    asyncio.run(run())
    assert client._subscribed_topics == set()
    assert client._intended_subscriptions == set()
    client._active_subscriptions.clear()
    assert client._active_subscriptions == set()

    restored = WebSocketBridgeClient("ws://example:8765")
    restored_socket = FakeWebSocket()
    _client_socket(restored, restored_socket)
    restored._advertised_channels[5] = {
        "id": 5,
        "topic": "/present",
        "encoding": "cdr",
        "schemaName": "example/Present",
        "schema": "",
    }
    restored._intended_subscriptions = {"/present", "/not-yet"}
    restored._wants_connection_graph = True
    asyncio.run(restored._restore_subscriptions())
    assert len(restored_socket.sent) == 2


def test_client_connection_failure_and_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    client = WebSocketBridgeClient(
        "ws://example:8765", min_retry_delay=0.25, max_retry_delay=1.0, backoff_factor=2.0
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("robo_ws_bridge.client.asyncio.sleep", sleep)
    client._consecutive_failures = 2
    asyncio.run(client._backoff_sleep())
    assert delays == [1.0]
    client._consecutive_failures = 0

    async def fail() -> None:
        raise OSError("refused")

    async def stop_backoff() -> None:
        client._should_connect = False

    client._attempt_connection = fail
    client._backoff_sleep = stop_backoff
    client._should_connect = True
    asyncio.run(client._connect_continuously())
    assert client._consecutive_failures == 1


def test_client_message_loop_handles_unknown_and_generic_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    websocket = FakeWebSocket([42, RuntimeError("bad frame")])
    _client_socket(client, websocket)
    client._should_connect = True

    async def sleep(_delay: float) -> None:
        client._should_connect = False

    monkeypatch.setattr("robo_ws_bridge.client.asyncio.sleep", sleep)

    original_status = client._set_connection_status

    async def status(status: ConnectionStatus) -> None:
        await original_status(status)
        if status is ConnectionStatus.RECONNECTING:
            client._should_connect = False

    client._set_connection_status = status
    asyncio.run(client._handle_messages_loop())
    assert not client._running


def test_client_message_loop_handles_normal_close_and_wait_break() -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    websocket = FakeWebSocket([ConnectionClosed(None, None)])
    _client_socket(client, websocket)
    client._should_connect = True

    async def status(status: ConnectionStatus) -> None:
        if status is ConnectionStatus.RECONNECTING:
            client._should_connect = False

    client._set_connection_status = status
    asyncio.run(client._handle_messages_loop())

    waiting = WebSocketBridgeClient("ws://example:8765")
    waiting._should_connect = True

    async def wait() -> None:
        waiting._should_connect = False

    waiting._connection_event.wait = wait
    asyncio.run(waiting._handle_messages_loop())
    assert not waiting._running


def test_client_json_dispatch_and_advertisement_edges() -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    client._handle_server_info = AsyncMock()
    client._handle_status = AsyncMock()
    client._handle_remove_status = AsyncMock()
    client._handle_advertise = AsyncMock()
    client._handle_unadvertise = AsyncMock()
    client._handle_advertise_services = AsyncMock()
    client._handle_unadvertise_services = AsyncMock()
    client._handle_connection_graph_update = AsyncMock()
    client._handle_service_call_failure = AsyncMock()

    async def run() -> None:
        messages = [
            {"op": "serverInfo"},
            {"op": "status"},
            {"op": "removeStatus"},
            {"op": "advertise"},
            {"op": "unadvertise"},
            {"op": "advertiseServices"},
            {"op": "unadvertiseServices"},
            {"op": "connectionGraphUpdate"},
            {"op": "serviceCallFailure"},
            {"op": "parameterValues", "parameters": []},
            {"op": "unknown"},
        ]
        for message in messages:
            await client._handle_json(json.dumps(message))
        await client._handle_json("[]")

    asyncio.run(run())
    for method in (
        client._handle_server_info,
        client._handle_status,
        client._handle_remove_status,
        client._handle_advertise,
        client._handle_unadvertise,
        client._handle_advertise_services,
        client._handle_unadvertise_services,
        client._handle_connection_graph_update,
        client._handle_service_call_failure,
    ):
        method.assert_awaited_once()

    client = WebSocketBridgeClient("ws://example:8765")
    websocket = FakeWebSocket()
    _client_socket(client, websocket)
    client._running = True
    client._subscribed_topics.add("/camera/image")
    advertised: list[int] = []
    unadvertised: list[int] = []
    client.on_advertised_channel(lambda channel: advertised.append(channel["id"]))
    client.on_channel_unadvertised(lambda channel: unadvertised.append(channel["id"]))

    async def advertise_and_remove() -> None:
        await client._handle_advertise(
            {
                "op": "advertise",
                "channels": [
                    {
                        "id": 8,
                        "topic": "/camera/image",
                        "encoding": "cdr",
                        "schemaName": "sensor_msgs/Image",
                        "schema": "data",
                    }
                ],
            }
        )
        await client._handle_unadvertise({"op": "unadvertise", "channelIds": [8, 99]})

    asyncio.run(advertise_and_remove())
    assert advertised == [8]
    assert unadvertised == [8]


def test_client_binary_dispatch_and_response_edge_cases() -> None:
    client = WebSocketBridgeClient("ws://example:8765")
    seen_messages: list[tuple[int, bytes]] = []
    seen_times: list[int] = []
    client.on_message(
        lambda _channel, timestamp, payload: seen_messages.append((timestamp, payload))
    )
    client.on_time_update(seen_times.append)

    async def run() -> None:
        await client._handle_binary(bytes([int(BinaryOpCodes.MESSAGE_DATA)]))
        await client._handle_binary(bytes([int(BinaryOpCodes.TIME)]))
        await client._handle_binary(bytes([int(BinaryOpCodes.SERVICE_CALL_RESPONSE)]))
        await client._handle_binary(bytes([int(BinaryOpCodes.FETCH_ASSET_RESPONSE)]))
        await client._handle_binary(bytes([255]))

    asyncio.run(run())

    frame = _message_frame(4, 123, b"data")
    asyncio.run(client._handle_message_data(frame))
    client._subscription_to_channel[4] = 9
    asyncio.run(client._handle_message_data(frame))
    client._advertised_channels[9] = {
        "id": 9,
        "topic": "/data",
        "encoding": "raw",
        "schemaName": "example/Raw",
        "schema": "",
    }
    asyncio.run(client._handle_message_data(frame))
    assert seen_messages == [(123, b"data")]

    async def response_edges() -> None:
        client._handle_service_call_response(
            struct.pack("<BIII", int(BinaryOpCodes.SERVICE_CALL_RESPONSE), 1, 99, 0)
        )
        done: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        done.set_result(None)
        client._pending_calls[100] = cast("object", done)
        client._handle_service_call_response(
            struct.pack("<BIII", int(BinaryOpCodes.SERVICE_CALL_RESPONSE), 1, 100, 0)
        )

        await client._handle_service_call_failure(
            {"op": "serviceCallFailure", "callId": 99, "message": "x"}
        )
        failure_done: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        failure_done.set_result(None)
        client._pending_calls[100] = cast("object", failure_done)
        await client._handle_service_call_failure(
            {"op": "serviceCallFailure", "callId": 100, "message": "x"}
        )

        client._handle_parameter_values({"op": "parameterValues", "parameters": []})
        client._handle_parameter_values(
            {"op": "parameterValues", "id": "missing", "parameters": []}
        )
        parameter_done: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        parameter_done.set_result(None)
        client._pending_param_requests["done"] = cast("object", parameter_done)
        client._handle_parameter_values({"op": "parameterValues", "id": "done", "parameters": []})

        client._handle_fetch_asset_response(b"short")
        client._handle_fetch_asset_response(_fetch_asset_frame(99, 0, payload=b"data"))
        asset_done: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        asset_done.set_result(None)
        client._pending_asset_requests[100] = cast("object", asset_done)
        client._handle_fetch_asset_response(_fetch_asset_frame(100, 0))

        client._pending_asset_requests[7] = asyncio.get_running_loop().create_future()
        client._handle_fetch_asset_response(_fetch_asset_frame(7, 1, b"not found"))
        with pytest.raises(Exception, match="not found"):
            client._pending_asset_requests[7].result()

        await client._handle_time_data(struct.pack("<BQ", int(BinaryOpCodes.TIME), 42))
        await client._handle_time_data(b"short")

    asyncio.run(response_edges())
    assert seen_times == [42]


def test_client_unadvertise_and_call_service_disconnect() -> None:
    client = WebSocketBridgeClient("ws://example:8765")

    async def run() -> None:
        await client.unadvertise(42)
        with pytest.raises(RuntimeError, match="Cannot call service"):
            await client.call_service(1, b"", encoding="cdr")

    asyncio.run(run())


def test_server_registration_metadata_and_broadcast_helpers() -> None:
    endpoint = WebSocketBridgeEndpoint(
        name="edge",
        metadata={"source": "test"},
        supported_encodings=["cdr"],
        session_id="session-1",
    )
    channel = Channel(1, "/data", "cdr", "example/Raw", "bytes data", "ros2msg")
    assert channel.as_channel_info()["schemaEncoding"] == "ros2msg"
    endpoint.register_channel(channel)
    endpoint.register_json_handler(JsonOpCodes.STATUS, lambda _state, _message: None)
    endpoint.register_binary_handler(BinaryOpCodes.TIME, lambda _state, _payload: None)
    endpoint.on_connect(lambda _state: None)
    endpoint.on_disconnect(lambda _state: None)
    endpoint.on_subscribe(lambda _state, _sub, _channel: None)
    endpoint.on_unsubscribe(lambda _state, _sub, _channel: None)

    websocket = FakeWebSocket()
    state = ConnectionState(websocket=cast("object", websocket))
    state.subscriptions.update({7: 1, 8: 2})
    endpoint._connections[cast("object", websocket)] = state

    async def run() -> None:
        await endpoint.close_connections()
        await endpoint.advertise_channel(channel)
        await endpoint.advertise_channel(channel, update_registry=False)
        await endpoint.advertise_channels([channel])
        await endpoint.advertise_channels([channel], update_registry=False)
        await endpoint.advertise_channels([], update_registry=False)
        await endpoint.advertise_all()
        await endpoint.unadvertise([1])
        await endpoint.send_status(2, "warning", status_id="status")
        await endpoint.remove_status(["status"])
        await endpoint.clear_session("session-2")

    asyncio.run(run())
    assert websocket.closed
    assert endpoint.get_subscriptions_for_channel(1) == [(cast("object", websocket), 7)]
    assert endpoint.get_connection() == [state]
    assert endpoint.get_connection(cast("object", websocket)) is state
    assert endpoint.dropped_frames == 0
    info = endpoint._server_info()
    assert info["metadata"] == {"source": "test"}
    assert info["supportedEncodings"] == ["cdr"]
    assert endpoint._session_id == "session-2"
    endpoint.unregister_channel(1)
    endpoint.unregister_channel(999)


def test_server_outbox_overflow_and_sender_error_paths() -> None:
    async def run() -> None:
        publish_endpoint = WebSocketBridgeEndpoint()
        publish_socket = FakeWebSocket()
        publish_state = ConnectionState(
            websocket=cast("object", publish_socket),
            subscriptions={7: 1},
            outbox=ConnectionOutbox(hard_limit_bytes=0),
        )
        publish_endpoint._connections[cast("object", publish_socket)] = publish_state
        publish_endpoint.register_channel(Channel(1, "/data", "raw", "example/Raw", ""))
        await publish_endpoint.publish_message(1, b"payload", timestamp_ns=123)
        await publish_endpoint.publish_message(2, b"ignored", timestamp_ns=123)
        assert publish_state.close_task is not None
        await publish_state.close_task

        playback_endpoint = WebSocketBridgeEndpoint(playback_time_range=(0, 10))
        playback_socket = FakeWebSocket()
        playback_state = ConnectionState(
            websocket=cast("object", playback_socket), outbox=ConnectionOutbox(hard_limit_bytes=0)
        )
        playback_endpoint._connections[cast("object", playback_socket)] = playback_state
        playback_endpoint.broadcast_playback_state(
            PlaybackState(PlaybackStatus.PLAYING, 1, 1.0, False)
        )
        assert playback_state.close_task is not None
        await playback_state.close_task

        control_endpoint = WebSocketBridgeEndpoint()
        control_socket = FakeWebSocket()
        control_state = ConnectionState(
            websocket=cast("object", control_socket), outbox=ConnectionOutbox(hard_limit_bytes=0)
        )
        control_endpoint._connections[cast("object", control_socket)] = control_state
        await control_endpoint.send_status(2, "overflow")
        assert control_state.close_task is not None
        await control_state.close_task

        error_socket = AsyncMock()
        error_socket.send.side_effect = OSError("send failed")
        error_state = ConnectionState(websocket=cast("object", error_socket))
        error_state.outbox.offer(1, b"payload", delivery="reliable")
        await control_endpoint._run_sender(error_state)
        error_socket.close.assert_awaited_once()

    asyncio.run(run())


def test_server_publish_helpers_and_playback_decode_edges() -> None:
    async def run() -> None:
        endpoint = WebSocketBridgeEndpoint()
        websocket = AsyncMock()
        await endpoint.send_message_to_subscription(
            cast("object", websocket),
            3,
            b"data",
            timestamp_ns=5,
        )
        websocket.send.side_effect = ConnectionClosed(None, None)
        await endpoint.send_message_to_subscription(
            cast("object", websocket),
            3,
            b"data",
            timestamp_ns=5,
        )

        no_playback = WebSocketBridgeEndpoint()
        state = ConnectionState(websocket=cast("object", FakeWebSocket()))
        await no_playback._handle_playback_control_request(state, b"x")

        no_handler = WebSocketBridgeEndpoint(playback_time_range=(0, 10))
        await no_handler._handle_playback_control_request(state, b"x")

        invalid = WebSocketBridgeEndpoint(playback_time_range=(0, 10))
        invalid.on_playback_control(
            lambda _request: PlaybackState(PlaybackStatus.PLAYING, 0, 1.0, False)
        )
        await invalid._handle_playback_control_request(state, b"\x03")

        rejecting = WebSocketBridgeEndpoint(playback_time_range=(0, 10))
        rejecting.on_playback_control(lambda _request: (_ for _ in ()).throw(ValueError("reject")))
        request_id = b"id"
        request = (
            struct.pack("<BBfBQI", 3, int(PlaybackCommand.PLAY), 1.0, 0, 0, len(request_id))
            + request_id
        )
        await rejecting._handle_playback_control_request(state, request)

    asyncio.run(run())

    with pytest.raises(ValueError, match="too short"):
        WebSocketBridgeEndpoint._decode_playback_control_request(b"\x03")
    truncated = struct.pack("<BBfBQI", 3, int(PlaybackCommand.PLAY), 1.0, 0, 0, 5) + b"id"
    with pytest.raises(ValueError, match="truncated"):
        WebSocketBridgeEndpoint._decode_playback_control_request(truncated)
    invalid_command = struct.pack("<BBfBQI", 3, 99, 1.0, 0, 0, 0)
    with pytest.raises(ValueError, match="invalid playback command"):
        WebSocketBridgeEndpoint._decode_playback_control_request(invalid_command)
    with pytest.raises(ValueError, match="out-of-range"):
        WebSocketBridgeEndpoint._encode_playback_state(
            PlaybackState(
                PlaybackStatus.PLAYING,
                -1,
                1.0,
                False,
            )
        )


def test_server_json_binary_handlers_and_subscription_replacement() -> None:
    endpoint = WebSocketBridgeEndpoint()
    websocket = FakeWebSocket()
    state = ConnectionState(websocket=cast("object", websocket), subscriptions={4: 9})
    seen: list[str] = []
    endpoint.on_subscribe(lambda _state, sub, channel: seen.append(f"subscribe:{sub}:{channel}"))
    endpoint.on_unsubscribe(
        lambda _state, sub, channel: seen.append(f"unsubscribe:{sub}:{channel}")
    )
    endpoint.register_json_handler("custom", lambda _state, _message: seen.append("json"))
    endpoint.register_binary_handler(42, lambda _state, _payload: seen.append("binary"))
    state.outbox.offer(4, b"stale", delivery="reliable")

    async def run() -> None:
        await endpoint._handle_json_frame(state, "not-json")
        await endpoint._handle_json_frame(state, "{}")
        await endpoint._handle_json_frame(state, json.dumps({"op": "custom"}))
        await endpoint._apply_subscriptions(
            state,
            {
                "op": "subscribe",
                "subscriptions": [
                    {"id": 4, "channelId": 10},
                    {"id": 5, "channelId": 11},
                ],
            },
        )
        await endpoint._remove_subscriptions(
            state, {"op": "unsubscribe", "subscriptionIds": [99, 5]}
        )
        await endpoint._handle_json_frame(
            state, json.dumps({"op": "unsubscribe", "subscriptionIds": [99]})
        )
        await endpoint._handle_binary_frame(state, b"")
        await endpoint._handle_binary_frame(state, bytes([42]) + b"payload")

    asyncio.run(run())
    assert seen == [
        "json",
        "unsubscribe:4:9",
        "subscribe:4:10",
        "subscribe:5:11",
        "unsubscribe:5:11",
        "binary",
    ]


def test_server_handle_connection_disconnect_callback_and_server_lifecycle() -> None:
    endpoint = WebSocketBridgeEndpoint()
    websocket = FakeWebSocket([ConnectionClosed(None, None)])
    callbacks: list[str] = []
    endpoint.on_connect(lambda _state: callbacks.append("connect"))
    endpoint.on_disconnect(lambda _state: callbacks.append("disconnect"))

    asyncio.run(endpoint.handle_connection(cast("object", websocket)))
    assert callbacks == ["connect", "disconnect"]
    assert endpoint.connections == []

    async def lifecycle() -> None:
        server = WebSocketBridgeServer(port=0)
        await server.stop()
        await server.start()
        with pytest.raises(RuntimeError, match="already running"):
            await server.start()
        await server.stop()
        await server.stop()

    asyncio.run(lifecycle())
