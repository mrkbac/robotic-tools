"""Tests for Foxglove WebSocket server publishing helpers."""

import asyncio
import json
import logging
import socket
import struct
from contextlib import suppress

import pytest
from robo_ws_bridge import WebSocketBridgeEndpoint, WebSocketBridgeServer
from robo_ws_bridge.server import Channel, ConnectionOutbox
from robo_ws_bridge.ws_types import BinaryOpCodes
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


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


def test_bridge_server_logs_non_websocket_clients_without_traceback(caplog) -> None:
    port = _free_port()

    async def run() -> None:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET / HTTP/1.0\r\nHost: example\r\n\r\n")
            await writer.drain()
            with suppress(ConnectionError):
                await reader.read()
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
        finally:
            await server.stop()

    with caplog.at_level(logging.WARNING, logger="websockets.server"):
        asyncio.run(run())

    records = [record for record in caplog.records if record.name == "websockets.server"]
    assert records, "expected the rejected handshake to be logged"
    assert all(record.levelno < logging.ERROR for record in records)
    assert all(record.exc_info is None for record in records)
    assert any("HTTP/1.0" in record.getMessage() for record in records)


def test_outbox_latest_channel_keeps_only_newest_frame() -> None:
    outbox = ConnectionOutbox()
    outbox.offer(7, b"a" * 100, delivery="latest")
    outbox.offer(7, b"b" * 100, delivery="latest")
    outbox.offer(8, b"c" * 100, delivery="latest")

    assert outbox.pending_bytes == 200
    assert outbox.dropped_frames == 1

    async def drain(count: int) -> list[bytes]:
        return [(await outbox.next_frame()).payload for _ in range(count)]

    frames = asyncio.run(drain(2))
    assert set(frames) == {b"b" * 100, b"c" * 100}
    assert outbox.pending_bytes == 0


def test_outbox_reliable_frames_keep_order_and_hard_cap_drops() -> None:
    outbox = ConnectionOutbox(hard_limit_bytes=250)
    outbox.offer(1, b"first", delivery="reliable")
    outbox.offer(1, b"second", delivery="reliable")
    outbox.offer(1, b"x" * 300, delivery="reliable")
    assert outbox.dropped_frames == 1

    async def drain(count: int) -> list[bytes]:
        return [(await outbox.next_frame()).payload for _ in range(count)]

    assert asyncio.run(drain(2)) == [b"first", b"second"]


def test_outbox_congestion_flag_follows_pending_bytes() -> None:
    outbox = ConnectionOutbox(soft_limit_bytes=150)
    assert not outbox.is_congested
    outbox.offer(1, b"y" * 200, delivery="reliable")
    assert outbox.is_congested

    async def drain() -> bytes:
        return (await outbox.next_frame()).payload

    asyncio.run(drain())
    assert not outbox.is_congested


def test_outbox_discard_removes_pending_subscription_frames() -> None:
    outbox = ConnectionOutbox()
    outbox.offer(7, b"latest", delivery="latest")
    outbox.offer(7, b"reliable", delivery="reliable")
    outbox.offer(8, b"keep", delivery="reliable")

    outbox.discard(7)

    async def drain() -> bytes:
        return (await outbox.next_frame()).payload

    assert asyncio.run(drain()) == b"keep"
    assert outbox.pending_bytes == 0


def test_outbox_clear_drops_pending_without_counting_as_dropped() -> None:
    outbox = ConnectionOutbox()
    outbox.offer(1, b"a" * 50, delivery="reliable")
    outbox.offer(1, b"b" * 50, delivery="reliable")
    outbox.offer(2, b"c" * 50, delivery="latest")
    assert outbox.pending_bytes == 150

    outbox.clear()

    assert outbox.pending_bytes == 0
    assert outbox.dropped_frames == 0

    async def has_no_pending_frame() -> bool:
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(outbox.next_frame(), timeout=0.05)
            return False
        return True

    assert asyncio.run(has_no_pending_frame())


def test_outbox_reports_subscription_busy_until_send_completes() -> None:
    outbox = ConnectionOutbox()
    outbox.offer(7, b"frame", delivery="latest")

    async def take() -> bytes:
        frame = await outbox.next_frame()
        assert outbox.is_key_busy(7)
        outbox.complete(7)
        return frame.payload

    assert asyncio.run(take()) == b"frame"
    assert not outbox.is_key_busy(7)


def test_outbox_reliable_overflow_is_reported_to_caller() -> None:
    outbox = ConnectionOutbox(hard_limit_bytes=10)

    assert outbox.offer(1, b"x" * 11, delivery="reliable") == "overflow"
    assert outbox.pending_bytes == 0


def test_publish_message_never_blocks_on_slow_client() -> None:
    port = _free_port()

    async def run() -> tuple[float, bool]:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port)
        server.register_channel(Channel(1, "/camera", "cdr", "pkg/Img", "data", delivery="latest"))
        subscribed = asyncio.Event()
        server.on_subscribe(lambda *_args: subscribed.set())
        await server.start()
        try:
            async with connect(
                f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"]
            ) as websocket:
                await websocket.recv()  # server info
                await websocket.recv()  # advertise
                await websocket.send(
                    json.dumps({"op": "subscribe", "subscriptions": [{"id": 5, "channelId": 1}]})
                )
                await asyncio.wait_for(subscribed.wait(), timeout=2)
                # Do not read from the socket: publish far more than kernel buffers hold.
                begin = asyncio.get_running_loop().time()
                for _ in range(50):
                    await server.publish_message(1, b"z" * 1_000_000, timestamp_ns=1)
                elapsed = asyncio.get_running_loop().time() - begin
                state = server.connections[0]
                return elapsed, state.outbox.pending_bytes <= 2_000_000
        finally:
            await server.stop()

    elapsed, bounded = asyncio.run(run())
    assert elapsed < 1.0, f"publish blocked for {elapsed:.2f}s"
    assert bounded, "latest-wins outbox should hold at most one pending frame per channel"


def test_publish_message_closes_client_on_reliable_overflow() -> None:
    port = _free_port()

    async def run() -> int:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port)
        server.register_channel(Channel(1, "/video", "cdr", "pkg/Video", "data"))
        subscribed = asyncio.Event()
        server.on_subscribe(lambda *_args: subscribed.set())
        await server.start()
        try:
            async with connect(
                f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"]
            ) as websocket:
                await websocket.recv()
                await websocket.recv()
                await websocket.send(
                    json.dumps({"op": "subscribe", "subscriptions": [{"id": 5, "channelId": 1}]})
                )
                await asyncio.wait_for(subscribed.wait(), timeout=2)
                await server.publish_message(1, b"x" * 2_000_000, timestamp_ns=1)
                with pytest.raises(ConnectionClosed) as closed:
                    await websocket.recv()
                assert closed.value.rcvd is not None
                return closed.value.rcvd.code
        finally:
            await server.stop()

    assert asyncio.run(run()) == 1013


def test_channel_is_busy_only_when_every_subscriber_is_busy() -> None:
    port = _free_port()

    async def run() -> tuple[bool, bool]:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port)
        server.register_channel(Channel(1, "/camera", "cdr", "pkg/Img", "data", delivery="latest"))
        both_subscribed = asyncio.Event()

        def on_subscribe(*_args: object) -> None:
            if len(server.get_subscriptions_for_channel(1)) == 2:
                both_subscribed.set()

        server.on_subscribe(on_subscribe)
        await server.start()
        first = await connect(f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"])
        second = await connect(f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"])
        try:
            for websocket in (first, second):
                await websocket.recv()
                await websocket.recv()
                await websocket.send(
                    json.dumps({"op": "subscribe", "subscriptions": [{"id": 1, "channelId": 1}]})
                )
            await asyncio.wait_for(both_subscribed.wait(), timeout=2)
            states = server.connections
            states[0].outbox.offer(1, b"first", delivery="latest")
            one_busy = server.are_all_subscribers_busy(1)
            states[1].outbox.offer(1, b"second", delivery="latest")
            all_busy = server.are_all_subscribers_busy(1)
            for state in states:
                state.outbox.discard(1)
            return one_busy, all_busy
        finally:
            await first.close()
            await second.close()
            await server.stop()

    assert asyncio.run(run()) == (False, True)


def test_publish_message_survives_disconnect_while_publishing() -> None:
    port = _free_port()

    async def run() -> int:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port)
        server.register_channel(Channel(1, "/data", "cdr", "pkg/Msg", "data"))
        both_subscribed = asyncio.Event()

        def _on_subscribe(*_args: object) -> None:
            if len(server.get_subscriptions_for_channel(1)) >= 2:
                both_subscribed.set()

        server.on_subscribe(_on_subscribe)
        await server.start()
        try:
            first = await connect(f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"])
            async with connect(
                f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"]
            ) as second:
                for websocket in (first, second):
                    await websocket.recv()
                    await websocket.recv()
                    await websocket.send(
                        json.dumps(
                            {"op": "subscribe", "subscriptions": [{"id": 1, "channelId": 1}]}
                        )
                    )
                await asyncio.wait_for(both_subscribed.wait(), timeout=2)
                received = 0

                async def read_second() -> None:
                    nonlocal received
                    with suppress(Exception):
                        while True:
                            frame = await second.recv()
                            if isinstance(frame, bytes) and frame[0] == int(
                                BinaryOpCodes.MESSAGE_DATA
                            ):
                                received += 1

                reader = asyncio.create_task(read_second())
                for index in range(200):
                    if index == 20:
                        await first.close()
                    await server.publish_message(1, b"payload", timestamp_ns=index)
                    await asyncio.sleep(0)
                for _ in range(200):
                    if received >= 150:
                        break
                    await asyncio.sleep(0.01)
                reader.cancel()
                return received
        finally:
            await server.stop()

    received = asyncio.run(run())
    assert received >= 150


def test_clear_pending_frames_empties_slow_client_outbox() -> None:
    port = _free_port()

    async def run() -> tuple[int, int]:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port)
        server.register_channel(Channel(1, "/camera", "cdr", "pkg/Img", "data", delivery="latest"))
        subscribed = asyncio.Event()
        server.on_subscribe(lambda *_args: subscribed.set())
        await server.start()
        try:
            async with connect(
                f"ws://127.0.0.1:{port}", subprotocols=["foxglove.websocket.v1"]
            ) as websocket:
                await websocket.recv()  # server info
                await websocket.recv()  # advertise
                await websocket.send(
                    json.dumps({"op": "subscribe", "subscriptions": [{"id": 5, "channelId": 1}]})
                )
                await asyncio.wait_for(subscribed.wait(), timeout=2)
                # Do not read: the sender blocks on the first large frame and the
                # newest frame stays queued in the outbox.
                for _ in range(10):
                    await server.publish_message(1, b"z" * 1_000_000, timestamp_ns=1)
                before = server.connections[0].outbox.pending_bytes
                server.clear_pending_frames()
                after = server.connections[0].outbox.pending_bytes
                return before, after
        finally:
            await server.stop()

    before, after = asyncio.run(run())
    assert before > 0
    assert after == 0
