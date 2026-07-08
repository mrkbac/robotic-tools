"""Tests for `pymcap-cli bridge delay`."""

from __future__ import annotations

import asyncio
import re
import socket
import struct
from dataclasses import dataclass

import pytest
from pymcap_cli.cmd.bridge._shared import BridgeFetchError
from pymcap_cli.cmd.bridge.delay import (
    DelayReference,
    DelayReport,
    _collect_delay_async,
    _delay_to_dict,
)
from robo_ws_bridge.server import Channel as ServerChannel
from robo_ws_bridge.server import WebSocketBridgeServer
from robo_ws_bridge.ws_types import BinaryOpCodes, ServerCapabilities

_JSON_STRING_SCHEMA = '{"type":"object","properties":{"data":{"type":"string"}}}'
_JSON_STAMPED_SCHEMA = (
    '{"type":"object","properties":'
    '{"header":{"type":"object","properties":'
    '{"stamp":{"type":"object","properties":'
    '{"sec":{"type":"integer"},"nanosec":{"type":"integer"}}}}}}}'
)


@dataclass
class FakeClock:
    values: list[int]

    def __call__(self) -> int:
        return self.values.pop(0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _json_channel(channel_id: int, topic: str, schema: str = _JSON_STRING_SCHEMA) -> ServerChannel:
    return ServerChannel(
        id=channel_id,
        topic=topic,
        encoding="json",
        schema_name="test_msgs/Stamped",
        schema=schema,
        schema_encoding="jsonschema",
    )


def test_collect_delay_async_without_topics_uses_time_updates_without_subscribing() -> None:
    port = _free_port()
    clock = FakeClock([1_125_000_000])

    async def run() -> DelayReport:
        server = WebSocketBridgeServer(
            host="127.0.0.1",
            port=port,
            name="test-bridge",
            capabilities=[ServerCapabilities.TIME.value],
        )
        server.register_channel(_json_channel(1, "/clock"))
        await server.start()

        connected = asyncio.Event()
        subscribed_channel_ids: list[int] = []
        server.on_connect(lambda *_args: connected.set())
        server.on_subscribe(
            lambda _state, _subscription_id, channel_id: subscribed_channel_ids.append(channel_id)
        )

        async def send_time_once() -> None:
            await connected.wait()
            frame = bytearray(9)
            frame[0] = int(BinaryOpCodes.TIME)
            struct.pack_into("<Q", frame, 1, 1_000_000_000)
            for state in server.connections:
                await state.websocket.send(bytes(frame))

        time_sender = asyncio.create_task(send_time_once())
        try:
            report = await _collect_delay_async(
                f"ws://127.0.0.1:{port}",
                topic_patterns=(),
                against=DelayReference.LOCAL,
                duration=0.2,
                connect_timeout=5.0,
                now_ns=clock,
            )
            assert subscribed_channel_ids == []
            return report
        finally:
            time_sender.cancel()
            await asyncio.gather(time_sender, return_exceptions=True)
            await server.stop()

    report = asyncio.run(run())
    assert report.total_messages == 0
    assert report.wants_header_age is False
    assert report.channels == ()
    assert report.time_offset.latest_ns == 125_000_000
    assert report.time_offset.count == 1


def test_collect_delay_async_without_topics_requires_time_capability() -> None:
    port = _free_port()

    async def run() -> None:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port, name="test-bridge")
        server.register_channel(_json_channel(1, "/clock"))
        await server.start()

        subscribed_channel_ids: list[int] = []
        server.on_subscribe(
            lambda _state, _subscription_id, channel_id: subscribed_channel_ids.append(channel_id)
        )

        try:
            with pytest.raises(BridgeFetchError, match="does not advertise"):
                await _collect_delay_async(
                    f"ws://127.0.0.1:{port}",
                    topic_patterns=(),
                    against=DelayReference.LOCAL,
                    duration=0.2,
                    connect_timeout=5.0,
                    now_ns=FakeClock([]),
                )
            assert subscribed_channel_ids == []
        finally:
            await server.stop()

    asyncio.run(run())


def test_collect_delay_async_with_topics_measures_header_age_against_bridge_time() -> None:
    port = _free_port()
    clock = FakeClock([1_500_000_000])

    async def run() -> DelayReport:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port, name="test-bridge")
        server.register_channel(_json_channel(1, "/stamped", _JSON_STAMPED_SCHEMA))
        await server.start()

        subscribed = asyncio.Event()
        server.on_subscribe(lambda *_args: subscribed.set())

        async def publish_once() -> None:
            await subscribed.wait()
            await server.publish_message(
                1,
                b'{"header":{"stamp":{"sec":1,"nanosec":100000000}}}',
                timestamp_ns=1_300_000_000,
            )

        publisher = asyncio.create_task(publish_once())
        try:
            return await _collect_delay_async(
                f"ws://127.0.0.1:{port}",
                topic_patterns=(re.compile("^/stamped$"),),
                against=DelayReference.BRIDGE,
                duration=0.2,
                connect_timeout=5.0,
                now_ns=clock,
            )
        finally:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
            await server.stop()

    report = asyncio.run(run())
    assert report.total_messages == 1
    assert report.wants_header_age is True
    stats = report.channels[0]
    assert stats.clock_offset.latest_ns == 200_000_000
    assert stats.header_age.latest_ns == 200_000_000
    assert stats.missing_header_stamp == 0

    payload = _delay_to_dict(report)
    assert payload["mode"] == "header_age"
    channels = payload["channels"]
    assert isinstance(channels, list)
    first_channel = channels[0]
    assert isinstance(first_channel, dict)
    header_age = first_channel["header_age"]
    assert isinstance(header_age, dict)
    assert header_age["latest_ms"] == 200.0
