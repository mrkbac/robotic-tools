"""Tests for shared live bridge topic measurements."""

from __future__ import annotations

import asyncio
import socket
from io import StringIO
from typing import TYPE_CHECKING

import pymcap_cli.cmd.bridge._topic_monitor as monitor_module
import pytest
from pymcap_cli.cmd.bridge._topic_monitor import (
    TopicMetrics,
    TopicMonitor,
    TopicMonitorSnapshot,
    TopicMonitorView,
    _build_hz_table,
    _collect_topic_metrics_async,
    run_topic_monitor,
)
from pymcap_cli.core.message_filter import MessageFilterOptions
from rich.console import Console
from robo_ws_bridge.server import Channel as ServerChannel
from robo_ws_bridge.server import WebSocketBridgeServer

if TYPE_CHECKING:
    from rich.table import Table
    from robo_ws_bridge.ws_types import ChannelInfo


def _channel(channel_id: int, topic: str) -> ChannelInfo:
    return {
        "id": channel_id,
        "topic": topic,
        "encoding": "json",
        "schemaName": "test_msgs/String",
        "schema": '{"type":"object","properties":{"data":{"type":"string"}}}',
        "schemaEncoding": "jsonschema",
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_topic_monitor_calculates_rate_bandwidth_period_and_bridge_age() -> None:
    monitor = TopicMonitor(window_seconds=10.0)
    channel = _channel(1, "/imu")
    monitor.register_channel(channel, now_ns=0)

    for second in range(1, 6):
        monitor.observe(
            channel,
            bridge_timestamp_ns=second * 1_000_000_000 - 20_000_000,
            payload_size=100,
            arrival_ns=second * 1_000_000_000,
            wall_time_ns=second * 1_000_000_000,
        )

    stats = monitor.snapshot(now_ns=5_000_000_000).topics[0]
    assert stats.topic == "/imu"
    assert stats.message_count == 5
    assert stats.total_messages == 5
    assert stats.hz == pytest.approx(1.0)
    assert stats.payload_bytes_per_second == pytest.approx(100.0)
    assert stats.message_size_mean == pytest.approx(100.0)
    assert stats.period_mean_ns == pytest.approx(1_000_000_000)
    assert stats.period_min_ns == 1_000_000_000
    assert stats.period_max_ns == 1_000_000_000
    assert stats.period_stddev_ns == 0.0
    assert stats.bridge_age_mean_ns == pytest.approx(20_000_000)
    assert stats.last_age_ns == 0


def test_topic_monitor_silent_topic_decays_to_zero_and_retains_last_age() -> None:
    monitor = TopicMonitor(window_seconds=10.0)
    channel = _channel(1, "/camera")
    monitor.register_channel(channel, now_ns=0)
    monitor.observe(
        channel,
        bridge_timestamp_ns=5_000_000_000,
        payload_size=1_000,
        arrival_ns=5_000_000_000,
        wall_time_ns=5_000_000_000,
    )

    stats = monitor.snapshot(now_ns=15_000_000_000).topics[0]
    assert stats.message_count == 0
    assert stats.total_messages == 1
    assert stats.hz == 0.0
    assert stats.payload_bytes_per_second == 0.0
    assert stats.message_size_mean is None
    assert stats.last_age_ns == 10_000_000_000


def test_topic_monitor_aggregates_multiple_channels_with_the_same_topic() -> None:
    monitor = TopicMonitor(window_seconds=10.0)
    first = _channel(1, "/shared")
    second = _channel(2, "/shared")
    monitor.register_channel(first, now_ns=0)
    monitor.register_channel(second, now_ns=0)

    monitor.observe(
        first,
        bridge_timestamp_ns=1_000_000_000,
        payload_size=10,
        arrival_ns=1_000_000_000,
        wall_time_ns=1_000_000_000,
    )
    monitor.observe(
        second,
        bridge_timestamp_ns=2_000_000_000,
        payload_size=20,
        arrival_ns=2_000_000_000,
        wall_time_ns=2_000_000_000,
    )

    snapshot = monitor.snapshot(now_ns=2_000_000_000)
    assert len(snapshot.topics) == 1
    assert snapshot.topics[0].topic == "/shared"
    assert snapshot.topics[0].message_count == 2
    assert snapshot.topics[0].message_size_mean == pytest.approx(15.0)


def test_run_topic_monitor_prints_every_text_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_snapshot = TopicMonitorSnapshot(
        sampled_at_ns=1,
        window_seconds=10.0,
        topics=(),
    )
    topic_snapshot = TopicMonitorSnapshot(
        sampled_at_ns=2,
        window_seconds=10.0,
        topics=(
            TopicMetrics(
                topic="/diagnostics",
                message_count=1,
                total_messages=1,
                hz=1.0,
                payload_bytes_per_second=10.0,
                message_size_mean=10.0,
                message_size_min=10,
                message_size_max=10,
                period_mean_ns=None,
                period_min_ns=None,
                period_max_ns=None,
                period_stddev_ns=None,
                bridge_age_mean_ns=0.0,
                bridge_age_min_ns=0,
                bridge_age_max_ns=0,
                last_age_ns=0,
            ),
        ),
    )

    async def collect(_url: str, **kwargs: object) -> TopicMonitorSnapshot:
        on_snapshot = kwargs["on_snapshot"]
        assert callable(on_snapshot)
        on_snapshot(empty_snapshot)
        on_snapshot(topic_snapshot)
        return topic_snapshot

    rendered: list[object] = []
    monkeypatch.setattr(monitor_module, "_collect_topic_metrics_async", collect)
    monkeypatch.setattr(monitor_module.console, "print", rendered.append)

    assert (
        run_topic_monitor(
            "localhost",
            topic=["/imu"],
            all_topics=False,
            exclude_topic=None,
            window=10.0,
            interval=1.0,
            duration=2.0,
            json_output=False,
            connect_timeout=5.0,
            view=TopicMonitorView.HZ,
        )
        == 0
    )
    assert len(rendered) == 2
    output = StringIO()
    output_console = Console(file=output, width=180, color_system=None)
    for renderable in rendered:
        output_console.print(renderable)
    text = output.getvalue()
    assert text.count("Topic frequency") == 1
    assert text.count("Time") == 1
    assert text.index("No matching channels advertised.") < text.index("Time")


def test_hz_snapshots_keep_column_positions_when_values_change() -> None:
    def table(topic: TopicMetrics) -> Table:
        return _build_hz_table(
            TopicMonitorSnapshot(
                sampled_at_ns=1,
                window_seconds=10.0,
                topics=(topic,),
            )
        )

    def rendered_row(topic: TopicMetrics) -> str:
        output = StringIO()
        Console(file=output, width=160, color_system=None).print(table(topic))
        return next(line for line in output.getvalue().splitlines() if "/imu" in line)

    compact = TopicMetrics(
        topic="/imu",
        message_count=2,
        total_messages=2,
        hz=3.0,
        payload_bytes_per_second=10.0,
        message_size_mean=5.0,
        message_size_min=5,
        message_size_max=5,
        period_mean_ns=1_000,
        period_min_ns=900,
        period_max_ns=1_100,
        period_stddev_ns=50,
        bridge_age_mean_ns=0.0,
        bridge_age_min_ns=0,
        bridge_age_max_ns=0,
        last_age_ns=0,
    )
    wide = TopicMetrics(
        topic="/imu",
        message_count=123_456,
        total_messages=123_456,
        hz=2.9,
        payload_bytes_per_second=10.0,
        message_size_mean=5.0,
        message_size_min=5,
        message_size_max=5,
        period_mean_ns=333_333_333,
        period_min_ns=250_000_000,
        period_max_ns=1_250_000_000,
        period_stddev_ns=123_456_789,
        bridge_age_mean_ns=0.0,
        bridge_age_min_ns=0,
        bridge_age_max_ns=0,
        last_age_ns=12_345_000_000,
    )

    assert all(column.width is not None for column in table(compact).columns[1:])
    assert all(column.header.justify == "center" for column in table(compact).columns)
    assert len(rendered_row(compact)) == len(rendered_row(wide))


def test_collect_topic_metrics_subscribes_to_every_matching_topic() -> None:
    port = _free_port()

    async def run() -> tuple[set[int], set[str]]:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port, name="test-bridge")
        server.register_channel(
            ServerChannel(
                id=1,
                topic="/imu/front",
                encoding="json",
                schema_name="test_msgs/String",
                schema='{"type":"object"}',
                schema_encoding="jsonschema",
            )
        )
        server.register_channel(
            ServerChannel(
                id=2,
                topic="/imu/rear",
                encoding="json",
                schema_name="test_msgs/String",
                schema='{"type":"object"}',
                schema_encoding="jsonschema",
            )
        )
        server.register_channel(
            ServerChannel(
                id=3,
                topic="/camera",
                encoding="json",
                schema_name="test_msgs/String",
                schema='{"type":"object"}',
                schema_encoding="jsonschema",
            )
        )
        await server.start()

        subscribed_ids: set[int] = set()
        all_selected = asyncio.Event()

        def on_subscribe(_state: object, _subscription_id: int, channel_id: int) -> None:
            subscribed_ids.add(channel_id)
            if subscribed_ids == {1, 2}:
                all_selected.set()

        server.on_subscribe(on_subscribe)

        async def publish() -> None:
            await all_selected.wait()
            await server.publish_message(1, b"front", timestamp_ns=1)
            await server.publish_message(2, b"rear", timestamp_ns=2)

        publisher = asyncio.create_task(publish())
        snapshots = []
        try:
            report = await _collect_topic_metrics_async(
                f"ws://127.0.0.1:{port}",
                message_filter=MessageFilterOptions.from_args(topic=["/imu/.*"]),
                window_seconds=1.0,
                interval_seconds=0.05,
                duration_seconds=0.15,
                connect_timeout=5.0,
                on_snapshot=snapshots.append,
            )
            return subscribed_ids, {topic.topic for topic in report.topics}
        finally:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
            await server.stop()

    subscribed_ids, topics = asyncio.run(run())
    assert subscribed_ids == {1, 2}
    assert topics == {"/imu/front", "/imu/rear"}
