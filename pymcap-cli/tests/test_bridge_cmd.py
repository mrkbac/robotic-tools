"""Unit tests for the `pymcap-cli bridge` command's pure logic."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import socket
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.bridge import record as record_module
from pymcap_cli.cmd.bridge import tf as tf_module
from pymcap_cli.cmd.bridge._shared import (
    BridgeInfo,
    BridgeStatus,
    SortChoice,
    _append_status,
    _remove_statuses,
    _sort_channels,
    bridge_to_dict,
    to_ws_url,
)
from pymcap_cli.cmd.bridge.cat import _cat_async, cat
from pymcap_cli.cmd.bridge.info import (
    _build_channels_table,
    _build_connection_graph_summary,
    _build_connection_graph_tree,
    _topic_connection_counts,
)
from pymcap_cli.cmd.bridge.record import (
    BridgeRecorder,
    TopicSelector,
    _build_record_status,
    _record_async,
)
from pymcap_cli.core.message_filter import MessageFilterOptions
from pymcap_cli.core.tf_tree import TfGraph
from pymcap_cli.display.message_render import BytesMode
from rich.console import Console, RenderableType
from robo_ws_bridge import ConnectionGraph
from robo_ws_bridge.server import Channel as ServerChannel
from robo_ws_bridge.server import ConnectionState, WebSocketBridgeServer
from small_mcap import JSONDecoderFactory, McapWriter, read_message_decoded

if TYPE_CHECKING:
    from pathlib import Path

    from robo_ws_bridge.ws_types import ChannelInfo


def _render(renderable: RenderableType) -> str:
    console = Console(record=True, width=120, color_system=None)
    console.print(renderable)
    return console.export_text()


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("ws://localhost:8765", "ws://localhost:8765"),
        ("wss://example.com", "wss://example.com"),
        ("1.2.3.4", "ws://1.2.3.4:8765"),
        ("1.2.3.4:9000", "ws://1.2.3.4:9000"),
        ("127.0.0.1", "ws://127.0.0.1:8765"),
        ("localhost", "ws://localhost:8765"),
        ("myhost.example.com", "ws://myhost.example.com:8765"),
        ("myhost.example.com:9000", "ws://myhost.example.com:9000"),
        ("::1", "ws://[::1]:8765"),
        ("[::1]", "ws://[::1]:8765"),
        ("[::1]:9000", "ws://[::1]:9000"),
    ],
)
def test_to_ws_url(arg: str, expected: str) -> None:
    assert to_ws_url(arg) == expected


def test_to_ws_url_custom_default_port() -> None:
    assert to_ws_url("10.0.0.1", default_port=12345) == "ws://10.0.0.1:12345"


def _make_channel(
    *, channel_id: int, topic: str, schema_name: str = "std_msgs/String"
) -> ChannelInfo:
    return {
        "id": channel_id,
        "topic": topic,
        "encoding": "cdr",
        "schemaName": schema_name,
        "schema": "",
        "schemaEncoding": "ros2msg",
    }


def test_sort_channels_by_topic() -> None:
    channels = [
        _make_channel(channel_id=2, topic="/b"),
        _make_channel(channel_id=1, topic="/a"),
        _make_channel(channel_id=3, topic="/c"),
    ]
    sorted_topics = [c["topic"] for c in _sort_channels(channels, SortChoice.TOPIC, reverse=False)]
    assert sorted_topics == ["/a", "/b", "/c"]


def test_sort_channels_by_id_reverse() -> None:
    channels = [
        _make_channel(channel_id=1, topic="/a"),
        _make_channel(channel_id=10, topic="/b"),
        _make_channel(channel_id=2, topic="/c"),
    ]
    sorted_ids = [c["id"] for c in _sort_channels(channels, SortChoice.ID, reverse=True)]
    assert sorted_ids == [10, 2, 1]


def test_sort_channels_by_schema() -> None:
    channels = [
        _make_channel(channel_id=1, topic="/a", schema_name="zoo/Z"),
        _make_channel(channel_id=2, topic="/b", schema_name="alpha/A"),
    ]
    sorted_schemas = [
        c["schemaName"] for c in _sort_channels(channels, SortChoice.SCHEMA, reverse=False)
    ]
    assert sorted_schemas == ["alpha/A", "zoo/Z"]


def test_bridge_to_dict_shape_without_graph() -> None:
    info = BridgeInfo(
        url="ws://example:8765",
        server_info={
            "op": "serverInfo",
            "name": "srv",
            "capabilities": ["clientPublish"],
            "supportedEncodings": ["cdr"],
            "metadata": {"k": "v"},
            "sessionId": "s1",
        },
        channels=(
            _make_channel(channel_id=1, topic="/a"),
            _make_channel(channel_id=2, topic="/b", schema_name="other/Msg"),
        ),
        statuses=(
            BridgeStatus(level=1, message="warming up", status_id="boot"),
            BridgeStatus(level=2, message="sensor offline"),
        ),
    )
    payload = bridge_to_dict(info)
    assert "connectionGraph" not in payload
    assert payload["services"] == []
    assert payload["statuses"] == [
        {"op": "status", "level": 1, "message": "warming up", "id": "boot"},
        {"op": "status", "level": 2, "message": "sensor offline"},
    ]


def test_bridge_to_dict_includes_connection_graph() -> None:
    info = BridgeInfo(
        url="ws://example:8765",
        server_info={
            "op": "serverInfo",
            "name": "srv",
            "capabilities": ["connectionGraph"],
        },
        channels=(),
        connection_graph=ConnectionGraph(
            published_topics=({"name": "/foo", "publisherIds": ["pub-1"]},),
            subscribed_topics=({"name": "/foo", "subscriberIds": ["sub-1", "sub-2"]},),
            advertised_services=({"name": "/svc", "providerIds": ["srv-1"]},),
        ),
    )
    payload = bridge_to_dict(info)
    assert payload["connectionGraph"] == {
        "publishedTopics": [{"name": "/foo", "publisherIds": ["pub-1"]}],
        "subscribedTopics": [{"name": "/foo", "subscriberIds": ["sub-1", "sub-2"]}],
        "advertisedServices": [{"name": "/svc", "providerIds": ["srv-1"]}],
    }


def test_connection_graph_tree_returns_none_when_empty() -> None:
    graph = ConnectionGraph(
        published_topics=(),
        subscribed_topics=(),
        advertised_services=(),
    )
    assert _build_connection_graph_tree(graph) is None


def test_connection_graph_tree_skips_empty_sections_for_sub_only_node() -> None:
    graph = ConnectionGraph(
        published_topics=(),
        subscribed_topics=({"name": "/chatter", "subscriberIds": ["/listener"]},),
        advertised_services=(),
    )
    tree = _build_connection_graph_tree(graph)
    assert tree is not None
    output = _render(tree)
    assert "/listener" in output
    assert "subscribes" in output
    assert "publishes" not in output
    assert "provides" not in output


def test_connection_graph_tree_includes_provides_only_node() -> None:
    graph = ConnectionGraph(
        published_topics=(),
        subscribed_topics=(),
        advertised_services=({"name": "/reset", "providerIds": ["/svc_node"]},),
    )
    tree = _build_connection_graph_tree(graph)
    assert tree is not None
    output = _render(tree)
    assert "/svc_node" in output
    assert "provides" in output
    assert "/reset" in output


def test_connection_graph_tree_shows_schema_for_pub_sub_topics() -> None:
    graph = ConnectionGraph(
        published_topics=({"name": "/chatter", "publisherIds": ["/talker"]},),
        subscribed_topics=({"name": "/chatter", "subscriberIds": ["/listener"]},),
        advertised_services=({"name": "/reset", "providerIds": ["/talker"]},),
    )
    channels = [_make_channel(channel_id=1, topic="/chatter", schema_name="std_msgs/String")]
    tree = _build_connection_graph_tree(graph, channels)
    assert tree is not None
    output = _render(tree)
    assert "std_msgs/String" in output
    reset_line = next(line for line in output.splitlines() if "/reset" in line)
    assert "std_msgs/String" not in reset_line


def test_connection_graph_tree_shows_service_type_for_provided_services() -> None:
    graph = ConnectionGraph(
        published_topics=(),
        subscribed_topics=(),
        advertised_services=({"name": "/reset", "providerIds": ["/svc_node"]},),
    )
    services = [{"id": 1, "name": "/reset", "type": "std_srvs/Empty"}]
    tree = _build_connection_graph_tree(graph, services=services)
    assert tree is not None
    output = _render(tree)
    reset_line = next(line for line in output.splitlines() if "/reset" in line)
    assert "std_srvs/Empty" in reset_line


def test_connection_graph_tree_sorts_nodes_and_topics() -> None:
    graph = ConnectionGraph(
        published_topics=(
            {"name": "/zeta", "publisherIds": ["/talker"]},
            {"name": "/alpha", "publisherIds": ["/talker"]},
        ),
        subscribed_topics=({"name": "/zeta", "subscriberIds": ["/listener"]},),
        advertised_services=(),
    )
    tree = _build_connection_graph_tree(graph)
    assert tree is not None
    output = _render(tree)

    listener_pos = output.index("/listener")
    talker_pos = output.index("/talker")
    assert listener_pos < talker_pos

    talker_section = output[talker_pos:]
    alpha_pos = talker_section.index("/alpha")
    zeta_pos = talker_section.index("/zeta")
    assert alpha_pos < zeta_pos


def test_connection_graph_summary_returns_none_when_empty() -> None:
    graph = ConnectionGraph(
        published_topics=(),
        subscribed_topics=(),
        advertised_services=(),
    )
    assert _build_connection_graph_summary(graph) is None


def test_connection_graph_summary_counts_per_node_without_listing_topics() -> None:
    graph = ConnectionGraph(
        published_topics=(
            {"name": "/zeta", "publisherIds": ["/talker"]},
            {"name": "/alpha", "publisherIds": ["/talker"]},
        ),
        subscribed_topics=({"name": "/zeta", "subscriberIds": ["/listener"]},),
        advertised_services=({"name": "/reset", "providerIds": ["/talker"]},),
    )
    summary = _build_connection_graph_summary(graph)
    assert summary is not None
    output = _render(summary)
    assert "2 nodes" in output
    assert "/talker" in output
    assert "/listener" in output
    # The compact summary reports counts only — it must not enumerate topic names.
    assert "/zeta" not in output
    assert "/alpha" not in output
    assert "/reset" not in output


def test_channels_table_is_compact_without_publisher_columns() -> None:
    channels = [
        _make_channel(channel_id=1, topic="/chatter", schema_name="std_msgs/String"),
    ]
    output = _render(_build_channels_table(channels))
    assert "/chatter" in output
    assert "std_msgs/String" in output
    assert "Publishers" not in output
    assert "Subscribers" not in output
    assert "Pub" not in output
    assert "Sub" not in output


def test_topic_connection_counts_maps_pub_and_sub_totals() -> None:
    graph = ConnectionGraph(
        published_topics=({"name": "/chatter", "publisherIds": ["/talker", "/talker2"]},),
        subscribed_topics=({"name": "/chatter", "subscriberIds": ["/listener"]},),
        advertised_services=(),
    )
    assert _topic_connection_counts(graph) == {"/chatter": (2, 1)}


def test_channels_table_shows_pub_sub_counts_when_graph_available() -> None:
    channels = [
        _make_channel(channel_id=1, topic="/chatter", schema_name="std_msgs/String"),
        _make_channel(channel_id=2, topic="/quiet", schema_name="std_msgs/String"),
    ]
    topic_counts = {"/chatter": (2, 1)}
    output = _render(_build_channels_table(channels, topic_counts))
    assert "Pub" in output
    assert "Sub" in output
    # A topic absent from the graph falls back to zero counts rather than being dropped.
    assert "/quiet" in output


def test_status_cache_replaces_and_removes_id_statuses() -> None:
    statuses: list[BridgeStatus] = []

    _append_status(statuses, 1, "warming up", "boot")
    _append_status(statuses, 2, "offline", "boot")
    _append_status(statuses, 0, "anonymous", None)
    _remove_statuses(statuses, ["boot"])

    assert statuses == [BridgeStatus(level=0, message="anonymous")]


def _make_recorder(
    *, selector: TopicSelector | None = None, message_limit: int | None = None
) -> tuple[BridgeRecorder, McapWriter, io.BytesIO]:
    buf = io.BytesIO()
    writer = McapWriter(buf, use_chunking=False)
    writer.start(profile="", library="test")
    recorder = BridgeRecorder(
        writer=writer,
        selector=selector if selector is not None else TopicSelector(),
        message_limit=message_limit,
    )
    return recorder, writer, buf


def test_topic_selector_all_topics_overrides_other_includes() -> None:
    selector = TopicSelector()
    assert selector.matches("/anything") is True


def test_topic_selector_exact_topics_match_strictly() -> None:
    selector = TopicSelector(MessageFilterOptions.from_args(topic=["/chatter"]))
    assert selector.matches("/chatter") is True
    assert selector.matches("/chatter/sub") is False


def test_topic_selector_include_patterns_use_fullmatch() -> None:
    selector = TopicSelector(
        MessageFilterOptions.from_args(topic=["/camera/.*", ".*imu.*"]),
    )
    assert selector.matches("/camera/front") is True
    assert selector.matches("/sensor/imu/data") is True
    assert selector.matches("/lidar/points") is False


def test_topic_selector_exclude_wins_over_include() -> None:
    selector = TopicSelector(
        MessageFilterOptions.from_args(exclude_topic=["/debug/.*"]),
    )
    assert selector.matches("/debug/log") is False
    assert selector.matches("/sensor/data") is True


def test_topic_selector_without_any_match_rule_rejects_everything() -> None:
    selector = TopicSelector(MessageFilterOptions.from_args(topic=["a^"]))
    assert selector.matches("/anything") is False


def test_topic_selector_exclude_topics_strict_match() -> None:
    selector = TopicSelector(
        MessageFilterOptions.from_args(exclude_topic=["/debug"]),
    )
    assert selector.matches("/debug") is False
    assert selector.matches("/debug/sub") is True


def test_topic_selector_combines_includes_and_excludes() -> None:
    selector = TopicSelector(
        MessageFilterOptions.from_args(
            topic=["/chatter", "/cam/.*"],
            exclude_topic=[".*debug.*"],
        ),
    )
    assert selector.matches("/chatter") is True
    assert selector.matches("/cam/front") is True
    assert selector.matches("/cam/debug") is False
    assert selector.matches("/imu/data") is False


def test_record_all_topics_skips_invalid_include_regex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_selectors: list[TopicSelector] = []

    async def fake_record_async(**kwargs) -> int:
        selector = kwargs["selector"]
        assert isinstance(selector, TopicSelector)
        seen_selectors.append(selector)
        return 0

    monkeypatch.setattr(record_module, "_record_async", fake_record_async)

    rc = record_module.record(
        target="localhost",
        output=tmp_path / "capture.mcap",
        all_topics=True,
        topic=["["],
    )

    assert rc == 0
    assert seen_selectors == [TopicSelector()]


def test_record_all_topics_still_validates_exclude_regex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_record_async(**_kwargs) -> int:
        raise AssertionError("recording should not start with an invalid exclude regex")

    monkeypatch.setattr(record_module, "_record_async", fake_record_async)

    rc = record_module.record(
        target="localhost",
        output=tmp_path / "capture.mcap",
        all_topics=True,
        exclude_topic=["["],
    )

    assert rc == 1


def test_recorder_dedups_schemas_and_channels() -> None:
    recorder, writer, _ = _make_recorder()
    channel: ChannelInfo = {
        "id": 1,
        "topic": "/foo",
        "encoding": "json",
        "schemaName": "Pkg/Msg",
        "schema": '{"type":"object"}',
        "schemaEncoding": "jsonschema",
    }

    sid_a = recorder.schema_id_for(channel)
    sid_b = recorder.schema_id_for(dict(channel) | {"id": 99})
    assert sid_a == sid_b == 1

    other_schema = dict(channel) | {"schemaName": "Pkg/Other", "id": 2}
    sid_c = recorder.schema_id_for(other_schema)
    assert sid_c == 2

    cid_a = recorder.channel_id_for(channel)
    cid_b = recorder.channel_id_for(dict(channel) | {"id": 7})
    assert cid_a == cid_b == 1
    writer.finish()


def test_recorder_returns_zero_schema_id_when_schemaless() -> None:
    recorder, writer, _ = _make_recorder()
    schemaless: ChannelInfo = {
        "id": 1,
        "topic": "/raw",
        "encoding": "raw",
        "schemaName": "",
        "schema": "",
    }
    assert recorder.schema_id_for(schemaless) == 0
    writer.finish()


def test_recorder_on_message_writes_records_and_respects_limit(tmp_path: Path) -> None:
    recorder, writer, buf = _make_recorder(message_limit=2)
    channel: ChannelInfo = {
        "id": 1,
        "topic": "/foo",
        "encoding": "json",
        "schemaName": "Pkg/Msg",
        "schema": '{"type":"object","properties":{"v":{"type":"integer"}}}',
        "schemaEncoding": "jsonschema",
    }
    recorder.on_message(channel, 1_000_000_000, b'{"v":1}')
    recorder.on_message(channel, 2_000_000_000, b'{"v":2}')
    recorder.on_message(channel, 3_000_000_000, b'{"v":3}')  # over limit, ignored
    recorder.on_message(
        {**channel, "topic": "/skipped"},
        4_000_000_000,
        b'{"v":4}',
    )  # over limit
    writer.finish()

    assert recorder.total_messages == 2
    assert recorder.message_counts == {"/foo": 2}
    assert recorder.payload_bytes == len(b'{"v":1}') + len(b'{"v":2}')

    out = tmp_path / "rec.mcap"
    out.write_bytes(buf.getvalue())
    with out.open("rb") as f:
        decoded = list(read_message_decoded(f, decoder_factories=[JSONDecoderFactory()]))
    payloads = [d.decoded_message for d in decoded]
    assert payloads == [{"v": 1}, {"v": 2}]
    assert {d.message.log_time for d in decoded} == {1_000_000_000, 2_000_000_000}


def test_recorder_skips_messages_for_topics_not_matching_filter() -> None:
    recorder, writer, _ = _make_recorder(
        selector=TopicSelector(
            MessageFilterOptions.from_args(topic=["/keep.*"]),
        )
    )
    channel: ChannelInfo = {
        "id": 1,
        "topic": "/skip",
        "encoding": "json",
        "schemaName": "",
        "schema": "",
    }
    recorder.on_message(channel, 1, b"{}")
    writer.finish()
    assert recorder.total_messages == 0
    assert recorder.message_counts == {}


def test_record_status_waiting_message_renders_without_markup(tmp_path: Path) -> None:
    recorder, writer, _ = _make_recorder()
    try:
        rendered = _render(
            _build_record_status(
                url="ws://example:8765",
                output=tmp_path / "capture.mcap",
                recorder=recorder,
                elapsed=1.5,
                duration=None,
                message_limit=None,
            )
        )
    finally:
        writer.finish()

    assert "Payload:" in rendered
    assert "Written:" not in rendered
    assert "Waiting for messages..." in rendered
    assert "[dim]" not in rendered


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_record_async_captures_messages_into_mcap(tmp_path: Path) -> None:
    port = _free_port()
    output = tmp_path / "capture.mcap"

    async def run() -> int:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port, name="test-bridge")
        server.register_channel(
            ServerChannel(
                id=1,
                topic="/chatter",
                encoding="json",
                schema_name="std_msgs/String",
                schema='{"type":"object","properties":{"data":{"type":"string"}}}',
                schema_encoding="jsonschema",
            )
        )
        await server.start()

        subscribed = asyncio.Event()
        server.on_subscribe(lambda *_args: subscribed.set())

        async def publish_until_done() -> None:
            await subscribed.wait()
            for i in range(3):
                await server.publish_message(
                    1,
                    f'{{"data":"hello-{i}"}}'.encode(),
                    timestamp_ns=1_000_000_000 + i,
                )
                await asyncio.sleep(0.02)

        publisher = asyncio.create_task(publish_until_done())
        try:
            return await _record_async(
                url=f"ws://127.0.0.1:{port}",
                output=output,
                selector=TopicSelector(MessageFilterOptions.from_args(topic=["/chatter"])),
                duration=None,
                message_limit=3,
                chunk_size=1024,
                compression_choice="none",
                connect_timeout=5.0,
                refresh_interval=0.05,
                show_status=False,
            )
        finally:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
            await server.stop()

    rc = asyncio.run(run())
    assert rc == 0
    assert output.exists()

    with output.open("rb") as f:
        decoded = list(read_message_decoded(f, decoder_factories=[JSONDecoderFactory()]))
    assert [d.decoded_message for d in decoded] == [
        {"data": "hello-0"},
        {"data": "hello-1"},
        {"data": "hello-2"},
    ]
    assert all(d.channel.topic == "/chatter" for d in decoded)
    assert decoded[0].schema is not None
    assert decoded[0].schema.name == "std_msgs/String"


def test_record_async_captures_duplicate_topic_channels(tmp_path: Path) -> None:
    port = _free_port()
    output = tmp_path / "capture.mcap"

    async def run() -> int:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port, name="test-bridge")
        server.register_channel(_json_string_channel(1, "/duplicated"))
        server.register_channel(_json_string_channel(2, "/duplicated"))
        await server.start()

        subscribed = asyncio.Event()
        subscribed_channels: set[int] = set()

        def _on_subscribe(_state: ConnectionState, _subscription_id: int, channel_id: int) -> None:
            subscribed_channels.add(channel_id)
            if {1, 2} <= subscribed_channels:
                subscribed.set()

        server.on_subscribe(_on_subscribe)

        async def publish_once_subscribed() -> None:
            await subscribed.wait()
            await server.publish_message(1, b'{"data":"first-channel"}', timestamp_ns=1)
            await server.publish_message(2, b'{"data":"second-channel"}', timestamp_ns=2)

        publisher = asyncio.create_task(publish_once_subscribed())
        try:
            return await _record_async(
                url=f"ws://127.0.0.1:{port}",
                output=output,
                selector=TopicSelector(MessageFilterOptions.from_args(topic=["/duplicated"])),
                duration=0.5,
                message_limit=2,
                chunk_size=1024,
                compression_choice="none",
                connect_timeout=5.0,
                refresh_interval=0.05,
                show_status=False,
            )
        finally:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
            await server.stop()

    rc = asyncio.run(run())
    assert rc == 0

    with output.open("rb") as f:
        decoded = list(read_message_decoded(f, decoder_factories=[JSONDecoderFactory()]))
    assert [d.decoded_message["data"] for d in decoded] == [
        "first-channel",
        "second-channel",
    ]
    assert all(d.channel.topic == "/duplicated" for d in decoded)


_JSON_STRING_SCHEMA = '{"type":"object","properties":{"data":{"type":"string"}}}'


def _json_string_channel(channel_id: int, topic: str) -> ServerChannel:
    return ServerChannel(
        id=channel_id,
        topic=topic,
        encoding="json",
        schema_name="std_msgs/String",
        schema=_JSON_STRING_SCHEMA,
        schema_encoding="jsonschema",
    )


def _run_cat_against_server(
    *,
    register: list[ServerChannel],
    publish: list[tuple[int, bytes, int]],
    cat_kwargs: dict,
    wait_for_channels: set[int] | None = None,
) -> int:
    port = _free_port()

    async def run() -> int:
        server = WebSocketBridgeServer(host="127.0.0.1", port=port, name="test-bridge")
        for channel in register:
            server.register_channel(channel)
        await server.start()

        subscribed = asyncio.Event()
        subscribed_channels: set[int] = set()

        def _on_subscribe(_state: ConnectionState, _subscription_id: int, channel_id: int) -> None:
            if wait_for_channels is None:
                subscribed.set()
                return
            subscribed_channels.add(channel_id)
            if wait_for_channels <= subscribed_channels:
                subscribed.set()

        server.on_subscribe(_on_subscribe)

        async def publisher_task() -> None:
            if wait_for_channels is None:
                await subscribed.wait()
            else:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(subscribed.wait(), timeout=0.2)
            for channel_id, payload, ts in publish:
                await server.publish_message(channel_id, payload, timestamp_ns=ts)
                await asyncio.sleep(0.01)

        publisher = asyncio.create_task(publisher_task())
        try:
            return await _cat_async(url=f"ws://127.0.0.1:{port}", **cat_kwargs)
        finally:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
            await server.stop()

    return asyncio.run(run())


def _default_cat_kwargs(**overrides: object) -> dict:
    base: dict = {
        "topic": None,
        "exclude_topic": None,
        "query": None,
        "grep": None,
        "grep_ignore_case": False,
        "limit": None,
        "duration": None,
        "bytes_mode": BytesMode.SMART,
        "connect_timeout": 5.0,
    }
    base.update(overrides)
    return base


def test_cat_async_streams_decoded_jsonl_when_piped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = _run_cat_against_server(
        register=[_json_string_channel(1, "/chatter")],
        publish=[
            (1, b'{"data":"hello-0"}', 1_000_000_000),
            (1, b'{"data":"hello-1"}', 1_000_000_001),
            (1, b'{"data":"hello-2"}', 1_000_000_002),
        ],
        cat_kwargs=_default_cat_kwargs(topic=[".*"], limit=3),
    )
    assert rc == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert [entry["message"] for entry in lines] == [
        {"data": "hello-0"},
        {"data": "hello-1"},
        {"data": "hello-2"},
    ]
    assert all(entry["topic"] == "/chatter" for entry in lines)
    assert all(entry["schema"] == "std_msgs/String" for entry in lines)
    assert [entry["log_time"] for entry in lines] == [1_000_000_000, 1_000_000_001, 1_000_000_002]


def test_cat_async_filters_topics_by_regex(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _run_cat_against_server(
        register=[
            _json_string_channel(1, "/included/topic"),
            _json_string_channel(2, "/excluded/topic"),
        ],
        publish=[
            (1, b'{"data":"in-0"}', 1),
            (2, b'{"data":"out-0"}', 2),
            (1, b'{"data":"in-1"}', 3),
        ],
        cat_kwargs=_default_cat_kwargs(topic=["/included/.*"], limit=2),
    )
    assert rc == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert all(entry["topic"] == "/included/topic" for entry in lines)
    assert [entry["message"]["data"] for entry in lines] == ["in-0", "in-1"]


def test_cat_async_grep_drops_non_matching_messages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = _run_cat_against_server(
        register=[_json_string_channel(1, "/log")],
        publish=[
            (1, b'{"data":"info: started"}', 1),
            (1, b'{"data":"ERROR: boom"}', 2),
            (1, b'{"data":"info: finished"}', 3),
        ],
        cat_kwargs=_default_cat_kwargs(
            topic=[".*"],
            grep="error",
            grep_ignore_case=True,
            limit=1,
        ),
    )
    assert rc == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["message"]["data"] == "ERROR: boom"


def test_cat_async_passes_variables_to_query(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _run_cat_against_server(
        register=[_json_string_channel(1, "/chatter")],
        publish=[
            (1, b'{"data":"skip"}', 1),
            (1, b'{"data":"keep"}', 2),
        ],
        cat_kwargs=_default_cat_kwargs(
            query=["/chatter.data{==$wanted}"],
            variables={"wanted": "keep"},
            limit=1,
        ),
    )

    assert rc == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert [entry["message"] for entry in lines] == ["keep"]


def test_cat_async_applies_repeatable_queries_by_topic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = _run_cat_against_server(
        register=[
            _json_string_channel(1, "/front"),
            _json_string_channel(2, "/rear"),
        ],
        publish=[
            (1, b'{"data":"front-value"}', 1),
            (2, b'{"data":"rear-value"}', 2),
        ],
        cat_kwargs=_default_cat_kwargs(
            query=["/front.data", "/rear.data"],
            limit=2,
        ),
    )

    assert rc == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert [(entry["topic"], entry["message"]) for entry in lines] == [
        ("/front", "front-value"),
        ("/rear", "rear-value"),
    ]


def test_cat_async_skips_channel_with_unknown_encoding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A channel whose encoding has no matching decoder is silently skipped at subscribe time."""
    rc = _run_cat_against_server(
        register=[
            ServerChannel(
                id=1,
                topic="/weird",
                encoding="protobuf",  # neither json nor cdr/ros2msg
                schema_name="some.Schema",
                schema="",
                schema_encoding="protobuf",
            ),
            _json_string_channel(2, "/ok"),
        ],
        publish=[
            (2, b'{"data":"first"}', 1),
        ],
        cat_kwargs=_default_cat_kwargs(topic=[".*"], limit=1),
    )
    assert rc == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert [entry["topic"] for entry in lines] == ["/ok"]
    assert lines[0]["message"]["data"] == "first"


def test_cat_async_swallows_malformed_schema_for_one_channel(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A channel whose schema fails to parse must not abort cat for sibling channels."""
    rc = _run_cat_against_server(
        register=[
            ServerChannel(
                id=1,
                topic="/broken",
                encoding="cdr",
                schema_name="bogus/Msg",
                schema="this is not valid ros2msg !!!",
                schema_encoding="ros2msg",
            ),
            _json_string_channel(2, "/ok"),
        ],
        publish=[
            (2, b'{"data":"survives"}', 1),
        ],
        cat_kwargs=_default_cat_kwargs(topic=[".*"], limit=1),
    )
    assert rc == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert [entry["topic"] for entry in lines] == ["/ok"]
    assert lines[0]["message"]["data"] == "survives"


def test_cat_async_subscribes_each_channel_for_duplicate_topics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = _run_cat_against_server(
        register=[
            _json_string_channel(1, "/duplicated"),
            _json_string_channel(2, "/duplicated"),
        ],
        publish=[
            (1, b'{"data":"first-channel"}', 1),
            (2, b'{"data":"second-channel"}', 2),
        ],
        cat_kwargs=_default_cat_kwargs(topic=["/duplicated"], limit=2, duration=1.0),
        wait_for_channels={1, 2},
    )
    assert rc == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert [entry["message"]["data"] for entry in lines] == [
        "first-channel",
        "second-channel",
    ]


def test_cat_command_unreachable_port_returns_one() -> None:
    port = _free_port()
    rc = cat(
        target=f"ws://127.0.0.1:{port}",
        topic=[".*"],
        connect_timeout=0.3,
        limit=1,
    )
    assert rc == 1


def test_cat_command_rejects_non_positive_limit() -> None:
    rc = cat(target="ws://127.0.0.1:1", topic=[".*"], limit=0)
    assert rc == 1


def test_cat_command_rejects_non_positive_duration() -> None:
    rc = cat(target="ws://127.0.0.1:1", topic=[".*"], duration=0.0)
    assert rc == 1


def test_bridge_tf_returns_error_for_cyclic_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_collect_tf_graph_async(
        _url: str,
        *,
        static_only: bool,
        connect_timeout: float,
        collect_seconds: float,
    ) -> TfGraph:
        _ = (static_only, connect_timeout, collect_seconds)
        graph = TfGraph()
        graph.add(
            static=True,
            stamp_ns=0,
            parent="map",
            child="odom",
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        graph.add(
            static=True,
            stamp_ns=0,
            parent="odom",
            child="map",
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        return graph

    monkeypatch.setattr(tf_module, "_collect_tf_graph_async", fake_collect_tf_graph_async)

    assert tf_module.tf("ws://example:8765", discover_seconds=0.1) == 1
