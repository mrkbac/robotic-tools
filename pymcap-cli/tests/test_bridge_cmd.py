"""Unit tests for the `pymcap-cli bridge` command's pure logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.bridge_cmd import (
    BridgeInfo,
    BridgeStatus,
    SortChoice,
    _append_status,
    _build_connection_graph_tree,
    _remove_statuses,
    _sort_channels,
    bridge_to_dict,
    to_ws_url,
)
from rich.console import Console
from robo_ws_bridge import ConnectionGraph

if TYPE_CHECKING:
    from rich.tree import Tree
    from robo_ws_bridge.ws_types import ChannelInfo


def _render(tree: Tree) -> str:
    console = Console(record=True, width=120, color_system=None)
    console.print(tree)
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


def test_status_cache_replaces_and_removes_id_statuses() -> None:
    statuses: list[BridgeStatus] = []

    _append_status(statuses, 1, "warming up", "boot")
    _append_status(statuses, 2, "offline", "boot")
    _append_status(statuses, 0, "anonymous", None)
    _remove_statuses(statuses, ["boot"])

    assert statuses == [BridgeStatus(level=0, message="anonymous")]
