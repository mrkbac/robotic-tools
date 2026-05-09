"""Unit tests for the Foxglove WebSocket bridge client."""

from __future__ import annotations

import asyncio

from robo_ws_bridge import ConnectionGraph, WebSocketBridgeClient


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
