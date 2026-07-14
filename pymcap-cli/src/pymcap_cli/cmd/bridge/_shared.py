"""Shared infrastructure for the `bridge` subcommands.

Connection helpers, dataclasses, status bookkeeping, and the JSON wire-format
converter are factored out so that the per-command modules in this package
stay focused on their own UX concerns.
"""

import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from ipaddress import ip_address
from urllib.parse import urlsplit

from rich.console import Console
from robo_ws_bridge import ConnectionGraph, WebSocketBridgeClient
from robo_ws_bridge.ws_types import ChannelInfo, ServerCapabilities, ServerInfoMessage, ServiceInfo
from small_mcap import Schema

logger = logging.getLogger(__name__)
console = Console()

_DEFAULT_PORT = 8765

_STATUS_LEVEL_LABELS = {0: "INFO", 1: "WARNING", 2: "ERROR"}
_STATUS_LEVEL_STYLES = {0: "cyan", 1: "yellow", 2: "red"}


class SortChoice(str, Enum):
    """Sort field choices for the channel display."""

    TOPIC = "topic"
    ID = "id"
    SCHEMA = "schema"


def channel_to_schema(channel: ChannelInfo) -> Schema:
    """Build a decoder-compatible MCAP `Schema` from a bridge channel advertisement."""
    return Schema(
        id=channel["id"],
        name=channel.get("schemaName", ""),
        encoding=channel.get("schemaEncoding", ""),
        data=channel.get("schema", "").encode("utf-8"),
    )


class ChannelSubscriptionManager:
    """Subscribe to matching bridge advertisements once per channel id."""

    def __init__(
        self,
        client: WebSocketBridgeClient,
        should_subscribe: Callable[[ChannelInfo], bool],
    ) -> None:
        self._client = client
        self._should_subscribe = should_subscribe
        self._subscribed_channel_ids: set[int] = set()
        self._next_subscription_id = 1

    def install(self) -> None:
        self._client.on_disconnect(self._clear_subscriptions)
        self._client.on_advertised_channel(self.subscribe_if_matching)
        self._client.on_channel_unadvertised(self._forget_channel)

    async def subscribe_existing(self) -> None:
        for channel in list(self._client.channels.values()):
            await self.subscribe_if_matching(channel)

    async def subscribe_if_matching(self, channel: ChannelInfo) -> None:
        channel_id = channel["id"]
        if channel_id in self._subscribed_channel_ids or not self._should_subscribe(channel):
            return

        subscription_id = self._next_subscription_id
        self._next_subscription_id += 1
        await self._client.subscribe_to_channel(subscription_id, channel_id)
        self._subscribed_channel_ids.add(channel_id)

    def _forget_channel(self, channel: ChannelInfo) -> None:
        self._subscribed_channel_ids.discard(channel["id"])

    def _clear_subscriptions(self) -> None:
        self._subscribed_channel_ids.clear()


@dataclass(frozen=True)
class BridgeStatus:
    """Status currently advertised by a bridge."""

    level: int
    message: str
    status_id: str | None = None


@dataclass(frozen=True)
class BridgeInfo:
    """Static introspection result for a Foxglove WebSocket bridge."""

    url: str
    server_info: ServerInfoMessage
    channels: tuple[ChannelInfo, ...]
    services: tuple[ServiceInfo, ...] = ()
    statuses: tuple[BridgeStatus, ...] = ()
    connection_graph: ConnectionGraph | None = None


class BridgeFetchError(RuntimeError):
    """Raised when the bridge cannot be reached or fails to advertise its server info."""


def to_ws_url(arg: str, default_port: int = _DEFAULT_PORT) -> str:
    """Normalize a bridge target into a ``ws://host:port`` URL.

    Anything not starting with ``ws://`` / ``wss://`` is treated as a host
    (or ``host:port``); the default port is appended when none is given.
    """
    if arg.startswith(("ws://", "wss://")):
        return arg

    try:
        host_ip = ip_address(arg)
    except ValueError:
        pass
    else:
        host = f"[{arg}]" if host_ip.version == 6 else arg
        return f"ws://{host}:{default_port}"

    parsed = urlsplit(f"//{arg}")
    try:
        has_port = parsed.port is not None
    except ValueError:
        has_port = True
    if has_port:
        return f"ws://{arg}"
    return f"ws://{parsed.netloc}:{default_port}"


def _append_status(
    statuses: list[BridgeStatus], level: int, message: str, status_id: str | None
) -> None:
    status = BridgeStatus(level=level, message=message, status_id=status_id)
    if status_id is not None:
        statuses[:] = [existing for existing in statuses if existing.status_id != status_id]
    statuses.append(status)


def _remove_statuses(statuses: list[BridgeStatus], status_ids: list[str]) -> None:
    ids = set(status_ids)
    statuses[:] = [existing for existing in statuses if existing.status_id not in ids]


async def _fetch_async(url: str, *, connect_timeout: float, discover_seconds: float) -> BridgeInfo:
    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=1.0)
    server_info_event = asyncio.Event()
    statuses: list[BridgeStatus] = []

    def _mark_ready(_name: str, _caps: list[str], _session: str | None) -> None:
        server_info_event.set()

    def _collect_status(level: int, message: str, status_id: str | None) -> None:
        _append_status(statuses, level, message, status_id)

    client.on_server_info(_mark_ready)
    client.on_status(_collect_status)
    client.on_remove_status(lambda status_ids: _remove_statuses(statuses, status_ids))
    await client.connect()
    try:
        try:
            await asyncio.wait_for(server_info_event.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError as exc:
            raise BridgeFetchError(
                f"Timed out after {connect_timeout:.1f}s waiting for serverInfo from {url}"
            ) from exc
        server_info = client.server_info
        if server_info is None:
            raise BridgeFetchError(f"No serverInfo received from {url}")

        graph_supported = ServerCapabilities.CONNECTION_GRAPH.value in server_info["capabilities"]
        if graph_supported:
            await client.subscribe_connection_graph()

        await asyncio.sleep(discover_seconds)

        return BridgeInfo(
            url=url,
            server_info=server_info,
            channels=tuple(client.channels.values()),
            services=tuple(client.services.values()),
            statuses=tuple(statuses),
            connection_graph=client.connection_graph if graph_supported else None,
        )
    finally:
        await client.disconnect()


def fetch_bridge_info(
    url: str, *, connect_timeout: float = 5.0, discover_seconds: float = 1.5
) -> BridgeInfo:
    """Connect to ``url``, wait for advertise messages to settle, return a snapshot."""
    return asyncio.run(
        _fetch_async(url, connect_timeout=connect_timeout, discover_seconds=discover_seconds)
    )


def bridge_status_to_dict(status: BridgeStatus) -> dict[str, int | str]:
    """JSON-serializable representation of a bridge status."""
    payload: dict[str, int | str] = {
        "op": "status",
        "level": status.level,
        "message": status.message,
    }
    if status.status_id is not None:
        payload["id"] = status.status_id
    return payload


def bridge_to_dict(info: BridgeInfo) -> dict[str, object]:
    """JSON-serializable representation of a `BridgeInfo` (wire-format keys)."""
    payload: dict[str, object] = {
        "url": info.url,
        "server": dict(info.server_info),
        "channels": [dict(ch) for ch in info.channels],
        "services": [dict(svc) for svc in info.services],
        "statuses": [bridge_status_to_dict(s) for s in info.statuses],
    }
    if info.connection_graph is not None:
        graph = info.connection_graph
        payload["connectionGraph"] = {
            "publishedTopics": [dict(t) for t in graph.published_topics],
            "subscribedTopics": [dict(t) for t in graph.subscribed_topics],
            "advertisedServices": [dict(s) for s in graph.advertised_services],
        }
    return payload


def _sort_channels(
    channels: Iterable[ChannelInfo], sort: SortChoice, reverse: bool
) -> list[ChannelInfo]:
    if sort is SortChoice.TOPIC:
        return sorted(channels, key=lambda c: c["topic"], reverse=reverse)
    if sort is SortChoice.ID:
        return sorted(channels, key=lambda c: c["id"], reverse=reverse)
    return sorted(channels, key=lambda c: c.get("schemaName", ""), reverse=reverse)
