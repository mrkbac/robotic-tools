"""Bridge command - inspect or record from a live Foxglove WebSocket bridge."""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from ipaddress import ip_address
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import urlsplit

from cyclopts import App, Parameter
from cyclopts import Group as CycloptsGroup
from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from robo_ws_bridge import ConnectionGraph, WebSocketBridgeClient
from robo_ws_bridge.ws_types import ServerCapabilities
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    MessagePath,
    MessagePathError,
    ValidationError,
    parse_message_path,
)
from small_mcap import JSONDecoderFactory, McapWriter

from pymcap_cli.display.display_utils import _format_parts_with_colors, _format_schema_with_link
from pymcap_cli.display.message_render import (
    SMART_BYTES_INLINE_LIMIT,
    TTY_BYTES_TRUNCATE,
    BytesMode,
    EnumPlan,
    build_enum_plan,
    message_matches_grep,
    message_to_dict,
    render_message_tree,
    resolve_msgdef_by_name,
)
from pymcap_cli.log_setup import ERR
from pymcap_cli.types.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    OutputPathOption,
    str_to_compression_type,
)
from pymcap_cli.utils import compile_topic_patterns, confirm_output_overwrite

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from re import Pattern

    from robo_ws_bridge.ws_types import ChannelInfo, ServerInfoMessage, ServiceInfo
    from ros_parser.models import MessageDefinition

logger = logging.getLogger(__name__)
console = Console()

CONNECTION_GROUP = CycloptsGroup("Connection Options")
DISPLAY_GROUP = CycloptsGroup("Display Options")

_DEFAULT_PORT = 8765


class SortChoice(str, Enum):
    """Sort field choices for the channel display."""

    TOPIC = "topic"
    ID = "id"
    SCHEMA = "schema"


_STATUS_LEVEL_LABELS = {0: "INFO", 1: "WARNING", 2: "ERROR"}
_STATUS_LEVEL_STYLES = {0: "cyan", 1: "yellow", 2: "red"}


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


def _sort_channels(
    channels: Iterable[ChannelInfo], sort: SortChoice, reverse: bool
) -> list[ChannelInfo]:
    if sort is SortChoice.TOPIC:
        return sorted(channels, key=lambda c: c["topic"], reverse=reverse)
    if sort is SortChoice.ID:
        return sorted(channels, key=lambda c: c["id"], reverse=reverse)
    return sorted(channels, key=lambda c: c.get("schemaName", ""), reverse=reverse)


def _build_summary(info: BridgeInfo) -> Table:
    server = info.server_info
    capabilities = server.get("capabilities") or []
    encodings = server.get("supportedEncodings") or []
    session_id = server.get("sessionId")

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold blue")
    summary.add_column()
    summary.add_row("Bridge:", f"[green]{info.url}[/]")
    summary.add_row("Server:", f"[yellow]{server['name']}[/]")
    if capabilities:
        summary.add_row("Capabilities:", f"[cyan]{', '.join(capabilities)}[/]")
    if encodings:
        summary.add_row("Encodings:", f"[cyan]{', '.join(encodings)}[/]")
    if session_id:
        summary.add_row("Session:", f"[dim]{session_id}[/]")
    summary.add_row("Channels:", f"[green]{len(info.channels):,}[/]")
    return summary


def _build_statuses_table(statuses: tuple[BridgeStatus, ...]) -> Table:
    table = Table(title="Status messages", title_justify="left", title_style="bold cyan")
    table.add_column("Level")
    table.add_column("Message", style="white")
    table.add_column("ID", style="dim")
    for status in statuses:
        level = status.level
        label = _STATUS_LEVEL_LABELS.get(level, str(level))
        style = _STATUS_LEVEL_STYLES.get(level, "white")
        table.add_row(
            f"[{style}]{label}[/]",
            status.message,
            status.status_id or "",
        )
    return table


def _build_connection_graph_tree(
    graph: ConnectionGraph,
    channels: Iterable[ChannelInfo] = (),
    services: Iterable[ServiceInfo] = (),
) -> Tree | None:
    schemas_by_topic: dict[str, str] = {}
    for ch in channels:
        schema = ch.get("schemaName") or ""
        if schema:
            schemas_by_topic.setdefault(ch["topic"], schema)

    types_by_service: dict[str, str] = {}
    for svc in services:
        svc_type = svc.get("type") or ""
        if svc_type:
            types_by_service.setdefault(svc["name"], svc_type)

    nodes: dict[str, dict[str, list[str]]] = {}

    def _bucket(node_id: str) -> dict[str, list[str]]:
        return nodes.setdefault(node_id, {"publishes": [], "subscribes": [], "provides": []})

    for topic in graph.published_topics:
        for pub_id in topic["publisherIds"]:
            _bucket(pub_id)["publishes"].append(topic["name"])
    for topic in graph.subscribed_topics:
        for sub_id in topic["subscriberIds"]:
            _bucket(sub_id)["subscribes"].append(topic["name"])
    for svc in graph.advertised_services:
        for provider_id in svc["providerIds"]:
            _bucket(provider_id)["provides"].append(svc["name"])

    if not nodes:
        return None

    tree = Tree("[bold cyan]Connection graph[/]")
    for node_id in sorted(nodes):
        node_branch = tree.add(_format_parts_with_colors(node_id))
        for section in ("publishes", "subscribes", "provides"):
            items = sorted(nodes[node_id][section])
            if not items:
                continue
            section_branch = node_branch.add(f"[bold]{section}[/]")
            for item in items:
                label = _format_parts_with_colors(item)
                if section == "provides":
                    svc_type = types_by_service.get(item)
                    if svc_type:
                        label = f"{label} [dim]{svc_type}[/]"
                else:
                    schema = schemas_by_topic.get(item)
                    if schema:
                        label = f"{label} [dim]{schema}[/]"
                section_branch.add(label)
    return tree


def _build_channels_table(channels: list[ChannelInfo], graph: ConnectionGraph | None) -> Table:
    table = Table()
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Topic")
    table.add_column("Schema")

    pub_ids: dict[str, list[str]] = {}
    sub_ids: dict[str, list[str]] = {}
    if graph is not None:
        pub_ids = {t["name"]: list(t["publisherIds"]) for t in graph.published_topics}
        sub_ids = {t["name"]: list(t["subscriberIds"]) for t in graph.subscribed_topics}
        table.add_column("Publishers")
        table.add_column("Subscribers")

    seen_topics: set[str] = set()
    for ch in channels:
        seen_topics.add(ch["topic"])
        row = [
            str(ch["id"]),
            _format_parts_with_colors(ch["topic"]),
            _format_schema_with_link(ch.get("schemaName")),
        ]
        if graph is not None:
            row.append(_format_node_ids(pub_ids.get(ch["topic"], [])))
            row.append(_format_node_ids(sub_ids.get(ch["topic"], [])))
        table.add_row(*row)

    if graph is not None:
        for topic in sorted((set(pub_ids) | set(sub_ids)) - seen_topics):
            table.add_row(
                "",
                _format_parts_with_colors(topic),
                "",
                _format_node_ids(pub_ids.get(topic, [])),
                _format_node_ids(sub_ids.get(topic, [])),
            )

    return table


def _format_node_ids(node_ids: list[str]) -> str:
    if not node_ids:
        return ""
    return "\n".join(_format_parts_with_colors(node_id) for node_id in sorted(node_ids))


def _build_display(info: BridgeInfo, sort: SortChoice, reverse: bool) -> RenderableType:
    sorted_channels = _sort_channels(info.channels, sort, reverse)
    parts: list[RenderableType] = [_build_summary(info), Text("")]
    if info.statuses:
        parts.extend([_build_statuses_table(info.statuses), Text("")])
    parts.append(_build_channels_table(sorted_channels, info.connection_graph))
    if info.connection_graph is not None:
        graph_tree = _build_connection_graph_tree(
            info.connection_graph, info.channels, info.services
        )
        if graph_tree is not None:
            parts.extend([Text(""), graph_tree])
    return Group(*parts)


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


def _output_json(info: BridgeInfo, compress: bool) -> None:
    payload = json.dumps(bridge_to_dict(info))
    if compress:
        compressed = gzip.compress(payload.encode("utf-8"))
        print(base64.b64encode(compressed).decode("ascii"))  # noqa: T201
    else:
        print(payload)  # noqa: T201


async def _watch_async(
    url: str,
    sort: SortChoice,
    reverse: bool,
    interval: float,
    connect_timeout: float,
) -> int:
    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
    connected = asyncio.Event()
    statuses: list[BridgeStatus] = []

    def _collect_status(level: int, message: str, status_id: str | None) -> None:
        _append_status(statuses, level, message, status_id)

    client.on_server_info(lambda *_: connected.set())
    client.on_status(_collect_status)
    client.on_remove_status(lambda status_ids: _remove_statuses(statuses, status_ids))
    await client.connect()
    try:
        try:
            await asyncio.wait_for(connected.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError:
            ERR.print(f"[red]Error:[/] Timed out connecting to {url}")
            return 1

        graph_supported = ServerCapabilities.CONNECTION_GRAPH.value in (
            client.server_info["capabilities"] if client.server_info else []
        )
        if graph_supported:
            await client.subscribe_connection_graph()

        with Live(console=console, refresh_per_second=4) as live:
            while True:
                server_info = client.server_info
                if server_info is None:
                    await asyncio.sleep(interval)
                    continue
                info = BridgeInfo(
                    url=url,
                    server_info=server_info,
                    channels=tuple(client.channels.values()),
                    services=tuple(client.services.values()),
                    statuses=tuple(statuses),
                    connection_graph=client.connection_graph if graph_supported else None,
                )
                now = datetime.now(tz=timezone.utc).astimezone().strftime("%H:%M:%S")
                status = Text.from_markup(
                    f"\n[dim]Watching... Last update: {now}"
                    f" | Channels: {len(info.channels)} | Ctrl+C to stop[/]"
                )
                live.update(Group(_build_display(info, sort, reverse), status))
                await asyncio.sleep(interval)
    finally:
        await client.disconnect()


def _watch(
    url: str,
    sort: SortChoice,
    reverse: bool,
    interval: float,
    connect_timeout: float,
) -> int:
    try:
        return asyncio.run(_watch_async(url, sort, reverse, interval, connect_timeout))
    except KeyboardInterrupt:
        console.print("\n[dim]Watch stopped.[/]")
        return 0


def inspect(
    target: str,
    *,
    json_output: Annotated[
        bool,
        Parameter(name=["--json"], group=DISPLAY_GROUP),
    ] = False,
    compress: Annotated[
        bool,
        Parameter(name=["--compress"], group=DISPLAY_GROUP),
    ] = False,
    sort: Annotated[
        SortChoice,
        Parameter(name=["-s", "--sort"], group=DISPLAY_GROUP),
    ] = SortChoice.TOPIC,
    reverse: Annotated[
        bool,
        Parameter(name=["--reverse"], group=DISPLAY_GROUP),
    ] = False,
    watch: Annotated[
        bool,
        Parameter(name=["-w", "--watch"], group=CONNECTION_GROUP),
    ] = False,
    watch_interval: Annotated[
        float,
        Parameter(name=["--watch-interval"], group=CONNECTION_GROUP),
    ] = 0.5,
    connect_timeout: Annotated[
        float,
        Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
    ] = 5.0,
    discover_seconds: Annotated[
        float,
        Parameter(name=["--discover-seconds"], group=CONNECTION_GROUP),
    ] = 1.5,
) -> int:
    """Inspect a live Foxglove WebSocket bridge.

    Connects to a Foxglove WebSocket bridge, prints the server metadata and
    advertised channel list, then disconnects. Use `info` for MCAP files.

    Parameters
    ----------
    target
        Bridge address. Accepts ``ws://host:port``, ``wss://host:port``,
        a hostname, an IP, or ``host:port``. Defaults to port 8765 when none
        is given.
    json_output
        Output as JSON instead of Rich tables.
    compress
        Gzip+base64 the JSON payload (requires --json).
    sort
        Sort channels by ``topic`` (default), ``id``, or ``schema``.
    reverse
        Reverse sort order (descending).
    watch
        Stay connected and refresh on advertise/unadvertise events. Ctrl+C to stop.
    watch_interval
        Refresh cadence in seconds for watch mode (default: 0.5).
    connect_timeout
        Seconds to wait for the bridge's serverInfo before giving up (default: 5.0).
    discover_seconds
        Seconds to wait after connect for advertise messages to settle
        (default: 1.5).

    Examples
    --------
    ```
    pymcap-cli bridge ws://localhost:8765
    pymcap-cli bridge 127.0.0.1:8765
    pymcap-cli bridge 192.168.1.10           # default port 8765
    pymcap-cli bridge ws://localhost:8765 --json
    pymcap-cli bridge ws://localhost:8765 --watch
    ```
    """
    if compress and not json_output:
        logger.error("--compress requires --json")
        return 1
    if watch and json_output:
        logger.error("--watch is incompatible with --json")
        return 1

    url = to_ws_url(target)

    if watch:
        return _watch(url, sort, reverse, watch_interval, connect_timeout)

    try:
        info = fetch_bridge_info(
            url, connect_timeout=connect_timeout, discover_seconds=discover_seconds
        )
    except BridgeFetchError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Failed to connect to {url}: {exc}")
        return 1

    if json_output:
        _output_json(info, compress)
        return 0

    console.print(_build_display(info, sort, reverse))
    return 0


RECORD_GROUP = CycloptsGroup("Record Options")


@dataclass(frozen=True)
class _RecorderChannelKey:
    topic: str
    schema_name: str
    schema_encoding: str
    schema_data: bytes
    message_encoding: str


@dataclass(frozen=True)
class TopicSelector:
    """Decide which topics to record, mirroring `ros2 bag record` semantics."""

    all_topics: bool = False
    exact_topics: frozenset[str] = field(default_factory=frozenset)
    include_patterns: tuple[Pattern[str], ...] = ()
    exclude_topics: frozenset[str] = field(default_factory=frozenset)
    exclude_patterns: tuple[Pattern[str], ...] = ()

    def matches(self, topic: str) -> bool:
        if topic in self.exclude_topics:
            return False
        if any(pattern.search(topic) for pattern in self.exclude_patterns):
            return False
        if self.all_topics:
            return True
        if topic in self.exact_topics:
            return True
        return any(pattern.search(topic) for pattern in self.include_patterns)


@dataclass
class BridgeRecorder:
    """Translate `(channel, timestamp, payload)` callbacks into MCAP records."""

    writer: McapWriter
    selector: TopicSelector
    message_limit: int | None = None
    done_event: asyncio.Event | None = None
    schema_ids: dict[tuple[str, str, bytes], int] = field(default_factory=dict)
    channel_ids: dict[_RecorderChannelKey, int] = field(default_factory=dict)
    message_counts: dict[str, int] = field(default_factory=dict)
    payload_bytes: int = 0
    total_messages: int = 0
    _next_schema_id: int = 1
    _next_channel_id: int = 1

    def matches_topic(self, topic: str) -> bool:
        return self.selector.matches(topic)

    def schema_id_for(self, channel: ChannelInfo) -> int:
        schema_name = channel.get("schemaName", "")
        schema_data = channel.get("schema", "").encode("utf-8")
        schema_encoding = channel.get("schemaEncoding", "")
        if not schema_name and not schema_data:
            return 0
        key = (schema_name, schema_encoding, schema_data)
        existing = self.schema_ids.get(key)
        if existing is not None:
            return existing
        schema_id = self._next_schema_id
        self._next_schema_id += 1
        self.schema_ids[key] = schema_id
        self.writer.add_schema(
            schema_id=schema_id,
            name=schema_name,
            encoding=schema_encoding,
            data=schema_data,
        )
        return schema_id

    def channel_id_for(self, channel: ChannelInfo) -> int:
        key = _RecorderChannelKey(
            topic=channel["topic"],
            schema_name=channel.get("schemaName", ""),
            schema_encoding=channel.get("schemaEncoding", ""),
            schema_data=channel.get("schema", "").encode("utf-8"),
            message_encoding=channel["encoding"],
        )
        existing = self.channel_ids.get(key)
        if existing is not None:
            return existing
        channel_id = self._next_channel_id
        self._next_channel_id += 1
        schema_id = self.schema_id_for(channel)
        self.channel_ids[key] = channel_id
        self.writer.add_channel(
            channel_id=channel_id,
            topic=key.topic,
            message_encoding=key.message_encoding,
            schema_id=schema_id,
        )
        return channel_id

    def on_message(self, channel: ChannelInfo, timestamp: int, payload: bytes) -> None:
        if not self.matches_topic(channel["topic"]):
            return
        if self.message_limit is not None and self.total_messages >= self.message_limit:
            return
        channel_id = self.channel_id_for(channel)
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp,
            data=payload,
            publish_time=timestamp,
        )
        self.total_messages += 1
        self.payload_bytes += len(payload)
        self.message_counts[channel["topic"]] = self.message_counts.get(channel["topic"], 0) + 1
        if (
            self.done_event is not None
            and self.message_limit is not None
            and self.total_messages >= self.message_limit
        ):
            self.done_event.set()


def _format_byte_size(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def _build_record_status(
    *,
    url: str,
    output: Path,
    recorder: BridgeRecorder,
    elapsed: float,
    duration: float | None,
    message_limit: int | None,
) -> RenderableType:
    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold blue")
    summary.add_column()
    summary.add_row("Bridge:", f"[green]{url}[/]")
    summary.add_row("Output:", f"[green]{output}[/]")
    elapsed_str = f"{elapsed:.1f}s"
    if duration is not None:
        elapsed_str = f"{elapsed:.1f}/{duration:.1f}s"
    summary.add_row("Elapsed:", elapsed_str)
    counter = f"[green]{recorder.total_messages:,}[/]"
    if message_limit is not None:
        counter = f"{counter} / {message_limit:,}"
    summary.add_row("Messages:", counter)
    summary.add_row("Payload:", _format_byte_size(recorder.payload_bytes))

    table = Table(title="Per-topic", title_justify="left", title_style="bold cyan")
    table.add_column("Topic")
    table.add_column("Messages", justify="right")
    for topic in sorted(recorder.message_counts):
        count = recorder.message_counts[topic]
        table.add_row(_format_parts_with_colors(topic), f"{count:,}")

    if not recorder.message_counts:
        return Group(summary, Text(""), Text("Waiting for messages...", style="dim"))
    return Group(summary, Text(""), table)


async def _subscribe_if_matching(
    client: WebSocketBridgeClient, recorder: BridgeRecorder, channel: ChannelInfo
) -> None:
    if recorder.matches_topic(channel["topic"]):
        await client.subscribe(channel["topic"])


async def _subscribe_matching_channels(
    client: WebSocketBridgeClient, recorder: BridgeRecorder
) -> None:
    async def _on_advertise(channel: ChannelInfo) -> None:
        await _subscribe_if_matching(client, recorder, channel)

    client.on_advertised_channel(_on_advertise)
    for channel in list(client.channels.values()):
        await _subscribe_if_matching(client, recorder, channel)


async def _record_async(
    *,
    url: str,
    output: Path,
    selector: TopicSelector,
    duration: float | None,
    message_limit: int | None,
    chunk_size: int,
    compression_choice: str,
    connect_timeout: float,
    refresh_interval: float,
    show_status: bool,
) -> int:
    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
    server_info_event = asyncio.Event()
    client.on_server_info(lambda *_: server_info_event.set())

    await client.connect()
    try:
        try:
            await asyncio.wait_for(server_info_event.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError:
            ERR.print(f"[red]Error:[/] Timed out connecting to {url}")
            return 1

        compression_type = str_to_compression_type(compression_choice)
        try:
            with output.open("wb") as out:
                writer = McapWriter(out, chunk_size=chunk_size, compression=compression_type)
                writer.start(profile="", library="pymcap-cli bridge record")

                done = asyncio.Event()
                recorder = BridgeRecorder(
                    writer=writer,
                    selector=selector,
                    message_limit=message_limit,
                    done_event=done,
                )

                client.on_message(recorder.on_message)
                await _subscribe_matching_channels(client, recorder)

                start = time.monotonic()
                background: list[asyncio.Task[None]] = []
                if duration is not None:

                    async def _stop_after_duration() -> None:
                        await asyncio.sleep(duration)
                        done.set()

                    background.append(asyncio.create_task(_stop_after_duration()))

                async def _wait_for_done_with_status() -> None:
                    if not show_status:
                        await done.wait()
                        return
                    with Live(console=console, refresh_per_second=4) as live:
                        while not done.is_set():
                            elapsed = time.monotonic() - start
                            live.update(
                                _build_record_status(
                                    url=url,
                                    output=output,
                                    recorder=recorder,
                                    elapsed=elapsed,
                                    duration=duration,
                                    message_limit=message_limit,
                                )
                            )
                            try:
                                await asyncio.wait_for(
                                    asyncio.shield(done.wait()), timeout=refresh_interval
                                )
                            except asyncio.TimeoutError:
                                continue

                try:
                    await _wait_for_done_with_status()
                finally:
                    for task in background:
                        task.cancel()
                    if background:
                        await asyncio.gather(*background, return_exceptions=True)
                    writer.finish()
        except OSError as exc:
            ERR.print(f"[red]Error:[/] Failed to write to {output}: {exc}")
            return 1

        console.print(f"[green]Wrote {recorder.total_messages:,} messages to[/] [bold]{output}[/]")
        return 0
    finally:
        await client.disconnect()


def record(
    target: str,
    *,
    output: OutputPathOption,
    topics: Annotated[
        list[str],
        Parameter(
            name=["--topics"],
            group=RECORD_GROUP,
            help="Space-delimited list of topics to record (exact names).",
        ),
    ] = [],  # noqa: B006
    all_topics: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            group=RECORD_GROUP,
            negative="--no-all",
            help="Record every advertised topic.",
        ),
    ] = False,
    regex: Annotated[
        str | None,
        Parameter(
            name=["--regex", "-e"],
            group=RECORD_GROUP,
            help=(
                "Record topics matching this regex (case-insensitive, ``re.search``)."
                " Note: --all overrides --regex."
            ),
        ),
    ] = None,
    exclude_topics: Annotated[
        list[str],
        Parameter(
            name=["--exclude-topics"],
            group=RECORD_GROUP,
            help="Space-delimited list of topics to skip (exact names).",
        ),
    ] = [],  # noqa: B006
    exclude_regex: Annotated[
        str | None,
        Parameter(
            name=["--exclude-regex"],
            group=RECORD_GROUP,
            help="Skip topics matching this regex (case-insensitive). Wins over includes.",
        ),
    ] = None,
    duration: Annotated[
        float | None,
        Parameter(
            name=["--duration", "-d"],
            group=RECORD_GROUP,
            help=(
                "Stop after this many seconds. (Differs from ros2 bag's -d, which splits the bag.)"
            ),
        ),
    ] = None,
    message_limit: Annotated[
        int | None,
        Parameter(
            name=["--message-limit"],
            group=RECORD_GROUP,
            help="Stop after writing this many messages.",
        ),
    ] = None,
    connect_timeout: Annotated[
        float,
        Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
    ] = 5.0,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    progress: Annotated[
        bool,
        Parameter(
            name=["--progress"],
            group=DISPLAY_GROUP,
            negative="--no-progress",
            help="Show a live status panel while recording.",
        ),
    ] = True,
) -> int:
    """Record messages from a live Foxglove WebSocket bridge into an MCAP file.

    Mirrors the ``ros2 bag record`` topic-selection surface: ``--topics`` for
    a list of exact names, ``-e/--regex`` for a regex include, ``-a/--all`` for
    everything, with ``--exclude-topics`` and ``--exclude-regex`` filtering on
    top. Stops on ``--duration`` / ``--message-limit`` or Ctrl+C.

    Parameters
    ----------
    target
        Bridge address — same forms accepted by ``bridge`` (URL, host, or
        ``host:port``); defaults to port 8765 when none is given.
    output
        MCAP file to write.
    topics
        Space-delimited list of topics to record (exact names).
    all_topics
        Record every advertised topic.
    regex
        Record topics matching this regex. ``--all`` overrides this.
    exclude_topics
        Topics to skip (exact names).
    exclude_regex
        Skip topics matching this regex. Wins over includes.
    duration
        Stop after this many seconds.
    message_limit
        Stop after writing this many messages.
    connect_timeout
        Seconds to wait for the bridge's ``serverInfo`` (default: 5.0).
    chunk_size
        MCAP chunk size in bytes.
    compression
        Output compression algorithm.
    force
        Overwrite ``output`` without prompting.
    progress
        Show a live status panel while recording.

    Examples
    --------
    ```
    pymcap-cli bridge record ws://localhost:8765 -a -o capture.mcap
    pymcap-cli bridge record localhost --topics /chatter /imu/data -o capture.mcap
    pymcap-cli bridge record localhost -e '^/camera/' -o capture.mcap -d 30
    pymcap-cli bridge record localhost -a --exclude-regex '^/debug/' -o capture.mcap
    ```
    """
    if duration is not None and duration <= 0:
        ERR.print("[red]Error:[/] --duration must be positive")
        return 1
    if message_limit is not None and message_limit <= 0:
        ERR.print("[red]Error:[/] --message-limit must be positive")
        return 1
    if not all_topics and not topics and regex is None:
        ERR.print("[red]Error:[/] specify --topics, --all, or --regex.")
        return 1

    output_path = Path(output)
    confirm_output_overwrite(output_path, force)

    try:
        include_patterns = (
            tuple(compile_topic_patterns([regex])) if regex is not None and not all_topics else ()
        )
        exclude_patterns = (
            tuple(compile_topic_patterns([exclude_regex])) if exclude_regex is not None else ()
        )
    except ValueError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1

    selector = TopicSelector(
        all_topics=all_topics,
        exact_topics=frozenset(topics),
        include_patterns=include_patterns,
        exclude_topics=frozenset(exclude_topics),
        exclude_patterns=exclude_patterns,
    )

    url = to_ws_url(target)
    refresh_interval = 0.25

    try:
        return asyncio.run(
            _record_async(
                url=url,
                output=output_path,
                selector=selector,
                duration=duration,
                message_limit=message_limit,
                chunk_size=chunk_size,
                compression_choice=compression.value,
                connect_timeout=connect_timeout,
                refresh_interval=refresh_interval,
                show_status=progress,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Recording stopped.[/]")
        return 0
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Recording failed for {url}: {exc}")
        return 1


CAT_FILTER_GROUP = CycloptsGroup("Filtering")
CAT_OUTPUT_GROUP = CycloptsGroup("Output")


@dataclass(frozen=True)
class _BridgeSchema:
    """MCAP-style Schema view of a bridge ChannelInfo, satisfying the decoder Protocol."""

    id: int
    name: str
    encoding: str
    data: bytes


def _channel_to_schema(channel: ChannelInfo) -> _BridgeSchema:
    return _BridgeSchema(
        id=channel["id"],
        name=channel.get("schemaName", ""),
        encoding=channel.get("schemaEncoding", ""),
        data=channel.get("schema", "").encode("utf-8"),
    )


async def _cat_async(
    *,
    url: str,
    topics: list[str],
    exclude_topics: list[str],
    query: str | None,
    grep: str | None,
    grep_ignore_case: bool,
    limit: int | None,
    duration: float | None,
    bytes_mode: BytesMode,
    connect_timeout: float,
) -> int:
    parsed_query: MessagePath | None = None
    if query:
        try:
            parsed_query = parse_message_path(query)
        except Exception:
            logger.exception("Invalid --query syntax")
            return 1

    grep_pattern: re.Pattern[str] | None = None
    if grep:
        try:
            grep_pattern = re.compile(grep, re.IGNORECASE if grep_ignore_case else 0)
        except re.error:
            logger.exception("Invalid --grep regex")
            return 1

    try:
        topic_patterns = [re.compile(p) for p in topics]
        exclude_topic_patterns = [re.compile(p) for p in exclude_topics]
    except re.error:
        logger.exception("Invalid topic regex")
        return 1

    is_tty = sys.stdout.isatty()
    factories = [JSONDecoderFactory(), DecoderFactory()]

    decoders: dict[int, Callable[[bytes | memoryview], Any] | None] = {}
    schemas_by_channel: dict[int, _BridgeSchema] = {}
    parsed_schemas: dict[int, dict[str, MessageDefinition] | None] = {}
    enum_plans: dict[int, EnumPlan | None] = {}
    validated_topics: set[str] = set()
    warned_channels: set[int] = set()
    subscribed: set[int] = set()
    total = 0
    return_code = 0
    done = asyncio.Event()

    def _topic_matches(topic: str) -> bool:
        if parsed_query is not None:
            return topic == parsed_query.topic and not any(
                p.search(topic) for p in exclude_topic_patterns
            )
        if topic_patterns and not any(p.search(topic) for p in topic_patterns):
            return False
        return not any(p.search(topic) for p in exclude_topic_patterns)

    def _decoder_for(channel: ChannelInfo) -> Callable[[bytes | memoryview], Any] | None:
        cid = channel["id"]
        if cid in decoders:
            return decoders[cid]
        schema = _channel_to_schema(channel)
        message_encoding = channel["encoding"]
        decoder: Callable[[bytes | memoryview], Any] | None = None
        for factory in factories:
            try:
                decoder = factory.decoder_for(message_encoding, schema)
            except Exception:
                # `mcap_ros2_support_fast.DecoderFactory` parses ROS2 schemas eagerly
                # and raises on malformed ros2msg. Treat as "no decoder" so a single
                # bad schema doesn't take down the whole cat session.
                logger.exception(
                    f"Decoder construction failed for {channel['topic']} "
                    f"(schema={schema.name!r}, encoding={message_encoding!r})"
                )
                decoder = None
            if decoder is not None:
                break
        decoders[cid] = decoder
        if decoder is not None:
            schemas_by_channel[cid] = schema
        return decoder

    def _get_parsed_schema(cid: int) -> dict[str, MessageDefinition] | None:
        if cid in parsed_schemas:
            return parsed_schemas[cid]
        schema = schemas_by_channel.get(cid)
        if schema is None:
            return None
        try:
            parsed = parse_schema_to_definitions(schema.name, schema.data)
        except Exception:
            logger.exception(f"Failed to parse schema '{schema.name}'")
            parsed = None
        parsed_schemas[cid] = parsed
        return parsed

    def _get_enum_plan(cid: int) -> EnumPlan | None:
        if cid in enum_plans:
            return enum_plans[cid]
        parsed = _get_parsed_schema(cid)
        schema = schemas_by_channel.get(cid)
        plan = build_enum_plan(schema.name, parsed) if parsed and schema else None
        enum_plans[cid] = plan
        return plan

    def _validate_query(query_path: MessagePath, cid: int, topic: str) -> bool:
        all_definitions = _get_parsed_schema(cid)
        if all_definitions is None:
            return True
        schema = schemas_by_channel.get(cid)
        if schema is None:
            return True
        try:
            root_msgdef = resolve_msgdef_by_name(schema.name, all_definitions)
            if root_msgdef is None:
                logger.warning(f"Could not find message definition for schema '{schema.name}'")
            else:
                query_path.validate(root_msgdef, all_definitions)
        except ValidationError:
            logger.exception(f"Query validation error for topic '{topic}'")
            logger.exception(f"Query: {query}  Schema: {schema.name}")
            return False
        return True

    def _emit_jsonl(channel: ChannelInfo, log_time_ns: int, data: Any) -> None:
        entry: dict[str, Any] = {
            "topic": channel["topic"],
            "log_time": log_time_ns,
        }
        schema_name = channel.get("schemaName")
        if schema_name:
            entry["schema"] = schema_name
        entry["message"] = message_to_dict(data, bytes_mode=bytes_mode)
        sys.stdout.write(json.dumps(entry, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def _emit_tty(channel: ChannelInfo, log_time_ns: int, data: Any) -> None:
        schema_name = channel.get("schemaName") or "unknown"
        header = Text()
        header.append(channel["topic"], style="bold cyan")
        header.append(" @ ", style="dim")
        header.append(str(log_time_ns), style="green")
        header.append(" [", style="dim")
        header.append(schema_name, style="yellow")
        header.append("]", style="dim")
        plan = None if parsed_query is not None else _get_enum_plan(channel["id"])
        tree = render_message_tree(
            data,
            plan,
            title=header,
            bytes_mode=bytes_mode,
            truncate_bytes=TTY_BYTES_TRUNCATE,
        )
        console.print(Panel(tree, border_style="blue", expand=False))

    def _on_message(channel: ChannelInfo, log_time_ns: int, payload: bytes) -> None:
        nonlocal total, return_code
        if done.is_set():
            return
        decoder = _decoder_for(channel)
        if decoder is None:
            cid = channel["id"]
            if cid not in warned_channels:
                warned_channels.add(cid)
                logger.warning(
                    f"No decoder for {channel['topic']} "
                    f"(message_encoding={channel['encoding']!r}, "
                    f"schemaEncoding={channel.get('schemaEncoding', '')!r})"
                )
            return
        try:
            decoded = decoder(payload)
        except Exception:
            logger.exception(f"Failed to decode message on {channel['topic']}")
            return

        topic = channel["topic"]
        if parsed_query is not None and topic not in validated_topics:
            validated_topics.add(topic)
            if not _validate_query(parsed_query, channel["id"], topic):
                return_code = 1
                done.set()
                return

        if parsed_query is not None:
            try:
                data = parsed_query.apply(decoded)
            except MessagePathError as exc:
                logger.warning(f"Filter error on {topic}: {exc}")
                return
            if data is None:
                return
        else:
            data = decoded

        if grep_pattern is not None and not message_matches_grep(data, grep_pattern):
            return

        if is_tty:
            _emit_tty(channel, log_time_ns, data)
        else:
            _emit_jsonl(channel, log_time_ns, data)

        total += 1
        if limit is not None and total >= limit:
            done.set()

    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
    server_info_event = asyncio.Event()
    client.on_server_info(lambda *_: server_info_event.set())

    async def _maybe_subscribe(channel: ChannelInfo) -> None:
        cid = channel["id"]
        if cid in subscribed or not _topic_matches(channel["topic"]):
            return
        if _decoder_for(channel) is None:
            return
        subscribed.add(cid)
        await client.subscribe(channel["topic"])

    async def _on_advertise(channel: ChannelInfo) -> None:
        await _maybe_subscribe(channel)

    client.on_advertised_channel(_on_advertise)
    client.on_message(_on_message)

    await client.connect()
    try:
        try:
            await asyncio.wait_for(server_info_event.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError:
            ERR.print(f"[red]Error:[/] Timed out connecting to {url}")
            return 1

        for channel in list(client.channels.values()):
            await _maybe_subscribe(channel)

        background: list[asyncio.Task[None]] = []
        if duration is not None:

            async def _stop_after_duration() -> None:
                await asyncio.sleep(duration)
                done.set()

            background.append(asyncio.create_task(_stop_after_duration()))

        try:
            await done.wait()
        finally:
            for task in background:
                task.cancel()
            if background:
                await asyncio.gather(*background, return_exceptions=True)

        return return_code
    finally:
        await client.disconnect()


def cat(
    target: str,
    *,
    topics: Annotated[
        list[str],
        Parameter(
            name=["-t", "--topics"],
            group=CAT_FILTER_GROUP,
            help="Topic regex(es) to include. Repeat or pass space-delimited.",
        ),
    ] = [],  # noqa: B006
    exclude_topics: Annotated[
        list[str],
        Parameter(
            name=["-x", "--exclude-topics", "-n"],
            group=CAT_FILTER_GROUP,
            help="Topic regex(es) to exclude. Wins over --topics and --query.",
        ),
    ] = [],  # noqa: B006
    query: Annotated[
        str | None,
        Parameter(
            name=["-q", "--query"],
            group=CAT_FILTER_GROUP,
            help="MessagePath expression scoping output to one topic and/or subfield.",
        ),
    ] = None,
    grep: Annotated[
        str | None,
        Parameter(
            name=["-g", "--grep"],
            group=CAT_FILTER_GROUP,
            help=(
                "Regex applied to every scalar value in the decoded message. "
                "Messages with no match are skipped. Bytes-like fields are not "
                "searched."
            ),
        ),
    ] = None,
    grep_ignore_case: Annotated[
        bool,
        Parameter(name=["-i", "--grep-ignore-case"], group=CAT_FILTER_GROUP),
    ] = False,
    limit: Annotated[
        int | None,
        Parameter(name=["-l", "--limit"], group=CAT_OUTPUT_GROUP, help="Stop after N messages."),
    ] = None,
    duration: Annotated[
        float | None,
        Parameter(
            name=["-d", "--duration"],
            group=CAT_OUTPUT_GROUP,
            help="Stop after this many seconds.",
        ),
    ] = None,
    bytes_mode: Annotated[
        BytesMode,
        Parameter(
            name=["--bytes"],
            group=CAT_OUTPUT_GROUP,
            help=(
                "How to render `bytes` fields. `smart` (default) inlines payloads "
                f"≤{SMART_BYTES_INLINE_LIMIT} bytes as int lists and collapses "
                "larger ones to `<N bytes>`. `ints` for the full list, `base64` "
                "for a string, or `skip` to drop the payload."
            ),
        ),
    ] = BytesMode.SMART,
    connect_timeout: Annotated[
        float,
        Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
    ] = 5.0,
) -> int:
    """Stream live decoded messages from a Foxglove WebSocket bridge to stdout.

    TTY: each message is rendered as a Rich panel with a per-field tree.
    Pipe: each message becomes one JSONL line `{topic, log_time, schema, message}`.
    Use `bridge record` instead if you want an MCAP file.

    Parameters
    ----------
    target
        Bridge address. Same forms accepted by ``bridge`` (URL, host, ``host:port``);
        defaults to port 8765.
    topics
        Topic regex(es) to include (case-sensitive ``re.search``). Empty means all.
    exclude_topics
        Topic regex(es) to skip. Wins over includes and over ``--query``.
    query
        MessagePath expression (e.g. ``/odom.pose.position.x``); restricts output
        to its topic and projects the addressed subfield.
    grep
        Regex applied to scalar values in the decoded message.
    grep_ignore_case
        Make ``--grep`` case-insensitive.
    limit
        Stop after this many messages.
    duration
        Stop after this many seconds.
    bytes_mode
        How to serialise `bytes` fields. See ``--bytes``.
    connect_timeout
        Seconds to wait for the bridge's ``serverInfo`` (default: 5.0).

    Examples
    --------
    ```
    pymcap-cli bridge cat ws://localhost:8765 -t '^/chatter' -l 3
    pymcap-cli bridge cat localhost:8765 -t '.*' -l 1 | jq
    pymcap-cli bridge cat localhost:8765 -q '/odom.pose.position.x' -l 5
    pymcap-cli bridge cat localhost:8765 -t '.*' -g 'error' -i -d 10
    ```
    """
    if duration is not None and duration <= 0:
        ERR.print("[red]Error:[/] --duration must be positive")
        return 1
    if limit is not None and limit <= 0:
        ERR.print("[red]Error:[/] --limit must be positive")
        return 1

    url = to_ws_url(target)

    try:
        return asyncio.run(
            _cat_async(
                url=url,
                topics=topics,
                exclude_topics=exclude_topics,
                query=query,
                grep=grep,
                grep_ignore_case=grep_ignore_case,
                limit=limit,
                duration=duration,
                bytes_mode=bytes_mode,
                connect_timeout=connect_timeout,
            )
        )
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Failed to connect to {url}: {exc}")
        return 1


bridge_app = App(
    name="bridge",
    help="Inspect or record from a live Foxglove WebSocket bridge.",
    help_format="rich",
)
bridge_app.default(inspect)
bridge_app.command(record, name="record")
bridge_app.command(cat, name="cat")
