"""`pymcap-cli bridge inspect` — server snapshot and live `--watch`."""

import asyncio
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Annotated

from cyclopts import Parameter
from rich.console import Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from robo_ws_bridge import ConnectionGraph, WebSocketBridgeClient
from robo_ws_bridge.ws_types import ChannelInfo, ServerCapabilities, ServiceInfo

from pymcap_cli.cmd.bridge._shared import (
    _STATUS_LEVEL_LABELS,
    _STATUS_LEVEL_STYLES,
    CONNECTION_GROUP,
    DISPLAY_GROUP,
    BridgeFetchError,
    BridgeInfo,
    BridgeStatus,
    SortChoice,
    _append_status,
    _output_json,
    _remove_statuses,
    _sort_channels,
    console,
    fetch_bridge_info,
    to_ws_url,
)
from pymcap_cli.display.display_utils import _format_parts_with_colors, _format_schema_with_link
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)


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


def _format_node_ids(node_ids: list[str]) -> str:
    if not node_ids:
        return ""
    return "\n".join(_format_parts_with_colors(node_id) for node_id in sorted(node_ids))


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
