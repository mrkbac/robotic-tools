"""`pymcap-cli bridge diag` — live ROS2 diagnostics overview from a bridge."""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Annotated, Any

from cyclopts import Group as CycloptsGroup
from cyclopts import Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from robo_ws_bridge import WebSocketBridgeClient
from robo_ws_bridge.ws_types import ChannelInfo

from pymcap_cli.cmd.bridge._shared import (
    CONNECTION_GROUP,
    DISPLAY_GROUP,
    BridgeFetchError,
    BridgeTarget,
    ChannelSubscriptionManager,
    channel_to_schema,
    console,
    to_ws_url,
)
from pymcap_cli.core.diagnostics import (
    DEFAULT_TOPICS,
    DiagEntry,
    add_diagnostic_message,
    filter_entries,
    level_totals,
)
from pymcap_cli.display.diag_render import (
    build_inspect_view,
    build_json_output,
    build_summary_table,
    build_tree_view,
)
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)

FILTER_GROUP = CycloptsGroup("Filtering")


def _compile_pattern(pattern: str, flag_name: str) -> re.Pattern[str] | None:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        logger.exception(f"Invalid regex for {flag_name}")
        return None


async def _collect_diagnostics_async(
    url: str,
    *,
    topics: list[str],
    connect_timeout: float,
    collect_seconds: float,
) -> dict[str, DiagEntry]:
    entries: dict[str, DiagEntry] = {}
    wanted = set(topics)

    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
    server_info_event = asyncio.Event()
    client.on_server_info(lambda *_: server_info_event.set())

    factory = DecoderFactory()
    decoders: dict[int, Callable[[bytes | memoryview], Any] | None] = {}

    def _decoder_for(channel: ChannelInfo) -> Callable[[bytes | memoryview], Any] | None:
        cid = channel["id"]
        if cid in decoders:
            return decoders[cid]
        try:
            decoder = factory.decoder_for(channel["encoding"], channel_to_schema(channel))
        except Exception:
            logger.exception(f"Decoder construction failed for {channel['topic']}")
            decoder = None
        decoders[cid] = decoder
        return decoder

    def _on_message(channel: ChannelInfo, log_time_ns: int, payload: bytes) -> None:
        decoder = _decoder_for(channel)
        if decoder is None:
            return
        try:
            decoded = decoder(payload)
        except Exception:
            logger.exception(f"Failed to decode diagnostics on {channel['topic']}")
            return
        add_diagnostic_message(entries, log_time_ns, decoded)

    subscriber = ChannelSubscriptionManager(
        client,
        lambda channel: channel["topic"] in wanted and _decoder_for(channel) is not None,
    )
    subscriber.install()
    client.on_message(_on_message)

    await client.connect()
    try:
        try:
            await asyncio.wait_for(server_info_event.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError as exc:
            raise BridgeFetchError(
                f"Timed out after {connect_timeout:.1f}s waiting for serverInfo from {url}"
            ) from exc

        await subscriber.subscribe_existing()

        await asyncio.sleep(collect_seconds)
        return entries
    finally:
        await client.disconnect()


def diag(
    target: BridgeTarget,
    *,
    level: Annotated[
        int | None,
        Parameter(name=["-l", "--level"], group=FILTER_GROUP),
    ] = None,
    show_all: Annotated[
        bool,
        Parameter(name=["-a", "--all"], group=FILTER_GROUP),
    ] = False,
    name: Annotated[
        str | None,
        Parameter(name=["-n", "--name"], group=FILTER_GROUP),
    ] = None,
    hardware_id: Annotated[
        str | None,
        Parameter(name=["--hardware-id", "--hw"], group=FILTER_GROUP),
    ] = None,
    inspect: Annotated[
        str | None,
        Parameter(name=["-i", "--inspect"], group=DISPLAY_GROUP),
    ] = None,
    inspect_all: Annotated[
        bool,
        Parameter(name=["-I", "--inspect-all"], group=DISPLAY_GROUP),
    ] = False,
    tree: Annotated[
        bool,
        Parameter(name=["--tree"], group=DISPLAY_GROUP),
    ] = False,
    json_output: Annotated[
        bool,
        Parameter(name=["--json"], group=DISPLAY_GROUP),
    ] = False,
    topics: Annotated[
        list[str] | None,
        Parameter(name=["-t", "--topics"], group=FILTER_GROUP),
    ] = None,
    connect_timeout: Annotated[
        float,
        Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
    ] = 5.0,
    discover_seconds: Annotated[
        float,
        Parameter(name=["--discover-seconds"], group=CONNECTION_GROUP),
    ] = 3.0,
) -> int:
    """Live ROS2 diagnostics overview from a Foxglove WebSocket bridge.

    Subscribes to ``/diagnostics`` (and ``/diagnostics_agg``), listens for
    ``--discover-seconds``, then prints the same health overview as
    ``pymcap-cli diag`` for a file. By default shows only components with issues.

    Parameters
    ----------
    target
        Bridge address (URL, host, or ``host:port``); falls back to ``$PYMCAP_BRIDGE``.
    level
        Minimum level to show (0=OK, 1=WARN, 2=ERROR, 3=STALE). Default: 1.
    show_all
        Show all components including OK.
    name
        Regex filter on component name (case-insensitive).
    hardware_id
        Regex filter on hardware ID (case-insensitive).
    inspect
        Show detailed view for components matching this regex.
    inspect_all
        Show detailed view for all components.
    tree
        Display as a hierarchical tree grouped by hardware ID.
    json_output
        Output as JSON for scripting.
    topics
        Diagnostics topic names. Defaults to /diagnostics and /diagnostics_agg.
    connect_timeout
        Seconds to wait for the bridge's serverInfo before giving up (default: 5.0).
    discover_seconds
        Seconds to accumulate diagnostics before printing (default: 3.0).

    Examples
    --------
    ```
    pymcap-cli bridge diag ws://localhost:8765
    pymcap-cli bridge diag 192.168.1.10 --all
    pymcap-cli bridge diag 192.168.1.10 --inspect encoder --discover-seconds 5
    ```
    """
    resolved_topics = topics if topics is not None else DEFAULT_TOPICS

    name_re = _compile_pattern(name, "--name") if name else None
    if name and name_re is None:
        return 1
    hw_re = _compile_pattern(hardware_id, "--hardware-id") if hardware_id else None
    if hardware_id and hw_re is None:
        return 1
    if inspect_all:
        inspect = "."
    inspect_re = _compile_pattern(inspect, "--inspect") if inspect else None
    if inspect and inspect_re is None:
        return 1

    url = to_ws_url(target)

    try:
        entries = asyncio.run(
            _collect_diagnostics_async(
                url,
                topics=resolved_topics,
                connect_timeout=connect_timeout,
                collect_seconds=discover_seconds,
            )
        )
    except BridgeFetchError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Failed to connect to {url}: {exc}")
        return 1

    if not entries:
        console.print(
            f"[yellow]No diagnostics received from {url} in {discover_seconds:.1f}s "
            f"(topics: {', '.join(resolved_topics)}).[/]"
        )
        return 0

    totals = level_totals(entries)
    min_level = 0 if (show_all or inspect_re) else (level if level is not None else 1)
    filtered = filter_entries(entries, min_level=min_level, name_pattern=name_re, hw_pattern=hw_re)

    if json_output:
        print(json.dumps(build_json_output(filtered, len(entries), totals), indent=2))  # noqa: T201
        return 0

    if inspect_re:
        matched = [e for e in filtered if inspect_re.search(e.name)]
        if not matched:
            logger.warning(f"No components matching '{inspect}'")
            return 0
        for renderable in build_inspect_view(matched):
            console.print(renderable)
        return 0

    if tree:
        console.print(build_tree_view(filtered))
        return 0

    console.print(build_summary_table(filtered, len(entries), totals, show_all))
    return 0
