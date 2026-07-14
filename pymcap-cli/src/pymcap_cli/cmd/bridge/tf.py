"""`pymcap-cli bridge tf` — reconstruct the TF frame tree from a live bridge."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from mcap_ros2_support_fast.decoder import DecoderFactory
from robo_ws_bridge import WebSocketBridgeClient
from robo_ws_bridge.ws_types import ChannelInfo

from pymcap_cli.cmd._cli_options import (
    BridgeTarget,
    ConnectTimeoutOption,
    DiscoverSecondsOption,
    StaticOnlyOption,
)
from pymcap_cli.cmd.bridge._shared import (
    BridgeFetchError,
    ChannelSubscriptionManager,
    channel_to_schema,
    console,
    to_ws_url,
)
from pymcap_cli.core.tf_findings import collect_tf_findings, has_error_findings
from pymcap_cli.core.tf_tree import TF_STATIC_TOPIC, TF_TOPIC, TfGraph, add_tf_message
from pymcap_cli.display.tf_render import TF_COMPACT_WIDTH, build_findings_table, build_tf_table
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)


async def _collect_tf_graph_async(
    url: str,
    *,
    static_only: bool,
    connect_timeout: float,
    collect_seconds: float,
) -> TfGraph:
    graph = TfGraph()
    tf_topics = {TF_STATIC_TOPIC} if static_only else {TF_TOPIC, TF_STATIC_TOPIC}

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

    def _on_message(channel: ChannelInfo, _log_time_ns: int, payload: bytes) -> None:
        decoder = _decoder_for(channel)
        if decoder is None:
            return
        try:
            decoded = decoder(payload)
        except Exception:
            logger.exception(f"Failed to decode TF message on {channel['topic']}")
            return
        add_tf_message(graph, channel["topic"], decoded)

    subscriber = ChannelSubscriptionManager(
        client,
        lambda channel: channel["topic"] in tf_topics and _decoder_for(channel) is not None,
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
        return graph
    finally:
        await client.disconnect()


def tf(
    target: BridgeTarget,
    *,
    static_only: StaticOnlyOption = False,
    connect_timeout: ConnectTimeoutOption = 5.0,
    discover_seconds: DiscoverSecondsOption = 2.0,
) -> int:
    """Reconstruct the TF frame tree from a live Foxglove WebSocket bridge.

    Subscribes to ``/tf`` (and ``/tf_static``), listens for ``--discover-seconds``,
    then prints the parent→child frame tree with the latest translation and
    roll/pitch/yaw per edge — the same table as ``pymcap-cli tftree`` for a file.

    Parameters
    ----------
    target
        Bridge address. Accepts ``ws://host:port``, ``wss://host:port``, a hostname,
        an IP, or ``host:port`` (default port 8765). Falls back to ``$PYMCAP_BRIDGE``.
    static_only
        Only subscribe to ``/tf_static``.
    connect_timeout
        Seconds to wait for the bridge's serverInfo before giving up (default: 5.0).
    discover_seconds
        Seconds to accumulate transforms before printing (default: 2.0).

    Examples
    --------
    ```
    pymcap-cli bridge tf ws://localhost:8765
    pymcap-cli bridge tf 192.168.1.10 --static-only
    pymcap-cli bridge tf 192.168.1.10 --discover-seconds 5
    ```
    """
    url = to_ws_url(target)

    try:
        graph = asyncio.run(
            _collect_tf_graph_async(
                url,
                static_only=static_only,
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

    findings = collect_tf_findings(graph)
    table = build_tf_table(graph.transforms, graph.counts, compact=console.width < TF_COMPACT_WIDTH)
    if table is not None:
        console.print(table)
    elif not graph.transforms:
        console.print(f"[yellow]No transforms received from {url} in {discover_seconds:.1f}s.[/]")
        return 0

    if findings:
        console.print()
        console.print(build_findings_table(findings))

    return 1 if has_error_findings(findings) else 0
