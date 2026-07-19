"""`pymcap-cli bridge pub` — publish a message to a topic on a live Foxglove bridge."""

import asyncio
import logging
from typing import Annotated

from cyclopts import Parameter
from robo_ws_bridge import WebSocketBridgeClient
from robo_ws_bridge.ws_types import ChannelInfo, ServerCapabilities

from pymcap_cli.cmd._cli_options import (
    BridgeTarget,
    ConnectTimeoutOption,
    DiscoverSecondsOption,
)
from pymcap_cli.cmd.bridge._codec import (
    CodecError,
    FieldSyntaxError,
    PayloadValue,
    encode_message,
    parse_field_args,
)
from pymcap_cli.cmd.bridge._shared import (
    BridgeFetchError,
    console,
    to_ws_url,
)
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)


async def _find_channel(
    client: WebSocketBridgeClient, topic: str, *, discover_seconds: float
) -> ChannelInfo:
    deadline = discover_seconds
    while True:
        for channel in client.channels.values():
            if channel["topic"] == topic:
                return channel
        if deadline <= 0:
            break
        step = min(0.1, deadline)
        await asyncio.sleep(step)
        deadline -= step

    available = sorted(channel["topic"] for channel in client.channels.values())
    listing = ", ".join(available) if available else "(none advertised)"
    raise BridgeFetchError(
        f"Topic {topic!r} not advertised, so its schema can't be inferred. Available: {listing}"
    )


async def _pub_async(
    url: str,
    topic: str,
    fields: dict[str, PayloadValue],
    *,
    count: int,
    rate: float,
    connect_timeout: float,
    discover_seconds: float,
) -> int:
    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
    server_info_event = asyncio.Event()
    client.on_server_info(lambda *_: server_info_event.set())

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
        if ServerCapabilities.CLIENT_PUBLISH.value not in server_info["capabilities"]:
            raise BridgeFetchError(
                f"Bridge at {url} does not advertise the 'clientPublish' capability"
            )

        channel = await _find_channel(client, topic, discover_seconds=discover_seconds)
        payload = encode_message(
            encoding=channel["encoding"],
            schema_name=channel.get("schemaName", ""),
            schema_encoding=channel.get("schemaEncoding", "ros2msg"),
            schema_text=channel.get("schema", ""),
            value=fields,
        )

        channel_id = await client.advertise(
            topic,
            encoding=channel["encoding"],
            schema_name=channel.get("schemaName", ""),
            schema=channel.get("schema", ""),
            schema_encoding=channel.get("schemaEncoding"),
        )
        try:
            interval = 1.0 / rate if rate > 0 else 0.0
            for i in range(count):
                await client.publish(channel_id, payload)
                if interval and i + 1 < count:
                    await asyncio.sleep(interval)
        finally:
            await client.unadvertise(channel_id)
        return count
    finally:
        await client.disconnect()


def pub(
    target: BridgeTarget,
    topic: str,
    fields: list[str] = [],  # noqa: B006
    *,
    count: Annotated[int, Parameter(name=["-c", "--count"])] = 1,
    rate: Annotated[float, Parameter(name=["-r", "--rate"])] = 0.0,
    connect_timeout: ConnectTimeoutOption = 5.0,
    discover_seconds: DiscoverSecondsOption = 2.0,
) -> int:
    """Publish a message to a topic on a live Foxglove WebSocket bridge.

    The message schema is borrowed from the bridge's existing advertisement for
    ``topic``, so the topic must already be advertised. The message is built from
    ``field:=value`` arguments (JSON values with a string fallback).

    Parameters
    ----------
    target
        Bridge address. Accepts ``ws://host:port``, ``wss://host:port``, a hostname,
        an IP, or ``host:port`` (default port 8765). Falls back to ``$PYMCAP_BRIDGE``.
    topic
        Topic to publish to; must be advertised by the bridge.
    fields
        Message fields as ``field:=value`` tokens; each value is parsed as JSON with a
        string fallback.
    count
        Number of messages to publish (default: 1).
    rate
        Publish rate in Hz when ``--count`` > 1 (default: 0 = as fast as possible).
    connect_timeout
        Seconds to wait for the bridge's serverInfo before giving up (default: 5.0).
    discover_seconds
        Seconds to wait for the topic advertisement before giving up (default: 2.0).

    Examples
    --------
    ```
    pymcap-cli bridge pub ws://localhost:8765 /enabled data:=true
    pymcap-cli bridge pub 192.168.1.10 /cmd linear:='{"x": 0.5}' --count 10 --rate 5
    ```
    """
    url = to_ws_url(target)

    if count <= 0:
        ERR.print("[red]Error:[/] --count must be positive")
        return 1
    if rate < 0:
        ERR.print("[red]Error:[/] --rate must not be negative")
        return 1
    if rate > 0 and count <= 1:
        ERR.print("[red]Error:[/] --rate has no effect unless --count is greater than 1")
        return 1

    try:
        message = parse_field_args(fields)
    except FieldSyntaxError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1

    try:
        published = asyncio.run(
            _pub_async(
                url,
                topic,
                message,
                count=count,
                rate=rate,
                connect_timeout=connect_timeout,
                discover_seconds=discover_seconds,
            )
        )
    except (BridgeFetchError, CodecError) as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Failed to connect to {url}: {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[dim]Interrupted.[/]")
        return 0

    plural = "s" if published != 1 else ""
    console.print(f"[green]Published[/] {published} message{plural} to {topic}.")
    return 0
