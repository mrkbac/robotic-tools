"""`pymcap-cli bridge fetch` — download an asset from a live Foxglove bridge."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from robo_ws_bridge import FetchAssetError, WebSocketBridgeClient
from robo_ws_bridge.ws_types import ServerCapabilities

from pymcap_cli.cmd.bridge._shared import (
    CONNECTION_GROUP,
    BridgeFetchError,
    BridgeTarget,
    console,
    to_ws_url,
)
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)


async def _fetch_async(
    url: str,
    uri: str,
    *,
    connect_timeout: float,
    call_timeout: float,
) -> bytes:
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
        if ServerCapabilities.ASSETS.value not in server_info["capabilities"]:
            raise BridgeFetchError(f"Bridge at {url} does not advertise the 'assets' capability")

        return await client.fetch_asset(uri, timeout=call_timeout)
    finally:
        await client.disconnect()


def fetch(
    target: BridgeTarget,
    uri: str,
    *,
    output: Annotated[Path | None, Parameter(name=["-o", "--output"])] = None,
    connect_timeout: Annotated[
        float,
        Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
    ] = 5.0,
    call_timeout: Annotated[
        float,
        Parameter(name=["--call-timeout"], group=CONNECTION_GROUP),
    ] = 10.0,
) -> int:
    """Download an asset (e.g. a URDF or mesh) from a live Foxglove WebSocket bridge.

    Writes the asset bytes to ``--output`` if given, otherwise to stdout when it is
    piped (refuses to dump binary to a terminal).

    Parameters
    ----------
    target
        Bridge address. Accepts ``ws://host:port``, ``wss://host:port``, a hostname,
        an IP, or ``host:port`` (default port 8765). Falls back to ``$PYMCAP_BRIDGE``.
    uri
        Asset URI to fetch, e.g. ``package://my_robot/urdf/robot.urdf``.
    output
        File to write the asset to. If omitted, bytes go to stdout (piped only).
    connect_timeout
        Seconds to wait for the bridge's serverInfo before giving up (default: 5.0).
    call_timeout
        Seconds to wait for the asset response (default: 10.0).

    Examples
    --------
    ```
    pymcap-cli bridge fetch ws://localhost:8765 package://robot/urdf/robot.urdf -o robot.urdf
    pymcap-cli bridge fetch 192.168.1.10 package://robot/meshes/base.stl > base.stl
    ```
    """
    url = to_ws_url(target)

    if output is None and sys.stdout.isatty():
        ERR.print("[red]Error:[/] refusing to write binary asset to a terminal; use --output")
        return 1

    try:
        data = asyncio.run(
            _fetch_async(url, uri, connect_timeout=connect_timeout, call_timeout=call_timeout)
        )
    except BridgeFetchError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except FetchAssetError as exc:
        ERR.print(f"[red]Fetch failed:[/] {exc}")
        return 1
    except asyncio.TimeoutError:
        ERR.print(f"[red]Error:[/] Asset {uri} was not returned within {call_timeout:.1f}s")
        return 1
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Failed to connect to {url}: {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[dim]Interrupted.[/]")
        return 0

    if output is None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    else:
        output.write_bytes(data)
        console.print(f"[green]Wrote[/] {len(data)} bytes to {output}.")
    return 0
