"""`pymcap-cli bridge params` — list, get, or set parameters on a live Foxglove bridge."""

import asyncio
import logging
from typing import Annotated

from cyclopts import Parameter
from robo_ws_bridge import WebSocketBridgeClient
from robo_ws_bridge.ws_types import Parameter as BridgeParameter
from robo_ws_bridge.ws_types import ServerCapabilities

from pymcap_cli.cmd._cli_options import (
    BridgeTarget,
    CallTimeoutOption,
    ConnectTimeoutOption,
)
from pymcap_cli.cmd.bridge._codec import FieldSyntaxError, parse_field_args
from pymcap_cli.cmd.bridge._shared import (
    BridgeFetchError,
    console,
    to_ws_url,
)
from pymcap_cli.display.param_render import build_parameters_table
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)


async def _params_async(
    url: str,
    names: list[str],
    updates: list[BridgeParameter],
    *,
    connect_timeout: float,
    call_timeout: float,
) -> list[BridgeParameter]:
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
        if ServerCapabilities.PARAMETERS.value not in server_info["capabilities"]:
            raise BridgeFetchError(
                f"Bridge at {url} does not advertise the 'parameters' capability"
            )

        if updates:
            return await client.set_parameters(updates, timeout=call_timeout)
        return await client.get_parameters(names or None, timeout=call_timeout)
    finally:
        await client.disconnect()


def params(
    target: BridgeTarget,
    names: list[str] = [],  # noqa: B006
    *,
    set_values: Annotated[list[str], Parameter(name=["-s", "--set"])] = [],  # noqa: B006
    connect_timeout: ConnectTimeoutOption = 5.0,
    call_timeout: CallTimeoutOption = 5.0,
) -> int:
    """List, get, or set parameters on a live Foxglove WebSocket bridge.

    With no arguments, lists all parameters. Positional ``names`` fetch specific
    parameters. ``--set name:=value`` sets parameters (repeatable) and prints the
    server-confirmed values.

    Parameters
    ----------
    target
        Bridge address. Accepts ``ws://host:port``, ``wss://host:port``, a hostname,
        an IP, or ``host:port`` (default port 8765). Falls back to ``$PYMCAP_BRIDGE``.
    names
        Parameter names to fetch; empty fetches all.
    set_values
        ``name:=value`` assignments to set; each value is parsed as JSON with a string
        fallback. When given, ``names`` is ignored.
    connect_timeout
        Seconds to wait for the bridge's serverInfo before giving up (default: 5.0).
    call_timeout
        Seconds to wait for the parameter response (default: 5.0).

    Examples
    --------
    ```
    pymcap-cli bridge params ws://localhost:8765
    pymcap-cli bridge params 192.168.1.10 /use_sim_time
    pymcap-cli bridge params 192.168.1.10 --set /max_speed:=2.5
    ```
    """
    url = to_ws_url(target)

    updates: list[BridgeParameter] = []
    if set_values:
        try:
            parsed = parse_field_args(set_values)
        except FieldSyntaxError as exc:
            ERR.print(f"[red]Error:[/] {exc}")
            return 1
        updates = [{"name": name, "value": value} for name, value in parsed.items()]

    try:
        result = asyncio.run(
            _params_async(
                url,
                names,
                updates,
                connect_timeout=connect_timeout,
                call_timeout=call_timeout,
            )
        )
    except BridgeFetchError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except asyncio.TimeoutError:
        ERR.print(f"[red]Error:[/] Bridge did not respond within {call_timeout:.1f}s")
        return 1
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Failed to connect to {url}: {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[dim]Interrupted.[/]")
        return 0

    if not result:
        console.print("[yellow]No parameters returned.[/]")
        return 0

    console.print(build_parameters_table(result))
    return 0
