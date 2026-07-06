"""`pymcap-cli bridge call` — call a service on a live Foxglove WebSocket bridge."""

import asyncio
import dataclasses
import logging
from typing import Annotated

from cyclopts import Parameter
from robo_ws_bridge import ServiceCallError, WebSocketBridgeClient
from robo_ws_bridge.ws_types import ServerCapabilities, ServiceInfo

from pymcap_cli.cmd.bridge._codec import (
    CodecError,
    FieldSyntaxError,
    PayloadValue,
    decode_message,
    encode_message,
    parse_field_args,
)
from pymcap_cli.cmd.bridge._shared import (
    CONNECTION_GROUP,
    BridgeFetchError,
    BridgeTarget,
    console,
    to_ws_url,
)
from pymcap_cli.display.service_render import build_service_response_table
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _MessageSpec:
    """Codec inputs for one side (request/response) of a service."""

    encoding: str
    schema_name: str
    schema_encoding: str
    schema_text: str


def _request_spec(service: ServiceInfo) -> _MessageSpec:
    request = service.get("request")
    if request is not None:
        return _MessageSpec(
            encoding=request["encoding"],
            schema_name=request["schemaName"],
            schema_encoding=request.get("schemaEncoding", "ros2msg"),
            schema_text=request.get("schema", ""),
        )
    legacy = service.get("requestSchema")
    if legacy is not None:
        return _MessageSpec("cdr", f"{service['type']}_Request", "ros2msg", legacy)
    raise BridgeFetchError(f"Service {service['name']} advertises no request schema")


def _response_spec(service: ServiceInfo) -> _MessageSpec:
    response = service.get("response")
    if response is not None:
        return _MessageSpec(
            encoding=response["encoding"],
            schema_name=response["schemaName"],
            schema_encoding=response.get("schemaEncoding", "ros2msg"),
            schema_text=response.get("schema", ""),
        )
    legacy = service.get("responseSchema")
    if legacy is not None:
        return _MessageSpec("cdr", f"{service['type']}_Response", "ros2msg", legacy)
    raise BridgeFetchError(f"Service {service['name']} advertises no response schema")


def _encode_request(service: ServiceInfo, request: dict[str, PayloadValue]) -> tuple[bytes, str]:
    spec = _request_spec(service)
    payload = encode_message(
        encoding=spec.encoding,
        schema_name=spec.schema_name,
        schema_encoding=spec.schema_encoding,
        schema_text=spec.schema_text,
        value=request,
    )
    return payload, spec.encoding


def _decode_response(service: ServiceInfo, payload: bytes) -> dict[str, PayloadValue]:
    spec = _response_spec(service)
    return decode_message(
        encoding=spec.encoding,
        schema_name=spec.schema_name,
        schema_encoding=spec.schema_encoding,
        schema_text=spec.schema_text,
        payload=payload,
    )


async def _resolve_service(
    client: WebSocketBridgeClient, service_name: str, *, discover_seconds: float
) -> ServiceInfo:
    deadline = discover_seconds
    while True:
        for service in client.services.values():
            if service["name"] == service_name:
                return service
        if deadline <= 0:
            break
        step = min(0.1, deadline)
        await asyncio.sleep(step)
        deadline -= step

    available = sorted(service["name"] for service in client.services.values())
    listing = ", ".join(available) if available else "(none advertised)"
    raise BridgeFetchError(f"Service {service_name!r} not found. Available: {listing}")


async def _call_service_async(
    url: str,
    service_name: str,
    request: dict[str, PayloadValue],
    *,
    connect_timeout: float,
    discover_seconds: float,
    call_timeout: float,
) -> tuple[ServiceInfo, dict[str, PayloadValue]]:
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
        if ServerCapabilities.SERVICES.value not in server_info["capabilities"]:
            raise BridgeFetchError(f"Bridge at {url} does not advertise the 'services' capability")

        service = await _resolve_service(client, service_name, discover_seconds=discover_seconds)
        payload, encoding = _encode_request(service, request)
        response = await client.call_service(
            service["id"], payload, encoding=encoding, timeout=call_timeout
        )
        return service, _decode_response(service, response.payload)
    finally:
        await client.disconnect()


def call(
    target: BridgeTarget,
    service: str,
    fields: list[str] = [],  # noqa: B006
    *,
    connect_timeout: Annotated[
        float,
        Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
    ] = 5.0,
    discover_seconds: Annotated[
        float,
        Parameter(name=["--discover-seconds"], group=CONNECTION_GROUP),
    ] = 2.0,
    call_timeout: Annotated[
        float,
        Parameter(name=["--call-timeout"], group=CONNECTION_GROUP),
    ] = 10.0,
) -> int:
    """Call a service advertised by a live Foxglove WebSocket bridge.

    Resolves ``service`` by name from the bridge's advertised services, encodes the
    request from ``field:=value`` arguments, calls the service, and prints the decoded
    response fields.

    Parameters
    ----------
    target
        Bridge address. Accepts ``ws://host:port``, ``wss://host:port``, a hostname,
        an IP, or ``host:port`` (default port 8765). Falls back to ``$PYMCAP_BRIDGE``.
    service
        Advertised service name, e.g. ``/set_bool``.
    fields
        Request fields as ``field:=value`` tokens; each value is parsed as JSON with a
        string fallback. Omit for services with an empty request.
    connect_timeout
        Seconds to wait for the bridge's serverInfo before giving up (default: 5.0).
    discover_seconds
        Seconds to wait for the service advertisement before giving up (default: 2.0).
    call_timeout
        Seconds to wait for the service response (default: 10.0).

    Examples
    --------
    ```
    pymcap-cli bridge call ws://localhost:8765 /set_bool data:=true
    pymcap-cli bridge call 192.168.1.10 /reset
    pymcap-cli bridge call 192.168.1.10 /set_pose pose:='{"x": 1.0, "y": 2.0}'
    ```
    """
    url = to_ws_url(target)

    try:
        request = parse_field_args(fields)
    except FieldSyntaxError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1

    try:
        resolved_service, response = asyncio.run(
            _call_service_async(
                url,
                service,
                request,
                connect_timeout=connect_timeout,
                discover_seconds=discover_seconds,
                call_timeout=call_timeout,
            )
        )
    except (BridgeFetchError, CodecError) as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except ServiceCallError as exc:
        ERR.print(f"[red]Service call failed:[/] {exc}")
        return 1
    except asyncio.TimeoutError:
        ERR.print(f"[red]Error:[/] Service {service} did not respond within {call_timeout:.1f}s")
        return 1
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Failed to connect to {url}: {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[dim]Interrupted.[/]")
        return 0

    if not response:
        console.print(f"[green]{resolved_service['name']}[/] returned an empty response.")
        return 0

    console.print(build_service_response_table(resolved_service, response))
    return 0
