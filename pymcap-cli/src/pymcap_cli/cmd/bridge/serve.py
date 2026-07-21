"""`pymcap-cli bridge serve` — host MCAP playback over Foxglove WebSocket."""

import asyncio
import os
import socket
import sys
import threading
import webbrowser
from contextlib import suppress
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from robo_ws_bridge import (
    ConnectionState,
    ServerConnection,
    WebSocketBridgeEndpoint,
    WebSocketBridgeServer,
)
from robo_ws_bridge.server import Channel as ServerChannel

from pymcap_cli.cmd._cli_options import (
    SERVER_GROUP,
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    NoBrowserOption,
    OptionalBackendOption,
    OptionalCodecOption,
    OptionalPointCloudDropInvalidOption,
    OptionalPointCloudSortFieldOption,
    ServerPortOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.cmd.bridge._playback import (
    PlaybackChannel,
    PlaybackClock,
    PlaybackError,
    is_frame_channel,
)
from pymcap_cli.cmd.bridge._playback_transforms import (
    OptionalAdaptiveQualityOption,
    OptionalDracoCompressionLevelOption,
    OptionalEncoderOption,
    OptionalImageFormatOption,
    OptionalJpegQualityOption,
    OptionalPointCloudCompressionOption,
    OptionalPointCloudEncodingOption,
    OptionalPointCloudFormatOption,
    OptionalPointCloudOption,
    OptionalPointCloudSchemaOption,
    OptionalPresetOption,
    OptionalQualityOption,
    OptionalResolutionOption,
    OptionalScaleOption,
    OptionalVideoFormatOption,
    OptionalVideoOption,
    apply_preset,
    resolve_playback_transform_config,
)
from pymcap_cli.cmd.bridge._shared import console
from pymcap_cli.cmd.bridge.play import LoopOption, SpeedOption
from pymcap_cli.core.rosbag2_layout import find_bag_splits
from pymcap_cli.log_setup import ERR

_CLOCK_INTERVAL_SECONDS = 1 / 30
_CLIENT_SETTLE_SECONDS = 0.05

# A list-valued ``--host`` so the flag may be given bare (``--host`` -> bind all
# interfaces, like ``vite --host``), with a value (``--host 0.0.0.0``), or omitted.
ServeHostOption = Annotated[
    list[str] | None,
    Parameter(
        name=["--host"],
        consume_multiple=True,
        group=SERVER_GROUP,
        help=(
            "Interface to bind. Bare --host (or 0.0.0.0) binds every interface "
            "and lists each reachable URL. Default: 127.0.0.1."
        ),
    ),
]


def _resolve_host(host: str | list[str] | None) -> str:
    if host is None:
        return "127.0.0.1"
    if isinstance(host, str):
        return host  # direct callers may pass a plain host string
    if not host:
        return ""  # bare --host binds all interfaces
    if len(host) == 1:
        return host[0]
    raise ValueError("--host accepts at most one address")


def _library_root(files: list[str]) -> Path | None:
    if len(files) != 1:
        return None
    candidate = Path(files[0])
    if not candidate.is_dir() or find_bag_splits(candidate):
        return None
    return candidate


def _bind_host(host: str) -> str:
    """An empty ``--host`` binds every interface, like ``vite --host``."""
    return "0.0.0.0" if host == "" else host  # noqa: S104


def _binds_all_interfaces(host: str) -> bool:
    return host in {"", "0.0.0.0", "::"}  # noqa: S104


def _url_host(host: str) -> str:
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host  # noqa: S104
    if ":" in client_host and not client_host.startswith("["):
        return f"[{client_host}]"
    return client_host


def _probe_route(family: socket.AddressFamily, target: tuple[str, int]) -> str | None:
    """Primary source address for reaching ``target`` (no packets are sent)."""
    with suppress(OSError), socket.socket(family, socket.SOCK_DGRAM) as probe:
        probe.connect(target)
        address = probe.getsockname()[0]
        if isinstance(address, str):
            return address
    return None


def _lan_ip_addresses() -> list[str]:
    """Best-effort routable IPv4 addresses of this host, loopback aside.

    Dependency-free: probes the default IPv4 route and the Tailscale/CGNAT
    range (100.64.0.0/10, reached via Tailscale's 100.100.100.100), then
    resolves the hostname. A machine on Tailscale (or any 100.64/10 CGNAT
    address) is picked up; hosts without a route there just re-probe the
    default route and dedupe. Interfaces on none of those may be missed —
    pass their address to ``--host`` explicitly.
    """
    addresses: set[str] = set()
    # Targets are never sent to; connect() only selects the source interface.
    for target in (("192.0.2.0", 9), ("100.100.100.100", 9)):
        address = _probe_route(socket.AF_INET, target)
        if address is not None:
            addresses.add(address)
    with suppress(OSError):
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if isinstance(address, str):
                addresses.add(address)
    return sorted(ip for ip in addresses if not ip.startswith("127."))


def _display_hosts(host: str) -> list[str]:
    """Client-reachable hostnames to advertise for a bound ``--host``."""
    if not _binds_all_interfaces(host):
        return [_url_host(host)]
    return ["localhost", *_lan_ip_addresses()]


def _is_graphical_session() -> bool:
    if sys.platform in {"darwin", "win32", "cygwin"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _launch_url(url: str) -> None:
    if not _is_graphical_session():
        # webbrowser would fall back to a terminal browser and hijack the server terminal.
        console.print(f"[dim]No graphical session; open[/] {url} [dim]from your desktop.[/]")
        return
    threading.Thread(
        target=webbrowser.open,
        args=(url,),
        daemon=True,
    ).start()


class BridgeServerPlaybackSink:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        endpoint: WebSocketBridgeEndpoint | None = None,
        url: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.url = url or f"ws://{host}:{port}"
        self.server = endpoint
        self._owns_server = endpoint is None
        self._is_started = False
        self.channel_ids: dict[PlaybackChannel, int] = {}
        self._subscriptions: set[tuple[ServerConnection, int, int]] = set()
        self._has_connection = asyncio.Event()
        self.started = asyncio.Event()
        self._clock_done = asyncio.Event()
        self._clock: PlaybackClock | None = None
        self._clock_task: asyncio.Task[None] | None = None

    async def start(self, channels: tuple[PlaybackChannel, ...]) -> None:
        if self._is_started:
            return
        encodings = sorted({channel.message_encoding for channel in channels})
        server = self.server
        if server is None:
            server = WebSocketBridgeServer(
                host=self.host,
                port=self.port,
                name="pymcap-cli bridge serve",
                capabilities=["time"],
                supported_encodings=encodings,
                metadata={"source": "pymcap-cli"},
            )
        self.server = server
        for channel_id, channel in enumerate(channels, start=1):
            self.channel_ids[channel] = channel_id
            server.register_channel(
                ServerChannel(
                    id=channel_id,
                    topic=channel.topic,
                    encoding=channel.message_encoding,
                    schema_name=channel.schema_name,
                    schema=channel.schema_text,
                    schema_encoding=channel.schema_encoding or None,
                    delivery="latest" if is_frame_channel(channel) else "reliable",
                )
            )

        def on_subscribe(state: ConnectionState, subscription_id: int, channel_id: int) -> None:
            websocket = state.websocket
            self._subscriptions.add((websocket, subscription_id, channel_id))

        def on_connect(_state: ConnectionState) -> None:
            self._has_connection.set()

        def on_unsubscribe(state: ConnectionState, subscription_id: int, channel_id: int) -> None:
            websocket = state.websocket
            self._subscriptions.discard((websocket, subscription_id, channel_id))

        def on_disconnect(state: ConnectionState) -> None:
            websocket = state.websocket
            self._subscriptions = {
                entry for entry in self._subscriptions if entry[0] is not websocket
            }
            if not server.connections:
                self._has_connection.clear()

        server.on_connect(on_connect)
        server.on_subscribe(on_subscribe)
        server.on_unsubscribe(on_unsubscribe)
        server.on_disconnect(on_disconnect)
        if self._owns_server:
            assert isinstance(server, WebSocketBridgeServer)
            await server.start()
        self._is_started = True
        self.started.set()
        console.print(f"[green]Serving playback at[/] [bold]{self.url}[/]")

    async def wait_until_ready(self) -> None:
        while True:
            await self._has_connection.wait()
            await asyncio.sleep(_CLIENT_SETTLE_SECONDS)
            server = self.server
            assert server is not None
            if server.connections:
                return

    async def publish(
        self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
    ) -> None:
        server = self.server
        assert server is not None
        await server.publish_message(
            self.channel_ids[channel], bytes(payload), timestamp_ns=timestamp_ns
        )

    async def timeline_started(self, clock: PlaybackClock) -> None:
        server = self.server
        assert server is not None
        self._clock = clock
        self._clock_done.clear()
        await server.publish_time(clock.record_origin_ns)

        async def publish_clock() -> None:
            while not self._clock_done.is_set():
                await server.publish_time(clock.current_time_ns())
                try:
                    await asyncio.wait_for(self._clock_done.wait(), timeout=_CLOCK_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    continue

        self._clock_task = asyncio.create_task(publish_clock())

    async def timeline_finished(self, timestamp_ns: int) -> None:
        self._clock_done.set()
        if self._clock_task is not None:
            await self._clock_task
            self._clock_task = None
        self._clock = None
        server = self.server
        assert server is not None
        await server.publish_time(timestamp_ns)

    @property
    def current_time_ns(self) -> int | None:
        """Current playback clock time, or ``None`` outside an active timeline."""
        return None if self._clock is None else self._clock.current_time_ns()

    async def close(self) -> None:
        self._clock_done.set()
        if self._clock_task is not None:
            await self._clock_task
            self._clock_task = None
        self._clock = None
        if self._owns_server and isinstance(self.server, WebSocketBridgeServer):
            await self.server.stop()

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        connections = 0 if self.server is None else len(self.server.connections)
        network_drops = 0 if self.server is None else self.server.dropped_frames
        return (
            ("Server", self.url),
            ("Clients", str(connections)),
            ("Subscriptions", str(len(self._subscriptions))),
            ("Dropped (network)", str(network_drops)),
        )

    def is_channel_active(self, channel: PlaybackChannel) -> bool:
        channel_id = self.channel_ids[channel]
        return any(
            active_channel_id == channel_id for _, _, active_channel_id in self._subscriptions
        )

    def is_channel_congested(self, channel: PlaybackChannel) -> bool:
        server = self.server
        if server is None:
            return False
        return server.are_all_subscribers_busy(self.channel_ids[channel])

    async def wait_until_active(self) -> float:
        return 0.0


def serve(
    files: list[str],
    *,
    host: ServeHostOption = None,
    port: ServerPortOption = 8765,
    preset: OptionalPresetOption = None,
    image_format: OptionalImageFormatOption = None,
    codec: OptionalCodecOption = None,
    quality: OptionalQualityOption = None,
    adaptive_quality: OptionalAdaptiveQualityOption = None,
    encoder: OptionalEncoderOption = None,
    backend: OptionalBackendOption = None,
    scale: OptionalScaleOption = None,
    jpeg_quality: OptionalJpegQualityOption = None,
    video: OptionalVideoOption = None,
    video_format: OptionalVideoFormatOption = None,
    pointcloud: OptionalPointCloudOption = None,
    resolution: OptionalResolutionOption = None,
    pc_format: OptionalPointCloudFormatOption = None,
    pc_schema: OptionalPointCloudSchemaOption = None,
    pc_encoding: OptionalPointCloudEncodingOption = None,
    pc_compression: OptionalPointCloudCompressionOption = None,
    draco_compression_level: OptionalDracoCompressionLevelOption = None,
    pointcloud_drop_invalid: OptionalPointCloudDropInvalidOption = None,
    pointcloud_sort_field: OptionalPointCloudSortFieldOption = None,
    speed: SpeedOption = 1.0,
    loop: LoopOption = True,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    no_browser: NoBrowserOption = False,
) -> int:
    """Host MCAP files or a recording directory for Foxglove clients.

    Every input opens through the same recording browser. Each Foxglove connection
    gets an independent playback timeline and accepts repeated ``file=`` query
    parameters at ``/ws``.

    Examples
    --------
    ```
    pymcap-cli bridge serve recording.mcap
    pymcap-cli bridge serve recording.mcap --preset fast
    pymcap-cli bridge serve part1.mcap part2.mcap --speed 2 --loop
    pymcap-cli bridge serve /data/recordings --port 9090
    pymcap-cli bridge serve recording.mcap --host 0.0.0.0 --port 8765
    ```
    """
    if not 1 <= port <= 65535:
        ERR.print("[red]Error:[/] --port must be in [1, 65535]")
        return 1
    try:
        resolved_host = _resolve_host(host)
        image_format, scale = apply_preset(
            preset,
            image_format=image_format,
            scale=scale,
        )
        resolved_adaptive_quality = adaptive_quality
        if (
            resolved_adaptive_quality is None
            and preset in {"compress", "fast", "low"}
            and image_format in {None, "video"}
        ):
            resolved_adaptive_quality = True
        transform_config = resolve_playback_transform_config(
            preset=preset,
            adaptive_quality=resolved_adaptive_quality,
            image_format=image_format,
            codec=codec,
            quality=quality,
            encoder=encoder,
            backend=backend,
            scale=scale,
            jpeg_quality=jpeg_quality,
            video=video,
            video_format=video_format,
            pointcloud=pointcloud,
            resolution=resolution,
            pc_format=pc_format,
            pc_schema=pc_schema,
            pc_encoding=pc_encoding,
            pc_compression=pc_compression,
            draco_compression_level=draco_compression_level,
            pointcloud_drop_invalid=pointcloud_drop_invalid,
            pointcloud_sort_field=pointcloud_sort_field,
        )
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
        from pymcap_cli.cmd.bridge._library import (  # noqa: PLC0415
            RecordingLibrary,
            RecordingLibraryServer,
        )

        library_root = _library_root(files)
        library = (
            RecordingLibrary(library_root)
            if library_root is not None
            else RecordingLibrary.from_paths(tuple(Path(file) for file in files))
        )
        library_server = RecordingLibraryServer(
            library,
            host=_bind_host(resolved_host),
            port=port,
            message_filter=message_filter,
            transform_config=transform_config,
            speed=speed,
            loop=loop,
        )

        async def run_library() -> None:
            await library_server.start()
            hosts = _display_hosts(resolved_host)
            urls = [f"http://{h}:{library_server.port}/" for h in hosts]
            console.print("[green]Serving recording library at[/]")
            for url in urls:
                console.print(f"  [bold]{url}[/]")
            if resolved_host not in {"127.0.0.1", "::1", "localhost"}:
                console.print(
                    "[yellow]Warning:[/] Foxglove playback sessions have no authentication; "
                    "protect remote access with a trusted network or reverse proxy."
                )
            if not no_browser:
                _launch_url(urls[0])
            try:
                await library_server.serve_forever()
            finally:
                await library_server.stop()

        asyncio.run(run_library())
    except ImportError as exc:
        missing = exc.name or str(exc)
        ERR.print(
            "[red]Error:[/] JIT ROS transform dependencies are missing. "
            "Install the required pymcap-cli video, pointcloud, image, or draco extra.\n"
            f"Missing: {missing}"
        )
        return 1
    except (PlaybackError, OSError, ValueError) as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/]")
        return 0
    return 0
