"""`pymcap-cli bridge serve` — host MCAP playback over Foxglove WebSocket."""

import asyncio
import os
import sys
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode

from robo_ws_bridge import (
    ConnectionState,
    ServerConnection,
    WebSocketBridgeEndpoint,
    WebSocketBridgeServer,
)
from robo_ws_bridge.server import Channel as ServerChannel

from pymcap_cli.cmd._cli_options import (
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    NoBrowserOption,
    OptionalBackendOption,
    OptionalCodecOption,
    OptionalPointCloudDropInvalidOption,
    OptionalPointCloudSortFieldOption,
    ProgressOption,
    ServerHostOption,
    ServerPortOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.cmd.bridge._playback import (
    _SETTLE_SECONDS,
    PlaybackChannel,
    PlaybackClock,
    PlaybackError,
    PlaybackStats,
    prepare_playback,
    run_playback,
)
from pymcap_cli.cmd.bridge._playback_transforms import (
    OptionalDracoCompressionLevelOption,
    OptionalEncoderOption,
    OptionalImageFormatOption,
    OptionalJpegQualityOption,
    OptionalPointCloudCompressionOption,
    OptionalPointCloudEncodingOption,
    OptionalPointCloudFormatOption,
    OptionalPointCloudOption,
    OptionalPointCloudSchemaOption,
    OptionalQualityOption,
    OptionalResolutionOption,
    OptionalScaleOption,
    OptionalVideoFormatOption,
    OptionalVideoOption,
    TransformModeOption,
    create_playback_transform_plan,
    resolve_playback_transform_config,
)
from pymcap_cli.cmd.bridge._shared import console
from pymcap_cli.cmd.bridge.play import LoopOption, SpeedOption
from pymcap_cli.core.rosbag2_layout import find_bag_splits
from pymcap_cli.log_setup import ERR

_CLOCK_INTERVAL_SECONDS = 1 / 30


def _library_root(files: list[str]) -> Path | None:
    if len(files) != 1:
        return None
    candidate = Path(files[0])
    if not candidate.is_dir() or find_bag_splits(candidate):
        return None
    return candidate


def _url_host(host: str) -> str:
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host  # noqa: S104
    if ":" in client_host and not client_host.startswith("["):
        return f"[{client_host}]"
    return client_host


def _foxglove_url(host: str, port: int) -> str:
    websocket_url = f"ws://{_url_host(host)}:{port}"
    query = urlencode(
        {
            "ds": "foxglove-websocket",
            "ds.url": websocket_url,
        }
    )
    return f"foxglove://open?{query}"


def _is_graphical_session() -> bool:
    if sys.platform in {"darwin", "win32", "cygwin"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _launch_url(url: str) -> None:
    if not _is_graphical_session():
        # webbrowser would fall back to a terminal browser (w3m, lynx, ...),
        # which hijacks the terminal and cannot open foxglove:// links.
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
        self._has_subscription = asyncio.Event()
        self._activity_handlers: list[Callable[[bool], None]] = []
        self._inactive_since: float | None = None
        self.started = asyncio.Event()
        self._clock_done = asyncio.Event()
        self._timeline_active = asyncio.Event()
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
                )
            )

        def on_subscribe(state: ConnectionState, subscription_id: int, channel_id: int) -> None:
            was_active = bool(self._subscriptions)
            websocket = state.websocket
            self._subscriptions.add((websocket, subscription_id, channel_id))
            self._update_subscription_activity(was_active)

        def on_unsubscribe(state: ConnectionState, subscription_id: int, channel_id: int) -> None:
            was_active = bool(self._subscriptions)
            websocket = state.websocket
            self._subscriptions.discard((websocket, subscription_id, channel_id))
            self._update_subscription_activity(was_active)

        def on_disconnect(state: ConnectionState) -> None:
            was_active = bool(self._subscriptions)
            websocket = state.websocket
            self._subscriptions = {
                entry for entry in self._subscriptions if entry[0] is not websocket
            }
            self._update_subscription_activity(was_active)

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
        console.print("[dim]Waiting for a client subscription...[/]")
        while True:
            await self._has_subscription.wait()
            await asyncio.sleep(_SETTLE_SECONDS)
            if self._subscriptions:
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
        if self._has_subscription.is_set():
            self._timeline_active.set()
            await server.publish_time(clock.record_origin_ns)

        async def publish_clock() -> None:
            while not self._clock_done.is_set():
                await self._timeline_active.wait()
                if self._clock_done.is_set():
                    break
                await server.publish_time(clock.current_time_ns())
                try:
                    await asyncio.wait_for(self._clock_done.wait(), timeout=_CLOCK_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    continue

        self._clock_task = asyncio.create_task(publish_clock())

    async def timeline_finished(self, timestamp_ns: int) -> None:
        self._clock_done.set()
        self._timeline_active.set()
        if self._clock_task is not None:
            await self._clock_task
            self._clock_task = None
        self._clock = None
        self._timeline_active.clear()
        server = self.server
        assert server is not None
        if self._has_subscription.is_set():
            await server.publish_time(timestamp_ns)

    async def close(self) -> None:
        self._clock_done.set()
        self._timeline_active.set()
        if self._clock_task is not None:
            await self._clock_task
            self._clock_task = None
        self._clock = None
        self._timeline_active.clear()
        if self._owns_server and isinstance(self.server, WebSocketBridgeServer):
            await self.server.stop()

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        connections = 0 if self.server is None else len(self.server.connections)
        return (
            ("Server", self.url),
            ("Clients", str(connections)),
            ("Subscriptions", str(len(self._subscriptions))),
        )

    def is_channel_active(self, channel: PlaybackChannel) -> bool:
        channel_id = self.channel_ids[channel]
        return any(
            active_channel_id == channel_id for _, _, active_channel_id in self._subscriptions
        )

    @property
    def has_subscriptions(self) -> bool:
        return bool(self._subscriptions)

    def on_activity_change(self, handler: Callable[[bool], None]) -> None:
        self._activity_handlers.append(handler)

    async def wait_until_active(self) -> float:
        await self._has_subscription.wait()
        delay = 0.0
        if self._inactive_since is not None:
            delay = asyncio.get_running_loop().time() - self._inactive_since
            self._inactive_since = None
        if self._clock is not None:
            self._clock.delay(delay)
            self._timeline_active.set()
            return 0.0
        return delay

    def _update_subscription_activity(self, was_active: bool) -> None:
        is_active = bool(self._subscriptions)
        if is_active == was_active:
            return
        if is_active:
            self._has_subscription.set()
        else:
            self._has_subscription.clear()
            self._inactive_since = asyncio.get_running_loop().time()
            self._timeline_active.clear()
        for handler in self._activity_handlers:
            handler(is_active)


def serve(
    files: list[str],
    *,
    host: ServerHostOption = "127.0.0.1",
    port: ServerPortOption = 8765,
    transform: TransformModeOption = "none",
    image_format: OptionalImageFormatOption = None,
    codec: OptionalCodecOption = None,
    quality: OptionalQualityOption = None,
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
    progress: ProgressOption = True,
    no_browser: NoBrowserOption = False,
) -> int:
    """Host MCAP files or a recording directory for Foxglove clients.

    Playback begins one second after the first client subscribes. All clients
    on the same file selection share one chronological playback timeline.
    Passing a directory that isn't a rosbag2 recording starts a small recording
    browser and accepts repeated ``file=`` query parameters at ``/ws``.

    Examples
    --------
    ```
    pymcap-cli bridge serve recording.mcap
    pymcap-cli bridge serve part1.mcap part2.mcap --speed 2 --loop
    pymcap-cli bridge serve /data/recordings --port 9090
    pymcap-cli bridge serve recording.mcap --host 0.0.0.0 --port 8765
    ```
    """
    if not 1 <= port <= 65535:
        ERR.print("[red]Error:[/] --port must be in [1, 65535]")
        return 1
    try:
        transform_config = resolve_playback_transform_config(
            transform=transform,
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
        library_root = _library_root(files)
        if library_root is not None:
            from pymcap_cli.cmd.bridge._library import (  # noqa: PLC0415
                RecordingLibrary,
                RecordingLibraryServer,
            )

            library_server = RecordingLibraryServer(
                RecordingLibrary(library_root),
                host=host,
                port=port,
                message_filter=message_filter,
                transform_config=transform_config,
                speed=speed,
                loop=loop,
            )

            async def run_library() -> None:
                await library_server.start()
                url = f"http://{_url_host(host)}:{library_server.port}/"
                console.print(f"[green]Serving recording library at[/] [bold]{url}[/]")
                if host not in {"127.0.0.1", "::1", "localhost"}:
                    console.print(
                        "[yellow]Warning:[/] playback controls have no authentication; "
                        "protect remote access with a trusted network or reverse proxy."
                    )
                if not no_browser:
                    _launch_url(url)
                try:
                    await library_server.serve_forever()
                finally:
                    await library_server.stop()

            asyncio.run(run_library())
            return 0
        prepared = prepare_playback(files, message_filter)
        transform_plan = create_playback_transform_plan(transform_config, prepared.channels)
        sink = BridgeServerPlaybackSink(host, port)

        async def run_direct() -> PlaybackStats:
            output_channels = (
                prepared.channels if transform_plan is None else transform_plan.channels
            )
            await sink.start(output_channels)
            if not no_browser:
                _launch_url(_foxglove_url(host, port))
            return await run_playback(
                prepared,
                sink,
                speed=speed,
                loop=loop,
                show_status=progress and console.is_terminal,
                transform_plan=transform_plan,
            )

        stats = asyncio.run(run_direct())
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
    console.print(f"[green]Served[/] {stats.messages:,} messages.")
    return 0
