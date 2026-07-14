"""`pymcap-cli bridge play` — publish MCAP playback into a live bridge."""

import asyncio
from typing import Annotated

from cyclopts import Group, Parameter
from robo_ws_bridge import ConnectionGraph, WebSocketBridgeClient
from robo_ws_bridge.ws_types import ServerCapabilities

from pymcap_cli.cmd._cli_options import (
    BridgeTarget,
    ConnectTimeoutOption,
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    OptionalBackendOption,
    OptionalCodecOption,
    OptionalPointCloudDropInvalidOption,
    OptionalPointCloudSortFieldOption,
    ProgressOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.cmd.bridge._playback import (
    _SETTLE_SECONDS,
    PlaybackChannel,
    PlaybackClock,
    PlaybackError,
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
from pymcap_cli.cmd.bridge._shared import console, to_ws_url
from pymcap_cli.log_setup import ERR

PLAYBACK_GROUP = Group("Playback")
SpeedOption = Annotated[
    float,
    Parameter(name=["--speed"], group=PLAYBACK_GROUP, help="Playback speed multiplier."),
]
LoopOption = Annotated[
    bool,
    Parameter(name=["--loop"], group=PLAYBACK_GROUP, help="Repeat playback until Ctrl+C."),
]
OnlySubscribedOption = Annotated[
    bool,
    Parameter(
        name=["--only-subscribed"],
        group=PLAYBACK_GROUP,
        help="Publish only topics with consumers reported by connectionGraph.",
    ),
]


class BridgeClientPlaybackSink:
    def __init__(self, url: str, *, connect_timeout: float, only_subscribed: bool = False) -> None:
        self.url = url
        self.connect_timeout = connect_timeout
        self.only_subscribed = only_subscribed
        self.client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
        self.channel_ids: dict[PlaybackChannel, int] = {}
        self._selected_topics: set[str] = set()
        self._consumer_topics: set[str] = set()
        self._server_info = asyncio.Event()
        self._graph_received = asyncio.Event()
        self._has_consumers = asyncio.Event()
        self._connection_lost = asyncio.Event()
        self.client.on_server_info(lambda *_: self._server_info.set())
        self.client.on_connection_graph_update(self._on_connection_graph_update)
        self.client.on_reconnecting(lambda *_: self._connection_lost.set())

    async def start(self, channels: tuple[PlaybackChannel, ...]) -> None:
        await self.client.connect()
        try:
            await asyncio.wait_for(self._server_info.wait(), timeout=self.connect_timeout)
        except asyncio.TimeoutError as exc:
            detail = (
                f"Timed out after {self.connect_timeout:.1f}s waiting for "
                f"serverInfo from {self.url}"
            )
            raise PlaybackError(detail) from exc
        server_info = self.client.server_info
        if server_info is None:
            raise PlaybackError(f"No serverInfo received from {self.url}")
        if ServerCapabilities.CLIENT_PUBLISH.value not in server_info["capabilities"]:
            raise PlaybackError(f"Bridge at {self.url} does not advertise clientPublish")
        if (
            self.only_subscribed
            and ServerCapabilities.CONNECTION_GRAPH.value not in server_info["capabilities"]
        ):
            raise PlaybackError(
                f"Bridge at {self.url} must advertise connectionGraph "
                "when --only-subscribed is used"
            )
        supported = set(server_info.get("supportedEncodings", []))
        unsupported = sorted({channel.message_encoding for channel in channels} - supported)
        if unsupported:
            raise PlaybackError(
                f"Bridge at {self.url} does not support message encoding(s): "
                + ", ".join(unsupported)
            )
        for channel in channels:
            channel_id = await self.client.advertise(
                channel.topic,
                encoding=channel.message_encoding,
                schema_name=channel.schema_name,
                schema=channel.schema_text,
                schema_encoding=channel.schema_encoding or None,
            )
            self.channel_ids[channel] = channel_id
        self._selected_topics = {channel.topic for channel in channels}
        if self.only_subscribed:
            await self.client.subscribe_connection_graph()
        console.print(f"[green]Connected to[/] [bold]{self.url}[/]")

    async def wait_until_ready(self) -> None:
        if self.only_subscribed:
            console.print("[dim]Waiting for the initial connection graph...[/]")
            try:
                await asyncio.wait_for(self._graph_received.wait(), timeout=self.connect_timeout)
            except asyncio.TimeoutError as exc:
                raise PlaybackError(
                    f"Timed out after {self.connect_timeout:.1f}s waiting for "
                    f"connectionGraphUpdate from {self.url}"
                ) from exc
            if not self._has_consumers.is_set():
                console.print("[dim]Waiting for a selected topic to gain a consumer...[/]")
            await self.wait_until_active()
        await asyncio.sleep(_SETTLE_SECONDS)

    async def publish(
        self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
    ) -> None:
        del timestamp_ns
        if self._connection_lost.is_set():
            raise PlaybackError(f"Connection to {self.url} was lost during playback")
        await self.client.publish(self.channel_ids[channel], bytes(payload))

    async def timeline_started(self, clock: PlaybackClock) -> None:
        del clock

    async def timeline_finished(self, timestamp_ns: int) -> None:
        del timestamp_ns

    async def close(self) -> None:
        for channel_id in self.channel_ids.values():
            try:
                await self.client.unadvertise(channel_id)
            except (OSError, RuntimeError):
                break
        await self.client.disconnect()

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        rows = [("Bridge", self.url)]
        if self.only_subscribed:
            rows.append(("Consumer topics", str(len(self._consumer_topics))))
        return tuple(rows)

    def is_channel_active(self, channel: PlaybackChannel) -> bool:
        return not self.only_subscribed or channel.topic in self._consumer_topics

    async def wait_until_active(self) -> float:
        if not self.only_subscribed or self._has_consumers.is_set():
            return 0.0
        started = asyncio.get_running_loop().time()
        await self._has_consumers.wait()
        return asyncio.get_running_loop().time() - started

    def _on_connection_graph_update(self, graph: ConnectionGraph) -> None:
        self._consumer_topics = {
            topic["name"]
            for topic in graph.subscribed_topics
            if topic["name"] in self._selected_topics and topic["subscriberIds"]
        }
        self._graph_received.set()
        if self._consumer_topics:
            self._has_consumers.set()
        else:
            self._has_consumers.clear()


def play(
    files: list[str],
    *,
    target: BridgeTarget,
    only_subscribed: OnlySubscribedOption = False,
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
    loop: LoopOption = False,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    connect_timeout: ConnectTimeoutOption = 5.0,
    progress: ProgressOption = True,
) -> int:
    """Play one or more MCAP files into a live Foxglove WebSocket bridge.

    Inputs are merged chronologically by log time. The bridge must advertise
    ``clientPublish`` and support every selected message encoding.

    Examples
    --------
    ```
    pymcap-cli bridge play recording.mcap --target localhost
    pymcap-cli bridge play part1.mcap part2.mcap --target localhost --speed 2
    pymcap-cli bridge play recording.mcap --target localhost --only-subscribed
    PYMCAP_BRIDGE=localhost pymcap-cli bridge play recording.mcap -t '/camera/.*'
    ```
    """
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
        prepared = prepare_playback(files, message_filter)
        transform_plan = create_playback_transform_plan(transform_config, prepared.channels)
        stats = asyncio.run(
            run_playback(
                prepared,
                BridgeClientPlaybackSink(
                    to_ws_url(target),
                    connect_timeout=connect_timeout,
                    only_subscribed=only_subscribed,
                ),
                speed=speed,
                loop=loop,
                show_status=progress and console.is_terminal,
                transform_plan=transform_plan,
            )
        )
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
        console.print("\n[dim]Playback stopped.[/]")
        return 0
    console.print(f"[green]Published[/] {stats.messages:,} messages.")
    return 0
