"""`pymcap-cli bridge proxy` — low-latency live Foxglove bridge proxy."""

import asyncio
import contextlib
import logging
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Protocol, cast

from cyclopts import Group as CycloptsGroup
from cyclopts import Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory, McapROS2DecodeError
from robo_ws_bridge import (
    ConnectionState,
    ServerConnection,
    WebSocketBridgeClient,
    WebSocketBridgeServer,
)
from robo_ws_bridge.server import Channel
from robo_ws_bridge.ws_types import ChannelInfo

from pymcap_cli.cmd._cli_options import (
    CONNECTION_GROUP,
    ENCODING_GROUP,
    IMAGE_POINTCLOUD_MODE_CONSTRAINT,
    POINTCLOUD_GROUP,
    BackendOption,
    BridgeTarget,
    CodecOption,
    DracoCompressionLevelOption,
    EncoderOption,
    ImageFormatOption,
    JpegQualityOption,
    PointCloudCompressionOption,
    PointCloudDropInvalidOption,
    PointCloudEncodingOption,
    PointCloudFormatOption,
    PointCloudOption,
    PointCloudSchemaOption,
    PointCloudSortFieldOption,
    QualityOption,
    ResolutionOption,
    ScaleOption,
    ServerHostOption,
    ServerPortOption,
)
from pymcap_cli.cmd._pointcloud_cleanup import resolve_pointcloud_cleanup
from pymcap_cli.cmd.bridge._adaptive import (
    AdaptiveQualityGovernor,
    RungTransition,
    adaptive_video_rungs,
)
from pymcap_cli.cmd.bridge._playback import is_frame_schema
from pymcap_cli.cmd.bridge._proxy_runtime import (
    MESSAGE_ENCODING,
    IncomingMessage,
    LiveTransformer,
    OutboundMessage,
    ProxyMetrics,
    TransformWorker,
    schema_from_channel,
)
from pymcap_cli.cmd.bridge._proxy_transforms import (
    COMPRESSED_VIDEO_SCHEMA,
    ImageConfig,
    PointCloudConfig,
    ProxyConfig,
    VideoTransformer,
    build_transform_rules,
    create_transformer,
    is_video_keyframe,
)
from pymcap_cli.cmd.bridge._shared import console, to_ws_url
from pymcap_cli.log_setup import ERR

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

LATENCY_GROUP = CycloptsGroup("Low Latency")


class _CompressedVideoMessageLike(Protocol):
    data: bytes | bytearray | memoryview
    format: str


@dataclass(slots=True)
class _ChannelState:
    upstream_info: ChannelInfo
    downstream_info: ChannelInfo
    downstream_id: int
    worker: TransformWorker | None
    throttle_hz: float
    transformer: LiveTransformer | None = None
    is_adaptive_video: bool = False
    last_sent_time: float | None = None


@dataclass(frozen=True, slots=True)
class ProxySnapshot:
    """Point-in-time proxy state for the live dashboard."""

    upstream_connected: bool
    upstream_channels: int
    transformed_channels: int
    client_count: int
    downgraded_channels: int


class BridgeProxy:
    """Low-latency Foxglove WebSocket proxy."""

    def __init__(
        self,
        *,
        upstream_url: str,
        listen_host: str,
        listen_port: int,
        config: ProxyConfig,
    ) -> None:
        self.upstream_url = upstream_url
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.config = config
        self.metrics = ProxyMetrics()
        self.upstream_client = WebSocketBridgeClient(
            upstream_url,
            max_message_size=config.max_message_size,
        )
        self.downstream_server = WebSocketBridgeServer(
            host=listen_host,
            port=listen_port,
            name="pymcap-cli bridge proxy",
            capabilities=[],
            metadata={"proxy": "true", "latency": "latest"},
            supported_encodings=[MESSAGE_ENCODING],
            max_message_size=config.max_message_size,
        )
        self._transform_rules = build_transform_rules(config)
        # Downstream delivery, backpressure, and slow-client drops are owned by
        # the transport's per-connection outbox; the governor keys adaptive
        # video rungs by downstream channel id.
        self._governor: AdaptiveQualityGovernor[int] = AdaptiveQualityGovernor()
        self._shutdown_event = asyncio.Event()
        self._channels: dict[int, _ChannelState] = {}
        self._downstream_to_upstream: dict[int, int] = {}
        self._client_subscriptions: dict[ServerConnection, dict[int, tuple[int, int]]] = {}
        self._upstream_subscriptions: dict[int, tuple[int, int]] = {}
        self._next_channel_id = 10000
        self._next_upstream_sub_id = 1
        self._video_decoders: dict[int, Callable[[bytes | memoryview], object]] = {}

        self.upstream_client.on_advertised_channel(self.handle_upstream_advertise)
        self.upstream_client.on_channel_unadvertised(self.handle_upstream_unadvertise)
        self.upstream_client.on_message(self.handle_upstream_message)
        self.upstream_client.on_disconnect(self.handle_upstream_disconnected)
        self.upstream_client.on_reconnecting(self.handle_upstream_disconnected)
        self.upstream_client.on_connect(self.handle_upstream_reconnected)
        self.downstream_server.on_connect(self._on_client_connected)
        self.downstream_server.on_subscribe(self._on_client_subscribe)
        self.downstream_server.on_unsubscribe(self._on_client_unsubscribe)
        self.downstream_server.on_disconnect(self._on_client_disconnected)

    async def start(self) -> None:
        logger.info(
            "Starting bridge proxy: %s -> ws://%s:%d",
            self.upstream_url,
            self.listen_host,
            self.listen_port,
        )
        for url in _network_urls(self.listen_port):
            logger.info("Downstream URL: %s", url)
        await asyncio.gather(self.upstream_client.connect(), self.downstream_server.start())
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        self._shutdown_event.set()
        for state in list(self._channels.values()):
            if state.worker is not None:
                await state.worker.close()
        self._channels.clear()
        await asyncio.gather(
            self.upstream_client.disconnect(),
            self.downstream_server.stop(),
            return_exceptions=True,
        )

    async def handle_upstream_advertise(self, channel: ChannelInfo) -> None:
        upstream_id = channel["id"]
        transformer = self._create_transformer(channel)
        downstream_channel: ChannelInfo
        worker: TransformWorker | None = None
        if transformer is None:
            downstream_id = upstream_id
            downstream_channel = channel
        else:
            downstream_id = self._next_channel_id
            self._next_channel_id += 1
            downstream_channel = {
                "id": downstream_id,
                "topic": channel["topic"],
                "encoding": transformer.output_message_encoding,
                "schemaName": transformer.output_schema_name,
                "schema": transformer.output_schema_text,
                "schemaEncoding": transformer.output_schema_encoding,
            }
            worker = TransformWorker(
                channel=channel,
                downstream_id=downstream_id,
                transformer=transformer,
                queue_size=self.config.transform_queue_size,
                emit=self._send_outbound,
                metrics=self.metrics,
            )

        is_adaptive_video = (
            self.config.image.adaptive_quality
            and isinstance(transformer, VideoTransformer)
            and downstream_channel.get("schemaName", "") == COMPRESSED_VIDEO_SCHEMA
        )
        if is_adaptive_video:
            self._governor.register(downstream_id, adaptive_video_rungs(self.config.image.quality))

        state = _ChannelState(
            upstream_info=channel,
            downstream_info=downstream_channel,
            downstream_id=downstream_id,
            worker=worker,
            throttle_hz=self.config.throttle_hz,
            transformer=transformer,
            is_adaptive_video=is_adaptive_video,
        )
        self._channels[upstream_id] = state
        self._downstream_to_upstream[downstream_id] = upstream_id
        self._refresh_transform_count()
        # Standalone frames (raw images, point clouds) may be dropped for a slow
        # client; encoded video must not, so it queues reliably and instead
        # sheds bitrate through adaptive quality.
        delivery = (
            "latest" if is_frame_schema(downstream_channel.get("schemaName", "")) else "reliable"
        )
        await self.downstream_server.advertise_channel(
            Channel(
                id=downstream_id,
                topic=downstream_channel["topic"],
                encoding=downstream_channel["encoding"],
                schema_name=downstream_channel.get("schemaName", ""),
                schema=downstream_channel.get("schema", ""),
                schema_encoding=downstream_channel.get("schemaEncoding"),
                delivery=delivery,
            )
        )

    async def handle_upstream_unadvertise(self, channel: ChannelInfo) -> None:
        upstream_id = channel["id"]
        state = self._channels.pop(upstream_id, None)
        if state is None:
            return
        if state.worker is not None:
            await state.worker.close()
        self._governor.unregister(state.downstream_id)
        self._downstream_to_upstream.pop(state.downstream_id, None)
        self._refresh_transform_count()
        await self.downstream_server.unadvertise([state.downstream_id])

    async def handle_upstream_message(
        self, channel: ChannelInfo, timestamp: int, payload: bytes
    ) -> None:
        self.metrics.upstream_messages_received += 1
        state = self._channels.get(channel["id"])
        if state is None or self._should_throttle(state):
            return
        if state.is_adaptive_video and self._adaptive_should_drop(state):
            return
        if state.worker is not None:
            state.worker.enqueue(IncomingMessage(timestamp_ns=timestamp, payload=payload))
            return
        await self._send_outbound(self._build_passthrough_outbound(state, timestamp, payload))

    def _adaptive_should_drop(self, state: _ChannelState) -> bool:
        """Update the adaptive rung for a video channel and apply its frame-rate cap.

        Congestion is inferred from the transport outbox: when every subscriber
        of the channel already has queued work, the encoder steps down a rung.
        """
        now = asyncio.get_running_loop().time()
        is_congested = self.downstream_server.are_all_subscribers_busy(state.downstream_id)
        transition = self._governor.observe(state.downstream_id, is_congested=is_congested, now=now)
        if transition is not None:
            self._apply_rung(state, transition)
        if self._governor.should_drop_frame(state.downstream_id, now=now):
            self.metrics.adaptive_frames_dropped += 1
            return True
        return False

    def _apply_rung(self, state: _ChannelState, transition: RungTransition) -> None:
        previous = transition.previous
        current = transition.current
        if previous.scale_factor != current.scale_factor and isinstance(
            state.transformer, VideoTransformer
        ):
            state.transformer.request_scale_factor(current.scale_factor)
        logger.info(
            "Adaptive video for %s: q%d @ %.1f%% / %s fps -> q%d @ %.1f%% / %s fps",
            state.downstream_info.get("topic", ""),
            previous.quality,
            previous.scale_factor * 100,
            "source" if previous.max_fps is None else f"{previous.max_fps:g}",
            current.quality,
            current.scale_factor * 100,
            "source" if current.max_fps is None else f"{current.max_fps:g}",
        )

    async def handle_upstream_disconnected(self) -> None:
        self._upstream_subscriptions.clear()

    async def handle_upstream_reconnected(self) -> None:
        ref_counts: dict[int, int] = {}
        for subscriptions in self._client_subscriptions.values():
            for channel_id, _upstream_sub_id in subscriptions.values():
                upstream_id = self._downstream_to_upstream.get(channel_id, channel_id)
                ref_counts[upstream_id] = ref_counts.get(upstream_id, 0) + 1
        existing_sub_ids = {
            upstream_id: upstream_sub_id
            for upstream_id, (upstream_sub_id, _ref_count) in self._upstream_subscriptions.items()
        }
        self._upstream_subscriptions.clear()
        for upstream_id, ref_count in ref_counts.items():
            sub_id = existing_sub_ids.get(upstream_id)
            if sub_id is None:
                sub_id = self._next_upstream_sub_id
                self._next_upstream_sub_id += 1
            await self._subscribe_upstream_if_connected(sub_id, upstream_id)
            self._upstream_subscriptions[upstream_id] = (sub_id, ref_count)

    async def _on_client_connected(self, state: ConnectionState) -> None:
        logger.info("Proxy client connected: %s", _remote_address(state.websocket))

    async def _on_client_subscribe(
        self, state: ConnectionState, subscription_id: int, channel_id: int
    ) -> None:
        await self.handle_client_subscribe(state.websocket, subscription_id, channel_id)

    async def _on_client_unsubscribe(
        self, state: ConnectionState, subscription_id: int, channel_id: int
    ) -> None:
        del channel_id
        await self.handle_client_unsubscribe(state.websocket, subscription_id)

    async def _on_client_disconnected(self, state: ConnectionState) -> None:
        await self.handle_client_disconnected(state.websocket)

    async def handle_client_subscribe(
        self, websocket: ServerConnection, subscription_id: int, channel_id: int
    ) -> None:
        self._client_subscriptions.setdefault(websocket, {})
        upstream_id = self._downstream_to_upstream.get(channel_id, channel_id)
        if upstream_id in self._upstream_subscriptions:
            upstream_sub_id, ref_count = self._upstream_subscriptions[upstream_id]
            self._upstream_subscriptions[upstream_id] = (upstream_sub_id, ref_count + 1)
        else:
            upstream_sub_id = self._next_upstream_sub_id
            self._next_upstream_sub_id += 1
            self._upstream_subscriptions[upstream_id] = (upstream_sub_id, 1)
            await self._subscribe_upstream_if_connected(upstream_sub_id, upstream_id)
        self._client_subscriptions[websocket][subscription_id] = (channel_id, upstream_sub_id)

    async def handle_client_unsubscribe(
        self, websocket: ServerConnection, subscription_id: int
    ) -> None:
        client_subs = self._client_subscriptions.get(websocket, {})
        info = client_subs.pop(subscription_id, None)
        if info is None:
            return
        channel_id, _upstream_sub_id = info
        upstream_id = self._downstream_to_upstream.get(channel_id, channel_id)
        existing = self._upstream_subscriptions.get(upstream_id)
        if existing is None:
            return
        sub_id, ref_count = existing
        if ref_count <= 1:
            await self._unsubscribe_upstream_if_connected(sub_id)
            self._upstream_subscriptions.pop(upstream_id, None)
        else:
            self._upstream_subscriptions[upstream_id] = (sub_id, ref_count - 1)

    async def handle_client_disconnected(self, websocket: ServerConnection) -> None:
        subscriptions = list(self._client_subscriptions.get(websocket, {}))
        for subscription_id in subscriptions:
            await self.handle_client_unsubscribe(websocket, subscription_id)
        self._client_subscriptions.pop(websocket, None)

    async def _send_outbound(self, message: OutboundMessage) -> None:
        # Fan out to every subscriber of the channel through the transport
        # outbox, which applies the channel's delivery policy and never blocks
        # on a slow client.
        await self.downstream_server.publish_message(
            message.channel_id, message.payload, timestamp_ns=message.timestamp_ns
        )

    async def _subscribe_upstream_if_connected(
        self, upstream_sub_id: int, upstream_id: int
    ) -> None:
        if not self.upstream_client.is_connected:
            return
        try:
            await self.upstream_client.subscribe_to_channel(upstream_sub_id, upstream_id)
        except RuntimeError:
            logger.debug("Upstream disconnected before subscription could be sent")

    async def _unsubscribe_upstream_if_connected(self, upstream_sub_id: int) -> None:
        if not self.upstream_client.is_connected:
            return
        try:
            await self.upstream_client.unsubscribe_from_channel(upstream_sub_id)
        except RuntimeError:
            logger.debug("Upstream disconnected before unsubscription could be sent")

    def _build_passthrough_outbound(
        self, state: _ChannelState, timestamp: int, payload: bytes
    ) -> OutboundMessage:
        is_video, is_keyframe = self._passthrough_video_metadata(state, payload)
        return OutboundMessage(
            channel_id=state.downstream_id,
            timestamp_ns=timestamp,
            payload=payload,
            is_compressed_video=is_video,
            is_keyframe=is_keyframe,
        )

    def _passthrough_video_metadata(
        self, state: _ChannelState, payload: bytes
    ) -> tuple[bool, bool]:
        schema_name = state.downstream_info.get("schemaName", "")
        if schema_name != COMPRESSED_VIDEO_SCHEMA:
            return False, False
        decoder = self._video_decoders.get(state.downstream_id)
        if decoder is None:
            schema = schema_from_channel(state.downstream_info)
            decoder = DecoderFactory().decoder_for(state.downstream_info["encoding"], schema)
            if decoder is None:
                return True, True
            self._video_decoders[state.downstream_id] = decoder
        try:
            decoded = decoder(payload)
        except (McapROS2DecodeError, ValueError, TypeError, RuntimeError):
            return True, True
        video = cast("_CompressedVideoMessageLike", decoded)
        video_data = bytes(video.data)
        return True, is_video_keyframe(video_data, str(video.format))

    def _should_throttle(self, state: _ChannelState) -> bool:
        if state.throttle_hz <= 0:
            return False
        now = asyncio.get_running_loop().time()
        min_interval = 1.0 / state.throttle_hz
        if state.last_sent_time is not None and now - state.last_sent_time < min_interval:
            self.metrics.upstream_messages_throttled += 1
            return True
        state.last_sent_time = now
        return False

    def _refresh_transform_count(self) -> None:
        self.metrics.transformed_channel_count = sum(
            1 for item in self._channels.values() if item.worker is not None
        )

    def snapshot(self) -> ProxySnapshot:
        return ProxySnapshot(
            upstream_connected=self.upstream_client.is_connected,
            upstream_channels=len(self._channels),
            transformed_channels=self.metrics.transformed_channel_count,
            client_count=len(self._client_subscriptions),
            downgraded_channels=self._governor.downgraded_count(),
        )

    def _create_transformer(self, channel: ChannelInfo) -> "LiveTransformer | None":
        return create_transformer(self._transform_rules, channel)


def proxy(
    target: BridgeTarget,
    *,
    port: ServerPortOption = 8766,
    host: ServerHostOption = "0.0.0.0",  # noqa: S104
    image_format: Annotated[
        ImageFormatOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = "video",
    codec: Annotated[
        CodecOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = "h264",
    quality: Annotated[
        QualityOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = 28,
    encoder: Annotated[
        EncoderOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = None,
    backend: Annotated[
        BackendOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = "auto",
    scale: Annotated[
        ScaleOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = None,
    jpeg_quality: Annotated[
        JpegQualityOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = 90,
    pointcloud: Annotated[
        PointCloudOption, Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = True,
    pc_format: Annotated[
        PointCloudFormatOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = "cloudini",
    pc_schema: Annotated[
        PointCloudSchemaOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = "auto",
    pc_encoding: Annotated[
        PointCloudEncodingOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = "lossy",
    pc_compression: Annotated[
        PointCloudCompressionOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = "zstd",
    resolution: Annotated[
        ResolutionOption, Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = 0.01,
    draco_compression_level: Annotated[
        DracoCompressionLevelOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = 7,
    pointcloud_drop_invalid: PointCloudDropInvalidOption = None,
    pointcloud_sort_field: PointCloudSortFieldOption = None,
    adaptive_quality: Annotated[
        bool | None,
        Parameter(
            name=["--adaptive-quality"],
            negative="--no-adaptive-quality",
            group=ENCODING_GROUP,
            help=(
                "Shed video bitrate when a client's link is congested: cap frame "
                "rate first, then resolution, keeping the requested quality. "
                "Requires --image-format video. Default: enabled for video."
            ),
        ),
    ] = None,
    transform_queue_size: Annotated[
        int,
        Parameter(name=["--transform-queue-size"], group=LATENCY_GROUP),
    ] = 1,
    throttle_hz: Annotated[
        float,
        Parameter(name=["--throttle-hz"], group=LATENCY_GROUP),
    ] = 0.0,
    max_message_size: Annotated[
        int,
        Parameter(name=["--max-message-size"], group=CONNECTION_GROUP),
    ] = 0,
    dashboard: Annotated[
        bool,
        Parameter(name=["--dashboard"], negative="--no-dashboard", group=CONNECTION_GROUP),
    ] = True,
) -> int:
    """Run a low-latency transforming proxy for a live Foxglove WebSocket bridge."""
    if port <= 0:
        ERR.print("[red]Error:[/] --port must be positive")
        return 1
    if transform_queue_size <= 0:
        ERR.print("[red]Error:[/] --transform-queue-size must be positive")
        return 1
    if throttle_hz < 0:
        ERR.print("[red]Error:[/] --throttle-hz must not be negative")
        return 1
    if adaptive_quality and image_format != "video":
        ERR.print("[red]Error:[/] --adaptive-quality requires --image-format video")
        return 1
    resolved_adaptive_quality = (
        image_format == "video" if adaptive_quality is None else adaptive_quality
    )
    try:
        cleanup = resolve_pointcloud_cleanup(
            pointcloud_compression_enabled=pointcloud,
            pointcloud_drop_invalid=pointcloud_drop_invalid,
            pointcloud_sort_field=pointcloud_sort_field,
        )
    except ValueError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    config = ProxyConfig(
        image=ImageConfig(
            image_format=image_format,
            codec=codec,
            quality=quality,
            encoder=encoder,
            backend=backend,
            scale=scale,
            jpeg_quality=jpeg_quality,
            adaptive_quality=resolved_adaptive_quality,
        ),
        pointcloud=PointCloudConfig(
            enabled=pointcloud,
            pc_format=pc_format,
            pc_schema=pc_schema,
            pc_encoding=pc_encoding,
            pc_compression=pc_compression,
            resolution=resolution,
            draco_compression_level=draco_compression_level,
            drop_invalid=cleanup.drop_invalid,
            sort_field=cleanup.sort_field,
        ),
        transform_queue_size=transform_queue_size,
        throttle_hz=throttle_hz,
        max_message_size=max_message_size if max_message_size > 0 else None,
    )
    try:
        _validate_optional_dependencies(config)
    except ImportError as exc:
        ERR.print(
            "[red]Error:[/] bridge proxy compression dependencies are missing. "
            "Install with: uv add 'pymcap-cli[bridge-proxy]'\n"
            f"Missing: {exc.name or exc}"
        )
        return 1

    url = to_ws_url(target)
    try:
        return asyncio.run(
            _proxy_async(url=url, host=host, port=port, config=config, show_dashboard=dashboard)
        )
    except KeyboardInterrupt:
        console.print("[dim]Interrupted.[/]")
        return 0


async def _proxy_async(
    *, url: str, host: str, port: int, config: ProxyConfig, show_dashboard: bool
) -> int:
    bridge = BridgeProxy(
        upstream_url=url,
        listen_host=host,
        listen_port=port,
        config=config,
    )
    dashboard_task: asyncio.Task[None] | None = None
    try:
        serve = asyncio.ensure_future(bridge.start())
        if show_dashboard and console.is_terminal:
            from pymcap_cli.cmd.bridge._proxy_dashboard import ProxyDashboard  # noqa: PLC0415

            dashboard_task = asyncio.ensure_future(ProxyDashboard(bridge, console).run())
        await serve
    finally:
        if dashboard_task is not None:
            dashboard_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await dashboard_task
        await bridge.stop()
    return 0


def _validate_optional_dependencies(config: ProxyConfig) -> None:
    if config.image.image_format != "none":
        import mcap_codec_support.video  # noqa: PLC0415

    if config.pointcloud.enabled:
        import mcap_codec_support.pointcloud  # noqa: F401, PLC0415


def _network_urls(port: int) -> list[str]:
    urls = [f"ws://localhost:{port}"]
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET)
    except OSError:
        return urls
    seen = {"127.0.0.1"}
    for info in infos:
        ip = str(info[4][0])
        if ip not in seen:
            urls.append(f"ws://{ip}:{port}")
            seen.add(ip)
    return urls


def _remote_address(websocket: ServerConnection) -> str:
    try:
        remote = websocket.remote_address
    except (AttributeError, OSError):
        return "unknown"
    if isinstance(remote, tuple):
        return f"{remote[0]}:{remote[1]}"
    return str(remote)


__all__ = [
    "BridgeProxy",
    "proxy",
]
