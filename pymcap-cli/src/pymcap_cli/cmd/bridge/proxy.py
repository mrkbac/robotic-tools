"""`pymcap-cli bridge proxy` — low-latency live Foxglove bridge proxy."""

import asyncio
import contextlib
import logging
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, cast

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

from pymcap_cli.cmd.bridge._proxy_runtime import (
    MESSAGE_ENCODING,
    IncomingMessage,
    OutboundMessage,
    ProxyMetrics,
    SendManager,
    TransformWorker,
    schema_from_channel,
)
from pymcap_cli.cmd.bridge._proxy_transforms import (
    COMPRESSED_VIDEO_SCHEMA,
    ImageConfig,
    PointCloudConfig,
    ProxyConfig,
    build_transform_rules,
    create_transformer,
    is_video_keyframe,
)
from pymcap_cli.cmd.bridge._shared import (
    CONNECTION_GROUP,
    BridgeTarget,
    console,
    to_ws_url,
)
from pymcap_cli.log_setup import ERR

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymcap_cli.cmd.bridge._proxy_runtime import LiveTransformer

logger = logging.getLogger(__name__)

IMAGE_GROUP = CycloptsGroup("Image Compression")
POINTCLOUD_GROUP = CycloptsGroup("Point Cloud Compression")
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
    last_sent_time: float | None = None


@dataclass(frozen=True, slots=True)
class ProxySnapshot:
    """Point-in-time proxy state for the live dashboard."""

    upstream_connected: bool
    upstream_channels: int
    transformed_channels: int
    client_count: int


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
        self._send_manager = SendManager(
            self.downstream_server, self.metrics, queue_size=config.send_queue_size
        )
        self._shutdown_event = asyncio.Event()
        self._channels: dict[int, _ChannelState] = {}
        self._downstream_to_upstream: dict[int, int] = {}
        self._channel_subscribers: dict[int, set[tuple[ServerConnection, int]]] = {}
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
        await self._send_manager.close()
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

        state = _ChannelState(
            upstream_info=channel,
            downstream_info=downstream_channel,
            downstream_id=downstream_id,
            worker=worker,
            throttle_hz=self.config.throttle_hz,
        )
        self._channels[upstream_id] = state
        self._downstream_to_upstream[downstream_id] = upstream_id
        self._refresh_transform_count()
        await self.downstream_server.advertise_channel(
            Channel(
                id=downstream_id,
                topic=downstream_channel["topic"],
                encoding=downstream_channel["encoding"],
                schema_name=downstream_channel.get("schemaName", ""),
                schema=downstream_channel.get("schema", ""),
                schema_encoding=downstream_channel.get("schemaEncoding"),
            )
        )

    async def handle_upstream_unadvertise(self, channel: ChannelInfo) -> None:
        upstream_id = channel["id"]
        state = self._channels.pop(upstream_id, None)
        if state is None:
            return
        if state.worker is not None:
            await state.worker.close()
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
        if state.worker is not None:
            state.worker.enqueue(IncomingMessage(timestamp_ns=timestamp, payload=payload))
            return
        self._send_outbound(self._build_passthrough_outbound(state, timestamp, payload))

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
        self._channel_subscribers.setdefault(channel_id, set()).add((websocket, subscription_id))

    async def handle_client_unsubscribe(
        self, websocket: ServerConnection, subscription_id: int
    ) -> None:
        client_subs = self._client_subscriptions.get(websocket, {})
        info = client_subs.pop(subscription_id, None)
        if info is None:
            return
        channel_id, _upstream_sub_id = info
        subscribers = self._channel_subscribers.get(channel_id)
        if subscribers is not None:
            subscribers.discard((websocket, subscription_id))
            if not subscribers:
                self._channel_subscribers.pop(channel_id, None)
        await self._send_manager.remove(websocket, subscription_id)
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
        await self._send_manager.remove_websocket(websocket)

    def _send_outbound(self, message: OutboundMessage) -> None:
        subscribers = self._channel_subscribers.get(message.channel_id)
        if not subscribers:
            return
        for websocket, subscription_id in list(subscribers):
            self._send_manager.enqueue(websocket, subscription_id, message)

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
        )

    def _create_transformer(self, channel: ChannelInfo) -> "LiveTransformer | None":
        return create_transformer(self._transform_rules, channel)


def proxy(
    target: BridgeTarget,
    *,
    port: Annotated[int, Parameter(name=["--port"], group=CONNECTION_GROUP)] = 8766,
    host: Annotated[str, Parameter(name=["--host"], group=CONNECTION_GROUP)] = "0.0.0.0",  # noqa: S104
    image_format: Annotated[
        Literal["video", "jpeg", "png", "none"],
        Parameter(name=["--image-format"], group=IMAGE_GROUP),
    ] = "video",
    image_codec: Annotated[
        Literal["h264", "h265", "vp9", "av1"],
        Parameter(name=["--image-codec"], group=IMAGE_GROUP),
    ] = "h264",
    image_quality: Annotated[
        int,
        Parameter(name=["--image-quality"], group=IMAGE_GROUP),
    ] = 28,
    image_encoder: Annotated[
        str | None,
        Parameter(name=["--image-encoder"], group=IMAGE_GROUP),
    ] = None,
    image_backend: Annotated[
        Literal["auto", "pyav", "ffmpeg-cli", "gstreamer"],
        Parameter(name=["--image-backend"], group=IMAGE_GROUP),
    ] = "auto",
    image_scale: Annotated[
        int | None,
        Parameter(name=["--image-scale"], group=IMAGE_GROUP),
    ] = None,
    jpeg_quality: Annotated[
        int,
        Parameter(name=["--jpeg-quality"], group=IMAGE_GROUP),
    ] = 90,
    pointcloud: Annotated[
        bool,
        Parameter(name=["--pointcloud"], negative="--no-pointcloud", group=POINTCLOUD_GROUP),
    ] = True,
    pc_format: Annotated[
        Literal["cloudini", "draco"],
        Parameter(name=["--pc-format"], group=POINTCLOUD_GROUP),
    ] = "cloudini",
    pc_schema: Annotated[
        Literal["auto", "pointcloud2", "foxglove"],
        Parameter(name=["--pc-schema"], group=POINTCLOUD_GROUP),
    ] = "auto",
    pc_encoding: Annotated[
        Literal["lossy", "lossless", "none"],
        Parameter(name=["--pc-encoding"], group=POINTCLOUD_GROUP),
    ] = "lossy",
    pc_compression: Annotated[
        Literal["zstd", "lz4", "none"],
        Parameter(name=["--pc-compression"], group=POINTCLOUD_GROUP),
    ] = "zstd",
    resolution: Annotated[
        float,
        Parameter(name=["--resolution"], group=POINTCLOUD_GROUP),
    ] = 0.01,
    draco_compression_level: Annotated[
        int,
        Parameter(name=["--draco-compression-level"], group=POINTCLOUD_GROUP),
    ] = 7,
    send_queue_size: Annotated[
        int,
        Parameter(name=["--send-queue-size"], group=LATENCY_GROUP),
    ] = 1,
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
    if send_queue_size <= 0 or transform_queue_size <= 0:
        ERR.print("[red]Error:[/] queue sizes must be positive")
        return 1
    if throttle_hz < 0:
        ERR.print("[red]Error:[/] --throttle-hz must not be negative")
        return 1
    if not 1 <= jpeg_quality <= 100:
        ERR.print("[red]Error:[/] --jpeg-quality must be in [1, 100]")
        return 1
    if not 0 <= draco_compression_level <= 10:
        ERR.print("[red]Error:[/] --draco-compression-level must be in [0, 10]")
        return 1
    config = ProxyConfig(
        image=ImageConfig(
            image_format=image_format,
            codec=image_codec,
            quality=image_quality,
            encoder=image_encoder,
            backend=image_backend,
            scale=image_scale,
            jpeg_quality=jpeg_quality,
        ),
        pointcloud=PointCloudConfig(
            enabled=pointcloud,
            pc_format=pc_format,
            pc_schema=pc_schema,
            pc_encoding=pc_encoding,
            pc_compression=pc_compression,
            resolution=resolution,
            draco_compression_level=draco_compression_level,
        ),
        transform_queue_size=transform_queue_size,
        send_queue_size=send_queue_size,
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
