"""`pymcap-cli bridge proxy` — low-latency live Foxglove bridge proxy."""

import asyncio
import contextlib
import logging
import socket
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol, cast

from cyclopts import Group as CycloptsGroup
from cyclopts import Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory, McapROS2DecodeError
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from robo_ws_bridge import (
    ConnectionState,
    ServerConnection,
    WebSocketBridgeClient,
    WebSocketBridgeServer,
)
from robo_ws_bridge.server import Channel
from robo_ws_bridge.ws_types import ChannelInfo
from small_mcap import Schema

from pymcap_cli.cmd.bridge._shared import (
    CONNECTION_GROUP,
    BridgeTarget,
    console,
    to_ws_url,
)
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)

IMAGE_GROUP = CycloptsGroup("Image Compression")
POINTCLOUD_GROUP = CycloptsGroup("Point Cloud Compression")
LATENCY_GROUP = CycloptsGroup("Low Latency")

_RAW_IMAGE_SCHEMAS = frozenset({"sensor_msgs/Image"})
_COMPRESSED_IMAGE_SCHEMAS = frozenset({"sensor_msgs/CompressedImage"})
_IMAGE_SCHEMAS = _RAW_IMAGE_SCHEMAS | _COMPRESSED_IMAGE_SCHEMAS
_POINTCLOUD2_SCHEMAS = frozenset({"sensor_msgs/PointCloud2"})
_COMPRESSED_VIDEO_SCHEMA = "foxglove_msgs/msg/CompressedVideo"
_COMPRESSED_IMAGE_SCHEMA = "sensor_msgs/msg/CompressedImage"

_MESSAGE_ENCODING = "cdr"
_SCHEMA_ENCODING = "ros2msg"


class _StampLike(Protocol):
    sec: int
    nanosec: int


class _HeaderLike(Protocol):
    stamp: _StampLike
    frame_id: str


class _HeaderMessageLike(Protocol):
    header: _HeaderLike


class _RawImageLike(_HeaderMessageLike, Protocol):
    width: int
    height: int
    encoding: str
    step: int
    data: bytes


class _PointCloud2Like(_HeaderMessageLike, Protocol):
    data: bytes


class _CompressedVideoMessageLike(Protocol):
    data: bytes | bytearray | memoryview
    format: str


class _VideoEncoderConfigLike(Protocol):
    codec_name: str


class _VideoEncoderLike(Protocol):
    config: _VideoEncoderConfigLike

    def encode(self, frame: object) -> bytes | None: ...

    def close(self) -> None: ...


class _VideoBackendLike(Protocol):
    def test_encoder(self, encoder_name: str) -> bool: ...

    def resolve_encoder(self, codec: str) -> str: ...

    def decode_image(
        self, msg: "_LiveDecodedMessage", schema_name: str
    ) -> tuple[object, int, int]: ...

    def create_encoder(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> _VideoEncoderLike: ...

    def get_pix_fmt(self, topic: str) -> str | None: ...


class _RawImageEncoder(Protocol):
    def __call__(
        self,
        decoded_message: _RawImageLike,
        *,
        image_format: Literal["jpeg", "png"],
        jpeg_quality: int,
        scale: int | None,
    ) -> tuple[bytes, int, int]: ...


class _PointCloudCompressorLike(Protocol):
    def compress(self, msg: _PointCloud2Like) -> bytes: ...


class _PointCloudBuilder(Protocol):
    def __call__(
        self, msg: _PointCloud2Like, compressed_data: bytes, *, fmt: str
    ) -> Mapping[str, object]: ...


class _LiveTransformer(Protocol):
    output_schema_name: str
    output_schema_text: str
    output_schema_encoding: str
    output_message_encoding: str

    def transform(self, decoded: object, timestamp_ns: int) -> "_TransformResult | None": ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _ImageConfig:
    image_format: Literal["video", "jpeg", "png", "none"]
    codec: str
    quality: int
    encoder: str | None
    backend: Literal["auto", "pyav", "ffmpeg-cli", "gstreamer"]
    scale: int | None
    jpeg_quality: int


@dataclass(frozen=True, slots=True)
class _PointCloudConfig:
    enabled: bool
    pc_format: Literal["cloudini", "draco"]
    pc_schema: Literal["auto", "pointcloud2", "foxglove"]
    pc_encoding: Literal["lossy", "lossless", "none"]
    pc_compression: Literal["zstd", "lz4", "none"]
    resolution: float
    draco_compression_level: int


@dataclass(frozen=True, slots=True)
class _ProxyConfig:
    image: _ImageConfig
    pointcloud: _PointCloudConfig
    transform_queue_size: int
    send_queue_size: int
    throttle_hz: float
    max_message_size: int | None


@dataclass(frozen=True, slots=True)
class _LiveChannel:
    topic: str


@dataclass(frozen=True, slots=True)
class _LiveDecodedMessage:
    decoded_message: object
    channel: _LiveChannel


@dataclass(frozen=True, slots=True)
class _TransformResult:
    payload: Mapping[str, object]
    is_compressed_video: bool = False
    is_keyframe: bool = False


@dataclass(frozen=True, slots=True)
class _IncomingMessage:
    timestamp_ns: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class _OutboundMessage:
    channel_id: int
    timestamp_ns: int
    payload: bytes
    is_compressed_video: bool = False
    is_keyframe: bool = False


@dataclass(slots=True)
class _ProxyMetrics:
    upstream_messages_received: int = 0
    upstream_messages_throttled: int = 0
    transformed_channel_count: int = 0
    transform_queue_drops: int = 0
    transform_errors: int = 0
    send_queue_drops: int = 0
    send_errors: int = 0
    video_packets_waiting_for_keyframe: int = 0


@dataclass(slots=True)
class _ChannelState:
    upstream_info: ChannelInfo
    downstream_info: ChannelInfo
    downstream_id: int
    worker: "_TransformWorker | None"
    throttle_hz: float
    last_sent_time: float | None = None


@dataclass(slots=True)
class _SendSlot:
    websocket: ServerConnection
    subscription_id: int
    server: WebSocketBridgeServer
    metrics: _ProxyMetrics
    queue_size: int
    queue: asyncio.Queue[_OutboundMessage] = field(init=False)
    task: asyncio.Task[None] = field(init=False)
    needs_keyframe: bool = False

    def __post_init__(self) -> None:
        self.queue = asyncio.Queue(maxsize=max(1, self.queue_size))
        self.task = asyncio.create_task(self._run())

    def enqueue(self, message: _OutboundMessage) -> None:
        while self.queue.full():
            dropped = self.queue.get_nowait()
            self.queue.task_done()
            self.metrics.send_queue_drops += 1
            if dropped.is_compressed_video:
                self.needs_keyframe = True
        self.queue.put_nowait(message)

    async def close(self) -> None:
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task

    async def _run(self) -> None:
        while True:
            message = await self.queue.get()
            try:
                if message.is_compressed_video and self.needs_keyframe and not message.is_keyframe:
                    self.metrics.video_packets_waiting_for_keyframe += 1
                    continue
                if message.is_compressed_video and message.is_keyframe:
                    self.needs_keyframe = False
                try:
                    await self.server.send_message_to_subscription(
                        self.websocket,
                        self.subscription_id,
                        message.payload,
                        timestamp_ns=message.timestamp_ns,
                    )
                except Exception:
                    self.metrics.send_errors += 1
                    logger.exception("Failed to send proxied message")
            finally:
                self.queue.task_done()


class _SendManager:
    def __init__(
        self,
        server: WebSocketBridgeServer,
        metrics: _ProxyMetrics,
        *,
        queue_size: int,
    ) -> None:
        self._server = server
        self._metrics = metrics
        self._queue_size = max(1, queue_size)
        self._slots: dict[tuple[ServerConnection, int], _SendSlot] = {}

    def enqueue(
        self,
        websocket: ServerConnection,
        subscription_id: int,
        message: _OutboundMessage,
    ) -> None:
        key = (websocket, subscription_id)
        slot = self._slots.get(key)
        if slot is None:
            slot = _SendSlot(
                websocket=websocket,
                subscription_id=subscription_id,
                server=self._server,
                metrics=self._metrics,
                queue_size=self._queue_size,
            )
            self._slots[key] = slot
        slot.enqueue(message)

    async def remove(self, websocket: ServerConnection, subscription_id: int) -> None:
        slot = self._slots.pop((websocket, subscription_id), None)
        if slot is not None:
            await slot.close()

    async def remove_websocket(self, websocket: ServerConnection) -> None:
        keys = [key for key in self._slots if key[0] is websocket]
        for key in keys:
            slot = self._slots.pop(key)
            await slot.close()

    async def close(self) -> None:
        slots = list(self._slots.values())
        self._slots.clear()
        for slot in slots:
            await slot.close()


class _TransformWorker:
    def __init__(
        self,
        *,
        channel: ChannelInfo,
        downstream_id: int,
        transformer: _LiveTransformer,
        queue_size: int,
        emit: Callable[[_OutboundMessage], None],
        metrics: _ProxyMetrics,
    ) -> None:
        self._channel = channel
        self._downstream_id = downstream_id
        self._transformer = transformer
        self._queue: asyncio.Queue[_IncomingMessage] = asyncio.Queue(maxsize=max(1, queue_size))
        self._emit = emit
        self._metrics = metrics
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._decoder_factory = DecoderFactory()
        self._encoder_factory = ROS2EncoderFactory()
        self._decoder = self._decoder_for(channel)
        self._encoder = self._encoder_for(transformer)
        self._task = asyncio.create_task(self._run())

    def enqueue(self, message: _IncomingMessage) -> None:
        while self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
            self._metrics.transform_queue_drops += 1
        self._queue.put_nowait(message)

    async def close(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._executor.shutdown(wait=False)
        self._transformer.close()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            message = await self._queue.get()
            try:
                try:
                    outbound = await loop.run_in_executor(self._executor, self._process, message)
                except Exception:
                    self._metrics.transform_errors += 1
                    logger.exception("Transform failed for %s", self._channel["topic"])
                    continue
                if outbound is not None:
                    self._emit(outbound)
            finally:
                self._queue.task_done()

    def _process(self, message: _IncomingMessage) -> _OutboundMessage | None:
        decoded = self._decoder(message.payload)
        result = self._transformer.transform(decoded, message.timestamp_ns)
        if result is None:
            return None
        payload = bytes(self._encoder(result.payload))
        return _OutboundMessage(
            channel_id=self._downstream_id,
            timestamp_ns=message.timestamp_ns,
            payload=payload,
            is_compressed_video=result.is_compressed_video,
            is_keyframe=result.is_keyframe,
        )

    def _decoder_for(self, channel: ChannelInfo) -> Callable[[bytes | memoryview], object]:
        schema = _schema_from_channel(channel)
        decoder = self._decoder_factory.decoder_for(channel["encoding"], schema)
        if decoder is None:
            raise RuntimeError(
                f"No CDR decoder for {channel['topic']} ({channel.get('schemaName', '')})"
            )
        return decoder

    def _encoder_for(self, transformer: _LiveTransformer) -> Callable[[object], bytes | memoryview]:
        schema = Schema(
            id=0,
            name=transformer.output_schema_name,
            encoding=transformer.output_schema_encoding,
            data=transformer.output_schema_text.encode(),
        )
        encoder = self._encoder_factory.encoder_for(schema)
        if encoder is None:
            raise RuntimeError(f"No CDR encoder for {transformer.output_schema_name}")
        return encoder


class _VideoTransformer:
    output_schema_encoding = _SCHEMA_ENCODING
    output_message_encoding = _MESSAGE_ENCODING

    def __init__(self, config: _ImageConfig, topic: str) -> None:
        from mcap_codec_support.video import (  # noqa: PLC0415
            COMPRESSED_VIDEO_SCHEMA,
            FOXGLOVE_COMPRESSED_VIDEO,
            EncoderMode,
            VideoEncoderError,
            calculate_downscale_dimensions,
            get_software_encoder,
        )

        from pymcap_cli.core.processors.video_compress import (  # noqa: PLC0415
            resolve_video_compression_backend,
        )

        self.output_schema_name = COMPRESSED_VIDEO_SCHEMA
        self.output_schema_text = FOXGLOVE_COMPRESSED_VIDEO
        self._config = config
        self._topic = topic
        self._calculate_downscale_dimensions = calculate_downscale_dimensions
        self._get_software_encoder = get_software_encoder
        self._error_type = VideoEncoderError
        resolved = resolve_video_compression_backend(
            codec=config.codec,
            encoder=config.encoder,
            backend=EncoderMode(config.backend),
        )
        self._backend = cast("_VideoBackendLike", resolved.backend)
        self._encoder_name = resolved.encoder_name
        self._encoder: _VideoEncoderLike | None = None
        self._width = 0
        self._height = 0
        self._pix_fmt: str | None = None
        self._scale_dims: tuple[int, int] | None = None

    def transform(self, decoded: object, timestamp_ns: int) -> _TransformResult | None:
        del timestamp_ns
        frame, width, height = self._decode_image(decoded)
        encoder = self._ensure_encoder(decoded, frame, width, height)
        try:
            video_data = encoder.encode(frame)
        except self._error_type:
            logger.exception("Video encoder failed for %s; trying software fallback", self._topic)
            video_data = self._fallback_encode(decoded)
        if video_data is None:
            return None
        header = _message_header(decoded)
        return _TransformResult(
            payload={
                "timestamp": {
                    "sec": header.stamp.sec,
                    "nanosec": header.stamp.nanosec,
                },
                "frame_id": header.frame_id,
                "data": video_data,
                "format": self._config.codec,
            },
            is_compressed_video=True,
            is_keyframe=_is_video_keyframe(video_data, self._config.codec),
        )

    def close(self) -> None:
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None

    def _decode_image(self, decoded: object) -> tuple[object, int, int]:
        schema_name = _normalize_schema_name(_message_type(decoded))
        live = _LiveDecodedMessage(decoded_message=decoded, channel=_LiveChannel(self._topic))
        return self._backend.decode_image(live, schema_name)

    def _ensure_encoder(
        self, decoded: object, frame: object, width: int, height: int
    ) -> _VideoEncoderLike:
        del decoded, frame
        if self._config.scale is not None:
            width, height = self._calculate_downscale_dimensions(width, height, self._config.scale)
        width -= width % 2
        height -= height % 2
        pix_fmt = self._backend.get_pix_fmt(self._topic)
        scale_dims = (width, height) if pix_fmt is None and self._config.scale is not None else None
        if (
            self._encoder is not None
            and self._width == width
            and self._height == height
            and self._pix_fmt == pix_fmt
            and self._scale_dims == scale_dims
        ):
            return self._encoder
        self.close()
        self._encoder = self._backend.create_encoder(
            width,
            height,
            self._encoder_name,
            self._config.quality,
            input_pix_fmt=pix_fmt,
            scale=scale_dims,
        )
        self._width = width
        self._height = height
        self._pix_fmt = pix_fmt
        self._scale_dims = scale_dims
        logger.info(
            "Proxy converting %s to %s %dx%d using %s",
            self._topic,
            self._config.codec,
            width,
            height,
            self._encoder_name,
        )
        return self._encoder

    def _fallback_encode(self, decoded: object) -> bytes | None:
        software_encoder = self._get_software_encoder(self._config.codec)
        if self._encoder is not None and self._encoder.config.codec_name != software_encoder:
            self.close()
            self._encoder_name = software_encoder
        frame, width, height = self._decode_image(decoded)
        encoder = self._ensure_encoder(decoded, frame, width, height)
        return encoder.encode(frame)


class _StillImageTransformer:
    output_schema_name = _COMPRESSED_IMAGE_SCHEMA
    output_schema_encoding = _SCHEMA_ENCODING
    output_message_encoding = _MESSAGE_ENCODING

    def __init__(self, config: _ImageConfig) -> None:
        from mcap_codec_support.video import (  # noqa: PLC0415
            COMPRESSED_IMAGE,
            VideoEncoderError,
            encode_raw_image_to_compressed,
        )

        self.output_schema_text = COMPRESSED_IMAGE
        self._config = config
        self._encode = cast("_RawImageEncoder", encode_raw_image_to_compressed)
        self._error_type = VideoEncoderError

    def transform(self, decoded: object, timestamp_ns: int) -> _TransformResult | None:
        del timestamp_ns
        configured_format = self._config.image_format
        if configured_format == "jpeg":
            image_format: Literal["jpeg", "png"] = "jpeg"
        elif configured_format == "png":
            image_format = "png"
        else:
            raise RuntimeError(f"Still-image transformer cannot encode {configured_format!r}")
        raw_image = cast("_RawImageLike", decoded)
        try:
            image_data, _width, _height = self._encode(
                raw_image,
                image_format=image_format,
                jpeg_quality=self._config.jpeg_quality,
                scale=self._config.scale,
            )
        except self._error_type:
            logger.exception("Still-image compression failed")
            return None
        header = _message_header(decoded)
        return _TransformResult(
            payload={
                "header": {
                    "stamp": {
                        "sec": header.stamp.sec,
                        "nanosec": header.stamp.nanosec,
                    },
                    "frame_id": header.frame_id,
                },
                "format": image_format,
                "data": image_data,
            }
        )

    def close(self) -> None:
        return


class _PointCloudTransformer:
    output_schema_encoding = _SCHEMA_ENCODING
    output_message_encoding = _MESSAGE_ENCODING

    def __init__(self, config: _PointCloudConfig) -> None:
        from mcap_codec_support.pointcloud import (  # noqa: PLC0415
            COMPRESSED_POINTCLOUD2,
            COMPRESSED_POINTCLOUD2_SCHEMA,
            FOXGLOVE_COMPRESSED_POINTCLOUD,
            FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
            PointCloudCompressionError,
            build_compressed_pointcloud2_message,
            build_foxglove_compressed_pointcloud_message,
        )

        from pymcap_cli.core.processors.pointcloud_compress import (  # noqa: PLC0415
            _make_compressor,
        )

        resolved_schema = config.pc_schema
        if resolved_schema == "auto":
            resolved_schema = "foxglove" if config.pc_format == "draco" else "pointcloud2"
        self._config = config
        self._compressor = cast(
            "_PointCloudCompressorLike",
            _make_compressor(
                config.pc_format,
                config.pc_encoding,
                config.pc_compression,
                config.resolution,
                config.draco_compression_level,
            ),
        )
        self._error_type = PointCloudCompressionError
        if resolved_schema == "foxglove":
            self.output_schema_name = FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA
            self.output_schema_text = FOXGLOVE_COMPRESSED_POINTCLOUD
            self._build = cast("_PointCloudBuilder", build_foxglove_compressed_pointcloud_message)
        else:
            self.output_schema_name = COMPRESSED_POINTCLOUD2_SCHEMA
            self.output_schema_text = COMPRESSED_POINTCLOUD2
            self._build = cast("_PointCloudBuilder", build_compressed_pointcloud2_message)

    def transform(self, decoded: object, timestamp_ns: int) -> _TransformResult | None:
        del timestamp_ns
        pointcloud = cast("_PointCloud2Like", decoded)
        try:
            compressed = self._compressor.compress(pointcloud)
        except self._error_type:
            logger.exception("Point-cloud compression failed")
            return None
        return _TransformResult(
            payload=self._build(pointcloud, compressed, fmt=self._config.pc_format)
        )

    def close(self) -> None:
        return


class BridgeProxy:
    """Low-latency Foxglove WebSocket proxy."""

    def __init__(
        self,
        *,
        upstream_url: str,
        listen_host: str,
        listen_port: int,
        config: _ProxyConfig,
    ) -> None:
        self.upstream_url = upstream_url
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.config = config
        self.metrics = _ProxyMetrics()
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
            supported_encodings=[_MESSAGE_ENCODING],
            max_message_size=config.max_message_size,
        )
        self._send_manager = _SendManager(
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
        worker: _TransformWorker | None = None
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
            worker = _TransformWorker(
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
        self.metrics.transformed_channel_count = sum(
            1 for item in self._channels.values() if item.worker is not None
        )
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
        self.metrics.transformed_channel_count = sum(
            1 for item in self._channels.values() if item.worker is not None
        )
        await self.downstream_server.unadvertise([state.downstream_id])

    async def handle_upstream_message(
        self, channel: ChannelInfo, timestamp: int, payload: bytes
    ) -> None:
        self.metrics.upstream_messages_received += 1
        state = self._channels.get(channel["id"])
        if state is None:
            return
        if state.throttle_hz > 0:
            now = asyncio.get_running_loop().time()
            min_interval = 1.0 / state.throttle_hz
            if state.last_sent_time is not None and now - state.last_sent_time < min_interval:
                self.metrics.upstream_messages_throttled += 1
                return
            state.last_sent_time = now
        if state.worker is not None:
            state.worker.enqueue(_IncomingMessage(timestamp_ns=timestamp, payload=payload))
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

    def _send_outbound(self, message: _OutboundMessage) -> None:
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
    ) -> _OutboundMessage:
        is_video, is_keyframe = self._passthrough_video_metadata(state, payload)
        return _OutboundMessage(
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
        if schema_name != _COMPRESSED_VIDEO_SCHEMA:
            return False, False
        decoder = self._video_decoders.get(state.downstream_id)
        if decoder is None:
            schema = _schema_from_channel(state.downstream_info)
            created = DecoderFactory().decoder_for(state.downstream_info["encoding"], schema)
            if created is None:
                return True, True
            decoder = created
            self._video_decoders[state.downstream_id] = decoder
        try:
            decoded = decoder(payload)
        except (McapROS2DecodeError, ValueError, TypeError, RuntimeError):
            return True, True
        video = cast("_CompressedVideoMessageLike", decoded)
        video_data = bytes(video.data)
        return True, _is_video_keyframe(video_data, str(video.format))

    def _create_transformer(self, channel: ChannelInfo) -> _LiveTransformer | None:
        if channel["encoding"] != _MESSAGE_ENCODING:
            return None
        if channel.get("schemaEncoding", _SCHEMA_ENCODING) != _SCHEMA_ENCODING:
            return None
        schema_name = _normalize_schema_name(channel.get("schemaName", ""))
        if self.config.image.image_format == "video" and schema_name in _IMAGE_SCHEMAS:
            return _VideoTransformer(self.config.image, channel["topic"])
        if self.config.image.image_format in {"jpeg", "png"} and schema_name in _RAW_IMAGE_SCHEMAS:
            return _StillImageTransformer(self.config.image)
        if self.config.pointcloud.enabled and schema_name in _POINTCLOUD2_SCHEMAS:
            return _PointCloudTransformer(self.config.pointcloud)
        return None


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
        Literal["h264", "h265"],
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
    config = _ProxyConfig(
        image=_ImageConfig(
            image_format=image_format,
            codec=image_codec,
            quality=image_quality,
            encoder=image_encoder,
            backend=image_backend,
            scale=image_scale,
            jpeg_quality=jpeg_quality,
        ),
        pointcloud=_PointCloudConfig(
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
        return asyncio.run(_proxy_async(url=url, host=host, port=port, config=config))
    except KeyboardInterrupt:
        console.print("[dim]Interrupted.[/]")
        return 0


async def _proxy_async(*, url: str, host: str, port: int, config: _ProxyConfig) -> int:
    bridge = BridgeProxy(
        upstream_url=url,
        listen_host=host,
        listen_port=port,
        config=config,
    )
    try:
        await bridge.start()
    finally:
        await bridge.stop()
    return 0


def _validate_optional_dependencies(config: _ProxyConfig) -> None:
    if config.image.image_format != "none":
        import mcap_codec_support.video  # noqa: PLC0415

    if config.pointcloud.enabled:
        import mcap_codec_support.pointcloud  # noqa: F401, PLC0415


def _schema_from_channel(channel: ChannelInfo) -> Schema:
    return Schema(
        id=channel["id"],
        name=channel.get("schemaName", ""),
        encoding=channel.get("schemaEncoding", _SCHEMA_ENCODING),
        data=channel.get("schema", "").encode(),
    )


def _normalize_schema_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) == 3 and parts[1] in {"msg", "srv", "action"}:
        return f"{parts[0]}/{parts[2]}"
    return name


def _message_type(message: object) -> str:
    msg_type = vars(type(message)).get("_type", "")
    return msg_type if isinstance(msg_type, str) else ""


def _message_header(message: object) -> _HeaderLike:
    return cast("_HeaderMessageLike", message).header


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


def _is_video_keyframe(data: bytes, video_format: str) -> bool:
    fmt = video_format.lower()
    if "265" in fmt or "hevc" in fmt:
        return _has_h265_keyframe(data)
    if "264" in fmt or "avc" in fmt:
        return _has_h264_keyframe(data)
    return True


def _start_code_positions(data: bytes) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    idx = 0
    while idx < len(data) - 3:
        if data[idx : idx + 3] == b"\x00\x00\x01":
            positions.append((idx, idx + 3))
            idx += 3
        elif idx < len(data) - 4 and data[idx : idx + 4] == b"\x00\x00\x00\x01":
            positions.append((idx, idx + 4))
            idx += 4
        else:
            idx += 1
    return positions


def _has_h264_keyframe(data: bytes) -> bool:
    for _start, header_pos in _start_code_positions(data):
        if header_pos >= len(data):
            continue
        nal_type = data[header_pos] & 0x1F
        if nal_type == 5:
            return True
    return False


def _has_h265_keyframe(data: bytes) -> bool:
    for _start, header_pos in _start_code_positions(data):
        if header_pos + 1 >= len(data):
            continue
        nal_type = (data[header_pos] >> 1) & 0x3F
        if 16 <= nal_type <= 21:
            return True
    return False


__all__ = [
    "BridgeProxy",
    "_OutboundMessage",
    "_SendManager",
    "_is_video_keyframe",
    "proxy",
]
