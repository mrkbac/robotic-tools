"""Low-latency queueing and transform runtime for live bridge proxying."""

import asyncio
import contextlib
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from robo_ws_bridge import ServerConnection, WebSocketBridgeServer
from robo_ws_bridge.ws_types import ChannelInfo
from small_mcap import Schema

MESSAGE_ENCODING = "cdr"
SCHEMA_ENCODING = "ros2msg"

logger = logging.getLogger(__name__)


class LiveTransformer(Protocol):
    output_schema_name: str
    output_schema_text: str
    output_schema_encoding: str
    output_message_encoding: str

    def transform(self, decoded: object, timestamp_ns: int) -> "TransformResult | None": ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TransformResult:
    payload: Mapping[str, object]
    is_compressed_video: bool = False
    is_keyframe: bool = False


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    timestamp_ns: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    channel_id: int
    timestamp_ns: int
    payload: bytes
    is_compressed_video: bool = False
    is_keyframe: bool = False


@dataclass(slots=True)
class ProxyMetrics:
    upstream_messages_received: int = 0
    upstream_messages_throttled: int = 0
    transformed_channel_count: int = 0
    transform_queue_drops: int = 0
    transform_errors: int = 0
    send_queue_drops: int = 0
    send_errors: int = 0
    video_packets_waiting_for_keyframe: int = 0


@dataclass(slots=True)
class SendSlot:
    websocket: ServerConnection
    subscription_id: int
    server: WebSocketBridgeServer
    metrics: ProxyMetrics
    queue_size: int
    queue: asyncio.Queue[OutboundMessage] = field(init=False)
    task: asyncio.Task[None] = field(init=False)
    needs_keyframe: bool = False

    def __post_init__(self) -> None:
        self.queue = asyncio.Queue(maxsize=max(1, self.queue_size))
        self.task = asyncio.create_task(self._run())

    def enqueue(self, message: OutboundMessage) -> None:
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


class SendManager:
    def __init__(
        self,
        server: WebSocketBridgeServer,
        metrics: ProxyMetrics,
        *,
        queue_size: int,
    ) -> None:
        self._server = server
        self._metrics = metrics
        self._queue_size = max(1, queue_size)
        self._slots: dict[tuple[ServerConnection, int], SendSlot] = {}

    def enqueue(
        self,
        websocket: ServerConnection,
        subscription_id: int,
        message: OutboundMessage,
    ) -> None:
        key = (websocket, subscription_id)
        slot = self._slots.get(key)
        if slot is None:
            slot = SendSlot(
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


class TransformWorker:
    def __init__(
        self,
        *,
        channel: ChannelInfo,
        downstream_id: int,
        transformer: LiveTransformer,
        queue_size: int,
        emit: Callable[[OutboundMessage], None],
        metrics: ProxyMetrics,
    ) -> None:
        self._channel = channel
        self._downstream_id = downstream_id
        self._transformer = transformer
        self._queue: asyncio.Queue[IncomingMessage] = asyncio.Queue(maxsize=max(1, queue_size))
        self._emit = emit
        self._metrics = metrics
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._decoder_factory = DecoderFactory()
        self._encoder_factory = ROS2EncoderFactory()
        self._decoder = self._decoder_for(channel)
        self._encoder = self._encoder_for(transformer)
        self._task = asyncio.create_task(self._run())

    def enqueue(self, message: IncomingMessage) -> None:
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

    def _process(self, message: IncomingMessage) -> OutboundMessage | None:
        decoded = self._decoder(message.payload)
        result = self._transformer.transform(decoded, message.timestamp_ns)
        if result is None:
            return None
        payload = bytes(self._encoder(result.payload))
        return OutboundMessage(
            channel_id=self._downstream_id,
            timestamp_ns=message.timestamp_ns,
            payload=payload,
            is_compressed_video=result.is_compressed_video,
            is_keyframe=result.is_keyframe,
        )

    def _decoder_for(self, channel: ChannelInfo) -> Callable[[bytes | memoryview], object]:
        schema = schema_from_channel(channel)
        decoder = self._decoder_factory.decoder_for(channel["encoding"], schema)
        if decoder is None:
            raise RuntimeError(
                f"No CDR decoder for {channel['topic']} ({channel.get('schemaName', '')})"
            )
        return decoder

    def _encoder_for(self, transformer: LiveTransformer) -> Callable[[object], bytes | memoryview]:
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


def schema_from_channel(channel: ChannelInfo) -> Schema:
    return Schema(
        id=channel["id"],
        name=channel.get("schemaName", ""),
        encoding=channel.get("schemaEncoding", SCHEMA_ENCODING),
        data=channel.get("schema", "").encode(),
    )
