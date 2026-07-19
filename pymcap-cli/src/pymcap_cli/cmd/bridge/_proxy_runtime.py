"""Low-latency transform runtime for live bridge proxying.

Downstream delivery is owned by the transport's per-connection outbox
(:class:`robo_ws_bridge.server.ConnectionOutbox`): the proxy publishes each
transformed message with :meth:`WebSocketBridgeServer.publish_message` and the
outbox handles slow-client backpressure, latest-wins replacement, and drop
counting. This module only decodes, transforms, and re-encodes upstream
messages on a thread pool before handing them off.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Protocol

from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
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
    worker_count: int

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
    arrival_index: int = 0


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
    transform_stale_drops: int = 0
    transform_errors: int = 0
    adaptive_frames_dropped: int = 0


class TransformWorker:
    def __init__(
        self,
        *,
        channel: ChannelInfo,
        downstream_id: int,
        transformer: LiveTransformer,
        queue_size: int,
        emit: Callable[[OutboundMessage], Awaitable[None]],
        metrics: ProxyMetrics,
    ) -> None:
        self._channel = channel
        self._downstream_id = downstream_id
        self._transformer = transformer
        self._queue: asyncio.Queue[IncomingMessage] = asyncio.Queue(maxsize=max(1, queue_size))
        self._emit = emit
        self._metrics = metrics
        self._executor = ThreadPoolExecutor(max_workers=max(1, transformer.worker_count))
        self._decoder_factory = DecoderFactory()
        self._encoder_factory = ROS2EncoderFactory()
        self._decoder = self._decoder_for(channel)
        self._encoder = self._encoder_for(transformer)
        self._next_arrival_index = 0
        self._latest_emitted_index = -1
        self._tasks = tuple(
            asyncio.create_task(self._run()) for _ in range(max(1, transformer.worker_count))
        )

    def enqueue(self, message: IncomingMessage) -> None:
        message = replace(message, arrival_index=self._next_arrival_index)
        self._next_arrival_index += 1
        while self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
            self._metrics.transform_queue_drops += 1
        self._queue.put_nowait(message)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
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
                    if message.arrival_index < self._latest_emitted_index:
                        self._metrics.transform_stale_drops += 1
                        continue
                    self._latest_emitted_index = message.arrival_index
                    await self._emit(outbound)
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
