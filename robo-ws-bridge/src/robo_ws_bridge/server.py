from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Literal, cast, overload

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed, InvalidHandshake
from websockets.typing import Subprotocol

from .ws_types import (
    AdvertiseMessage,
    BinaryOpCodes,
    ChannelInfo,
    JsonMessage,
    JsonOpCodes,
    PlaybackCommand,
    PlaybackControlRequest,
    PlaybackState,
    RemoveStatusMessage,
    SerializedTimestamp,
    ServerCapabilities,
    ServerInfoMessage,
    StatusLevel,
    StatusMessage,
    SubscribeMessage,
    UnadvertiseMessage,
    UnsubscribeMessage,
)

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]

Logger = logging.getLogger(__name__)


class _InvalidHandshakeLogFilter(logging.Filter):
    """Rewrite websockets' handshake-failure tracebacks into one-line warnings.

    Non-WebSocket clients (HTTP/1.0 text browsers, health checks, port scanners)
    otherwise produce a full ERROR traceback for every connection attempt.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info is None:
            return True
        exception = record.exc_info[1]
        if not isinstance(exception, InvalidHandshake):
            return True
        reason = str(exception)
        if exception.__cause__ is not None:
            reason = f"{reason} ({exception.__cause__})"
        record.msg = "rejected invalid WebSocket handshake: %s"
        record.args = (reason,)
        record.exc_info = None
        record.exc_text = None
        record.levelno = logging.WARNING
        record.levelname = logging.getLevelName(logging.WARNING)
        return True


_INVALID_HANDSHAKE_LOG_FILTER = _InvalidHandshakeLogFilter()


def install_invalid_handshake_log_filter() -> None:
    """Log invalid client handshakes as one-line warnings instead of tracebacks."""
    logging.getLogger("websockets.server").addFilter(_INVALID_HANDSHAKE_LOG_FILTER)


DeliveryPolicy = Literal["reliable", "latest"]
OfferResult = Literal["queued", "replaced", "overflow"]

_OUTBOX_SOFT_LIMIT_BYTES = 256 << 10
_OUTBOX_HARD_LIMIT_BYTES = 1 << 20
_TIME_OUTBOX_KEY = -1
_PLAYBACK_STATE_OUTBOX_KEY = -2
_CONTROL_OUTBOX_KEY = -3
_PLAYBACK_CONTROL_HEADER = struct.Struct("<BfBQI")
_PLAYBACK_STATE_HEADER = struct.Struct("<BBQfBI")
_MAX_TIMESTAMP_NS = ((1 << 32) - 1) * 1_000_000_000 + 999_999_999


@dataclass(frozen=True, slots=True)
class Channel:
    """Description of a server channel.

    ``delivery`` controls how frames queue for slow clients: ``"reliable"``
    delivers every published frame in order, ``"latest"`` keeps only the
    newest pending frame per channel (older unsent frames are replaced).
    """

    id: int
    topic: str
    encoding: str
    schema_name: str
    schema: str
    schema_encoding: str | None = None
    delivery: DeliveryPolicy = "reliable"

    def as_channel_info(self) -> ChannelInfo:
        payload: ChannelInfo = {
            "id": self.id,
            "topic": self.topic,
            "encoding": self.encoding,
            "schemaName": self.schema_name,
            "schema": self.schema,
        }
        if self.schema_encoding is not None:
            payload["schemaEncoding"] = self.schema_encoding
        return payload


@dataclass(frozen=True, slots=True)
class OutboxFrame:
    """One queued frame and the subscription key that owns it."""

    key: int
    payload: bytes | str


class ConnectionOutbox:
    """Bounded per-connection send queue that never blocks publishers.

    Frames for ``"latest"`` channels replace any unsent frame with the same
    key, so a slow client always receives the newest data. ``"reliable"``
    frames keep FIFO order; once ``hard_limit_bytes`` of frames are pending,
    overflow is reported so the endpoint can reset the slow connection.
    """

    def __init__(
        self,
        *,
        soft_limit_bytes: int = _OUTBOX_SOFT_LIMIT_BYTES,
        hard_limit_bytes: int = _OUTBOX_HARD_LIMIT_BYTES,
    ) -> None:
        self._soft_limit_bytes = soft_limit_bytes
        self._hard_limit_bytes = hard_limit_bytes
        self._reliable: deque[OutboxFrame] = deque()
        self._latest: dict[int, OutboxFrame] = {}
        self._in_flight: set[int] = set()
        self._has_frames = asyncio.Event()
        self._take_latest_next = False
        self.pending_bytes = 0
        self.dropped_frames = 0

    @property
    def is_congested(self) -> bool:
        """True while more bytes are pending than the soft limit allows."""
        return self.pending_bytes > self._soft_limit_bytes

    def offer(self, key: int, frame: bytes | str, *, delivery: DeliveryPolicy) -> OfferResult:
        """Queue one frame; replaces the pending frame for ``latest`` keys."""
        queued = OutboxFrame(key, frame)
        if delivery == "latest":
            previous = self._latest.get(key)
            if previous is not None:
                self.pending_bytes -= len(previous.payload)
                if key != _TIME_OUTBOX_KEY:
                    self.dropped_frames += 1
            self._latest[key] = queued
            result: OfferResult = "replaced" if previous is not None else "queued"
        else:
            if self.pending_bytes + len(frame) > self._hard_limit_bytes:
                self.dropped_frames += 1
                return "overflow"
            self._reliable.append(queued)
            result = "queued"
        self.pending_bytes += len(frame)
        self._has_frames.set()
        return result

    def discard(self, key: int) -> None:
        """Discard every pending frame owned by a subscription key."""
        removed = self._latest.pop(key, None)
        if removed is not None:
            self.pending_bytes -= len(removed.payload)
            self.dropped_frames += 1
        if not self._reliable:
            return
        remaining: deque[OutboxFrame] = deque()
        while self._reliable:
            frame = self._reliable.popleft()
            if frame.key == key:
                self.pending_bytes -= len(frame.payload)
                self.dropped_frames += 1
            else:
                remaining.append(frame)
        self._reliable = remaining

    def clear(self) -> None:
        """Drop every queued frame; in-flight sends are unaffected.

        Used on a hard seek to discard stale pre-seek frames. These are not
        counted as congestion drops because the caller is deliberately
        replacing the stream, not losing data to a slow client.
        """
        self._reliable.clear()
        self._latest.clear()
        self.pending_bytes = 0

    def clear_data(self) -> None:
        """Drop queued data and time frames while preserving control messages."""
        self._latest = {key: frame for key, frame in self._latest.items() if key < _TIME_OUTBOX_KEY}
        self._reliable = deque(frame for frame in self._reliable if frame.key < 0)
        self.pending_bytes = sum(
            len(frame.payload) for frame in (*self._reliable, *self._latest.values())
        )

    def is_key_busy(self, key: int) -> bool:
        """True while a subscription has an in-flight or pending frame."""
        return (
            key in self._in_flight
            or key in self._latest
            or any(frame.key == key for frame in self._reliable)
        )

    def complete(self, key: int) -> None:
        """Mark an in-flight frame as sent."""
        self._in_flight.discard(key)

    async def next_frame(self) -> OutboxFrame:
        """Wait for and remove the next frame to send."""
        while True:
            frame = self._pop()
            if frame is not None:
                self._in_flight.add(frame.key)
                return frame
            self._has_frames.clear()
            await self._has_frames.wait()

    def _pop(self) -> OutboxFrame | None:
        if not self._reliable and not self._latest:
            return None
        take_latest = self._latest and (self._take_latest_next or not self._reliable)
        self._take_latest_next = not self._take_latest_next
        if take_latest:
            key = next(iter(self._latest))
            frame = self._latest.pop(key)
        else:
            frame = self._reliable.popleft()
        self.pending_bytes -= len(frame.payload)
        return frame


@dataclass(slots=True)
class ConnectionState:
    """Mutable state for a connected client."""

    websocket: ServerConnection
    subscriptions: dict[int, int] = field(default_factory=dict)
    outbox: ConnectionOutbox = field(default_factory=ConnectionOutbox)
    close_task: asyncio.Task[None] | None = None


JsonHandler = Callable[[ConnectionState, JsonDict], Awaitable[None] | None]
BinaryHandler = Callable[[ConnectionState, bytes], Awaitable[None] | None]
ConnectionHandler = Callable[[ConnectionState], Awaitable[None] | None]
SubscriptionHandler = Callable[[ConnectionState, int, int], Awaitable[None] | None]
PlaybackControlHandler = Callable[
    [PlaybackControlRequest], Awaitable[PlaybackState] | PlaybackState
]


def _ensure_awaitable(result: Awaitable[None] | None) -> Awaitable[None]:
    if result is None:
        return asyncio.sleep(0)
    return result


class WebSocketBridgeEndpoint:
    """Foxglove WebSocket protocol state independent of a TCP listener."""

    def __init__(
        self,
        *,
        name: str = "websocket-bridge",
        capabilities: Iterable[str] = (),
        metadata: dict[str, str] | None = None,
        supported_encodings: Iterable[str] | None = None,
        session_id: str | None = None,
        playback_time_range: tuple[int, int] | None = None,
    ) -> None:
        self._name = name
        self._capabilities = list(capabilities)
        self._metadata = dict(metadata or {})
        self._supported_encodings = list(supported_encodings or [])
        self._session_id = uuid.uuid4().hex if session_id is None else session_id
        self._playback_time_range = self._validate_playback_time_range(playback_time_range)
        playback_capability = ServerCapabilities.PLAYBACK_CONTROL.value
        if self._playback_time_range is not None:
            if playback_capability not in self._capabilities:
                self._capabilities.append(playback_capability)
        elif playback_capability in self._capabilities:
            raise ValueError("playbackControl requires playback_time_range")

        self._channels: dict[int, Channel] = {}
        self._connections: dict[ServerConnection, ConnectionState] = {}
        self._json_handlers: dict[str, list[JsonHandler]] = {}
        self._binary_handlers: dict[int, list[BinaryHandler]] = {}

        self._on_connect: list[ConnectionHandler] = []
        self._on_disconnect: list[ConnectionHandler] = []
        self._on_subscribe: list[SubscriptionHandler] = []
        self._on_unsubscribe: list[SubscriptionHandler] = []
        self._playback_control_handler: PlaybackControlHandler | None = None
        self._playback_state: PlaybackState | None = None

        self._state_lock = asyncio.Lock()
        self._disconnected_dropped_frames = 0

    @staticmethod
    def _validate_playback_time_range(
        playback_time_range: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if playback_time_range is None:
            return None
        start_time, end_time = playback_time_range
        if not 0 <= start_time <= end_time <= _MAX_TIMESTAMP_NS:
            raise ValueError(
                "playback_time_range must be ordered, non-negative, and representable as "
                "Foxglove timestamps"
            )
        return playback_time_range

    async def close_connections(self) -> None:
        """Close every client currently attached to this endpoint."""
        to_close = list(self._connections.values())
        for state in to_close:
            await state.websocket.close()

    def register_channel(self, channel: Channel) -> None:
        """Register or replace a channel that should be advertised to clients."""
        self._channels[channel.id] = channel

    async def advertise_channel(self, channel: Channel, *, update_registry: bool = True) -> None:
        """Advertise a single channel to all clients."""
        if update_registry:
            self.register_channel(channel)
        message: AdvertiseMessage = {
            "op": JsonOpCodes.ADVERTISE.value,
            "channels": [channel.as_channel_info()],
        }
        await self._broadcast_json(message)

    async def advertise_channels(
        self,
        channels: Iterable[Channel],
        *,
        update_registry: bool = True,
    ) -> None:
        """Advertise multiple channels to all clients."""
        channel_infos: list[ChannelInfo] = []
        for channel in channels:
            if update_registry:
                self.register_channel(channel)
            channel_infos.append(channel.as_channel_info())
        if not channel_infos:
            return
        message: AdvertiseMessage = {
            "op": JsonOpCodes.ADVERTISE.value,
            "channels": channel_infos,
        }
        await self._broadcast_json(message)

    def unregister_channel(self, channel_id: int) -> None:
        """Remove a previously advertised channel."""
        self._channels.pop(channel_id, None)

    def register_json_handler(self, opcode: JsonOpCodes | str, handler: JsonHandler) -> None:
        """Register a coroutine to process JSON messages for a specific opcode."""
        op_value = opcode.value if isinstance(opcode, JsonOpCodes) else opcode
        self._json_handlers.setdefault(op_value, []).append(handler)

    def register_binary_handler(self, opcode: BinaryOpCodes | int, handler: BinaryHandler) -> None:
        """Register a coroutine to process binary frames for a specific opcode."""
        op_value = int(opcode)
        self._binary_handlers.setdefault(op_value, []).append(handler)

    def on_connect(self, handler: ConnectionHandler) -> None:
        """Attach a handler that runs when a client connects."""
        self._on_connect.append(handler)

    def on_disconnect(self, handler: ConnectionHandler) -> None:
        """Attach a handler that runs when a client disconnects."""
        self._on_disconnect.append(handler)

    def on_subscribe(self, handler: SubscriptionHandler) -> None:
        """Attach a handler that runs when a client subscribes to a channel."""
        self._on_subscribe.append(handler)

    def on_unsubscribe(self, handler: SubscriptionHandler) -> None:
        """Attach a handler that runs when a client unsubscribes from a channel."""
        self._on_unsubscribe.append(handler)

    def on_playback_control(self, handler: PlaybackControlHandler) -> None:
        """Set the handler for play, pause, speed, and seek requests."""
        self._playback_control_handler = handler

    @property
    def connections(self) -> list[ConnectionState]:
        """Return a snapshot of the active connections."""
        return list(self._connections.values())

    def get_subscriptions_for_channel(self, channel_id: int) -> list[tuple[ServerConnection, int]]:
        """Get all (websocket, subscription_id) pairs for a specific channel.

        This is useful for custom message routing in proxies and other advanced
        use cases where you need to know which clients are subscribed to a channel.

        Args:
            channel_id: The channel ID to query subscriptions for

        Returns:
            List of (websocket, subscription_id) tuples for all subscriptions to this channel
        """
        results: list[tuple[ServerConnection, int]] = []
        for state in self._connections.values():
            for sub_id, subscribed_channel in state.subscriptions.items():
                if subscribed_channel == channel_id:
                    results.append((state.websocket, sub_id))
        return results

    async def advertise_all(self) -> None:
        """Broadcast current channel advertisement to every connected client."""
        if not self._channels:
            return
        message: AdvertiseMessage = {
            "op": JsonOpCodes.ADVERTISE.value,
            "channels": [channel.as_channel_info() for channel in self._channels.values()],
        }
        await self._broadcast_json(message)

    async def unadvertise(self, channel_ids: Iterable[int]) -> None:
        """Broadcast unadvertise message to clients for the given channel ids."""
        message: UnadvertiseMessage = {
            "op": JsonOpCodes.UNADVERTISE.value,
            "channelIds": list(channel_ids),
        }
        await self._broadcast_json(message)

    async def publish_message(
        self,
        channel_id: int,
        payload: bytes,
        *,
        timestamp_ns: int | None = None,
    ) -> None:
        """Queue a binary message for all subscribers of a specific channel.

        Never blocks on slow clients: frames go through each connection's
        outbox and a per-connection sender task delivers them.
        """
        timestamp = timestamp_ns if timestamp_ns is not None else time.time_ns()
        frame_prefix = bytearray(1 + 4 + 8)
        frame_prefix[0] = int(BinaryOpCodes.MESSAGE_DATA)
        struct.pack_into("<Q", frame_prefix, 5, timestamp)
        channel = self._channels.get(channel_id)
        delivery: DeliveryPolicy = "reliable" if channel is None else channel.delivery

        for state in list(self._connections.values()):
            for subscription_id, subscribed_channel in state.subscriptions.items():
                if subscribed_channel != channel_id:
                    continue
                struct.pack_into("<I", frame_prefix, 1, subscription_id)
                result = state.outbox.offer(
                    subscription_id, bytes(frame_prefix) + payload, delivery=delivery
                )
                if result == "overflow" and state.close_task is None:
                    state.close_task = asyncio.create_task(
                        state.websocket.close(
                            code=1013,
                            reason="Reliable playback queue overflow",
                        )
                    )

    async def publish_time(self, timestamp_ns: int) -> None:
        """Broadcast the current server time to every connected client."""
        frame = struct.pack("<BQ", int(BinaryOpCodes.TIME), timestamp_ns)
        for state in list(self._connections.values()):
            state.outbox.offer(_TIME_OUTBOX_KEY, frame, delivery="latest")

    def broadcast_playback_state(self, playback_state: PlaybackState) -> None:
        """Queue a playback-state update for every connected client."""
        self._playback_state = replace(playback_state, request_id=None)
        for state in list(self._connections.values()):
            self._queue_playback_state(state, playback_state)

    def _queue_playback_state(
        self,
        state: ConnectionState,
        playback_state: PlaybackState,
    ) -> None:
        frame = self._encode_playback_state(playback_state)
        result = state.outbox.offer(
            _PLAYBACK_STATE_OUTBOX_KEY,
            frame,
            delivery="reliable",
        )
        if result == "overflow" and state.close_task is None:
            state.close_task = asyncio.create_task(
                state.websocket.close(
                    code=1013,
                    reason="Playback control queue overflow",
                )
            )

    def clear_pending_frames(self) -> None:
        """Drop every connection's queued frames, e.g. before a hard seek.

        Stale frames buffered for slow clients would otherwise arrive after the
        new stream restarts, making the client jump back to pre-seek data.
        """
        for state in self._connections.values():
            state.outbox.clear_data()

    def has_congested_subscriber(self, channel_id: int) -> bool:
        """True when a subscriber of the channel has a backed-up outbox."""
        for state in self._connections.values():
            if state.outbox.is_congested and channel_id in state.subscriptions.values():
                return True
        return False

    def are_all_subscribers_busy(self, channel_id: int) -> bool:
        """True when every subscription for a channel already has queued work."""
        subscriptions = [
            (state, subscription_id)
            for state in self._connections.values()
            for subscription_id, subscribed_channel in state.subscriptions.items()
            if subscribed_channel == channel_id
        ]
        return bool(subscriptions) and all(
            state.outbox.is_key_busy(subscription_id) for state, subscription_id in subscriptions
        )

    @property
    def dropped_frames(self) -> int:
        """Frames dropped or replaced across current and closed connections."""
        return self._disconnected_dropped_frames + sum(
            state.outbox.dropped_frames for state in self._connections.values()
        )

    async def _run_sender(self, state: ConnectionState) -> None:
        try:
            while True:
                frame = await state.outbox.next_frame()
                try:
                    await state.websocket.send(frame.payload)
                finally:
                    state.outbox.complete(frame.key)
        except ConnectionClosed:
            Logger.debug("Client disconnected; stopping outbox sender")
        except (OSError, RuntimeError):
            Logger.exception("Outbox sender failed; closing client connection")
            await state.websocket.close(code=1011, reason="Playback sender failed")

    async def send_message_to_subscription(
        self,
        websocket: ServerConnection,
        subscription_id: int,
        payload: bytes,
        *,
        timestamp_ns: int | None = None,
    ) -> None:
        """Send a binary message to a specific client subscription.

        This is useful for proxies and custom message routing logic where you need
        to send messages to specific clients rather than broadcasting to all
        subscribers of a channel.

        Args:
            websocket: The client websocket connection
            subscription_id: The subscription ID for this client
            payload: The message payload bytes
            timestamp_ns: Optional timestamp in nanoseconds (defaults to current time)
        """
        timestamp = timestamp_ns if timestamp_ns is not None else time.time_ns()
        frame_prefix = bytearray(1 + 4 + 8)
        frame_prefix[0] = int(BinaryOpCodes.MESSAGE_DATA)
        struct.pack_into("<I", frame_prefix, 1, subscription_id)
        struct.pack_into("<Q", frame_prefix, 5, timestamp)

        try:
            await websocket.send(bytes(frame_prefix) + payload)
        except ConnectionClosed:
            Logger.debug("Client disconnected while sending message")

    async def send_status(
        self,
        level: StatusLevel,
        message: str,
        *,
        status_id: str | None = None,
    ) -> None:
        """Broadcast a status JSON message to every client."""
        payload: StatusMessage = {
            "op": JsonOpCodes.STATUS.value,
            "level": int(level),
            "message": message,
        }
        if status_id is not None:
            payload["id"] = status_id
        await self._broadcast_json(payload)

    async def remove_status(self, status_ids: Iterable[str]) -> None:
        """Remove status messages from the Foxglove Problems panel by ID."""
        payload: RemoveStatusMessage = {
            "op": JsonOpCodes.REMOVE_STATUS.value,
            "statusIds": list(status_ids),
        }
        await self._broadcast_json(payload)

    async def clear_session(self, new_session_id: str | None = None) -> str:
        """Rotate the session ID and tell connected clients to clear cached state."""
        self._session_id = uuid.uuid4().hex if new_session_id is None else new_session_id
        await self._broadcast_json(self._server_info())
        return self._session_id

    async def _broadcast_json(self, message: JsonMessage) -> None:
        """Queue a reliable JSON control message for every connection."""
        frame = json.dumps(message)
        for state in list(self._connections.values()):
            result = state.outbox.offer(_CONTROL_OUTBOX_KEY, frame, delivery="reliable")
            if result == "overflow" and state.close_task is None:
                state.close_task = asyncio.create_task(
                    state.websocket.close(
                        code=1013,
                        reason="Control queue overflow",
                    )
                )

    async def handle_connection(self, websocket: ServerConnection) -> None:
        """Handle the lifetime of a single client connection."""
        state = ConnectionState(websocket=websocket)
        async with self._state_lock:
            self._connections[websocket] = state
        sender = asyncio.create_task(self._run_sender(state))
        try:
            await self._send_server_info(state)
            await self.advertise_all()
            if self._playback_state is not None:
                self._queue_playback_state(state, self._playback_state)
            for handler in self._on_connect:
                await _ensure_awaitable(handler(state))
            async for raw in websocket:
                if isinstance(raw, str):
                    await self._handle_json_frame(state, raw)
                else:
                    await self._handle_binary_frame(state, raw)
        except ConnectionClosed:
            Logger.debug("Connection closed for subprotocol %s", websocket.subprotocol)
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender
            if state.close_task is not None:
                with suppress(asyncio.CancelledError, ConnectionClosed):
                    await state.close_task
            async with self._state_lock:
                self._connections.pop(websocket, None)
                self._disconnected_dropped_frames += state.outbox.dropped_frames
            for handler in self._on_disconnect:
                await _ensure_awaitable(handler(state))

    async def _send_server_info(self, state: ConnectionState) -> None:
        await state.websocket.send(json.dumps(self._server_info()))

    def _server_info(self) -> ServerInfoMessage:
        message: ServerInfoMessage = {
            "op": JsonOpCodes.SERVER_INFO.value,
            "name": self._name,
            "capabilities": list(self._capabilities),
        }
        if self._supported_encodings:
            message["supportedEncodings"] = list(self._supported_encodings)
        if self._metadata:
            message["metadata"] = dict(self._metadata)
        message["sessionId"] = self._session_id
        if self._playback_time_range is not None:
            start_time, end_time = self._playback_time_range
            message["dataStartTime"] = self._serialize_timestamp(start_time)
            message["dataEndTime"] = self._serialize_timestamp(end_time)
        return message

    @staticmethod
    def _serialize_timestamp(timestamp_ns: int) -> SerializedTimestamp:
        return {
            "sec": timestamp_ns // 1_000_000_000,
            "nsec": timestamp_ns % 1_000_000_000,
        }

    async def _handle_json_frame(self, state: ConnectionState, payload: str) -> None:
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            Logger.warning("Invalid JSON payload from client")
            return

        if not isinstance(message, dict) or "op" not in message:
            Logger.debug("Ignoring JSON payload without opcode")
            return

        op_value = str(message["op"])

        if op_value == JsonOpCodes.SUBSCRIBE.value:
            await self._apply_subscriptions(state, cast("SubscribeMessage", message))
        elif op_value == JsonOpCodes.UNSUBSCRIBE.value:
            await self._remove_subscriptions(state, cast("UnsubscribeMessage", message))

        handlers = self._json_handlers.get(op_value, [])
        for handler in handlers:
            await _ensure_awaitable(handler(state, message))

    async def _handle_binary_frame(self, state: ConnectionState, payload: bytes) -> None:
        if not payload:
            return
        opcode = payload[0]
        if opcode == int(BinaryOpCodes.PLAYBACK_CONTROL_REQUEST):
            await self._handle_playback_control_request(state, payload)
        handlers = self._binary_handlers.get(opcode, [])
        for handler in handlers:
            await _ensure_awaitable(handler(state, payload))

    async def _handle_playback_control_request(
        self,
        state: ConnectionState,
        payload: bytes,
    ) -> None:
        if self._playback_time_range is None:
            await self._send_error(state, "Server does not support playback control")
            return
        if self._playback_control_handler is None:
            await self._send_error(state, "Server has no playback control handler")
            return
        try:
            request = self._decode_playback_control_request(payload)
        except (UnicodeDecodeError, ValueError):
            Logger.warning("Invalid playback control request")
            await state.websocket.close(code=1002, reason="Invalid playback control request")
            return

        try:
            result = self._playback_control_handler(request)
            if isinstance(result, PlaybackState):
                playback_state = result
            else:
                playback_state = await result
        except ValueError as error:
            await self._send_error(state, str(error))
            return
        self.broadcast_playback_state(replace(playback_state, request_id=request.request_id))

    @staticmethod
    def _decode_playback_control_request(payload: bytes) -> PlaybackControlRequest:
        if len(payload) < 1 + _PLAYBACK_CONTROL_HEADER.size:
            raise ValueError("playback control request is too short")
        command_value, playback_speed, had_seek, seek_time, request_id_length = (
            _PLAYBACK_CONTROL_HEADER.unpack_from(payload, 1)
        )
        request_id_start = 1 + _PLAYBACK_CONTROL_HEADER.size
        request_id_end = request_id_start + request_id_length
        if len(payload) < request_id_end:
            raise ValueError("playback control request ID is truncated")
        try:
            playback_command = PlaybackCommand(command_value)
        except ValueError as error:
            raise ValueError("invalid playback command") from error
        request_id = payload[request_id_start:request_id_end].decode("utf-8")
        return PlaybackControlRequest(
            playback_command=playback_command,
            playback_speed=playback_speed,
            seek_time=seek_time if had_seek else None,
            request_id=request_id,
        )

    @staticmethod
    def _encode_playback_state(playback_state: PlaybackState) -> bytes:
        request_id = playback_state.request_id.encode("utf-8") if playback_state.request_id else b""
        try:
            header = _PLAYBACK_STATE_HEADER.pack(
                int(BinaryOpCodes.PLAYBACK_STATE),
                int(playback_state.status),
                playback_state.current_time,
                playback_state.playback_speed,
                playback_state.did_seek,
                len(request_id),
            )
        except struct.error as error:
            raise ValueError("playback state contains an out-of-range value") from error
        return header + request_id

    @staticmethod
    async def _send_error(state: ConnectionState, message: str) -> None:
        payload: StatusMessage = {
            "op": JsonOpCodes.STATUS.value,
            "level": 2,
            "message": message,
        }
        await state.websocket.send(json.dumps(payload))

    async def _apply_subscriptions(self, state: ConnectionState, message: SubscribeMessage) -> None:
        subscriptions = message["subscriptions"]
        for entry in subscriptions:
            sub_id = entry["id"]
            channel_id = entry["channelId"]
            previous_channel_id = state.subscriptions.get(sub_id)
            if previous_channel_id is not None:
                state.outbox.discard(sub_id)
                for handler in self._on_unsubscribe:
                    await _ensure_awaitable(handler(state, sub_id, previous_channel_id))
            state.subscriptions[sub_id] = channel_id
            for handler in self._on_subscribe:
                await _ensure_awaitable(handler(state, sub_id, channel_id))

    async def _remove_subscriptions(
        self, state: ConnectionState, message: UnsubscribeMessage
    ) -> None:
        subscription_ids = message["subscriptionIds"]
        for sub_id in subscription_ids:
            channel_id = state.subscriptions.pop(sub_id, None)
            if channel_id is None:
                continue
            state.outbox.discard(sub_id)
            for handler in self._on_unsubscribe:
                await _ensure_awaitable(handler(state, sub_id, channel_id))

    @overload
    def get_connection(self, websocket: ServerConnection) -> ConnectionState | None: ...

    @overload
    def get_connection(self, websocket: None = ...) -> list[ConnectionState]: ...

    def get_connection(
        self,
        websocket: ServerConnection | None = None,
    ) -> ConnectionState | None | list[ConnectionState]:
        """Retrieve connection state for a specific websocket or all connections."""
        if websocket is None:
            return self.connections
        return self._connections.get(websocket)


class WebSocketBridgeServer(WebSocketBridgeEndpoint):
    """Foxglove protocol endpoint with its own WebSocket TCP listener."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        name: str = "websocket-bridge",
        subprotocol: str | None = None,
        capabilities: Iterable[str] = (),
        metadata: dict[str, str] | None = None,
        supported_encodings: Iterable[str] | None = None,
        session_id: str | None = None,
        playback_time_range: tuple[int, int] | None = None,
        max_message_size: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            capabilities=capabilities,
            metadata=metadata,
            supported_encodings=supported_encodings,
            session_id=session_id,
            playback_time_range=playback_time_range,
        )
        self._host = host
        self._port = port
        self._subprotocols = (
            (subprotocol,)
            if subprotocol is not None
            else ("foxglove.sdk.v1", "foxglove.websocket.v1")
        )
        self._max_message_size = max_message_size
        self._server: Server | None = None

    async def start(self) -> None:
        """Start listening for client connections."""
        if self._server is not None:
            raise RuntimeError("server already running")

        Logger.info("Starting WebSocket bridge server on %s:%d", self._host, self._port)
        install_invalid_handshake_log_filter()
        self._server = await serve(
            self.handle_connection,
            self._host,
            self._port,
            subprotocols=[Subprotocol(value) for value in self._subprotocols],
            max_size=self._max_message_size,
        )

    async def stop(self) -> None:
        """Stop accepting new connections and close existing ones."""
        if self._server is None:
            return

        Logger.info("Stopping WebSocket bridge server")
        await self.close_connections()
        self._server.close()
        await self._server.wait_closed()
        self._server = None
