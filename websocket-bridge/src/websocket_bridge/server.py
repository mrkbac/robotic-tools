from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import overload

from websockets.exceptions import ConnectionClosed
from websockets.server import Serve, WebSocketServerProtocol, serve

from .ws_types import AdvertiseMessage, BinaryOpCodes, ChannelInfo, JsonMessage, JsonOpCodes

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]

Logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Channel:
    """Description of a server channel."""

    id: int
    topic: str
    encoding: str
    schema_name: str
    schema: str
    schema_encoding: str | None = None

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


@dataclass(slots=True)
class ConnectionState:
    """Mutable state for a connected client."""

    websocket: WebSocketServerProtocol
    subscriptions: dict[int, int] = field(default_factory=dict)


JsonHandler = Callable[[ConnectionState, JsonDict], Awaitable[None] | None]
BinaryHandler = Callable[[ConnectionState, bytes], Awaitable[None] | None]
ConnectionHandler = Callable[[ConnectionState], Awaitable[None] | None]
SubscriptionHandler = Callable[[ConnectionState, int, int], Awaitable[None] | None]


def _ensure_awaitable(result: Awaitable[None] | None) -> Awaitable[None]:
    if result is None:
        return asyncio.sleep(0)
    return result


class WebSocketBridgeServer:
    """Asynchronous WebSocket server that speaks the Foxglove bridge protocol."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        name: str = "websocket-bridge",
        subprotocol: str = "foxglove.websocket.v1",
        capabilities: Iterable[str] = (),
        metadata: dict[str, str] | None = None,
        supported_encodings: Iterable[str] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._name = name
        self._subprotocol = subprotocol
        self._capabilities = list(capabilities)
        self._metadata = dict(metadata or {})
        self._supported_encodings = list(supported_encodings or [])

        self._channels: dict[int, Channel] = {}
        self._connections: dict[WebSocketServerProtocol, ConnectionState] = {}
        self._json_handlers: dict[str, list[JsonHandler]] = {}
        self._binary_handlers: dict[int, list[BinaryHandler]] = {}

        self._on_connect: list[ConnectionHandler] = []
        self._on_disconnect: list[ConnectionHandler] = []
        self._on_subscribe: list[SubscriptionHandler] = []
        self._on_unsubscribe: list[SubscriptionHandler] = []

        self._server: Serve | None = None
        self._state_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start listening for client connections."""
        if self._server is not None:
            raise RuntimeError("server already running")

        Logger.info("Starting WebSocket bridge server on %s:%d", self._host, self._port)
        self._server = await serve(
            self._handle_connection,
            self._host,
            self._port,
            subprotocols=[self._subprotocol],
        )

    async def stop(self) -> None:
        """Stop accepting new connections and close existing ones."""
        if self._server is None:
            return

        Logger.info("Stopping WebSocket bridge server")

        # Close all active connections first
        to_close = list(self._connections.values())
        for state in to_close:
            await state.websocket.close()

        self._server.close()
        await self._server.wait_closed()
        self._server = None

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

    @property
    def connections(self) -> list[ConnectionState]:
        """Return a snapshot of the active connections."""
        return list(self._connections.values())

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
        payload = {
            "op": JsonOpCodes.UNADVERTISE.value,
            "channelIds": list(channel_ids),
        }
        await self._broadcast_json(payload)

    async def publish_message(
        self,
        channel_id: int,
        payload: bytes,
        *,
        timestamp_ns: int | None = None,
    ) -> None:
        """Send a binary message to all subscribers of a specific channel."""
        timestamp = timestamp_ns if timestamp_ns is not None else time.time_ns()
        frame_prefix = bytearray(1 + 4 + 8)
        frame_prefix[0] = int(BinaryOpCodes.MESSAGE_DATA)
        struct.pack_into("<Q", frame_prefix, 5, timestamp)

        for state in self._connections.values():
            subscription_ids = [
                subscription_id
                for subscription_id, subscribed_channel in state.subscriptions.items()
                if subscribed_channel == channel_id
            ]
            if not subscription_ids:
                continue
            for subscription_id in subscription_ids:
                struct.pack_into("<I", frame_prefix, 1, subscription_id)
                try:
                    await state.websocket.send(bytes(frame_prefix) + payload)
                except ConnectionClosed:
                    Logger.debug("Skipping closed connection during publish")

    async def send_status(
        self,
        level: int,
        message: str,
        *,
        status_id: str | None = None,
    ) -> None:
        """Broadcast a status JSON message to every client."""
        payload: JsonMessage = {
            "op": JsonOpCodes.STATUS.value,
            "level": level,
            "message": message,
        }
        if status_id is not None:
            payload["id"] = status_id
        await self._broadcast_json(payload)

    async def _broadcast_json(self, message: JsonDict) -> None:
        """Send a JSON message to all connections."""
        frame = json.dumps(message)
        for state in list(self._connections.values()):
            try:
                await state.websocket.send(frame)
            except ConnectionClosed:
                Logger.debug("Failed broadcast to closed connection")

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle the lifetime of a single client connection."""
        state = ConnectionState(websocket=websocket)
        async with self._state_lock:
            self._connections[websocket] = state
        try:
            await self._send_server_info(state)
            await self.advertise_all()
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
            async with self._state_lock:
                self._connections.pop(websocket, None)
            for handler in self._on_disconnect:
                await _ensure_awaitable(handler(state))

    async def _send_server_info(self, state: ConnectionState) -> None:
        message: JsonMessage = {
            "op": JsonOpCodes.SERVER_INFO.value,
            "name": self._name,
            "capabilities": list(self._capabilities),
        }
        if self._supported_encodings:
            message["supportedEncodings"] = list(self._supported_encodings)
        if self._metadata:
            message["metadata"] = dict(self._metadata)
        await state.websocket.send(json.dumps(message))

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
            await self._apply_subscriptions(state, message)
        elif op_value == JsonOpCodes.UNSUBSCRIBE.value:
            await self._remove_subscriptions(state, message)

        handlers = self._json_handlers.get(op_value, [])
        for handler in handlers:
            await _ensure_awaitable(handler(state, message))

    async def _handle_binary_frame(self, state: ConnectionState, payload: bytes) -> None:
        if not payload:
            return
        opcode = payload[0]
        handlers = self._binary_handlers.get(opcode, [])
        for handler in handlers:
            await _ensure_awaitable(handler(state, payload))

    async def _apply_subscriptions(self, state: ConnectionState, message: JsonDict) -> None:
        subscriptions = message.get("subscriptions")
        if not isinstance(subscriptions, list):
            return
        for entry in subscriptions:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("id"), int)
                and isinstance(entry.get("channelId"), int)
            ):
                sub_id = entry["id"]
                channel_id = entry["channelId"]
                state.subscriptions[sub_id] = channel_id
                for handler in self._on_subscribe:
                    await _ensure_awaitable(handler(state, sub_id, channel_id))

    async def _remove_subscriptions(self, state: ConnectionState, message: JsonDict) -> None:
        subscription_ids = message.get("subscriptionIds")
        if not isinstance(subscription_ids, list):
            return
        for sub_id in subscription_ids:
            if isinstance(sub_id, int):
                channel_id = state.subscriptions.pop(sub_id, None)
                if channel_id is None:
                    continue
                for handler in self._on_unsubscribe:
                    await _ensure_awaitable(handler(state, sub_id, channel_id))

    @overload
    def get_connection(self, websocket: WebSocketServerProtocol) -> ConnectionState | None: ...

    @overload
    def get_connection(self, websocket: None = ...) -> list[ConnectionState]: ...

    def get_connection(
        self,
        websocket: WebSocketServerProtocol | None = None,
    ) -> ConnectionState | None | list[ConnectionState]:
        """Retrieve connection state for a specific websocket or all connections."""
        if websocket is None:
            return self.connections
        return self._connections.get(websocket)
