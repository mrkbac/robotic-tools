from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
from typing import Any

import websockets
from websockets.client import WebSocketClientProtocol

from .ws_types import (
    BinaryOpCodes,
    ChannelInfo,
    ConnectionStatus,
    JsonOpCodes,
    RemoveStatusMessage,
    ServerInfoMessage,
    StatusMessage,
    SubscribeMessage,
    UnsubscribeMessage,
)

logger = logging.getLogger(__name__)


# Constants for binary message structure
_MESSAGE_DATA_HEADER_SIZE = 13  # 1 byte opcode + 4 bytes sub_id + 8 bytes timestamp
_TIME_MESSAGE_SIZE = 9  # 1 byte opcode + 8 bytes timestamp


class WebSocketBridgeClient:
    """Asynchronous client for the Foxglove WebSocket bridge protocol."""

    def __init__(
        self,
        url: str,
        *,
        subprotocol: str = "foxglove.websocket.v1",
        min_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        backoff_factor: float = 2.0,
    ) -> None:
        self._url = url
        self._subprotocol = subprotocol

        # Connection retry configuration
        self._min_retry_delay = min_retry_delay
        self._max_retry_delay = max_retry_delay
        self._backoff_factor = backoff_factor

        self._websocket: WebSocketClientProtocol | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._connection_task: asyncio.Task[None] | None = None
        self._connection_event = asyncio.Event()

        self._subscription_to_channel: dict[int, int] = {}
        self._next_subscription_id = 1
        self._active_subscriptions: set[int] = set()
        self._channel_to_subscription: dict[int, int] = {}

        # Subscription state tracking
        self._advertised_channels: dict[int, ChannelInfo] = {}

        self._subscribed_topics: set[str] = set()  # User's intended subscriptions
        self._intended_subscriptions: set[str] = set()  # Persist across disconnections

        # Connection state management
        self._connection_status = ConnectionStatus.DISCONNECTED
        self._should_connect = False
        self._running = False
        self._consecutive_failures = 0

        # Server info
        self._server_info: ServerInfoMessage | None = None

        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open the websocket connection and start receiving frames."""
        async with self._lock:
            if self._running:
                return
            logger.info("Connecting to %s", self._url)

            # Enable persistent connection attempts
            self._should_connect = True
            self._running = True

            # Start the persistent connection and message handling in the background
            self._connection_task = asyncio.create_task(self._connect_continuously())
            self._receiver_task = asyncio.create_task(self._handle_messages_loop())

    async def disconnect(self) -> None:
        """Close the connection and stop background tasks."""
        async with self._lock:
            if not self._running:
                return
            logger.info("Disconnecting from %s", self._url)

            # Stop persistent connection attempts
            self._should_connect = False
            self._running = False
            self._set_connection_status(ConnectionStatus.DISCONNECTED)

            # Cancel connection task
            if self._connection_task:
                self._connection_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._connection_task

            # Cancel message task
            if self._receiver_task:
                self._receiver_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._receiver_task

            if self._websocket and self._active_subscriptions:
                try:
                    unsubscribe_msg: UnsubscribeMessage = {
                        "op": JsonOpCodes.UNSUBSCRIBE.value,
                        "subscriptionIds": list(self._active_subscriptions),
                    }
                    await self._websocket.send(json.dumps(unsubscribe_msg))
                except (websockets.ConnectionClosed, OSError):
                    logger.debug("Failed to send unsubscribe on close")

            if self._websocket:
                await self._websocket.close()

            self._websocket = None
            self._receiver_task = None
            self._connection_task = None

    @property
    def channels(self) -> dict[int, ChannelInfo]:
        """Return the currently advertised channels."""
        return dict(self._advertised_channels)

    @property
    def server_info(self) -> ServerInfoMessage | None:
        """Return the cached serverInfo message."""
        return self._server_info

    def get_status(self) -> ConnectionStatus:
        """Get the current connection status."""
        return self._connection_status

    def _set_connection_status(self, status: ConnectionStatus) -> None:
        """Update connection status and notify via callback."""
        if self._connection_status != status:
            self._connection_status = status
            logger.debug(f"Connection status changed to: {status.value}")
            # Call the callback method for subclasses to override
            asyncio.create_task(self._on_connection_status_changed(status))

    async def _on_connection_status_changed(self, status: ConnectionStatus) -> None:
        """Callback for connection status changes. Override in subclasses."""
        logger.debug(f"Connection status: {status.value}")

    async def _backoff_sleep(self) -> None:
        """Sleep with exponential backoff based on consecutive failures."""
        delay = min(
            self._min_retry_delay * (self._backoff_factor**self._consecutive_failures),
            self._max_retry_delay,
        )
        logger.debug(
            f"Backing off for {delay:.1f} seconds (attempt #{self._consecutive_failures + 1})"
        )
        await asyncio.sleep(delay)

    async def _attempt_connection(self) -> None:
        """Attempt a single connection to the WebSocket server."""
        subprotocol = websockets.Subprotocol(self._subprotocol)
        self._websocket = await websockets.connect(self._url, subprotocols=[subprotocol])
        logger.info(f"Connected to {self._url}")

    async def _connect_continuously(self) -> None:
        """Keep trying to connect forever until successful or closed."""
        while self._should_connect:
            if not self._websocket:
                self._connection_event.clear()  # Clear event until connected
                self._set_connection_status(ConnectionStatus.CONNECTING)

                try:
                    await self._attempt_connection()
                    self._consecutive_failures = 0
                    self._set_connection_status(ConnectionStatus.CONNECTED)
                    self._connection_event.set()  # Signal connection established
                    logger.info("✅ WebSocket connected successfully")

                    # Restore subscriptions after successful connection
                    await self._restore_subscriptions()

                except Exception as e:  # noqa: BLE001
                    self._consecutive_failures += 1
                    logger.warning(f"Connection attempt #{self._consecutive_failures} failed: {e}")
                    if self._should_connect:  # Only sleep if we should still try
                        await self._backoff_sleep()
            else:
                # Wait while connected
                await asyncio.sleep(1.0)

    async def _restore_subscriptions(self) -> None:
        """Restore subscriptions after reconnection."""
        if not self._intended_subscriptions:
            return

        logger.info(
            f"Restoring {len(self._intended_subscriptions)} subscriptions after reconnection"
        )

        # Copy the set to avoid modification during iteration
        intended_subs = self._intended_subscriptions.copy()
        for topic in intended_subs:
            # Check if the topic is still advertised
            channel_id = None
            for channel in self._advertised_channels.values():
                if channel["topic"] == topic:
                    channel_id = channel["id"]
                    break

            if channel_id is not None:
                await self._subscribe_to_channel(channel_id)
            else:
                logger.debug(f"Topic {topic} not yet re-advertised, will subscribe when available")

    async def subscribe(self, topic: str) -> None:
        """Subscribe to messages from a topic."""
        if topic in self._subscribed_topics:
            logger.debug(f"Already subscribed to topic {topic}")
            return

        # Track the subscription intent persistently
        self._subscribed_topics.add(topic)
        self._intended_subscriptions.add(topic)

        if not self._running or not self._websocket:
            logger.debug("WebSocket not connected, subscription will be attempted when connected")
            return

        # Find channel for this topic
        channel_id = None
        for channel in self._advertised_channels.values():
            if channel["topic"] == topic:
                channel_id = channel["id"]
                break

        if channel_id is None:
            logger.debug(
                f"Topic {topic} not yet advertised by server, will subscribe when available"
            )
            return

        await self._subscribe_to_channel(channel_id)

    async def _subscribe_to_channel(self, channel_id: int) -> None:
        """Subscribe to a specific channel."""
        if not self._websocket:
            logger.warning("Cannot subscribe: not connected")
            return

        sub_id = self._next_subscription_id
        self._next_subscription_id += 1

        msg: SubscribeMessage = {
            "op": JsonOpCodes.SUBSCRIBE.value,
            "subscriptions": [{"id": sub_id, "channelId": channel_id}],
        }

        await self._websocket.send(json.dumps(msg))
        logger.info(f"Subscribed to channel {channel_id} with subscription ID {sub_id}")

        self._active_subscriptions.add(sub_id)
        self._subscription_to_channel[sub_id] = channel_id
        self._channel_to_subscription[channel_id] = sub_id

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from messages from a topic."""
        if topic not in self._subscribed_topics:
            logger.warning(f"Not subscribed to topic {topic}")
            return

        # Remove from both tracking sets
        self._subscribed_topics.remove(topic)
        self._intended_subscriptions.discard(topic)

        # Find and unsubscribe from channel if connected
        sub_id = None
        for channel in self._advertised_channels.values():
            if channel["topic"] == topic:
                sub_id = self._channel_to_subscription.get(channel["id"])
                break

        if sub_id is not None:
            await self._unsubscribe_from_channel(sub_id, topic)

    async def _unsubscribe_from_channel(self, sub_id: int, topic: str) -> None:
        """Unsubscribe from a specific channel."""
        if not self._websocket:
            logger.warning("Cannot unsubscribe: not connected")
            return

        msg: UnsubscribeMessage = {
            "op": JsonOpCodes.UNSUBSCRIBE.value,
            "subscriptionIds": [sub_id],
        }

        await self._websocket.send(json.dumps(msg))
        logger.info(f"Unsubscribed from topic {topic} (subscription ID {sub_id})")

        # Clean up tracking
        self._active_subscriptions.discard(sub_id)
        channel_id = self._subscription_to_channel.pop(sub_id, None)
        if channel_id is not None:
            self._channel_to_subscription.pop(channel_id, None)

    async def _handle_messages_loop(self) -> None:
        """Main message handling loop with automatic reconnection."""
        while self._should_connect:
            try:
                # Wait for connection to be established
                if not self._websocket:
                    await self._connection_event.wait()

                if not self._should_connect:
                    break

                # Process messages while connected
                if self._websocket is not None:
                    async for raw in self._websocket:
                        if isinstance(raw, bytes):
                            await self._handle_binary(raw)
                        elif isinstance(raw, str):
                            await self._handle_json(raw)
                        else:
                            logger.warning(f"Received unknown message type: {type(raw)}")

            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, will reconnect...")
                self._websocket = None
                self._connection_event.clear()  # Clear event when disconnected
                self._set_connection_status(ConnectionStatus.RECONNECTING)
                # Connection will be re-established by the connection task

            except Exception:
                logger.exception("Error in message loop")
                self._websocket = None
                self._connection_event.clear()  # Clear event on error
                self._set_connection_status(ConnectionStatus.RECONNECTING)
                # Brief pause before continuing loop
                await asyncio.sleep(1.0)

        self._running = False
        logger.info("Message loop stopped")

    async def _handle_json(self, text: str) -> None:
        """Handle JSON messages from the server."""
        try:
            msg = json.loads(text)
            op = msg.get("op")

            if op == JsonOpCodes.SERVER_INFO.value:
                await self._handle_server_info(msg)
            elif op == JsonOpCodes.STATUS.value:
                await self._handle_status(msg)
            elif op == JsonOpCodes.REMOVE_STATUS.value:
                await self._handle_remove_status(msg)
            elif op == JsonOpCodes.ADVERTISE.value:
                await self._handle_advertise(msg)
            elif op == JsonOpCodes.UNADVERTISE.value:
                await self._handle_unadvertise(msg)
            elif op == JsonOpCodes.PARAMETER_VALUES.value:
                pass  # TODO: Parameter handling
            elif op == JsonOpCodes.ADVERTISE_SERVICES.value:
                pass  # TODO: Advertise services handling
            elif op == JsonOpCodes.UNADVERTISE_SERVICES.value:
                pass  # TODO: Unadvertise services handling
            elif op == JsonOpCodes.SERVICE_CALL_FAILURE.value:
                pass  # TODO: Service call failure handling
            elif op == JsonOpCodes.CONNECTION_GRAPH_UPDATE.value:
                pass  # TODO: Connection graph handling
            else:
                logger.debug(f"Unknown JSON operation: {op}")
        except Exception:
            logger.exception(f"Failed to handle JSON message: {text}")

    async def _handle_server_info(self, msg: ServerInfoMessage) -> None:
        """Handle server info message."""
        self._server_info = msg

        # Extract and preprocess
        name = msg["name"]
        capabilities = msg["capabilities"]
        session_id = msg.get("sessionId")

        logger.info(f"Server: {name}")
        logger.info(f"Capabilities: {', '.join(capabilities)}")
        if session_id:
            logger.info(f"Session ID: {session_id}")

        # Call the callback with processed data
        await self.on_server_info(name, capabilities, session_id)

    async def on_server_info(
        self, name: str, capabilities: list[str], session_id: str | None
    ) -> None:
        """Callback for server info. Override in subclasses."""
        logger.debug(f"Server info: {name} with {len(capabilities)} capabilities")

    async def _handle_status(self, msg: StatusMessage) -> None:
        """Handle status message."""
        # Extract and preprocess
        level = msg["level"]
        message = msg["message"]
        status_id = msg.get("id")

        logger.debug(f"Status {status_id}: {message}")

        # Call the callback with processed data
        await self.on_status(level, message, status_id)

    async def on_status(self, level: int, message: str, status_id: str | None) -> None:
        """Callback for status messages. Override in subclasses."""
        logger.debug(f"Status message (level={level}): {message}")

    async def _handle_remove_status(self, msg: RemoveStatusMessage) -> None:
        """Handle remove status message."""
        logger.debug(f"Removing status messages: {', '.join(msg['statusIds'])}")

    async def _handle_advertise(self, msg: dict[str, Any]) -> None:
        """Handle topic advertisement from the server."""
        new_topics = []

        for ch in msg.get("channels", []):
            self._advertised_channels[ch["id"]] = ch
            new_topics.append(ch)

            logger.info(f"Topic advertised: {ch['topic']} (ID: {ch['id']})")

            # Subscribe if we were waiting for this topic
            if ch["topic"] in self._subscribed_topics:
                await self._subscribe_to_channel(ch["id"])

        if new_topics:
            for channel in new_topics:
                await self.on_advertised_channel(channel)

    async def on_advertised_channel(self, channel: ChannelInfo) -> None:
        """Callback for when a channel is advertised. Override in subclasses."""
        logger.debug(f"Channel advertised: {channel['topic']} (ID: {channel['id']})")

    async def _handle_unadvertise(self, msg: dict[str, Any]) -> None:
        """Handle topic unadvertisement from the server."""
        for channel_id in msg.get("channelIds", []):
            channel = self._advertised_channels.pop(channel_id, None)
            if channel:
                logger.info(f"Topic unadvertised: {channel['topic']}")
                await self.on_channel_unadvertised(channel)

    async def on_channel_unadvertised(self, channel: ChannelInfo) -> None:
        """Callback for when a channel is unadvertised. Override in subclasses."""
        logger.debug(f"Channel unadvertised: {channel['topic']} (ID: {channel['id']})")

    async def _handle_binary(self, data: bytes) -> None:
        """Handle binary message data."""
        opcode = data[0]
        if opcode == BinaryOpCodes.MESSAGE_DATA:
            await self._handle_message_data(data)
        elif opcode == BinaryOpCodes.TIME:
            await self._handle_time_data(data)
        elif opcode == BinaryOpCodes.SERVICE_CALL_RESPONSE:
            pass  # TODO: Implement service call response handling
        elif opcode == BinaryOpCodes.FETCH_ASSET_RESPONSE:
            pass  # TODO: Implement fetch asset response handling
        else:
            logger.debug(f"Unknown binary opcode: {opcode}")

    async def _handle_message_data(self, data: bytes) -> None:
        """Handle binary message data."""
        if len(data) < _MESSAGE_DATA_HEADER_SIZE:
            logger.warning("Invalid message data format")
            return

        sub_id = struct.unpack_from("<I", data, 1)[0]
        timestamp = struct.unpack_from("<Q", data, 5)[0]
        payload = data[_MESSAGE_DATA_HEADER_SIZE:]

        # Get the channel for this subscription
        channel_id = self._subscription_to_channel.get(sub_id)
        if channel_id is None:
            logger.warning(f"No channel mapping for subscription {sub_id}")
            return

        channel = self._advertised_channels.get(channel_id)
        if channel is None:
            logger.warning(f"No channel info for channel {channel_id}")
            return

        await self.on_message(channel, timestamp, payload)

    async def on_message(
        self,
        channel: ChannelInfo,
        timestamp: int,
        payload: bytes,
    ) -> None:
        """Callback for received messages. Override in subclasses."""
        logger.debug(
            f"Received message on topic {channel['topic']} "
            f"(channel ID: {channel['id']}, timestamp: {timestamp}, "
            f"payload size: {len(payload)} bytes)"
        )

    async def _handle_time_data(self, data: bytes) -> None:
        """Handle server time updates."""
        if len(data) >= _TIME_MESSAGE_SIZE:
            server_time = struct.unpack_from("<Q", data, 1)[0]
            await self.on_time_update(server_time)
        else:
            logger.warning("Invalid time message format")

    async def on_time_update(self, server_time: int) -> None:
        """Callback for server time updates. Override in subclasses."""
        logger.debug(f"Server time updated: {server_time}")

    # TODO: Implement service call support (protocol: Service Call Response/Request binary messages)

    # TODO: Implement fetch asset support (protocol: Fetch Asset Response binary message)
