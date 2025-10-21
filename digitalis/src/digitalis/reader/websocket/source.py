"""WebSocket source implementation with dynamic topic discovery."""

import asyncio
import contextlib
import json
import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import websockets
from mcap.well_known import SchemaEncoding
from mcap_ros2_support_fast.decoder import DecoderFunction, generate_dynamic

from digitalis.reader.source import Source, SourceStatus
from digitalis.reader.types import MessageEvent, SourceInfo, Topic

from .types import (
    AdvertiseServicesMessage,
    BinaryOpCodes,
    ConnectionGraphUpdateMessage,
    JsonOpCodes,
    ParameterValuesMessage,
    RemoveStatusMessage,
    ServerCapabilities,
    ServerInfoMessage,
    ServiceCallFailureMessage,
    StatusLevel,
    StatusMessage,
    SubscribeMessage,
    UnadvertiseServicesMessage,
    UnsubscribeMessage,
)

logger = logging.getLogger(__name__)

# Constants for binary message structure
MESSAGE_DATA_HEADER_SIZE = 13  # 1 byte opcode + 4 bytes sub_id + 8 bytes timestamp
TIME_MESSAGE_SIZE = 9  # 1 byte opcode + 8 bytes timestamp
SERVICE_RESPONSE_MIN_SIZE = 13  # 1 + 4 + 4 + 4 minimum
ASSET_RESPONSE_MIN_SIZE = 10  # 1 + 4 + 1 + 4 minimum


@dataclass
class AdvertisedChannel:
    """Information about an advertised channel."""

    id: int
    topic: str
    encoding: str
    schema_name: str
    schema: str
    schema_encoding: str


class ROS2DecodeError(Exception):
    """Raised if a message cannot be decoded as a ROS2 message."""


def get_decoder(schema: AdvertisedChannel, cache: dict[int, DecoderFunction]) -> DecoderFunction:
    """Get or create a decoder for a schema."""
    if schema is None or schema.schema_encoding != SchemaEncoding.ROS2:
        msg = f"Invalid schema for ROS2 decoding: {schema.schema_encoding}"
        raise ROS2DecodeError(msg)

    decoder = cache.get(schema.id)
    if decoder is None:
        decoder = generate_dynamic(schema.schema_name, schema.schema)
        cache[schema.id] = decoder
    return decoder


class WebSocketSource(Source):
    """WebSocket source for real-time data streaming with dynamic topic discovery."""

    def __init__(
        self,
        url: str,
        subprotocol: str = "foxglove.websocket.v1",
        min_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        backoff_factor: float = 2.0,
    ) -> None:
        self.url = url
        self.subprotocol = subprotocol

        # Connection retry configuration
        self._min_retry_delay = min_retry_delay
        self._max_retry_delay = max_retry_delay
        self._backoff_factor = backoff_factor

        self._ws: websockets.ClientConnection | None = None
        self._advertised_channels: dict[int, AdvertisedChannel] = {}
        self._topics: dict[str, Topic] = {}
        self._next_sub_id = 0
        self._active_subscriptions: set[int] = set()
        self._subscription_to_channel: dict[int, int] = {}
        self._channel_to_subscription: dict[int, int] = {}
        self._decoder_cache: dict[int, DecoderFunction] = {}
        # Subscription state tracking
        self._subscribed_topics: set[str] = set()  # User's intended subscriptions
        self._intended_subscriptions: set[str] = set()  # Persist across disconnections

        # Server capabilities and info
        self._server_capabilities: set[str] = set()
        self._server_name: str = ""
        self._server_session_id: str | None = None

        # Connection state management
        self._connection_status = SourceStatus.DISCONNECTED
        self._should_connect = False
        self._consecutive_failures = 0
        self._last_connection_attempt = 0.0
        self._connection_task: asyncio.Task | None = None
        self._connection_event = asyncio.Event()  # Signal when connection is established

        self._running = False
        self._play_back = True
        self._message_task: asyncio.Task | None = None

        # Callback handlers
        self._message_handler: Callable[[MessageEvent], None] | None = None
        self._source_info_handler: Callable[[SourceInfo], None] | None = None
        self._time_handler: Callable[[int], None] | None = None
        self._status_handler: Callable[[SourceStatus], None] | None = None

    async def initialize(self) -> SourceInfo:
        """Initialize the WebSocket connection with persistent retry logic."""
        logger.info(f"Initializing WebSocket source: {self.url}")

        # Notify that we're initializing
        if self._status_handler:
            self._status_handler(SourceStatus.INITIALIZING)

        # Enable persistent connection attempts
        self._should_connect = True
        self._running = True

        # Start the persistent connection and message handling in the background
        self._connection_task = asyncio.create_task(self._connect_continuously())
        self._message_task = asyncio.create_task(self._handle_messages_loop())

        logger.info("WebSocket source initialized - connecting in background")

        # Return initial empty source info - topics will be discovered dynamically
        source_info = SourceInfo(topics=[])

        # Notify source info handler
        if self._source_info_handler:
            self._source_info_handler(source_info)

        return source_info

    def start_playback(self) -> None:
        """Start or resume playback."""
        self._play_back = True

    def pause_playback(self) -> None:
        """Pause playback."""
        self._play_back = False

    async def subscribe(self, topic: str) -> None:
        """Subscribe to messages from a topic."""
        if topic in self._subscribed_topics:
            logger.debug(f"Already subscribed to topic {topic}")
            return

        # Track the subscription intent persistently
        self._subscribed_topics.add(topic)
        self._intended_subscriptions.add(topic)

        if not self._running or not self._ws:
            logger.debug("WebSocket not connected, subscription will be attempted when connected")
            return

        # Find channel for this topic
        channel_id = None
        for channel in self._advertised_channels.values():
            if channel.topic == topic:
                channel_id = channel.id
                break

        if channel_id is None:
            logger.debug(
                f"Topic {topic} not yet advertised by server, will subscribe when available"
            )
            return

        await self._subscribe_to_channel(channel_id)

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
            if channel.topic == topic:
                sub_id = self._channel_to_subscription.get(channel.id)
                break

        if sub_id is not None:
            await self._unsubscribe_from_channel(sub_id, topic)

    def set_message_handler(self, handler: Callable[[MessageEvent], None]) -> None:
        """Set the callback for handling incoming messages."""
        self._message_handler = handler

    def set_source_info_handler(self, handler: Callable[[SourceInfo], None]) -> None:
        """Set the callback for handling source info updates."""
        self._source_info_handler = handler

    def set_time_handler(self, handler: Callable[[int], None]) -> None:
        """Set the callback for handling time updates."""
        self._time_handler = handler

    def get_server_capabilities(self) -> set[str]:
        """Get the server capabilities."""
        return self._server_capabilities.copy()

    def get_server_info(self) -> dict[str, str | None]:
        """Get basic server information."""
        return {
            "name": self._server_name,
            "sessionId": self._server_session_id,
        }

    def has_capability(self, capability: ServerCapabilities) -> bool:
        """Check if the server has a specific capability."""
        return capability.value in self._server_capabilities

    def set_status_handler(self, handler: Callable[[SourceStatus], None]) -> None:
        """Set the callback for handling source status updates."""
        self._status_handler = handler

    def get_status(self) -> SourceStatus:
        """Get the current source status."""
        return self._connection_status

    async def close(self) -> None:
        """Clean up resources and close the connection."""
        logger.info("Closing WebSocket source")

        # Stop persistent connection attempts
        self._should_connect = False
        self._running = False
        self._set_connection_status(SourceStatus.DISCONNECTED)

        # Cancel connection task
        if self._connection_task:
            self._connection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._connection_task

        # Cancel message task
        if self._message_task:
            self._message_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._message_task

        if self._ws and self._active_subscriptions:
            try:
                unsubscribe_msg: UnsubscribeMessage = {
                    "op": JsonOpCodes.UNSUBSCRIBE.value,
                    "subscriptionIds": list(self._active_subscriptions),
                }
                await self._ws.send(json.dumps(unsubscribe_msg))
            except (websockets.ConnectionClosed, OSError):
                logger.debug("Failed to send unsubscribe on close")

        if self._ws:
            await self._ws.close()

        logger.info("WebSocket source closed")

    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        subprotocol = websockets.Subprotocol(self.subprotocol)
        self._ws = await websockets.connect(self.url, subprotocols=[subprotocol])
        logger.info(f"Connected to {self.url}")

    async def _subscribe_to_channel(self, channel_id: int) -> None:
        """Subscribe to a specific channel."""
        if not self._ws:
            logger.warning("Cannot subscribe: not connected")
            return

        sub_id = self._next_sub_id
        self._next_sub_id += 1

        msg: SubscribeMessage = {
            "op": JsonOpCodes.SUBSCRIBE.value,
            "subscriptions": [{"id": sub_id, "channelId": channel_id}],
        }

        await self._ws.send(json.dumps(msg))
        logger.info(f"Subscribed to channel {channel_id} with subscription ID {sub_id}")

        self._active_subscriptions.add(sub_id)
        self._subscription_to_channel[sub_id] = channel_id
        self._channel_to_subscription[channel_id] = sub_id

    async def _unsubscribe_from_channel(self, sub_id: int, topic: str) -> None:
        """Unsubscribe from a specific channel."""
        if not self._ws:
            logger.warning("Cannot unsubscribe: not connected")
            return

        msg: UnsubscribeMessage = {
            "op": JsonOpCodes.UNSUBSCRIBE.value,
            "subscriptionIds": [sub_id],
        }

        await self._ws.send(json.dumps(msg))
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
                if not self._ws:
                    await self._connection_event.wait()

                if not self._should_connect:
                    break

                # Process messages while connected
                if self._ws is not None:
                    async for raw in self._ws:
                        if isinstance(raw, bytes):
                            await self._handle_binary(raw)
                        elif isinstance(raw, str):
                            await self._handle_json(raw)
                        else:
                            logger.warning(f"Received unknown message type: {type(raw)}")

            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, will reconnect...")
                self._ws = None
                self._connection_event.clear()  # Clear event when disconnected
                self._set_connection_status(SourceStatus.RECONNECTING)
                # Connection will be re-established by the connection task

            except Exception:
                logger.exception("Error in message loop")
                self._ws = None
                self._connection_event.clear()  # Clear event on error
                self._set_connection_status(SourceStatus.RECONNECTING)
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
                await self._handle_parameter_values(msg)
            elif op == JsonOpCodes.ADVERTISE_SERVICES.value:
                await self._handle_advertise_services(msg)
            elif op == JsonOpCodes.UNADVERTISE_SERVICES.value:
                await self._handle_unadvertise_services(msg)
            elif op == JsonOpCodes.CONNECTION_GRAPH_UPDATE.value:
                await self._handle_connection_graph_update(msg)
            elif op == JsonOpCodes.SERVICE_CALL_FAILURE.value:
                await self._handle_service_call_failure(msg)
            else:
                logger.debug(f"Unknown JSON operation: {op}")
        except Exception:
            logger.exception(f"Failed to handle JSON message: {text}")

    async def _handle_advertise(self, msg: dict[str, Any]) -> None:
        """Handle topic advertisement from the server."""
        new_topics = []

        for ch in msg.get("channels", []):
            channel = AdvertisedChannel(
                id=ch["id"],
                topic=ch["topic"],
                encoding=ch["encoding"],
                schema_name=ch["schemaName"],
                schema=ch["schema"],
                schema_encoding=ch["schemaEncoding"],
            )

            self._advertised_channels[ch["id"]] = channel

            topic = Topic(
                name=ch["topic"],
                schema_name=ch["schemaName"],
                topic_id=ch["id"],
            )
            self._topics[ch["topic"]] = topic
            new_topics.append(topic)

            logger.info(f"Topic advertised: {ch['topic']} (ID: {ch['id']})")

            # Subscribe if we were waiting for this topic
            if ch["topic"] in self._subscribed_topics:
                await self._subscribe_to_channel(ch["id"])

        if new_topics:
            # Send all currently available topics
            all_topics = list(self._topics.values())

            # Update source info when topics change
            if self._source_info_handler:
                source_info = SourceInfo(topics=all_topics)
                self._source_info_handler(source_info)

    async def _handle_unadvertise(self, msg: dict[str, Any]) -> None:
        """Handle topic unadvertisement from the server."""
        for channel_id in msg.get("channelIds", []):
            channel = self._advertised_channels.pop(channel_id, None)
            if channel:
                self._topics.pop(channel.topic, None)
                logger.info(f"Topic unadvertised: {channel.topic}")

    async def _handle_binary(self, data: bytes) -> None:
        """Handle binary message data."""
        if not self._message_handler:
            return

        if not self._play_back:
            # If playback is paused, ignore incoming messages
            return

        opcode = data[0]
        if opcode == BinaryOpCodes.MESSAGE_DATA:
            await self._handle_message_data(data)
        elif opcode == BinaryOpCodes.TIME:
            await self._handle_time_data(data)
        elif opcode == BinaryOpCodes.SERVICE_CALL_RESPONSE:
            await self._handle_service_call_response(data)
        elif opcode == BinaryOpCodes.FETCH_ASSET_RESPONSE:
            await self._handle_fetch_asset_response(data)
        else:
            logger.debug(f"Unknown binary opcode: {opcode}")

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

    def _set_connection_status(self, status: SourceStatus) -> None:
        """Update connection status and notify handler."""
        if self._connection_status != status:
            self._connection_status = status
            logger.debug(f"Connection status changed to: {status.value}")

            # Notify generic status handler
            if self._status_handler:
                self._status_handler(status)

    async def _attempt_connection(self) -> None:
        """Attempt a single connection to the WebSocket server."""
        self._last_connection_attempt = time.time()
        subprotocol = websockets.Subprotocol(self.subprotocol)
        self._ws = await websockets.connect(self.url, subprotocols=[subprotocol])
        logger.info(f"Connected to {self.url}")

    async def _connect_continuously(self) -> None:
        """Keep trying to connect forever until successful or closed."""
        while self._should_connect:
            if not self._ws:
                self._connection_event.clear()  # Clear event until connected
                self._set_connection_status(SourceStatus.CONNECTING)

                try:
                    await self._attempt_connection()
                    self._consecutive_failures = 0
                    self._set_connection_status(SourceStatus.CONNECTED)
                    self._connection_event.set()  # Signal connection established
                    logger.info("✅ WebSocket connected successfully")

                    # Restore subscriptions after successful connection
                    await self._restore_subscriptions()

                except Exception as e:  # noqa: BLE001 TODO: improve
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
                if channel.topic == topic:
                    channel_id = channel.id
                    break

            if channel_id is not None:
                await self._subscribe_to_channel(channel_id)
            else:
                logger.debug(f"Topic {topic} not yet re-advertised, will subscribe when available")

    async def _handle_message_data(self, data: bytes) -> None:
        """Handle binary message data."""
        if len(data) < MESSAGE_DATA_HEADER_SIZE:
            logger.warning("Invalid message data format")
            return

        sub_id = struct.unpack_from("<I", data, 1)[0]
        timestamp = struct.unpack_from("<Q", data, 5)[0]
        payload = data[MESSAGE_DATA_HEADER_SIZE:]

        # Get the channel for this subscription
        channel_id = self._subscription_to_channel.get(sub_id)
        if channel_id is None:
            logger.warning(f"No channel mapping for subscription {sub_id}")
            return

        channel = self._advertised_channels.get(channel_id)
        if channel is None:
            logger.warning(f"No channel info for channel {channel_id}")
            return

        # Decode message
        message_obj: Any = payload
        try:
            decoder = get_decoder(channel, self._decoder_cache)
            message_obj = decoder(payload)
        except (ROS2DecodeError, ValueError):
            logger.debug(f"Failed to decode message for topic {channel.topic}")

        # Create message event
        message_event = MessageEvent(
            topic=channel.topic,
            message=message_obj,
            timestamp_ns=timestamp,
            schema_name=channel.schema_name,
        )

        if self._message_handler:
            self._message_handler(message_event)

        # Notify time handler with message timestamp
        if self._time_handler:
            self._time_handler(timestamp)

    async def _handle_time_data(self, data: bytes) -> None:
        """Handle server time updates."""
        if len(data) >= TIME_MESSAGE_SIZE:
            server_time = struct.unpack_from("<Q", data, 1)[0]
            logger.debug(f"Received server time: {server_time}")
            if self._time_handler:
                self._time_handler(server_time)
        else:
            logger.warning("Invalid time message format")

    async def _handle_server_info(self, msg: ServerInfoMessage) -> None:
        """Handle server info message."""
        self._server_name = msg["name"]
        self._server_capabilities = set(msg["capabilities"])
        self._server_session_id = msg.get("sessionId")

        logger.info(f"Server: {self._server_name}")
        logger.info(f"Capabilities: {', '.join(self._server_capabilities)}")
        if self._server_session_id:
            logger.info(f"Session ID: {self._server_session_id}")

    async def _handle_status(self, msg: StatusMessage) -> None:
        """Handle status message."""
        level_names = {
            StatusLevel.INFO.value: "INFO",
            StatusLevel.WARNING.value: "WARNING",
            StatusLevel.ERROR.value: "ERROR",
        }
        level_name = level_names.get(msg["level"], f"LEVEL_{msg['level']}")
        status_id = msg.get("id", "")
        id_part = f" [{status_id}]" if status_id else ""

        logger.log(
            logging.INFO
            if msg["level"] == StatusLevel.INFO.value
            else logging.WARNING
            if msg["level"] == StatusLevel.WARNING.value
            else logging.ERROR,
            f"Server {level_name}{id_part}: {msg['message']}",
        )

    async def _handle_remove_status(self, msg: RemoveStatusMessage) -> None:
        """Handle remove status message."""
        logger.debug(f"Removing status messages: {', '.join(msg['statusIds'])}")

    async def _handle_parameter_values(self, msg: ParameterValuesMessage) -> None:
        """Handle parameter values message."""
        request_id = msg.get("id", "")
        id_part = f" [{request_id}]" if request_id else ""
        logger.debug(f"Received parameter values{id_part}: {len(msg['parameters'])} parameters")

    async def _handle_advertise_services(self, msg: AdvertiseServicesMessage) -> None:
        """Handle advertise services message."""
        for service in msg["services"]:
            logger.info(
                f"Service advertised: {service['name']} "
                f"(ID: {service['id']}, Type: {service['type']})"
            )

    async def _handle_unadvertise_services(self, msg: UnadvertiseServicesMessage) -> None:
        """Handle unadvertise services message."""
        logger.info(f"Services unadvertised: {', '.join(map(str, msg['serviceIds']))}")

    async def _handle_connection_graph_update(self, msg: ConnectionGraphUpdateMessage) -> None:
        """Handle connection graph update message."""
        updates = []
        if "publishedTopics" in msg:
            updates.append(f"{len(msg['publishedTopics'])} published topics")
        if "subscribedTopics" in msg:
            updates.append(f"{len(msg['subscribedTopics'])} subscribed topics")
        if "advertisedServices" in msg:
            updates.append(f"{len(msg['advertisedServices'])} advertised services")
        if "removedTopics" in msg:
            updates.append(f"{len(msg['removedTopics'])} removed topics")
        if "removedServices" in msg:
            updates.append(f"{len(msg['removedServices'])} removed services")

        logger.debug(f"Connection graph update: {', '.join(updates)}")

    async def _handle_service_call_failure(self, msg: ServiceCallFailureMessage) -> None:
        """Handle service call failure message."""
        logger.warning(
            f"Service call failed (service: {msg['serviceId']}, "
            f"call: {msg['callId']}): {msg['message']}"
        )

    async def _handle_service_call_response(self, data: bytes) -> None:
        """Handle service call response binary message."""
        if len(data) < SERVICE_RESPONSE_MIN_SIZE:
            logger.warning("Invalid service call response format")
            return

        service_id = struct.unpack_from("<I", data, 1)[0]
        call_id = struct.unpack_from("<I", data, 5)[0]
        encoding_length = struct.unpack_from("<I", data, 9)[0]

        if len(data) < SERVICE_RESPONSE_MIN_SIZE + encoding_length:
            logger.warning("Invalid service call response format: encoding length")
            return

        encoding = data[
            SERVICE_RESPONSE_MIN_SIZE : SERVICE_RESPONSE_MIN_SIZE + encoding_length
        ].decode("utf-8")
        payload = data[SERVICE_RESPONSE_MIN_SIZE + encoding_length :]

        logger.debug(
            f"Service call response (service: {service_id}, call: {call_id}, "
            f"encoding: {encoding}, payload size: {len(payload)} bytes)"
        )

    async def _handle_fetch_asset_response(self, data: bytes) -> None:
        """Handle fetch asset response binary message."""
        if len(data) < ASSET_RESPONSE_MIN_SIZE:
            logger.warning("Invalid fetch asset response format")
            return

        request_id = struct.unpack_from("<I", data, 1)[0]
        status = data[5]
        error_msg_length = struct.unpack_from("<I", data, 6)[0]

        if len(data) < ASSET_RESPONSE_MIN_SIZE + error_msg_length:
            logger.warning("Invalid fetch asset response format: error message length")
            return

        error_message = (
            data[10 : 10 + error_msg_length].decode("utf-8") if error_msg_length > 0 else ""
        )
        asset_data = data[10 + error_msg_length :]

        if status == 0:
            logger.debug(
                f"Asset fetched successfully (request: {request_id}, size: {len(asset_data)} bytes)"
            )
        else:
            logger.warning(f"Asset fetch failed (request: {request_id}): {error_message}")
