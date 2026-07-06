from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import websockets
from websockets.typing import Subprotocol

from .ws_types import (
    AdvertisedService,
    AdvertiseMessage,
    AdvertiseServicesMessage,
    BinaryOpCodes,
    ChannelInfo,
    ConnectionGraphUpdateMessage,
    ConnectionStatus,
    FetchAssetMessage,
    GetParametersMessage,
    JsonOpCodes,
    Parameter,
    ParameterValuesMessage,
    PublishedTopic,
    RemoveStatusMessage,
    ServerInfoMessage,
    ServiceCallFailureMessage,
    ServiceInfo,
    SetParametersMessage,
    StatusMessage,
    SubscribeConnectionGraphMessage,
    SubscribedTopic,
    SubscribeMessage,
    UnadvertiseMessage,
    UnadvertiseServicesMessage,
    UnsubscribeConnectionGraphMessage,
    UnsubscribeMessage,
)

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


# Handler type aliases
ConnectHandler = Callable[[], Awaitable[None] | None]
DisconnectHandler = Callable[[], Awaitable[None] | None]
ReconnectingHandler = Callable[[], Awaitable[None] | None]
ServerInfoHandler = Callable[[str, list[str], str | None], Awaitable[None] | None]
StatusHandler = Callable[[int, str, str | None], Awaitable[None] | None]
RemoveStatusHandler = Callable[[list[str]], Awaitable[None] | None]
ChannelHandler = Callable[[ChannelInfo], Awaitable[None] | None]
MessageHandler = Callable[[ChannelInfo, int, bytes], Awaitable[None] | None]
TimeUpdateHandler = Callable[[int], Awaitable[None] | None]


@dataclass(frozen=True)
class ConnectionGraph:
    """Snapshot of the most recently observed connection-graph state."""

    published_topics: tuple[PublishedTopic, ...]
    subscribed_topics: tuple[SubscribedTopic, ...]
    advertised_services: tuple[AdvertisedService, ...]


ConnectionGraphHandler = Callable[[ConnectionGraph], Awaitable[None] | None]


@dataclass(frozen=True)
class ServiceCallResponse:
    """Successful response to a service call. The payload encoding matches the request."""

    service_id: int
    call_id: int
    encoding: str
    payload: bytes


class ServiceCallError(RuntimeError):
    """Raised when the bridge reports a ``serviceCallFailure`` for our call."""


class FetchAssetError(RuntimeError):
    """Raised when the bridge reports a failed ``fetchAsset`` response."""


# Constants for binary message structure
_MESSAGE_DATA_HEADER_SIZE = 13  # 1 byte opcode + 4 bytes sub_id + 8 bytes timestamp
_TIME_MESSAGE_SIZE = 9  # 1 byte opcode + 8 bytes timestamp
_SERVICE_CALL_RESPONSE_HEADER_SIZE = (
    13  # 1 byte opcode + 4 bytes service id + 4 call id + 4 enc len
)
_FETCH_ASSET_RESPONSE_HEADER_SIZE = 10  # 1 byte opcode + 4 request id + 1 status + 4 error len


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
        max_message_size: int | None = None,
    ) -> None:
        self._url = url
        self._subprotocol = subprotocol

        # Connection retry configuration
        self._min_retry_delay = min_retry_delay
        self._max_retry_delay = max_retry_delay
        self._backoff_factor = backoff_factor
        self._max_message_size = max_message_size

        self._websocket: ClientConnection | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._connection_task: asyncio.Task[None] | None = None
        self._connection_event = asyncio.Event()

        self._subscription_to_channel: dict[int, int] = {}
        self._next_subscription_id = 1
        self._active_subscriptions: set[int] = set()
        self._channel_to_subscription: dict[int, int] = {}

        # Service-call correlation: call id -> future awaiting the response.
        self._next_call_id = 1
        self._pending_calls: dict[int, asyncio.Future[ServiceCallResponse]] = {}

        # Parameter get/set correlation: request id -> future awaiting parameterValues.
        self._next_param_request_id = 1
        self._pending_param_requests: dict[str, asyncio.Future[list[Parameter]]] = {}

        # Asset-fetch correlation: request id -> future awaiting the asset bytes.
        self._next_asset_request_id = 1
        self._pending_asset_requests: dict[int, asyncio.Future[bytes]] = {}

        # Client-published channels use ids the client allocates.
        self._next_client_channel_id = 1

        # Subscription state tracking
        self._advertised_channels: dict[int, ChannelInfo] = {}
        self._advertised_services: dict[int, ServiceInfo] = {}

        self._subscribed_topics: set[str] = set()  # User's intended subscriptions
        self._intended_subscriptions: set[str] = set()  # Persist across disconnections

        # Connection state management
        self._connection_status = ConnectionStatus.DISCONNECTED
        self._should_connect = False
        self._running = False
        self._consecutive_failures = 0

        # Server info
        self._server_info: ServerInfoMessage | None = None

        # Connection graph state (populated when subscribed; keyed by name)
        self._graph_published: dict[str, PublishedTopic] = {}
        self._graph_subscribed: dict[str, SubscribedTopic] = {}
        self._graph_services: dict[str, AdvertisedService] = {}
        self._wants_connection_graph = False

        self._lock = asyncio.Lock()

        # Handler storage lists
        self._on_connect: list[ConnectHandler] = []
        self._on_disconnect: list[DisconnectHandler] = []
        self._on_reconnecting: list[ReconnectingHandler] = []
        self._on_server_info: list[ServerInfoHandler] = []
        self._on_status: list[StatusHandler] = []
        self._on_remove_status: list[RemoveStatusHandler] = []
        self._on_advertised_channel: list[ChannelHandler] = []
        self._on_channel_unadvertised: list[ChannelHandler] = []
        self._on_message: list[MessageHandler] = []
        self._on_time_update: list[TimeUpdateHandler] = []
        self._on_connection_graph_update: list[ConnectionGraphHandler] = []

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
            await self._set_connection_status(ConnectionStatus.DISCONNECTED)

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
    def services(self) -> dict[int, ServiceInfo]:
        """Return the currently advertised services."""
        return dict(self._advertised_services)

    @property
    def server_info(self) -> ServerInfoMessage | None:
        """Return the cached serverInfo message."""
        return self._server_info

    @property
    def connection_graph(self) -> ConnectionGraph:
        """Return a snapshot of the most recent connection-graph state."""
        return ConnectionGraph(
            published_topics=tuple(self._graph_published.values()),
            subscribed_topics=tuple(self._graph_subscribed.values()),
            advertised_services=tuple(self._graph_services.values()),
        )

    def get_connection_status(self) -> ConnectionStatus:
        """Get the current connection status."""
        return self._connection_status

    def on_connect(self, handler: ConnectHandler) -> None:
        """Register a callback for connection established events."""
        self._on_connect.append(handler)

    def on_disconnect(self, handler: DisconnectHandler) -> None:
        """Register a callback for connection closed events."""
        self._on_disconnect.append(handler)

    def on_reconnecting(self, handler: ReconnectingHandler) -> None:
        """Register a callback for reconnection attempt events."""
        self._on_reconnecting.append(handler)

    def on_server_info(self, handler: ServerInfoHandler) -> None:
        """Register a callback for server info messages."""
        self._on_server_info.append(handler)

    def on_status(self, handler: StatusHandler) -> None:
        """Register a callback for status messages."""
        self._on_status.append(handler)

    def on_remove_status(self, handler: RemoveStatusHandler) -> None:
        """Register a callback for removeStatus messages."""
        self._on_remove_status.append(handler)

    def on_advertised_channel(self, handler: ChannelHandler) -> None:
        """Register a callback for channel advertisement events."""
        self._on_advertised_channel.append(handler)

    def on_channel_unadvertised(self, handler: ChannelHandler) -> None:
        """Register a callback for channel unadvertisement events."""
        self._on_channel_unadvertised.append(handler)

    def on_message(self, handler: MessageHandler) -> None:
        """Register a callback for received messages."""
        self._on_message.append(handler)

    def on_time_update(self, handler: TimeUpdateHandler) -> None:
        """Register a callback for server time updates."""
        self._on_time_update.append(handler)

    def on_connection_graph_update(self, handler: ConnectionGraphHandler) -> None:
        """Register a callback for connection graph changes."""
        self._on_connection_graph_update.append(handler)

    async def subscribe_connection_graph(self) -> None:
        """Ask the server to start sending connectionGraphUpdate messages."""
        self._wants_connection_graph = True
        if not self._websocket:
            logger.debug("WebSocket not connected, connection-graph subscription deferred")
            return
        msg: SubscribeConnectionGraphMessage = {"op": JsonOpCodes.SUBSCRIBE_CONNECTION_GRAPH.value}
        await self._websocket.send(json.dumps(msg))

    async def unsubscribe_connection_graph(self) -> None:
        """Stop receiving connectionGraphUpdate messages from the server."""
        self._wants_connection_graph = False
        self._graph_published.clear()
        self._graph_subscribed.clear()
        self._graph_services.clear()
        if not self._websocket:
            return
        msg: UnsubscribeConnectionGraphMessage = {
            "op": JsonOpCodes.UNSUBSCRIBE_CONNECTION_GRAPH.value
        }
        await self._websocket.send(json.dumps(msg))

    async def _invoke_handlers(
        self, handlers: list[Callable[..., Awaitable[None] | None]], *args: object
    ) -> None:
        """Invoke a list of handlers with the given arguments.

        Handles both sync and async handlers.
        """
        for handler in handlers:
            try:
                result = handler(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Error in handler")

    async def _set_connection_status(self, status: ConnectionStatus) -> None:
        """Update connection status and notify via callback."""
        if self._connection_status != status:
            self._connection_status = status
            logger.debug("Connection status changed to: %s", status.value)

            # Call appropriate handlers based on status
            if status == ConnectionStatus.CONNECTED:
                await self._invoke_handlers(self._on_connect)
            elif status == ConnectionStatus.DISCONNECTED:
                await self._invoke_handlers(self._on_disconnect)
            elif status == ConnectionStatus.RECONNECTING:
                await self._invoke_handlers(self._on_reconnecting)

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
        self._websocket = await websockets.connect(
            self._url,
            subprotocols=[Subprotocol("foxglove.websocket.v1"), Subprotocol("foxglove.sdk.v1")],
            max_size=self._max_message_size,
        )
        logger.info(f"Connected to {self._url}")

    async def _connect_continuously(self) -> None:
        """Keep trying to connect forever until successful or closed."""
        while self._should_connect:
            if not self._websocket:
                self._connection_event.clear()  # Clear event until connected
                await self._set_connection_status(ConnectionStatus.CONNECTING)

                try:
                    await self._attempt_connection()
                    self._consecutive_failures = 0
                    await self._set_connection_status(ConnectionStatus.CONNECTED)
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
        if self._intended_subscriptions:
            logger.info(
                f"Restoring {len(self._intended_subscriptions)} subscriptions after reconnection"
            )
            for topic in self._intended_subscriptions.copy():
                channel_id = None
                for channel in self._advertised_channels.values():
                    if channel["topic"] == topic:
                        channel_id = channel["id"]
                        break

                if channel_id is not None:
                    await self._subscribe_to_channel(channel_id)
                else:
                    logger.debug(
                        "Topic %s not yet re-advertised, will subscribe when available", topic
                    )
        if self._wants_connection_graph:
            await self.subscribe_connection_graph()

    async def subscribe(self, topic: str) -> None:
        """Subscribe to messages from a topic."""
        if topic in self._subscribed_topics:
            logger.debug("Already subscribed to topic %s", topic)
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

    @property
    def is_connected(self) -> bool:
        """Check if the client is currently connected to the server.

        Returns:
            True if connected, False otherwise
        """
        return self._websocket is not None and self._running

    async def subscribe_to_channel(self, subscription_id: int, channel_id: int) -> None:
        """Subscribe to a specific channel with a custom subscription ID.

        This is an advanced method for proxy/bridge use cases where you need
        control over subscription IDs. For normal usage, use subscribe(topic) instead.

        Args:
            subscription_id: Custom subscription ID to use
            channel_id: Channel ID to subscribe to

        Raises:
            RuntimeError: If not connected to server
        """
        if not self._websocket:
            raise RuntimeError("Not connected to server")

        msg: SubscribeMessage = {
            "op": JsonOpCodes.SUBSCRIBE.value,
            "subscriptions": [{"id": subscription_id, "channelId": channel_id}],
        }
        await self._websocket.send(json.dumps(msg))
        logger.info(f"Subscribed to channel {channel_id} with subscription ID {subscription_id}")

        # Update internal state tracking (required for message routing in _handle_message_data)
        self._active_subscriptions.add(subscription_id)
        self._subscription_to_channel[subscription_id] = channel_id
        self._channel_to_subscription[channel_id] = subscription_id

    async def unsubscribe_from_channel(self, subscription_id: int) -> None:
        """Unsubscribe using a specific subscription ID.

        This is an advanced method for proxy/bridge use cases. For normal usage,
        use unsubscribe(topic) instead.

        Args:
            subscription_id: Subscription ID to unsubscribe

        Raises:
            RuntimeError: If not connected to server
        """
        if not self._websocket:
            raise RuntimeError("Not connected to server")

        msg: UnsubscribeMessage = {
            "op": JsonOpCodes.UNSUBSCRIBE.value,
            "subscriptionIds": [subscription_id],
        }
        await self._websocket.send(json.dumps(msg))
        logger.info(f"Unsubscribed subscription ID {subscription_id}")

        # Clean up internal state tracking (mirrors _unsubscribe_from_channel behavior)
        self._active_subscriptions.discard(subscription_id)
        channel_id = self._subscription_to_channel.pop(subscription_id, None)
        if channel_id is not None:
            self._channel_to_subscription.pop(channel_id, None)

    async def call_service(
        self,
        service_id: int,
        payload: bytes,
        *,
        encoding: str,
        timeout: float = 10.0,
    ) -> ServiceCallResponse:
        """Call an advertised service and await its response.

        Sends a ``SERVICE_CALL_REQUEST`` binary frame and resolves once the matching
        ``SERVICE_CALL_RESPONSE`` arrives. The payload is treated as opaque bytes; the
        caller is responsible for encoding the request and decoding the response
        according to ``encoding`` (one of the server's supported encodings).

        Args:
            service_id: Advertised service id (see :attr:`services`).
            payload: Encoded request bytes.
            encoding: Payload encoding, echoed back on the response.
            timeout: Seconds to wait for the response.

        Returns:
            The service response.

        Raises:
            RuntimeError: If the client is not connected.
            ServiceCallError: If the server reports a service-call failure.
            asyncio.TimeoutError: If no response arrives within ``timeout``.
        """
        if self._websocket is None:
            raise RuntimeError("Cannot call service while disconnected")

        call_id = self._next_call_id
        self._next_call_id += 1
        future: asyncio.Future[ServiceCallResponse] = asyncio.get_running_loop().create_future()
        self._pending_calls[call_id] = future

        encoding_bytes = encoding.encode("ascii")
        frame = (
            struct.pack(
                "<BIII",
                int(BinaryOpCodes.SERVICE_CALL_REQUEST),
                service_id,
                call_id,
                len(encoding_bytes),
            )
            + encoding_bytes
            + payload
        )
        try:
            await self._websocket.send(frame)
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending_calls.pop(call_id, None)

    async def advertise(
        self,
        topic: str,
        *,
        encoding: str,
        schema_name: str,
        schema: str = "",
        schema_encoding: str | None = None,
    ) -> int:
        """Advertise a client channel so this client can publish to ``topic``.

        Only allowed if the server declared the ``clientPublish`` capability. Returns
        the client-chosen channel id to pass to :meth:`publish`.
        """
        if self._websocket is None:
            raise RuntimeError("Cannot advertise while disconnected")
        channel_id = self._next_client_channel_id
        self._next_client_channel_id += 1
        channel: ChannelInfo = {
            "id": channel_id,
            "topic": topic,
            "encoding": encoding,
            "schemaName": schema_name,
            "schema": schema,
        }
        if schema_encoding is not None:
            channel["schemaEncoding"] = schema_encoding
        message: AdvertiseMessage = {"op": JsonOpCodes.ADVERTISE.value, "channels": [channel]}
        await self._websocket.send(json.dumps(message))
        return channel_id

    async def publish(self, channel_id: int, payload: bytes) -> None:
        """Publish an encoded message on a previously :meth:`advertise`\\ d channel."""
        if self._websocket is None:
            raise RuntimeError("Cannot publish while disconnected")
        frame = struct.pack("<BI", int(BinaryOpCodes.CLIENT_MESSAGE_DATA), channel_id) + payload
        await self._websocket.send(frame)

    async def unadvertise(self, channel_id: int) -> None:
        """Withdraw a client channel previously created with :meth:`advertise`."""
        if self._websocket is None:
            return
        message: UnadvertiseMessage = {
            "op": JsonOpCodes.UNADVERTISE.value,
            "channelIds": [channel_id],
        }
        await self._websocket.send(json.dumps(message))

    async def get_parameters(
        self, names: list[str] | None = None, *, timeout: float = 5.0
    ) -> list[Parameter]:
        """Fetch parameter values from the server (all parameters if ``names`` is empty).

        Only allowed if the server declared the ``parameters`` capability.
        """
        if self._websocket is None:
            raise RuntimeError("Cannot get parameters while disconnected")
        request_id = f"get-{self._next_param_request_id}"
        self._next_param_request_id += 1
        future: asyncio.Future[list[Parameter]] = asyncio.get_running_loop().create_future()
        self._pending_param_requests[request_id] = future
        message: GetParametersMessage = {
            "op": JsonOpCodes.GET_PARAMETERS.value,
            "parameterNames": names or [],
            "id": request_id,
        }
        try:
            await self._websocket.send(json.dumps(message))
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending_param_requests.pop(request_id, None)

    async def set_parameters(
        self, parameters: list[Parameter], *, timeout: float = 5.0
    ) -> list[Parameter]:
        """Set parameters and return the server-confirmed values.

        Only allowed if the server declared the ``parameters`` capability.
        """
        if self._websocket is None:
            raise RuntimeError("Cannot set parameters while disconnected")
        request_id = f"set-{self._next_param_request_id}"
        self._next_param_request_id += 1
        future: asyncio.Future[list[Parameter]] = asyncio.get_running_loop().create_future()
        self._pending_param_requests[request_id] = future
        message: SetParametersMessage = {
            "op": JsonOpCodes.SET_PARAMETERS.value,
            "parameters": parameters,
            "id": request_id,
        }
        try:
            await self._websocket.send(json.dumps(message))
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending_param_requests.pop(request_id, None)

    async def fetch_asset(self, uri: str, *, timeout: float = 10.0) -> bytes:
        """Fetch an asset (e.g. a URDF or mesh) by URI and return its bytes.

        Only allowed if the server declared the ``assets`` capability.

        Raises:
            FetchAssetError: If the server reports a failed fetch.
            asyncio.TimeoutError: If no response arrives within ``timeout``.
        """
        if self._websocket is None:
            raise RuntimeError("Cannot fetch asset while disconnected")
        request_id = self._next_asset_request_id
        self._next_asset_request_id += 1
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._pending_asset_requests[request_id] = future
        message: FetchAssetMessage = {
            "op": JsonOpCodes.FETCH_ASSET.value,
            "uri": uri,
            "requestId": request_id,
        }
        try:
            await self._websocket.send(json.dumps(message))
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending_asset_requests.pop(request_id, None)

    async def _send_json(self, message: str) -> bool:
        """Send a raw JSON message to the server (internal use only).

        Args:
            message: JSON string to send

        Returns:
            True if message was sent, False if not connected
        """
        if not self._websocket:
            return False
        await self._websocket.send(message)
        return True

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
                logger.exception("WebSocket connection closed, will reconnect...")
                self._websocket = None
                self._connection_event.clear()  # Clear event when disconnected
                await self._set_connection_status(ConnectionStatus.RECONNECTING)
                # Connection will be re-established by the connection task

            except Exception:
                logger.exception("Error in message loop")
                self._websocket = None
                self._connection_event.clear()  # Clear event on error
                await self._set_connection_status(ConnectionStatus.RECONNECTING)
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
            elif op == JsonOpCodes.ADVERTISE_SERVICES.value:
                await self._handle_advertise_services(msg)
            elif op == JsonOpCodes.UNADVERTISE_SERVICES.value:
                await self._handle_unadvertise_services(msg)
            elif op == JsonOpCodes.CONNECTION_GRAPH_UPDATE.value:
                await self._handle_connection_graph_update(msg)
            elif op == JsonOpCodes.SERVICE_CALL_FAILURE.value:
                await self._handle_service_call_failure(msg)
            elif op == JsonOpCodes.PARAMETER_VALUES.value:
                self._handle_parameter_values(msg)
            else:
                logger.debug("Unknown JSON operation: %s", op)
        except Exception:
            logger.exception("Failed to handle JSON message: %s", text)

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

        # Call registered handlers
        await self._invoke_handlers(self._on_server_info, name, capabilities, session_id)

    async def _handle_status(self, msg: StatusMessage) -> None:
        """Handle status message."""
        # Extract and preprocess
        level = msg["level"]
        message = msg["message"]
        status_id = msg.get("id")

        logger.debug("Status %s: %s", status_id, message)

        # Call registered handlers
        await self._invoke_handlers(self._on_status, level, message, status_id)

    async def _handle_remove_status(self, msg: RemoveStatusMessage) -> None:
        """Handle remove status message."""
        status_ids = msg["statusIds"]
        logger.debug("Removing status messages: %s", ", ".join(status_ids))
        await self._invoke_handlers(self._on_remove_status, status_ids)

    async def _handle_advertise(self, msg: AdvertiseMessage) -> None:
        """Handle topic advertisement from the server."""
        new_channels: list[ChannelInfo] = []

        for ch in msg["channels"]:
            self._advertised_channels[ch["id"]] = ch
            new_channels.append(ch)

            logger.info(f"Topic advertised: {ch['topic']} (ID: {ch['id']})")

            if ch["topic"] in self._subscribed_topics:
                await self._subscribe_to_channel(ch["id"])

        for channel in new_channels:
            await self._invoke_handlers(self._on_advertised_channel, channel)

    async def _handle_unadvertise(self, msg: UnadvertiseMessage) -> None:
        """Handle topic unadvertisement from the server."""
        for channel_id in msg["channelIds"]:
            channel = self._advertised_channels.pop(channel_id, None)
            if channel:
                logger.info(f"Topic unadvertised: {channel['topic']}")
                await self._invoke_handlers(self._on_channel_unadvertised, channel)

    async def _handle_advertise_services(self, msg: AdvertiseServicesMessage) -> None:
        """Handle service advertisement from the server."""
        for svc in msg["services"]:
            self._advertised_services[svc["id"]] = svc
            logger.info(f"Service advertised: {svc['name']} (ID: {svc['id']})")

    async def _handle_unadvertise_services(self, msg: UnadvertiseServicesMessage) -> None:
        """Handle service unadvertisement from the server."""
        for service_id in msg["serviceIds"]:
            svc = self._advertised_services.pop(service_id, None)
            if svc:
                logger.info(f"Service unadvertised: {svc['name']}")

    async def _handle_connection_graph_update(self, msg: ConnectionGraphUpdateMessage) -> None:
        """Apply an incremental connection-graph update and notify handlers."""
        for entry in msg.get("publishedTopics", []) or []:
            self._graph_published[entry["name"]] = entry
        for entry in msg.get("subscribedTopics", []) or []:
            self._graph_subscribed[entry["name"]] = entry
        for entry in msg.get("advertisedServices", []) or []:
            self._graph_services[entry["name"]] = entry
        for name in msg.get("removedTopics", []) or []:
            self._graph_published.pop(name, None)
            self._graph_subscribed.pop(name, None)
        for name in msg.get("removedServices", []) or []:
            self._graph_services.pop(name, None)
        await self._invoke_handlers(self._on_connection_graph_update, self.connection_graph)

    async def _handle_binary(self, data: bytes) -> None:
        """Handle binary message data."""
        opcode = data[0]
        if opcode == BinaryOpCodes.MESSAGE_DATA:
            await self._handle_message_data(data)
        elif opcode == BinaryOpCodes.TIME:
            await self._handle_time_data(data)
        elif opcode == BinaryOpCodes.SERVICE_CALL_RESPONSE:
            self._handle_service_call_response(data)
        elif opcode == BinaryOpCodes.FETCH_ASSET_RESPONSE:
            self._handle_fetch_asset_response(data)
        else:
            logger.debug("Unknown binary opcode: %d", opcode)

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

        await self._invoke_handlers(self._on_message, channel, timestamp, payload)

    def _handle_service_call_response(self, data: bytes) -> None:
        """Resolve the pending call awaiting this service-call response."""
        if len(data) < _SERVICE_CALL_RESPONSE_HEADER_SIZE:
            logger.warning("Invalid service call response format")
            return
        service_id, call_id = struct.unpack_from("<II", data, 1)
        encoding_len = struct.unpack_from("<I", data, 9)[0]
        encoding = data[13 : 13 + encoding_len].decode("ascii")
        payload = bytes(data[13 + encoding_len :])

        future = self._pending_calls.get(call_id)
        if future is None or future.done():
            logger.debug("No pending call for service-call response call_id=%d", call_id)
            return
        future.set_result(
            ServiceCallResponse(
                service_id=service_id,
                call_id=call_id,
                encoding=encoding,
                payload=payload,
            )
        )

    async def _handle_service_call_failure(self, msg: ServiceCallFailureMessage) -> None:
        """Reject the pending call the server reports as failed."""
        call_id = msg["callId"]
        future = self._pending_calls.get(call_id)
        if future is None or future.done():
            logger.debug("No pending call for serviceCallFailure call_id=%d", call_id)
            return
        future.set_exception(ServiceCallError(msg["message"]))

    def _handle_parameter_values(self, msg: ParameterValuesMessage) -> None:
        """Resolve the pending get/set request awaiting these parameter values."""
        request_id = msg.get("id")
        if request_id is None:
            return
        future = self._pending_param_requests.get(request_id)
        if future is None or future.done():
            logger.debug("No pending parameter request for id=%s", request_id)
            return
        future.set_result(msg["parameters"])

    def _handle_fetch_asset_response(self, data: bytes) -> None:
        """Resolve the pending fetch awaiting this asset response."""
        if len(data) < _FETCH_ASSET_RESPONSE_HEADER_SIZE:
            logger.warning("Invalid fetch asset response format")
            return
        request_id = struct.unpack_from("<I", data, 1)[0]
        status = data[5]
        error_len = struct.unpack_from("<I", data, 6)[0]
        error_message = data[10 : 10 + error_len].decode("utf-8", "replace")
        asset_data = bytes(data[10 + error_len :])

        future = self._pending_asset_requests.get(request_id)
        if future is None or future.done():
            logger.debug("No pending fetch for asset response request_id=%d", request_id)
            return
        if status == 0:
            future.set_result(asset_data)
        else:
            future.set_exception(
                FetchAssetError(error_message or f"fetch failed with status {status}")
            )

    async def _handle_time_data(self, data: bytes) -> None:
        """Handle server time updates."""
        if len(data) >= _TIME_MESSAGE_SIZE:
            server_time = struct.unpack_from("<Q", data, 1)[0]
            await self._invoke_handlers(self._on_time_update, server_time)
        else:
            logger.warning("Invalid time message format")
