"""Foxglove WebSocket proxy bridge implementation."""

import asyncio
import contextlib
import json
import logging
import struct
from collections import defaultdict
from typing import Any

import websockets
from mcap_ros2._dynamic import generate_dynamic, serialize_dynamic
from websockets.asyncio.server import ServerConnection, serve

from .schemas import get_schema
from .transformers import TransformerRegistry, TransformError
from .types import (
    AdvertiseMessage,
    BinaryOpCodes,
    ChannelInfo,
    JsonOpCodes,
    ServerInfoMessage,
    SubscribeMessage,
    UnadvertiseMessage,
    UnsubscribeMessage,
)

logger = logging.getLogger(__name__)


class ProxyBridge:
    """Foxglove WebSocket proxy bridge that forwards messages between upstream and downstream."""

    def __init__(
        self,
        upstream_url: str,
        listen_host: str = "0.0.0.0",  # noqa: S104
        listen_port: int = 8766,
        transformer_registry: TransformerRegistry | None = None,
        default_throttle_hz: float | None = 1.0,
        topic_throttle_overrides: dict[str, float | None] | None = None,
    ) -> None:
        """Initialize the proxy bridge.

        Args:
            upstream_url: WebSocket URL of the upstream Foxglove bridge
            listen_host: Host to listen on for downstream clients
            listen_port: Port to listen on for downstream clients
            transformer_registry: Optional transformer registry for message transformations
            default_throttle_hz: Default throttle rate in Hz for all topics
                (None disables throttling)
            topic_throttle_overrides: Optional per-topic throttle overrides in Hz
        """
        self.upstream_url = upstream_url
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.transformer_registry = transformer_registry or TransformerRegistry()
        self.default_throttle_hz = default_throttle_hz
        self.topic_throttle_overrides = dict(topic_throttle_overrides or {})

        # Upstream connection
        self.upstream_ws: websockets.ClientConnection | None = None
        self.upstream_connected = asyncio.Event()

        # Server info from upstream
        self.server_info: ServerInfoMessage | None = None

        # Channel tracking - stores DOWNSTREAM channels (what clients see)
        # Key is downstream channel ID (original or transformed)
        self.advertised_channels: dict[int, dict[str, Any]] = {}

        # Track upstream channel info for transformation lookup
        self.upstream_channels: dict[int, dict[str, Any]] = {}

        # Transformed channel tracking
        # Maps upstream channel ID -> downstream transformed channel ID
        self.transformed_channels: dict[int, int] = {}
        # Maps transformed channel ID -> upstream channel ID
        self.transformed_to_upstream: dict[int, int] = {}
        self.next_channel_id = 10000  # Start transformed channels at high ID to avoid conflicts

        # Topic throttling (per upstream channel)
        self.channel_throttle_hz: dict[int, float | None] = {}
        self.channel_last_sent_time: dict[int, float] = {}

        # Downstream client tracking
        self.downstream_clients: set[ServerConnection] = set()
        self.client_subscriptions: dict[
            ServerConnection, dict[int, int]
        ] = {}  # client -> {client_sub_id: channel_id}

        # Upstream subscription tracking (reference counting)
        self.upstream_subscriptions: dict[int, int] = {}  # channel_id -> upstream_sub_id
        self.upstream_sub_refcount: dict[int, int] = defaultdict(int)  # channel_id -> count
        self.next_upstream_sub_id = 0

        # Message encoder/decoder cache (generated from schema)
        self.message_decoders: dict[str, Any] = {}  # schema_name -> decoder function
        self.message_encoders: dict[str, Any] = {}  # schema_name -> encoder function

        # Running state
        self.running = False
        self.server_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the proxy bridge."""
        logger.info(
            f"Starting Foxglove proxy bridge: {self.upstream_url} -> ws://{self.listen_host}:{self.listen_port}"
        )
        self.running = True

        # Start upstream connection
        upstream_task = asyncio.create_task(self._connect_upstream())

        # Start downstream server
        self.server_task = asyncio.create_task(self._run_server())

        # Wait for both tasks
        await asyncio.gather(upstream_task, self.server_task)

    async def stop(self) -> None:
        """Stop the proxy bridge."""
        logger.info("Stopping proxy bridge")
        self.running = False

        # Close all downstream clients
        if self.downstream_clients:
            await asyncio.gather(
                *[client.close() for client in self.downstream_clients],
                return_exceptions=True,
            )

        # Close upstream connection
        if self.upstream_ws:
            await self.upstream_ws.close()

        # Cancel server task
        if self.server_task:
            self.server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.server_task

    async def _connect_upstream(self) -> None:
        """Connect to upstream Foxglove bridge with reconnection logic."""
        retry_delay = 1.0
        max_retry_delay = 30.0

        while self.running:
            try:
                logger.info(f"Connecting to upstream: {self.upstream_url}")
                subprotocol = websockets.Subprotocol("foxglove.websocket.v1")
                self.upstream_ws = await websockets.connect(
                    self.upstream_url, subprotocols=[subprotocol]
                )
                logger.info("Connected to upstream bridge")

                # Signal connection established
                self.upstream_connected.set()
                retry_delay = 1.0  # Reset retry delay on successful connection

                # Handle upstream messages
                async for message in self.upstream_ws:
                    if isinstance(message, str):
                        await self._handle_upstream_json(message)
                    elif isinstance(message, bytes):
                        await self._handle_upstream_binary(message)

            except websockets.ConnectionClosed:
                logger.warning("Upstream connection closed, reconnecting...")
                self.upstream_connected.clear()
                self.upstream_ws = None
            except Exception:
                logger.exception("Error in upstream connection")
                self.upstream_connected.clear()
                self.upstream_ws = None

            if self.running:
                logger.info(f"Retrying upstream connection in {retry_delay}s")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

    async def _handle_upstream_json(self, message: str) -> None:
        """Handle JSON messages from upstream."""
        try:
            msg = json.loads(message)
            op = msg.get("op")

            if op == JsonOpCodes.SERVER_INFO.value:
                # Store server info and forward to all clients
                self.server_info = msg
                logger.info(f"Received server info: {msg.get('name')}")
                await self._broadcast_to_clients(message)

            elif op == JsonOpCodes.ADVERTISE.value:
                # Track advertised channels and create transformed channels
                await self._handle_advertise_upstream(msg)

            elif op == JsonOpCodes.UNADVERTISE.value:
                downstream_ids: list[int] = []

                for upstream_channel_id in msg.get("channelIds", []):
                    transformed_id = self.transformed_channels.pop(upstream_channel_id, None)

                    if transformed_id is not None:
                        channel = self.advertised_channels.pop(transformed_id, None)
                        self.transformed_to_upstream.pop(transformed_id, None)
                        if channel:
                            downstream_ids.append(transformed_id)
                            logger.info(
                                "Transformed channel unadvertised: "
                                "%s (upstream_id=%s, downstream_id=%s)",
                                channel.get("topic", "?"),
                                upstream_channel_id,
                                transformed_id,
                            )
                    else:
                        channel = self.advertised_channels.pop(upstream_channel_id, None)
                        if channel:
                            downstream_ids.append(upstream_channel_id)
                            logger.info(
                                "Channel unadvertised: %s (id=%s)",
                                channel.get("topic", "?"),
                                upstream_channel_id,
                            )

                    self.upstream_channels.pop(upstream_channel_id, None)
                    self.channel_throttle_hz.pop(upstream_channel_id, None)
                    self.channel_last_sent_time.pop(upstream_channel_id, None)

                if downstream_ids:
                    unadvertise_msg: UnadvertiseMessage = {
                        "op": JsonOpCodes.UNADVERTISE.value,
                        "channelIds": downstream_ids,
                    }
                    await self._broadcast_to_clients(json.dumps(unadvertise_msg))

            elif op in (
                JsonOpCodes.STATUS.value,
                JsonOpCodes.REMOVE_STATUS.value,
            ):
                # Forward status messages
                await self._broadcast_to_clients(message)

            else:
                logger.debug(f"Ignoring upstream JSON message: {op}")

        except Exception:
            logger.exception("Error handling upstream JSON message")

    async def _handle_upstream_binary(self, data: bytes) -> None:
        """Handle binary messages from upstream."""
        if len(data) < 1:
            return

        opcode = data[0]

        if opcode == BinaryOpCodes.MESSAGE_DATA:
            # Forward message data to subscribed clients
            await self._forward_message_data(data)
        elif opcode == BinaryOpCodes.TIME:
            # Forward time messages to all clients
            await self._broadcast_binary_to_clients(data)
        else:
            logger.debug(f"Ignoring upstream binary opcode: {opcode}")

    async def _forward_message_data(self, data: bytes) -> None:
        """Forward message data from upstream to subscribed clients."""
        if len(data) < 5:
            return

        # Extract upstream subscription ID from binary message
        upstream_sub_id = struct.unpack_from("<I", data, 1)[0]

        # Find which channel this subscription belongs to
        channel_id = None
        for ch_id, up_sub_id in self.upstream_subscriptions.items():
            if up_sub_id == upstream_sub_id:
                channel_id = ch_id
                break

        if channel_id is None:
            logger.warning(f"Received message for unknown upstream subscription: {upstream_sub_id}")
            return

        # Check if this channel has a transformer
        transformed_channel_id = self.transformed_channels.get(channel_id)

        # Determine which downstream channel ID to use
        downstream_channel_id = transformed_channel_id if transformed_channel_id else channel_id

        # Apply throttling before performing any transformations or forwarding
        throttle_hz = self.channel_throttle_hz.get(channel_id, self.default_throttle_hz)
        if throttle_hz is not None and throttle_hz > 0:
            now = asyncio.get_running_loop().time()
            min_interval = 1.0 / throttle_hz if throttle_hz > 0 else 0.0
            last_sent = self.channel_last_sent_time.get(channel_id)

            if last_sent is not None and (now - last_sent) < min_interval:
                logger.debug(
                    "Dropping message due to throttle: channel_id=%s, interval=%.3fs",
                    channel_id,
                    min_interval,
                )
                return

            self.channel_last_sent_time[channel_id] = now

        # Forward to all clients subscribed to the downstream channel
        for client, subs in self.client_subscriptions.items():
            for client_sub_id, client_channel_id in subs.items():
                # Check if client subscribed to this channel's downstream representation
                if client_channel_id == downstream_channel_id:
                    # Apply transformation if needed
                    if transformed_channel_id:
                        # This channel is transformed - apply transformation
                        try:
                            transformed_data = await self._transform_message(channel_id, data)
                            if transformed_data:
                                # Rewrite subscription ID for transformed message
                                rewritten_data = bytearray(transformed_data)
                                struct.pack_into("<I", rewritten_data, 1, client_sub_id)

                                await client.send(bytes(rewritten_data))
                        except TransformError as e:
                            logger.warning(f"Transform failed: {e}")
                        except websockets.ConnectionClosed:
                            logger.debug("Client disconnected while sending transformed message")
                        except Exception:
                            logger.exception("Unexpected error during transformation")
                    else:
                        # No transformer - forward as-is
                        rewritten_data = bytearray(data)
                        struct.pack_into("<I", rewritten_data, 1, client_sub_id)

                        try:
                            await client.send(bytes(rewritten_data))
                        except websockets.ConnectionClosed:
                            logger.debug("Client disconnected while sending message")

    async def _broadcast_to_clients(self, message: str) -> None:
        """Broadcast a JSON message to all connected clients."""
        if not self.downstream_clients:
            return

        # Send to all clients concurrently
        await asyncio.gather(
            *[self._send_to_client(client, message) for client in self.downstream_clients],
            return_exceptions=True,
        )

    async def _broadcast_binary_to_clients(self, data: bytes) -> None:
        """Broadcast a binary message to all connected clients."""
        if not self.downstream_clients:
            return

        await asyncio.gather(
            *[self._send_binary_to_client(client, data) for client in self.downstream_clients],
            return_exceptions=True,
        )

    async def _send_to_client(self, client: ServerConnection, message: str) -> None:
        """Send a message to a specific client."""
        try:
            await client.send(message)
        except websockets.ConnectionClosed:
            logger.debug("Client disconnected while sending")

    async def _send_binary_to_client(self, client: ServerConnection, data: bytes) -> None:
        """Send binary data to a specific client."""
        try:
            await client.send(data)
        except websockets.ConnectionClosed:
            logger.debug("Client disconnected while sending binary")

    async def _run_server(self) -> None:
        """Run the downstream WebSocket server."""
        logger.info(f"Starting downstream server on ws://{self.listen_host}:{self.listen_port}")

        async with serve(
            self._handle_client,
            self.listen_host,
            self.listen_port,
            subprotocols=["foxglove.websocket.v1"],
        ):
            # Keep server running
            await asyncio.Future()  # Run forever

    async def _handle_client(self, websocket: ServerConnection) -> None:
        """Handle a downstream client connection."""
        logger.info(f"Client connected: {websocket.remote_address}")
        self.downstream_clients.add(websocket)
        self.client_subscriptions[websocket] = {}

        try:
            # Wait for upstream to be connected
            await self.upstream_connected.wait()

            # Send server info to new client
            if self.server_info:
                await websocket.send(json.dumps(self.server_info))

            # Send current advertised channels
            if self.advertised_channels:
                advertise_msg: AdvertiseMessage = {
                    "op": "advertise",
                    "channels": list(self.advertised_channels.values()),
                }
                await websocket.send(json.dumps(advertise_msg))

            # Handle client messages
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_client_json(websocket, message)
                elif isinstance(message, bytes):
                    await self._handle_client_binary(websocket, message)

        except websockets.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        except Exception:
            logger.exception("Error handling client")
        finally:
            await self._cleanup_client(websocket)

    async def _handle_client_json(self, client: ServerConnection, message: str) -> None:
        """Handle JSON messages from downstream client."""
        try:
            msg = json.loads(message)
            op = msg.get("op")

            if op == JsonOpCodes.SUBSCRIBE.value:
                await self._handle_client_subscribe(client, msg)
            elif op == JsonOpCodes.UNSUBSCRIBE.value:
                await self._handle_client_unsubscribe(client, msg)
            else:
                logger.debug(f"Ignoring client JSON message: {op}")

        except Exception:
            logger.exception("Error handling client JSON message")

    async def _handle_client_binary(self, _client: ServerConnection, _data: bytes) -> None:
        """Handle binary messages from downstream client."""
        # For initial pass-through, we don't handle client publishing
        logger.debug("Ignoring client binary message (client publish not supported)")

    async def _handle_client_subscribe(
        self, client: ServerConnection, msg: SubscribeMessage
    ) -> None:
        """Handle subscribe request from client."""
        for sub in msg.get("subscriptions", []):
            client_sub_id = sub["id"]
            channel_id = sub["channelId"]

            # Track client subscription
            self.client_subscriptions[client][client_sub_id] = channel_id

            # Check if this is a transformed channel
            upstream_channel_id = self.transformed_to_upstream.get(channel_id, channel_id)

            # Subscribe to upstream if not already subscribed
            if upstream_channel_id not in self.upstream_subscriptions:
                await self._subscribe_upstream(upstream_channel_id)

            # Increment reference count for the upstream channel
            self.upstream_sub_refcount[upstream_channel_id] += 1

            channel_info = self.upstream_channels.get(upstream_channel_id, {})
            is_transformed = channel_id in self.transformed_to_upstream
            logger.info(
                f"Client subscribed to {channel_info.get('topic', '?')} "
                f"(channel={channel_id}, upstream={upstream_channel_id}, transformed={is_transformed})"  # noqa: E501
            )

    async def _handle_client_unsubscribe(
        self, client: ServerConnection, msg: UnsubscribeMessage
    ) -> None:
        """Handle unsubscribe request from client."""
        client_subs = self.client_subscriptions.get(client, {})

        for client_sub_id in msg.get("subscriptionIds", []):
            channel_id = client_subs.pop(client_sub_id, None)

            if channel_id is not None:
                # Check if this is a transformed channel
                upstream_channel_id = self.transformed_to_upstream.get(channel_id, channel_id)

                # Decrement reference count for upstream channel
                self.upstream_sub_refcount[upstream_channel_id] -= 1

                # Unsubscribe from upstream if no more clients need it
                if self.upstream_sub_refcount[upstream_channel_id] <= 0:
                    await self._unsubscribe_upstream(upstream_channel_id)
                    del self.upstream_sub_refcount[upstream_channel_id]

                channel_info = self.upstream_channels.get(upstream_channel_id, {})
                logger.info(
                    f"Client unsubscribed from {channel_info.get('topic', '?')} (channel={channel_id})"  # noqa: E501
                )

    async def _subscribe_upstream(self, channel_id: int) -> None:
        """Subscribe to a channel on the upstream bridge."""
        if not self.upstream_ws:
            logger.warning("Cannot subscribe upstream: not connected")
            return

        upstream_sub_id = self.next_upstream_sub_id
        self.next_upstream_sub_id += 1

        subscribe_msg: SubscribeMessage = {
            "op": "subscribe",
            "subscriptions": [{"id": upstream_sub_id, "channelId": channel_id}],
        }

        await self.upstream_ws.send(json.dumps(subscribe_msg))
        self.upstream_subscriptions[channel_id] = upstream_sub_id

        channel = self.upstream_channels.get(channel_id, {})
        logger.info(
            f"Subscribed to upstream {channel.get('topic', '?')} (channel={channel_id}, upstream_sub={upstream_sub_id})"  # noqa: E501
        )

    async def _unsubscribe_upstream(self, channel_id: int) -> None:
        """Unsubscribe from a channel on the upstream bridge."""
        if not self.upstream_ws:
            return

        upstream_sub_id = self.upstream_subscriptions.pop(channel_id, None)
        if upstream_sub_id is None:
            return

        unsubscribe_msg: UnsubscribeMessage = {
            "op": "unsubscribe",
            "subscriptionIds": [upstream_sub_id],
        }

        await self.upstream_ws.send(json.dumps(unsubscribe_msg))

        channel = self.upstream_channels.get(channel_id, {})
        logger.info(
            f"Unsubscribed from upstream {channel.get('topic', '?')} (channel={channel_id})"
        )

    async def _cleanup_client(self, client: ServerConnection) -> None:
        """Clean up after a client disconnects."""
        # Remove from tracking
        self.downstream_clients.discard(client)

        # Unsubscribe from all channels this client was subscribed to
        client_subs = self.client_subscriptions.pop(client, {})
        for channel_id in client_subs.values():
            # Check if this is a transformed channel
            upstream_channel_id = self.transformed_to_upstream.get(channel_id, channel_id)

            # Decrement reference count for upstream channel
            self.upstream_sub_refcount[upstream_channel_id] -= 1

            # Unsubscribe from upstream if no more clients need it
            if self.upstream_sub_refcount[upstream_channel_id] <= 0:
                await self._unsubscribe_upstream(upstream_channel_id)
                del self.upstream_sub_refcount[upstream_channel_id]

        logger.info(f"Client cleanup complete: {client.remote_address}")

    async def _handle_advertise_upstream(self, msg: dict[str, Any]) -> None:
        """Handle advertise message from upstream, creating transformed channels.

        Args:
            msg: The advertise message from upstream
        """
        downstream_channels = []

        for channel in msg.get("channels", []):
            channel_id = channel["id"]
            schema_name = channel["schemaName"]

            # Track the upstream channel for transformation lookup
            self.upstream_channels[channel_id] = channel
            logger.info(
                f"Upstream channel advertised: {channel['topic']} (id={channel_id}, schema={schema_name})"  # noqa: E501
            )

            # Initialize throttling for this channel
            throttle_hz = self.topic_throttle_overrides.get(
                channel["topic"], self.default_throttle_hz
            )
            self.channel_throttle_hz[channel_id] = throttle_hz
            self.channel_last_sent_time.pop(channel_id, None)

            # Check if we have a transformer for this schema
            transformer = self.transformer_registry.get_transformer(schema_name)
            logger.debug(f"Transformer lookup for schema {schema_name}: {transformer}")

            if transformer:
                # Create a transformed channel
                transformed_id = self.next_channel_id
                self.next_channel_id += 1

                output_schema = transformer.get_output_schema()
                output_schema_def = get_schema(output_schema)

                # Create transformed channel info
                transformed_channel = {
                    "id": transformed_id,
                    "topic": channel["topic"],  # Same topic name
                    "encoding": channel["encoding"],  # Same encoding (cdr)
                    "schemaName": output_schema,
                    "schema": output_schema_def,
                    "schemaEncoding": channel.get("schemaEncoding", "ros2msg"),
                }

                # Track the transformation mapping
                self.transformed_channels[channel_id] = transformed_id
                self.transformed_to_upstream[transformed_id] = channel_id

                # Store DOWNSTREAM (transformed) channel
                self.advertised_channels[transformed_id] = transformed_channel
                downstream_channels.append(transformed_channel)

                logger.info(
                    f"Created transformed channel: {schema_name} -> {output_schema} "
                    f"(upstream_id={channel_id}, downstream_id={transformed_id})"
                )
            else:
                # No transformer - forward original channel
                # Store DOWNSTREAM (original) channel
                self.advertised_channels[channel_id] = channel
                downstream_channels.append(channel)

                logger.info(f"Forwarding original channel: {schema_name} (id={channel_id})")

        # Send advertise message to downstream clients
        if downstream_channels:
            advertise_msg: AdvertiseMessage = {
                "op": "advertise",
                "channels": downstream_channels,
            }
            await self._broadcast_to_clients(json.dumps(advertise_msg))

    async def _transform_message(self, channel_id: int, data: bytes) -> bytes | None:
        """Transform a binary message using the registered transformer.

        Args:
            channel_id: The upstream channel ID
            data: The binary message data (including header)

        Returns:
            Transformed binary message data, or None if transformation failed
        """
        # Get upstream channel info (for the original schema)
        channel = self.upstream_channels.get(channel_id)
        if not channel:
            return None

        schema_name = channel["schemaName"]

        transformer = self.transformer_registry.get_transformer(schema_name)
        if not transformer:
            return None

        try:
            # Extract the message payload (skip opcode, sub_id, timestamp)
            if len(data) < 13:
                return None

            opcode = data[0]
            sub_id = struct.unpack_from("<I", data, 1)[0]
            timestamp = struct.unpack_from("<Q", data, 5)[0]
            payload = data[13:]

            # Decode the input message
            input_msg = self._decode_message(channel, payload)

            # Transform the message
            output_msg = transformer.transform(input_msg)

            # Get output schema
            output_schema = transformer.get_output_schema()

            # Encode the output message
            output_payload = self._encode_message(output_schema, output_msg)

            # Reconstruct binary message with same header
            result = bytearray()
            result.append(opcode)
            result.extend(struct.pack("<I", sub_id))
            result.extend(struct.pack("<Q", timestamp))
            result.extend(output_payload)

            return bytes(result)

        except Exception:
            logger.exception("Message transformation failed")
            return None

    def _decode_message(self, channel: ChannelInfo, payload: bytes) -> Any:
        """Decode a CDR-encoded message.

        Args:
            channel: The channel information
            payload: The CDR-encoded payload

        Returns:
            Decoded message as a dictionary
        """
        schema_name = channel["schemaName"]

        # Get or create decoder for this schema
        if schema_name not in self.message_decoders:
            schema_def = channel["schema"]
            # Generate decoder functions for this schema
            decoders = generate_dynamic(schema_name, schema_def)
            # Get the decoder for the main message type
            if schema_name not in decoders:
                raise ValueError(f"Decoder not generated for {schema_name}")
            self.message_decoders[schema_name] = decoders[schema_name]

        decoder = self.message_decoders[schema_name]

        # Decode the message
        return decoder(payload)

    def _encode_message(self, schema_name: str, message: dict[str, Any]) -> bytes:
        """Encode a message to CDR format.

        Args:
            schema_name: The message schema name
            message: The message as a dictionary

        Returns:
            CDR-encoded bytes
        """
        # Get or create encoder for this schema
        if schema_name not in self.message_encoders:
            schema_def = get_schema(schema_name)
            # Generate encoder functions for this schema
            encoders = serialize_dynamic(schema_name, schema_def)
            # Get the encoder for the main message type
            if schema_name not in encoders:
                raise ValueError(f"Encoder not generated for {schema_name}")
            self.message_encoders[schema_name] = encoders[schema_name]

        encoder = self.message_encoders[schema_name]

        # Encode the message
        return encoder(message)
