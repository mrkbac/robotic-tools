"""Foxglove WebSocket proxy bridge implementation."""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import TYPE_CHECKING, Any

from mcap_ros2_support_fast._planner import generate_dynamic, serialize_dynamic
from websocket_bridge import (
    ConnectionState,
    ServerConnection,
    WebSocketBridgeClient,
    WebSocketBridgeServer,
)
from websocket_bridge.server import Channel

from .metrics import MetricsCollector
from .schemas import get_schema
from .transformers import TransformerRegistry, TransformError

if TYPE_CHECKING:
    from collections.abc import Callable

    from websocket_bridge.ws_types import ChannelInfo

logger = logging.getLogger(__name__)


def _get_network_interfaces(port: int) -> list[str]:
    """Get all accessible network interface addresses for the given port.

    Args:
        port: The port number to display

    Returns:
        List of connection URLs (e.g., ws://192.168.1.100:8766)
    """
    addresses = []

    # Always add localhost
    addresses.append(f"ws://localhost:{port}")

    try:
        # Get all network interface addresses
        hostname = socket.gethostname()
        all_ips = socket.getaddrinfo(hostname, None, socket.AF_INET)

        # Extract unique IPv4 addresses
        seen_ips: set[str] = {"127.0.0.1"}
        for addr_info in all_ips:
            ip = str(addr_info[4][0])
            if ip not in seen_ips:
                addresses.append(f"ws://{ip}:{port}")
                seen_ips.add(ip)
    except OSError:
        # If we can't get network interfaces, just show 0.0.0.0
        # This can happen due to network configuration issues
        pass

    return addresses


class MessageCodecCache:
    """Cache for message encoders and decoders with lazy generation."""

    def __init__(self) -> None:
        """Initialize the codec cache."""
        self._encoders: dict[str, Callable[[dict[str, Any]], bytes]] = {}
        self._decoders: dict[str, Callable[[bytes], Any]] = {}

    def get_decoder(self, schema_name: str, schema_def: str) -> Callable[[bytes], Any]:
        """Get or create a decoder for the given schema.

        Args:
            schema_name: Name of the message schema
            schema_def: Schema definition string

        Returns:
            Decoder function that takes bytes and returns decoded message
        """
        if schema_name not in self._decoders:
            self._decoders[schema_name] = generate_dynamic(schema_name, schema_def)
        return self._decoders[schema_name]

    def get_encoder(self, schema_name: str, schema_def: str) -> Callable[[dict[str, Any]], bytes]:
        """Get or create an encoder for the given schema.

        Args:
            schema_name: Name of the message schema
            schema_def: Schema definition string

        Returns:
            Encoder function that takes a message dict and returns CDR bytes
        """
        if schema_name not in self._encoders:
            self._encoders[schema_name] = serialize_dynamic(schema_name, schema_def)
        return self._encoders[schema_name]


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
        max_message_size: int | None = None,
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
            max_message_size: Maximum allowed websocket frame size in bytes (None disables limit)
        """
        self.upstream_url = upstream_url
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.transformer_registry = transformer_registry or TransformerRegistry()
        self.default_throttle_hz = default_throttle_hz
        self.topic_throttle_overrides = dict(topic_throttle_overrides or {})

        # Create upstream client and downstream server
        self.upstream_client = WebSocketBridgeClient(
            url=upstream_url,
            max_message_size=max_message_size,
        )
        self.upstream_client.on_advertised_channel(self.handle_upstream_advertise)
        self.upstream_client.on_channel_unadvertised(self.handle_upstream_unadvertise)
        self.upstream_client.on_message(self.handle_upstream_message)
        self.upstream_client.on_disconnect(self.handle_upstream_disconnected)
        self.upstream_client.on_connect(self.handle_upstream_reconnected)

        self.downstream_server = WebSocketBridgeServer(
            host=listen_host,
            port=listen_port,
            name="FoxBridge Proxy",
            capabilities=[],
            metadata={"proxy": "true"},
            supported_encodings=["cdr"],
            max_message_size=max_message_size,
        )

        # Register callbacks for downstream server
        self.downstream_server.on_connect(self._on_client_connected)
        self.downstream_server.on_subscribe(self._on_client_subscribe)
        self.downstream_server.on_unsubscribe(self._on_client_unsubscribe)
        self.downstream_server.on_disconnect(self._on_client_disconnected)

        # Channel tracking - stores DOWNSTREAM channels (what clients see)
        # Key is downstream channel ID (original or transformed)
        self.advertised_channels: dict[int, ChannelInfo] = {}

        # Track upstream channel info for transformation lookup
        self.upstream_channels: dict[int, ChannelInfo] = {}

        # Transformed channel tracking
        # Maps upstream channel ID -> downstream transformed channel ID
        self.transformed_channels: dict[int, int] = {}
        # Maps transformed channel ID -> upstream channel ID
        self.transformed_to_upstream: dict[int, int] = {}
        self.next_channel_id = 10000  # Start transformed channels at high ID to avoid conflicts

        # Topic throttling (per upstream channel)
        self.channel_throttle_hz: dict[int, float | None] = {}
        self.channel_last_sent_time: dict[int, float] = {}

        # Downstream client subscription tracking
        # Maps client websocket -> {client_sub_id: (channel_id, upstream_sub_id)}
        self.client_subscriptions: dict[ServerConnection, dict[int, tuple[int, int]]] = {}

        # Upstream subscription tracking (reference counting)
        # Maps channel_id -> (upstream_sub_id, ref_count)
        self.upstream_subscriptions: dict[int, tuple[int, int]] = {}
        self.next_upstream_sub_id = 0

        # Message encoder/decoder cache (generated from schema)
        self.codec_cache = MessageCodecCache()

        # Metrics collector
        self.metrics = MetricsCollector()

        # Shutdown event to keep start() alive until stop() is called
        self._shutdown_event = asyncio.Event()

    def _get_client_id(self, websocket: ServerConnection) -> str:
        """Generate a unique client ID from websocket connection."""
        return f"client_{id(websocket)}"

    def _get_remote_address(self, websocket: ServerConnection) -> str:
        """Get remote address from websocket connection."""
        try:
            remote = websocket.remote_address
            if isinstance(remote, tuple):
                return f"{remote[0]}:{remote[1]}"
            return str(remote)
        except (OSError, AttributeError):
            # OSError: connection issues; AttributeError: remote_address not available
            return "unknown"

    async def _on_client_connected(self, state: ConnectionState) -> None:
        """Handle client connection (callback for downstream server).

        Args:
            state: Client connection state
        """
        client_id = self._get_client_id(state.websocket)
        remote_address = self._get_remote_address(state.websocket)

        # Add client to metrics
        self.metrics.add_client(client_id, remote_address)
        logger.info(f"Client connected: {client_id} from {remote_address}")

    async def _on_client_subscribe(
        self,
        state: ConnectionState,
        subscription_id: int,
        channel_id: int,
    ) -> None:
        """Handle client subscription (callback for downstream server).

        Args:
            state: Client connection state
            subscription_id: Client subscription ID
            channel_id: Channel ID to subscribe to
        """
        await self.handle_client_subscribe(state.websocket, subscription_id, channel_id)

    async def _on_client_unsubscribe(
        self,
        state: ConnectionState,
        subscription_id: int,
        channel_id: int,  # noqa: ARG002
    ) -> None:
        """Handle client unsubscribe (callback for downstream server).

        Args:
            state: Client connection state
            subscription_id: Client subscription ID to unsubscribe
            channel_id: Channel ID (not used in current implementation)
        """
        await self.handle_client_unsubscribe(state.websocket, subscription_id)

    async def _on_client_disconnected(self, state: ConnectionState) -> None:
        """Handle client disconnection (callback for downstream server).

        Args:
            state: Client connection state
        """
        client_id = self._get_client_id(state.websocket)
        self.metrics.remove_client(client_id)
        await self.handle_client_disconnected(state.websocket)

    async def start(self) -> None:
        """Start the proxy bridge and run until stop() is called."""
        # Get all network interface addresses
        interfaces = _get_network_interfaces(self.listen_port)

        logger.info(f"Starting Foxglove proxy bridge: {self.upstream_url} -> downstream clients")
        logger.info("Connect using any of these addresses:")
        for addr in interfaces:
            logger.info(f"  {addr}")

        # Start both upstream client and downstream server
        await asyncio.gather(
            self.upstream_client.connect(),
            self.downstream_server.start(),
        )

        # Wait until stop() is called (which will set the shutdown event)
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the proxy bridge."""
        logger.info("Stopping proxy bridge")

        # Signal the shutdown event to unblock start()
        self._shutdown_event.set()

        await asyncio.gather(
            self.upstream_client.disconnect(),
            self.downstream_server.stop(),
            return_exceptions=True,
        )

    async def handle_upstream_advertise(self, channel: ChannelInfo) -> None:
        """Handle advertise message from upstream, creating transformed channels.

        Args:
            channel: The advertised channel from upstream
        """
        channel_id = channel["id"]
        schema_name = channel["schemaName"]

        # Track the upstream channel for transformation lookup
        self.upstream_channels[channel_id] = channel
        logger.debug(
            f"Upstream channel advertised: {channel['topic']} "
            f"(id={channel_id}, schema={schema_name})"
        )

        # Initialize throttling for this channel
        throttle_hz = self.topic_throttle_overrides.get(channel["topic"], self.default_throttle_hz)
        self.channel_throttle_hz[channel_id] = throttle_hz
        self.channel_last_sent_time.pop(channel_id, None)

        # Check if we have a transformer for this schema
        transformer = self.transformer_registry.get_transformer(schema_name)
        logger.debug(f"Transformer lookup for schema {schema_name}: {transformer}")

        downstream_channel: ChannelInfo
        if transformer:
            # Create a transformed channel
            transformed_id = self.next_channel_id
            self.next_channel_id += 1

            output_schema = transformer.get_output_schema()
            output_schema_def = get_schema(output_schema)

            # Create transformed channel info
            downstream_channel = {
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

            logger.debug(
                f"Created transformed channel: {schema_name} -> {output_schema} "
                f"(upstream_id={channel_id}, downstream_id={transformed_id})"
            )
        else:
            # No transformer - forward original channel
            downstream_channel = channel
            logger.debug(f"Forwarding original channel: {schema_name} (id={channel_id})")

        # Store DOWNSTREAM channel
        self.advertised_channels[downstream_channel["id"]] = downstream_channel

        # Convert ChannelInfo to Channel object and advertise to all downstream clients
        channel_obj = Channel(
            id=downstream_channel["id"],
            topic=downstream_channel["topic"],
            encoding=downstream_channel["encoding"],
            schema_name=downstream_channel["schemaName"],
            schema=downstream_channel["schema"],
            schema_encoding=downstream_channel.get("schemaEncoding"),
        )
        await self.downstream_server.advertise_channel(channel_obj)

    async def handle_upstream_unadvertise(self, channel: ChannelInfo) -> None:
        """Handle unadvertise message from upstream.

        Args:
            channel: The unadvertised channel from upstream
        """
        upstream_channel_id = channel["id"]

        # Check if this channel was transformed
        transformed_id = self.transformed_channels.pop(upstream_channel_id, None)

        if transformed_id is not None:
            # This was a transformed channel
            downstream_channel = self.advertised_channels.pop(transformed_id, None)
            self.transformed_to_upstream.pop(transformed_id, None)
            downstream_channel_id = transformed_id
            if downstream_channel:
                logger.info(
                    f"Transformed channel unadvertised: {downstream_channel.get('topic', '?')} "
                    f"(upstream_id={upstream_channel_id}, downstream_id={transformed_id})"
                )
        else:
            # Original channel (not transformed)
            downstream_channel = self.advertised_channels.pop(upstream_channel_id, None)
            downstream_channel_id = upstream_channel_id
            if downstream_channel:
                logger.info(
                    f"Channel unadvertised: {downstream_channel.get('topic', '?')} "
                    f"(id={upstream_channel_id})"
                )

        # Clean up tracking
        self.upstream_channels.pop(upstream_channel_id, None)
        self.channel_throttle_hz.pop(upstream_channel_id, None)
        self.channel_last_sent_time.pop(upstream_channel_id, None)

        # Unadvertise from downstream clients
        if downstream_channel:
            await self.downstream_server.unadvertise([downstream_channel_id])

    async def handle_upstream_message(
        self,
        channel: ChannelInfo,
        timestamp: int,
        payload: bytes,
    ) -> None:
        """Handle message from upstream.

        Args:
            channel: The channel this message belongs to
            timestamp: Message timestamp in nanoseconds
            payload: Message payload
        """
        channel_id = channel["id"]

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

        # Check if this channel has a transformer
        transformed_channel_id = self.transformed_channels.get(channel_id)

        if transformed_channel_id:
            # This channel is transformed - apply transformation
            try:
                transformed_msg = self._transform_message_dict(channel, payload)
                if transformed_msg:
                    # Get output schema and encode
                    transformer = self.transformer_registry.get_transformer(channel["schemaName"])
                    if transformer:
                        output_schema = transformer.get_output_schema()
                        output_payload = self._encode_message(output_schema, transformed_msg)

                        # Send to all subscribed clients
                        await self._send_to_subscribed_clients(
                            transformed_channel_id,
                            timestamp,
                            output_payload,
                        )
            except TransformError as e:
                logger.warning(f"Transform failed: {e}")
            except Exception:
                logger.exception("Unexpected error during transformation")
        else:
            # No transformer - forward as-is
            await self._send_to_subscribed_clients(channel_id, timestamp, payload)

    async def _send_to_subscribed_clients(
        self,
        channel_id: int,
        timestamp: int,
        payload: bytes,
    ) -> None:
        """Send message to all clients subscribed to a channel.

        Args:
            channel_id: The downstream channel ID
            timestamp: Message timestamp in nanoseconds
            payload: Message payload
        """
        # Find all clients subscribed to this channel
        # Create a snapshot to avoid "dictionary changed size during iteration" errors
        subscriber_count = 0
        for websocket, subs in list(self.client_subscriptions.items()):
            for client_sub_id, (subscribed_channel_id, _) in subs.items():
                if subscribed_channel_id == channel_id:
                    # Send message to this client
                    try:
                        await self.downstream_server.send_message_to_subscription(
                            websocket,
                            client_sub_id,
                            payload,
                            timestamp_ns=timestamp,
                        )
                        subscriber_count += 1

                        # Track message sent to client
                        client_id = self._get_client_id(websocket)
                        client_metrics = self.metrics.get_client(client_id)
                        if client_metrics:
                            client_metrics.record_message(len(payload))
                    except Exception:
                        logger.exception("Failed to send message to client")
                        # Track error
                        client_id = self._get_client_id(websocket)
                        client_metrics = self.metrics.get_client(client_id)
                        if client_metrics:
                            client_metrics.record_error()

    async def handle_client_subscribe(
        self,
        websocket: ServerConnection,
        subscription_id: int,
        channel_id: int,
    ) -> None:
        """Handle subscribe request from client.

        Args:
            websocket: The client websocket connection
            subscription_id: Client subscription ID
            channel_id: Channel ID to subscribe to
        """
        # Initialize client subscription dict if needed
        if websocket not in self.client_subscriptions:
            self.client_subscriptions[websocket] = {}

        # Check if this is a transformed channel
        upstream_channel_id = self.transformed_to_upstream.get(channel_id, channel_id)

        # Subscribe to upstream if not already subscribed (or increment ref count)
        upstream_sub_id: int
        if upstream_channel_id in self.upstream_subscriptions:
            # Already subscribed, increment ref count
            existing_sub_id, ref_count = self.upstream_subscriptions[upstream_channel_id]
            upstream_sub_id = existing_sub_id
            self.upstream_subscriptions[upstream_channel_id] = (existing_sub_id, ref_count + 1)
        else:
            # Subscribe to upstream
            upstream_sub_id = self.next_upstream_sub_id
            self.next_upstream_sub_id += 1

            # Subscribe to upstream channel with custom subscription ID
            await self.upstream_client.subscribe_to_channel(upstream_sub_id, upstream_channel_id)

            self.upstream_subscriptions[upstream_channel_id] = (upstream_sub_id, 1)

            logger.info(
                f"Subscribed to upstream channel {upstream_channel_id} "
                f"(upstream_sub={upstream_sub_id})"
            )

        # Track client subscription
        self.client_subscriptions[websocket][subscription_id] = (channel_id, upstream_sub_id)

        # Update client metrics (subscription count and topics)
        client_id = self._get_client_id(websocket)
        client_metrics = self.metrics.get_client(client_id)
        if client_metrics:
            client_metrics.subscription_count = len(self.client_subscriptions[websocket])
            channel_info = self.advertised_channels.get(channel_id)
            if channel_info:
                client_metrics.subscribed_topics.add(channel_info["topic"])

        channel_info = self.upstream_channels.get(upstream_channel_id)
        is_transformed = channel_id in self.transformed_to_upstream
        logger.info(
            f"Client subscribed to {channel_info.get('topic', '?') if channel_info else '?'} "
            f"(channel={channel_id}, upstream={upstream_channel_id}, transformed={is_transformed})"
        )

    async def handle_client_unsubscribe(
        self,
        websocket: ServerConnection,
        subscription_id: int,
    ) -> None:
        """Handle unsubscribe request from client.

        Args:
            websocket: The client websocket connection
            subscription_id: Client subscription ID to unsubscribe
        """
        client_subs = self.client_subscriptions.get(websocket, {})
        subscription_info = client_subs.pop(subscription_id, None)

        if subscription_info is not None:
            channel_id, upstream_sub_id = subscription_info  # noqa: RUF059

            # Update client metrics
            client_id = self._get_client_id(websocket)
            client_metrics = self.metrics.get_client(client_id)
            if client_metrics:
                client_metrics.subscription_count = len(client_subs)
                # Recalculate subscribed topics
                client_metrics.subscribed_topics.clear()
                for sub_channel_id, _ in client_subs.values():
                    channel_info = self.advertised_channels.get(sub_channel_id)
                    if channel_info:
                        client_metrics.subscribed_topics.add(channel_info["topic"])

            # Check if this is a transformed channel
            upstream_channel_id = self.transformed_to_upstream.get(channel_id, channel_id)

            # Decrement reference count for upstream channel
            if upstream_channel_id in self.upstream_subscriptions:
                sub_id, ref_count = self.upstream_subscriptions[upstream_channel_id]
                new_ref_count = ref_count - 1

                if new_ref_count <= 0:
                    # Unsubscribe from upstream
                    await self.upstream_client.unsubscribe_from_channel(sub_id)

                    del self.upstream_subscriptions[upstream_channel_id]
                    logger.info(
                        f"Unsubscribed from upstream channel {upstream_channel_id} "
                        f"(upstream_sub={sub_id})"
                    )
                else:
                    # Update ref count
                    self.upstream_subscriptions[upstream_channel_id] = (sub_id, new_ref_count)

            channel_info = self.upstream_channels.get(upstream_channel_id)
            topic_name = channel_info.get("topic", "?") if channel_info else "?"
            logger.info(f"Client unsubscribed from {topic_name} (channel={channel_id})")

    async def handle_client_disconnected(self, websocket: ServerConnection) -> None:
        """Handle client disconnection.

        Args:
            websocket: The client websocket connection
        """
        # Unsubscribe from all channels this client was subscribed to
        client_subs = self.client_subscriptions.pop(websocket, {})
        for subscription_id in list(client_subs.keys()):
            await self.handle_client_unsubscribe(websocket, subscription_id)

        logger.info("Client cleanup complete")

    async def handle_upstream_disconnected(self) -> None:
        """Handle upstream connection loss.

        Clears all upstream subscription state since subscription IDs
        are no longer valid on the disconnected connection.
        """
        logger.warning(
            "Upstream connection lost, clearing %d stale subscription(s)",
            len(self.upstream_subscriptions),
        )
        # Clear all upstream subscriptions - they're no longer valid
        self.upstream_subscriptions.clear()

    async def handle_upstream_reconnected(self) -> None:
        """Handle upstream reconnection.

        Restores all active subscriptions after reconnection by:
        1. Collecting all unique channels that downstream clients are subscribed to
        2. Creating new upstream subscriptions for each channel
        3. Restoring proper reference counts
        """
        # Collect all unique channels that clients are subscribed to
        # Map: upstream_channel_id -> ref_count
        channel_ref_counts: dict[int, int] = {}

        for client_subs in self.client_subscriptions.values():
            for channel_id, _ in client_subs.values():
                # Get the upstream channel ID (handle transformed channels)
                upstream_channel_id = self.transformed_to_upstream.get(channel_id, channel_id)
                channel_ref_counts[upstream_channel_id] = (
                    channel_ref_counts.get(upstream_channel_id, 0) + 1
                )

        if not channel_ref_counts:
            logger.info("Upstream reconnected, no subscriptions to restore")
            return

        logger.info(
            "Upstream reconnected, restoring %d subscription(s)",
            len(channel_ref_counts),
        )

        # Re-subscribe to all channels
        for upstream_channel_id, ref_count in channel_ref_counts.items():
            upstream_sub_id = self.next_upstream_sub_id
            self.next_upstream_sub_id += 1

            try:
                await self.upstream_client.subscribe_to_channel(
                    upstream_sub_id, upstream_channel_id
                )
                self.upstream_subscriptions[upstream_channel_id] = (upstream_sub_id, ref_count)

                channel_info = self.upstream_channels.get(upstream_channel_id)
                topic_name = channel_info.get("topic", "?") if channel_info else "?"
                logger.info(
                    f"Restored subscription to {topic_name} "
                    f"(channel={upstream_channel_id}, sub_id={upstream_sub_id}, refs={ref_count})"
                )
            except Exception:
                logger.exception(f"Failed to restore subscription to channel {upstream_channel_id}")

    def _transform_message_dict(
        self, channel: ChannelInfo, payload: bytes
    ) -> dict[str, Any] | None:
        """Transform a message using the registered transformer.

        Args:
            channel: The channel info
            payload: The message payload (CDR-encoded)

        Returns:
            Transformed message as a dictionary, or None if transformation failed
        """
        schema_name = channel["schemaName"]
        transformer = self.transformer_registry.get_transformer(schema_name)
        if not transformer:
            return None

        try:
            # Decode the input message
            input_msg = self._decode_message(channel, payload)

            # Transform the message
            return transformer.transform(input_msg)
        except TransformError as e:
            # TransformError is expected for invalid input frames
            # Log at debug level to avoid spam
            if "Invalid input frame data" in str(e):
                logger.debug(f"Skipping invalid frame: {e}")
            else:
                logger.warning(f"Transform failed: {e}")
            return None
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
        schema_def = channel["schema"]
        decoder = self.codec_cache.get_decoder(schema_name, schema_def)
        return decoder(payload)

    def _encode_message(self, schema_name: str, message: dict[str, Any]) -> bytes:
        """Encode a message to CDR format.

        Args:
            schema_name: The message schema name
            message: The message as a dictionary

        Returns:
            CDR-encoded bytes
        """
        schema_def = get_schema(schema_name)
        encoder = self.codec_cache.get_encoder(schema_name, schema_def)
        return encoder(message)
