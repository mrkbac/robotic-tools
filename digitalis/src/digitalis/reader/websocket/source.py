"""WebSocket source implementation with dynamic topic discovery."""

import logging
from collections.abc import Callable
from typing import Any

from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap.data import Schema
from websocket_bridge import WebSocketBridgeClient
from websocket_bridge.ws_types import ChannelInfo, ConnectionStatus, ServerCapabilities

from digitalis.reader.source import Source, SourceStatus
from digitalis.reader.types import MessageEvent, SourceInfo, Topic

logger = logging.getLogger(__name__)


class ROS2DecodeError(Exception):
    """Raised if a message cannot be decoded as a ROS2 message."""


class WebSocketSource(WebSocketBridgeClient, Source):
    """WebSocket source for real-time data streaming with dynamic topic discovery."""

    def __init__(
        self,
        url: str,
        subprotocol: str = "foxglove.websocket.v1",
        min_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        backoff_factor: float = 2.0,
    ) -> None:
        super().__init__(
            url=url,
            subprotocol=subprotocol,
            min_retry_delay=min_retry_delay,
            max_retry_delay=max_retry_delay,
            backoff_factor=backoff_factor,
        )

        self._decoder_factory = DecoderFactory()
        self._topics: dict[str, Topic] = {}
        self._play_back = True

        # Handler callbacks for Source interface
        self._message_handler: Callable[[MessageEvent], None] | None = None
        self._source_info_handler: Callable[[SourceInfo], None] | None = None
        self._time_handler: Callable[[int], None] | None = None
        self._status_handler: Callable[[SourceStatus], None] | None = None

    async def initialize(self) -> SourceInfo:
        """Initialize the WebSocket connection with persistent retry logic."""
        logger.info(f"Initializing WebSocket source: {self._url}")

        # Notify that we're initializing
        if self._status_handler:
            self._status_handler(SourceStatus.INITIALIZING)

        # Start connection
        await self.connect()

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

    def set_message_handler(self, handler: Callable[[MessageEvent], None]) -> None:
        """Set the callback for handling incoming messages."""
        self._message_handler = handler

    def set_source_info_handler(self, handler: Callable[[SourceInfo], None]) -> None:
        """Set the callback for handling source info updates."""
        self._source_info_handler = handler

    def set_time_handler(self, handler: Callable[[int], None]) -> None:
        """Set the callback for handling time updates."""
        self._time_handler = handler

    def set_status_handler(self, handler: Callable[[SourceStatus], None]) -> None:
        """Set the callback for handling source status updates."""
        self._status_handler = handler

    def has_capability(self, capability: ServerCapabilities) -> bool:
        """Check if the server has a specific capability."""
        if self.server_info is None:
            return False
        return capability.value in self.server_info["capabilities"]

    async def close(self) -> None:
        """Clean up resources and close the connection."""
        logger.info("Closing WebSocket source")
        await self.disconnect()
        logger.info("WebSocket source closed")

    # Override WebSocketBridgeClient callbacks

    async def _on_connection_status_changed(self, status: ConnectionStatus) -> None:
        """Map ConnectionStatus to SourceStatus and notify handler."""
        status_map = {
            ConnectionStatus.DISCONNECTED: SourceStatus.DISCONNECTED,
            ConnectionStatus.CONNECTING: SourceStatus.CONNECTING,
            ConnectionStatus.CONNECTED: SourceStatus.CONNECTED,
            ConnectionStatus.RECONNECTING: SourceStatus.RECONNECTING,
        }

        source_status = status_map.get(status, SourceStatus.DISCONNECTED)
        if self._status_handler:
            self._status_handler(source_status)

    async def on_server_info(
        self, name: str, capabilities: list[str], session_id: str | None
    ) -> None:
        """Handle server info."""
        logger.info(f"Server: {name}")
        logger.info(f"Capabilities: {', '.join(capabilities)}")
        if session_id:
            logger.info(f"Session ID: {session_id}")

    async def on_advertised_channel(self, channel: ChannelInfo) -> None:
        """Handle newly advertised channels and convert to Topics."""
        topic = Topic(
            name=channel["topic"],
            schema_name=channel["schemaName"],
            topic_id=channel["id"],
        )
        self._topics[channel["topic"]] = topic
        logger.info(f"Topic advertised: {channel['topic']} (ID: {channel['id']})")

        # Send updated source info with all topics
        if self._source_info_handler:
            all_topics = list(self._topics.values())
            source_info = SourceInfo(topics=all_topics)
            self._source_info_handler(source_info)

    async def on_channel_unadvertised(self, channel: ChannelInfo) -> None:
        """Handle unadvertised channels."""
        self._topics.pop(channel["topic"], None)
        logger.info(f"Topic unadvertised: {channel['topic']}")

    async def on_message(
        self,
        channel: ChannelInfo,
        timestamp: int,
        payload: bytes,
    ) -> None:
        """Decode and handle incoming messages."""
        if not self._message_handler or not self._play_back:
            return

        # Decode message
        message_obj: Any = payload
        try:
            decoder = self._decoder_factory.decoder_for(
                channel["encoding"],
                Schema(
                    id=channel["id"],
                    name=channel["schemaName"],
                    encoding=channel.get("schemaEncoding", "ros2msg"),
                    data=channel["schema"].encode(),
                ),
            )
            message_obj = decoder(payload)
        except (ROS2DecodeError, ValueError):
            logger.debug(f"Failed to decode message for topic {channel['topic']}")

        # Create message event
        message_event = MessageEvent(
            topic=channel["topic"],
            message=message_obj,
            timestamp_ns=timestamp,
            schema_name=channel["schemaName"],
        )

        self._message_handler(message_event)

        # Notify time handler with message timestamp
        if self._time_handler:
            self._time_handler(timestamp)

    async def on_time_update(self, server_time: int) -> None:
        """Handle server time updates."""
        logger.debug(f"Received server time: {server_time}")
        if self._time_handler:
            self._time_handler(server_time)
