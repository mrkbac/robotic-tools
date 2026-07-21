"""WebSocket bridge for Foxglove WebSocket protocol."""

from websockets.asyncio.server import ServerConnection

from .client import (
    ConnectionGraph,
    FetchAssetError,
    ServiceCallError,
    ServiceCallResponse,
    WebSocketBridgeClient,
)
from .server import (
    ConnectionState,
    WebSocketBridgeEndpoint,
    WebSocketBridgeServer,
    install_invalid_handshake_log_filter,
)
from .ws_types import (
    PlaybackCommand,
    PlaybackControlRequest,
    PlaybackState,
    PlaybackStatus,
    StatusLevel,
)

__all__ = [
    "ConnectionGraph",
    "ConnectionState",
    "FetchAssetError",
    "PlaybackCommand",
    "PlaybackControlRequest",
    "PlaybackState",
    "PlaybackStatus",
    "ServerConnection",
    "ServiceCallError",
    "ServiceCallResponse",
    "StatusLevel",
    "WebSocketBridgeClient",
    "WebSocketBridgeEndpoint",
    "WebSocketBridgeServer",
    "install_invalid_handshake_log_filter",
]
