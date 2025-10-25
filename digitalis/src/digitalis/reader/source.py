"""Core source interfaces for data access and playback control."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum

from .types import MessageEvent, SourceInfo


class SourceStatus(Enum):
    """Generic source status for UI display."""

    INITIALIZING = "initializing"
    READY = "ready"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class Source(ABC):
    """Base interface for data sources that provide topic subscription and message delivery."""

    def __init__(
        self,
        on_message: Callable[[MessageEvent], None],
        on_source_info: Callable[[SourceInfo], None],
        on_time: Callable[[int], None],
        on_status: Callable[[SourceStatus], None],
    ) -> None:
        """Initialize the source with callback handlers.

        Args:
            on_message: Callback for incoming messages
            on_source_info: Callback for source info updates
            on_time: Callback for time updates
            on_status: Callback for status updates
        """
        self._notify_message = on_message
        self._notify_source_info = on_source_info
        self._notify_time = on_time
        self._notify_status = on_status

    @abstractmethod
    async def initialize(self) -> SourceInfo:
        """Initialize the source and return available topics and metadata."""
        ...

    @abstractmethod
    def start_playback(self) -> None:
        """Start or resume playback."""
        ...

    @abstractmethod
    def pause_playback(self) -> None:
        """Pause playback."""
        ...

    @abstractmethod
    async def subscribe(self, topic: str) -> None:
        """Subscribe to messages from a topic."""
        ...

    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from messages from a topic."""
        ...

    def get_status(self) -> SourceStatus:
        """Get current source status."""
        # Default implementation for sources without connection status
        return SourceStatus.READY

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources and close the source."""
        ...


class PlaybackSource(Source):
    """Extended interface for sources that support playback control (e.g., file-based sources)."""

    @abstractmethod
    def set_playback_speed(self, speed: float) -> None:
        """Set playback speed multiplier (1.0 = normal speed, 2.0 = 2x speed, etc.)."""
        ...

    @abstractmethod
    async def seek_to_time(self, timestamp_ns: int) -> None:
        """Seek to a specific timestamp in nanoseconds."""
        ...

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """Return True if playback is currently active."""
        ...

    @property
    @abstractmethod
    def time_range(self) -> tuple[int, int] | None:
        """Return (start_time_ns, end_time_ns) tuple, or None if not available."""
        ...

    @property
    @abstractmethod
    def current_time(self) -> int | None:
        """Return current playback time in nanoseconds, or None if not available."""
        ...
