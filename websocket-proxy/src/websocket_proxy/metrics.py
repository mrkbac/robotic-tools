"""Metrics collection for websocket proxy server."""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ClientMetrics:
    """Metrics for a single client connection."""

    client_id: str
    remote_address: str
    connected_at: datetime
    user_agent: str | None = None

    # Traffic metrics
    messages_sent: int = 0
    bytes_sent: int = 0
    errors: int = 0
    last_message_at: datetime | None = None

    # Subscriptions
    subscription_count: int = 0
    subscribed_topics: set[str] = field(default_factory=set)

    # Rate tracking (samples stored for windowed calculations)
    _message_samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=60))
    _byte_samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=60))

    def record_message(self, byte_count: int) -> None:
        """Record a message sent to this client."""
        now = time.time()
        self.messages_sent += 1
        self.bytes_sent += byte_count
        self.last_message_at = datetime.now(timezone.utc)
        self._message_samples.append((now, 1))
        self._byte_samples.append((now, byte_count))

    def record_error(self) -> None:
        """Record an error for this client."""
        self.errors += 1

    def get_message_rate(self, window_seconds: float = 5.0) -> float:
        """Calculate messages per second over the last N seconds."""
        return self._calculate_rate(self._message_samples, window_seconds)

    def get_bandwidth(self, window_seconds: float = 5.0) -> float:
        """Calculate bytes per second over the last N seconds."""
        return self._calculate_rate(self._byte_samples, window_seconds)

    @staticmethod
    def _calculate_rate(samples: deque[tuple[float, int]], window_seconds: float) -> float:
        """Calculate rate from time-stamped samples within window."""
        if not samples:
            return 0.0

        now = time.time()
        cutoff = now - window_seconds
        total = sum(count for ts, count in samples if ts >= cutoff)

        # Find actual time span of samples in window
        valid_samples = [(ts, count) for ts, count in samples if ts >= cutoff]
        if not valid_samples:
            return 0.0

        oldest_ts = valid_samples[0][0]
        time_span = now - oldest_ts

        if time_span <= 0:
            return 0.0

        return total / time_span

    @property
    def connected_duration(self) -> float:
        """Get connection duration in seconds."""
        return (datetime.now(timezone.utc) - self.connected_at).total_seconds()


class MetricsCollector:
    """Central metrics collector for the proxy server."""

    def __init__(self) -> None:
        self.clients: dict[str, ClientMetrics] = {}
        self._start_time = datetime.now(timezone.utc)

    def add_client(
        self, client_id: str, remote_address: str, user_agent: str | None = None
    ) -> ClientMetrics:
        """Add a new client to track."""
        metrics = ClientMetrics(
            client_id=client_id,
            remote_address=remote_address,
            connected_at=datetime.now(timezone.utc),
            user_agent=user_agent,
        )
        self.clients[client_id] = metrics
        return metrics

    def remove_client(self, client_id: str) -> None:
        """Remove a client from tracking."""
        self.clients.pop(client_id, None)

    def get_client(self, client_id: str) -> ClientMetrics | None:
        """Get client metrics by ID."""
        return self.clients.get(client_id)

    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()
