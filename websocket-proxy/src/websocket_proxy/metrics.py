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


@dataclass
class ChannelMetrics:
    """Metrics for a single topic/channel."""

    channel_id: int
    topic: str
    schema_name: str

    # Throughput metrics
    messages_received: int = 0  # From upstream
    messages_sent: int = 0  # To downstream clients
    bytes_sent: int = 0

    # Subscriber tracking
    subscriber_ids: set[str] = field(default_factory=set)

    # Throttling metrics
    messages_dropped: int = 0

    # Transformation metrics
    transform_successes: int = 0
    transform_failures: int = 0
    is_transformed: bool = False

    # Rate tracking
    _message_received_samples: deque[tuple[float, int]] = field(
        default_factory=lambda: deque(maxlen=60)
    )
    _message_sent_samples: deque[tuple[float, int]] = field(
        default_factory=lambda: deque(maxlen=60)
    )
    _byte_samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=60))

    def record_received_message(self) -> None:
        """Record a message received from upstream."""
        now = time.time()
        self.messages_received += 1
        self._message_received_samples.append((now, 1))

    def record_sent_message(self, byte_count: int, subscriber_count: int) -> None:
        """Record a message sent to downstream clients."""
        now = time.time()
        self.messages_sent += subscriber_count  # Count per subscriber
        self.bytes_sent += byte_count * subscriber_count
        self._message_sent_samples.append((now, subscriber_count))
        self._byte_samples.append((now, byte_count * subscriber_count))

    def record_dropped_message(self) -> None:
        """Record a message dropped due to throttling."""
        self.messages_dropped += 1

    def record_transform_success(self) -> None:
        """Record a successful transformation."""
        self.transform_successes += 1

    def record_transform_failure(self) -> None:
        """Record a failed transformation."""
        self.transform_failures += 1

    def get_receive_rate(self, window_seconds: float = 5.0) -> float:
        """Calculate messages received per second."""
        return self._calculate_rate(self._message_received_samples, window_seconds)

    def get_send_rate(self, window_seconds: float = 5.0) -> float:
        """Calculate messages sent per second."""
        return self._calculate_rate(self._message_sent_samples, window_seconds)

    def get_bandwidth(self, window_seconds: float = 5.0) -> float:
        """Calculate bytes per second."""
        return self._calculate_rate(self._byte_samples, window_seconds)

    @staticmethod
    def _calculate_rate(samples: deque[tuple[float, int]], window_seconds: float) -> float:
        """Calculate rate from time-stamped samples within window."""
        if not samples:
            return 0.0

        now = time.time()
        cutoff = now - window_seconds
        total = sum(count for ts, count in samples if ts >= cutoff)

        valid_samples = [(ts, count) for ts, count in samples if ts >= cutoff]
        if not valid_samples:
            return 0.0

        oldest_ts = valid_samples[0][0]
        time_span = now - oldest_ts

        if time_span <= 0:
            return 0.0

        return total / time_span

    @property
    def subscriber_count(self) -> int:
        """Get current subscriber count."""
        return len(self.subscriber_ids)

    @property
    def throttle_efficiency(self) -> float:
        """Calculate throttle efficiency (% of messages sent vs received)."""
        total = self.messages_received + self.messages_dropped
        if total == 0:
            return 100.0
        return (self.messages_received / total) * 100.0


class MetricsCollector:
    """Central metrics collector for the proxy server."""

    def __init__(self) -> None:
        self.clients: dict[str, ClientMetrics] = {}
        self.channels: dict[int, ChannelMetrics] = {}
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

    def add_channel(
        self, channel_id: int, topic: str, schema_name: str, is_transformed: bool = False
    ) -> ChannelMetrics:
        """Add a new channel to track."""
        metrics = ChannelMetrics(
            channel_id=channel_id,
            topic=topic,
            schema_name=schema_name,
            is_transformed=is_transformed,
        )
        self.channels[channel_id] = metrics
        return metrics

    def get_channel(self, channel_id: int) -> ChannelMetrics | None:
        """Get channel metrics by ID."""
        return self.channels.get(channel_id)

    def remove_channel(self, channel_id: int) -> None:
        """Remove a channel from tracking."""
        self.channels.pop(channel_id, None)

    def get_total_message_rate(self) -> float:
        """Get total message rate across all channels."""
        return sum(ch.get_send_rate() for ch in self.channels.values())

    def get_total_bandwidth(self) -> float:
        """Get total bandwidth across all channels."""
        return sum(ch.get_bandwidth() for ch in self.channels.values())

    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()
