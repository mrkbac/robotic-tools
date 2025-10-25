"""Type definitions for the reader module."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class Topic:
    """Represents a topic with its schema information."""

    name: str
    schema_name: str | None
    topic_id: int
    message_count: int | None = None


@dataclass(slots=True, frozen=True)
class MessageEvent:
    """Represents a message event from a topic."""

    topic: str
    message: Any  # The deserialized message object
    timestamp_ns: int
    schema_name: str | None = None


@dataclass(slots=True, frozen=True)
class SourceInfo:
    """Information about a data source after initialization."""

    topics: list[Topic]
    start_time_ns: int | None = None
    end_time_ns: int | None = None
