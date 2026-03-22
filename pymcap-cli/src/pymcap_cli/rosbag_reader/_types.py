"""Data types for ROS1 bag format v2.0."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BagConnection:
    """A connection record from a ROS1 bag file.

    Each connection represents a unique publisher on a topic.
    The message_definition contains the full recursive .msg text,
    suitable for use as an MCAP schema.
    """

    conn_id: int
    topic: str
    msg_type: str
    md5sum: str
    message_definition: str
    callerid: str | None = None
    latching: bool = False


@dataclass(slots=True)
class BagMessage:
    """A raw message from a ROS1 bag file.

    The data field contains the raw ROS1-serialized bytes,
    passed through without deserialization.
    """

    conn_id: int
    time_ns: int  # Nanoseconds since epoch (sec * 1_000_000_000 + nsec)
    data: bytes


@dataclass(slots=True)
class BagInfo:
    """Summary information from a ROS1 bag file."""

    conn_count: int
    chunk_count: int
    connections: dict[int, BagConnection] = field(default_factory=dict)
    message_count: int = 0
    start_time_ns: int | None = None
    end_time_ns: int | None = None
