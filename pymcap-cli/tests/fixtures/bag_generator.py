"""Programmatic ROS1 bag v2.0 file generator for tests.

Generates valid bag files with controlled content for testing
the rosbag_reader without external dependencies.
"""

from __future__ import annotations

import bz2
import io
import struct
from dataclasses import dataclass

# Op codes
_OP_MESSAGE_DATA = 0x02
_OP_BAG_HEADER = 0x03
_OP_INDEX_DATA = 0x04
_OP_CHUNK = 0x05
_OP_CHUNK_INFO = 0x06
_OP_CONNECTION = 0x07

_MAGIC = b"#ROSBAG V2.0\n"
_NSEC_PER_SEC = 1_000_000_000


@dataclass
class TestConnection:
    """Definition of a topic/connection for test bag generation."""

    conn_id: int
    topic: str
    msg_type: str
    md5sum: str
    message_definition: str
    callerid: str | None = None
    latching: bool = False


@dataclass
class TestMessage:
    """A test message to write into a bag."""

    conn_id: int
    time_secs: int
    time_nsecs: int
    data: bytes


def _encode_field(name: str, value: bytes) -> bytes:
    """Encode a single header field: [4B len][name=value]."""
    field_data = name.encode("ascii") + b"=" + value
    return struct.pack("<I", len(field_data)) + field_data


def _encode_header(fields: list[tuple[str, bytes]]) -> bytes:
    """Encode a full header from a list of (name, value) pairs."""
    parts = b"".join(_encode_field(name, value) for name, value in fields)
    return struct.pack("<I", len(parts)) + parts


def _encode_time(secs: int, nsecs: int) -> bytes:
    """Encode a ROS time as 4B secs + 4B nsecs."""
    return struct.pack("<II", secs, nsecs)


def _build_connection_record(conn: TestConnection) -> bytes:
    """Build a complete connection record (header + data)."""
    header = _encode_header(
        [
            ("op", bytes([_OP_CONNECTION])),
            ("conn", struct.pack("<I", conn.conn_id)),
            ("topic", conn.topic.encode("utf-8")),
        ]
    )

    # Connection data contains topic metadata as header-style fields
    data_fields: list[tuple[str, bytes]] = [
        ("topic", conn.topic.encode("utf-8")),
        ("type", conn.msg_type.encode("utf-8")),
        ("md5sum", conn.md5sum.encode("utf-8")),
        ("message_definition", conn.message_definition.encode("utf-8")),
    ]
    if conn.callerid:
        data_fields.append(("callerid", conn.callerid.encode("utf-8")))
    if conn.latching:
        data_fields.append(("latching", b"1"))

    # Data is the encoded fields (without the outer length prefix from _encode_header)
    data_parts = b"".join(_encode_field(name, value) for name, value in data_fields)
    data_with_len = struct.pack("<I", len(data_parts)) + data_parts

    # But the record format is: [header][data_len][data]
    # header already includes its own length prefix, so we just need data_len + data
    # Actually _encode_header already adds the header_len prefix.
    # The full record is: [header_len][header][data_len][data]
    # _encode_header returns [header_len][header_bytes]
    # We need to append [data_len][data]
    return header + data_with_len


def _build_message_record(msg: TestMessage) -> bytes:
    """Build a complete message data record (header + data)."""
    header = _encode_header(
        [
            ("op", bytes([_OP_MESSAGE_DATA])),
            ("conn", struct.pack("<I", msg.conn_id)),
            ("time", _encode_time(msg.time_secs, msg.time_nsecs)),
        ]
    )
    data_with_len = struct.pack("<I", len(msg.data)) + msg.data
    return header + data_with_len


def _build_chunk(
    connections: list[TestConnection],
    messages: list[TestMessage],
    compression: str = "none",
) -> tuple[bytes, list[tuple[int, int, int]]]:
    """Build a chunk record containing connections and messages.

    Returns (chunk_record_bytes, index_entries) where index_entries
    is a list of (conn_id, time_secs, time_nsecs) for building index records.
    """
    # Build inner records (connections + messages)
    inner = io.BytesIO()
    for conn in connections:
        inner.write(_build_connection_record(conn))

    index_entries: list[tuple[int, int, int]] = []
    for msg in messages:
        offset = inner.tell()
        inner.write(_build_message_record(msg))
        index_entries.append((msg.conn_id, msg.time_secs, msg.time_nsecs))
        # Store offset for index data - we don't really use it in tests
        _ = offset

    uncompressed = inner.getvalue()
    uncompressed_size = len(uncompressed)

    # Compress
    if compression == "bz2":
        compressed = bz2.compress(uncompressed)
    elif compression == "lz4":
        import lz4.frame  # noqa: PLC0415

        compressed = lz4.frame.compress(uncompressed)
    elif compression == "none":
        compressed = uncompressed
    else:
        raise ValueError(f"Unknown compression: {compression}")

    # Build chunk record
    header = _encode_header(
        [
            ("op", bytes([_OP_CHUNK])),
            ("compression", compression.encode("ascii")),
            ("size", struct.pack("<I", uncompressed_size)),
        ]
    )
    chunk_record = header + struct.pack("<I", len(compressed)) + compressed

    return chunk_record, index_entries


def _build_index_data(
    conn_id: int,
    entries: list[tuple[int, int]],  # (time_secs, time_nsecs)
) -> bytes:
    """Build an index data record for a connection within a chunk."""
    header = _encode_header(
        [
            ("op", bytes([_OP_INDEX_DATA])),
            ("ver", struct.pack("<I", 1)),
            ("conn", struct.pack("<I", conn_id)),
            ("count", struct.pack("<I", len(entries))),
        ]
    )
    # Data: repeating [time (8B)][offset (4B)]
    data = b""
    for secs, nsecs in entries:
        data += _encode_time(secs, nsecs) + struct.pack("<I", 0)

    return header + struct.pack("<I", len(data)) + data


def _build_chunk_info(
    chunk_pos: int,
    connections_in_chunk: dict[int, int],  # conn_id -> msg_count
    start_secs: int,
    start_nsecs: int,
    end_secs: int,
    end_nsecs: int,
) -> bytes:
    """Build a chunk info record."""
    header = _encode_header(
        [
            ("op", bytes([_OP_CHUNK_INFO])),
            ("ver", struct.pack("<I", 1)),
            ("chunk_pos", struct.pack("<Q", chunk_pos)),
            ("start_time", _encode_time(start_secs, start_nsecs)),
            ("end_time", _encode_time(end_secs, end_nsecs)),
            ("count", struct.pack("<I", len(connections_in_chunk))),
        ]
    )
    # Data: repeating [conn_id (4B)][count (4B)]
    data = b""
    for conn_id, count in connections_in_chunk.items():
        data += struct.pack("<II", conn_id, count)

    return header + struct.pack("<I", len(data)) + data


def generate_bag(
    connections: list[TestConnection],
    messages: list[TestMessage],
    compression: str = "none",
) -> bytes:
    """Generate a complete ROS1 bag v2.0 file.

    Args:
        connections: Topic/connection definitions.
        messages: Messages to write, should be sorted by time.
        compression: Chunk compression ("none", "bz2", "lz4").

    Returns:
        Complete bag file as bytes.

    """
    buf = io.BytesIO()

    # Magic
    buf.write(_MAGIC)

    # Placeholder bag header (will be updated with index_pos)
    bag_header_start = buf.tell()
    bag_header_fields = _encode_header(
        [
            ("op", bytes([_OP_BAG_HEADER])),
            ("index_pos", struct.pack("<Q", 0)),  # placeholder
            ("conn_count", struct.pack("<I", len(connections))),
            ("chunk_count", struct.pack("<I", 1)),
        ]
    )
    # Pad data to 4096 bytes
    padding_needed = 4096 - 4  # minus 4 for data_len field
    bag_header_data = b" " * padding_needed
    buf.write(bag_header_fields + struct.pack("<I", len(bag_header_data)) + bag_header_data)

    # Single chunk containing all connections and messages
    chunk_pos = buf.tell()
    chunk_record, index_entries = _build_chunk(connections, messages, compression)
    buf.write(chunk_record)

    # Index data records (one per connection)
    conn_entries: dict[int, list[tuple[int, int]]] = {}
    for conn_id, secs, nsecs in index_entries:
        conn_entries.setdefault(conn_id, []).append((secs, nsecs))

    for conn_id, entries in conn_entries.items():
        buf.write(_build_index_data(conn_id, entries))

    # Index section starts here
    index_pos = buf.tell()

    # Connection records in index section
    for conn in connections:
        buf.write(_build_connection_record(conn))

    # Chunk info
    if messages:
        start_secs, start_nsecs = messages[0].time_secs, messages[0].time_nsecs
        end_secs, end_nsecs = messages[-1].time_secs, messages[-1].time_nsecs
    else:
        start_secs = start_nsecs = end_secs = end_nsecs = 0

    conn_msg_counts: dict[int, int] = {}
    for msg in messages:
        conn_msg_counts[msg.conn_id] = conn_msg_counts.get(msg.conn_id, 0) + 1

    buf.write(
        _build_chunk_info(chunk_pos, conn_msg_counts, start_secs, start_nsecs, end_secs, end_nsecs)
    )

    # Patch bag header with correct index_pos
    result = bytearray(buf.getvalue())
    # Find index_pos field in bag header and update it
    # The bag header record starts right after magic. We need to find the index_pos field.
    # Easier: just rebuild the bag header with the correct index_pos
    buf2 = io.BytesIO()
    bag_header_fields2 = _encode_header(
        [
            ("op", bytes([_OP_BAG_HEADER])),
            ("index_pos", struct.pack("<Q", index_pos)),
            ("conn_count", struct.pack("<I", len(connections))),
            ("chunk_count", struct.pack("<I", 1)),
        ]
    )
    buf2.write(bag_header_fields2)
    new_header = buf2.getvalue()
    old_header_len = bag_header_start  # length of magic
    # Replace the header portion (same size since field values have fixed sizes)
    result[old_header_len : old_header_len + len(new_header)] = new_header

    return bytes(result)


# Pre-built test connections and message generators


def make_string_connection(conn_id: int = 0, topic: str = "/chatter") -> TestConnection:
    """Create a std_msgs/String connection."""
    return TestConnection(
        conn_id=conn_id,
        topic=topic,
        msg_type="std_msgs/String",
        md5sum="992ce8a1687cec8c8bd883ec73ca41d1",
        message_definition="string data\n",
    )


def make_int_connection(conn_id: int = 1, topic: str = "/counter") -> TestConnection:
    """Create a std_msgs/Int32 connection."""
    return TestConnection(
        conn_id=conn_id,
        topic=topic,
        msg_type="std_msgs/Int32",
        md5sum="da5909fbe378aeaf85e547e830cc1bb7",
        message_definition="int32 data\n",
    )


def make_string_message(conn_id: int, secs: int, nsecs: int, text: str) -> TestMessage:
    """Create a std_msgs/String message with ROS1 serialization."""
    # ROS1 string serialization: [4B len][string bytes]
    encoded = text.encode("utf-8")
    data = struct.pack("<I", len(encoded)) + encoded
    return TestMessage(conn_id=conn_id, time_secs=secs, time_nsecs=nsecs, data=data)


def make_int_message(conn_id: int, secs: int, nsecs: int, value: int) -> TestMessage:
    """Create a std_msgs/Int32 message with ROS1 serialization."""
    data = struct.pack("<i", value)
    return TestMessage(conn_id=conn_id, time_secs=secs, time_nsecs=nsecs, data=data)


def generate_simple_bag(compression: str = "none") -> bytes:
    """Generate a simple bag with one topic and 5 string messages."""
    conn = make_string_connection()
    messages = [make_string_message(0, 1000, i * 100_000_000, f"hello {i}") for i in range(5)]
    return generate_bag([conn], messages, compression)


def generate_multi_topic_bag(compression: str = "none") -> bytes:
    """Generate a bag with two topics (string + int32) and interleaved messages."""
    string_conn = make_string_connection(conn_id=0, topic="/chatter")
    int_conn = make_int_connection(conn_id=1, topic="/counter")

    messages = []
    for i in range(5):
        messages.append(make_string_message(0, 1000, i * 200_000_000, f"msg {i}"))
        messages.append(make_int_message(1, 1000, i * 200_000_000 + 100_000_000, i * 10))

    return generate_bag([string_conn, int_conn], messages, compression)


def generate_empty_bag() -> bytes:
    """Generate a valid bag file with no messages."""
    return generate_bag([], [])
