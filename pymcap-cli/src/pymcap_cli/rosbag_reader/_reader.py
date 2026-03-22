"""ROS1 bag format v2.0 reader.

Implements reading of ROS1 bag files per the specification at:
https://wiki.ros.org/Bags/Format/2.0
"""

from __future__ import annotations

import bz2
import io
import struct
from typing import TYPE_CHECKING, BinaryIO

from pymcap_cli.rosbag_reader._types import BagConnection, BagInfo, BagMessage

if TYPE_CHECKING:
    from collections.abc import Iterator

# Magic string at the start of every v2.0 bag file
_MAGIC = b"#ROSBAG V2.0\n"

# Record op codes
_OP_MESSAGE_DATA = 0x02
_OP_BAG_HEADER = 0x03
_OP_INDEX_DATA = 0x04
_OP_CHUNK = 0x05
_OP_CHUNK_INFO = 0x06
_OP_CONNECTION = 0x07

_NSEC_PER_SEC = 1_000_000_000


def _read_uint32(f: BinaryIO) -> int:
    """Read a little-endian uint32."""
    data = f.read(4)
    if len(data) < 4:
        raise EOFError("Unexpected end of file reading uint32")
    return struct.unpack("<I", data)[0]


def _parse_time(raw: bytes, offset: int = 0) -> int:
    """Parse a ROS time (4-byte secs + 4-byte nsecs) into nanoseconds."""
    secs, nsecs = struct.unpack_from("<II", raw, offset)
    return secs * _NSEC_PER_SEC + nsecs


def _read_header_fields(data: bytes) -> dict[str, bytes]:
    """Parse header fields from raw header bytes.

    Each field: [4-byte field_len][name=value]
    Split on first '=' only since values can contain '='.
    """
    fields: dict[str, bytes] = {}
    offset = 0
    while offset < len(data):
        if offset + 4 > len(data):
            break
        field_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        field_data = data[offset : offset + field_len]
        offset += field_len

        eq_pos = field_data.index(b"=")
        name = field_data[:eq_pos].decode("ascii")
        value = field_data[eq_pos + 1 :]
        fields[name] = value

    return fields


def _read_record(f: BinaryIO) -> tuple[dict[str, bytes], bytes] | None:
    """Read a single record (header + data).

    Returns None at EOF.
    """
    header_len_raw = f.read(4)
    if len(header_len_raw) < 4:
        return None

    header_len = struct.unpack("<I", header_len_raw)[0]
    header_data = f.read(header_len)
    if len(header_data) < header_len:
        return None

    data_len = _read_uint32(f)
    data = f.read(data_len)
    if len(data) < data_len:
        return None

    fields = _read_header_fields(header_data)
    return fields, data


def _parse_connection(header: dict[str, bytes], data: bytes) -> BagConnection:
    """Parse a connection record into a BagConnection."""
    conn_id = struct.unpack("<I", header["conn"])[0]
    topic = header.get("topic", b"").decode("utf-8")

    # The data section contains connection header fields in the same format
    conn_fields = _read_header_fields(data)

    msg_type = conn_fields.get("type", b"").decode("utf-8")
    md5sum = conn_fields.get("md5sum", b"").decode("utf-8")
    message_definition = conn_fields.get("message_definition", b"").decode("utf-8")
    callerid_raw = conn_fields.get("callerid")
    callerid = callerid_raw.decode("utf-8") if callerid_raw else None
    latching_raw = conn_fields.get("latching")
    latching = latching_raw == b"1" if latching_raw else False

    # Prefer the topic from the connection header data (original topic)
    topic_from_data = conn_fields.get("topic")
    if topic_from_data:
        topic = topic_from_data.decode("utf-8")

    return BagConnection(
        conn_id=conn_id,
        topic=topic,
        msg_type=msg_type,
        md5sum=md5sum,
        message_definition=message_definition,
        callerid=callerid,
        latching=latching,
    )


def _get_header_op(header: dict[str, bytes]) -> int:
    """Extract the op code from a record header."""
    return header["op"][0]


def _lz4_decompress(data: bytes) -> bytes:
    """Decompress LZ4 data, raising a clear error if lz4 is not installed."""
    try:
        import lz4.frame  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "lz4 package is required to read LZ4-compressed bag files. "
            "Install it with: pip install lz4"
        ) from None
    return lz4.frame.decompress(data)


def _decompress(compression: str, data: bytes) -> bytes:
    """Decompress chunk data based on compression type."""
    if compression == "none":
        return data
    if compression == "bz2":
        return bz2.decompress(data)
    if compression == "lz4":
        return _lz4_decompress(data)
    raise ValueError(f"Unknown compression type: {compression!r}")


def _parse_records_from_bytes(data: bytes) -> Iterator[tuple[dict[str, bytes], bytes]]:
    """Parse records from a byte buffer (e.g., decompressed chunk data)."""
    f = io.BytesIO(data)
    while True:
        record = _read_record(f)
        if record is None:
            break
        yield record


def read_bag_info(f: BinaryIO) -> BagInfo:
    """Read bag header, connections, and chunk info from a bag file.

    Seeks to the index section (using index_pos from the bag header)
    to read connection and chunk_info records without decompressing chunks.
    Falls back to sequential scanning if index_pos is 0.

    Args:
        f: An open binary file positioned at the start.

    Returns:
        BagInfo with connections and summary statistics.

    Raises:
        ValueError: If the file is not a valid ROS bag v2.0 file.

    """
    # Verify magic
    magic = f.read(len(_MAGIC))
    if magic != _MAGIC:
        raise ValueError(f"Not a ROS bag v2.0 file: expected {_MAGIC!r}, got {magic!r}")

    # Read bag header record
    record = _read_record(f)
    if record is None:
        raise ValueError("Bag file is empty (no bag header record)")

    header, _data = record
    op = _get_header_op(header)
    if op != _OP_BAG_HEADER:
        raise ValueError(f"Expected bag header record (op=0x03), got op=0x{op:02x}")

    index_pos = struct.unpack("<Q", header["index_pos"])[0]
    conn_count = struct.unpack("<I", header["conn_count"])[0]
    chunk_count = struct.unpack("<I", header["chunk_count"])[0]

    connections: dict[int, BagConnection] = {}
    total_messages = 0
    start_time_ns: int | None = None
    end_time_ns: int | None = None

    if index_pos > 0:
        # Seek to the index section and read connection + chunk_info records
        f.seek(index_pos)
        while True:
            record = _read_record(f)
            if record is None:
                break

            rec_header, rec_data = record
            rec_op = _get_header_op(rec_header)

            if rec_op == _OP_CONNECTION:
                conn = _parse_connection(rec_header, rec_data)
                connections[conn.conn_id] = conn

            elif rec_op == _OP_CHUNK_INFO:
                # Extract message counts and time range
                count = struct.unpack("<I", rec_header["count"])[0]

                st = _parse_time(rec_header["start_time"])
                et = _parse_time(rec_header["end_time"])

                if st > 0 and (start_time_ns is None or st < start_time_ns):
                    start_time_ns = st
                if et > 0 and (end_time_ns is None or et > end_time_ns):
                    end_time_ns = et

                # Sum message counts from chunk info data
                offset = 0
                for _ in range(count):
                    _conn_id, msg_count = struct.unpack_from("<II", rec_data, offset)
                    offset += 8
                    total_messages += msg_count
    else:
        # No index section — scan chunks to discover connections
        _scan_connections_from_chunks(f, connections)

    return BagInfo(
        conn_count=conn_count,
        chunk_count=chunk_count,
        connections=connections,
        message_count=total_messages,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
    )


def _scan_connections_from_chunks(
    f: BinaryIO,
    connections: dict[int, BagConnection],
) -> None:
    """Scan chunks sequentially to discover connection records (fallback)."""
    while True:
        record = _read_record(f)
        if record is None:
            break

        rec_header, rec_data = record
        rec_op = _get_header_op(rec_header)

        if rec_op == _OP_CONNECTION:
            conn = _parse_connection(rec_header, rec_data)
            connections[conn.conn_id] = conn
        elif rec_op == _OP_CHUNK:
            compression = rec_header["compression"].decode("ascii")
            decompressed = _decompress(compression, rec_data)
            for sub_header, sub_data in _parse_records_from_bytes(decompressed):
                sub_op = _get_header_op(sub_header)
                if sub_op == _OP_CONNECTION:
                    conn = _parse_connection(sub_header, sub_data)
                    connections[conn.conn_id] = conn


def read_bag_messages(
    f: BinaryIO,
    info: BagInfo | None = None,  # noqa: ARG001
) -> Iterator[BagMessage]:
    """Lazily iterate all messages from a bag file.

    Reads chunks sequentially from the start, decompresses them,
    and yields BagMessage for each message data record.

    Args:
        f: An open binary file positioned at the start.
        info: Optional BagInfo (used to know where index section starts).
              If not provided, reads until EOF.

    Yields:
        BagMessage instances in chronological order.

    """
    # Skip magic
    magic = f.read(len(_MAGIC))
    if magic != _MAGIC:
        raise ValueError(f"Not a ROS bag v2.0 file: expected {_MAGIC!r}, got {magic!r}")

    # Skip bag header record
    record = _read_record(f)
    if record is None:
        return

    while True:
        record = _read_record(f)
        if record is None:
            break

        rec_header, rec_data = record
        rec_op = _get_header_op(rec_header)

        if rec_op == _OP_CHUNK:
            compression = rec_header["compression"].decode("ascii")
            decompressed = _decompress(compression, rec_data)
            yield from _iter_messages_from_chunk(decompressed)

        elif rec_op == _OP_MESSAGE_DATA:
            # Messages outside chunks (rare but valid)
            yield _parse_message(rec_header, rec_data)

        elif rec_op in (_OP_INDEX_DATA, _OP_CHUNK_INFO, _OP_CONNECTION):
            # Index section or top-level connection records — skip
            continue


def _iter_messages_from_chunk(data: bytes) -> Iterator[BagMessage]:
    """Iterate message records from decompressed chunk data."""
    for sub_header, sub_data in _parse_records_from_bytes(data):
        sub_op = _get_header_op(sub_header)
        if sub_op == _OP_MESSAGE_DATA:
            yield _parse_message(sub_header, sub_data)


def _parse_message(header: dict[str, bytes], data: bytes) -> BagMessage:
    """Parse a message data record into a BagMessage."""
    conn_id = struct.unpack("<I", header["conn"])[0]
    time_ns = _parse_time(header["time"])
    return BagMessage(conn_id=conn_id, time_ns=time_ns, data=data)
