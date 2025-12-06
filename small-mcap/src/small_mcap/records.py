import io
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import IO, ClassVar, Final, Protocol, TypeVar, cast


class WritableBuffer(Protocol):
    """Protocol for objects that can receive binary data."""

    def write(self, data: bytes | bytearray | memoryview, /) -> int:
        """Write data to the buffer and return bytes written."""
        ...


MAGIC = b"\x89MCAP0\r\n"
MAGIC_SIZE = len(MAGIC)

OPCODE_AND_LEN_STRUCT = struct.Struct("<BQ")


@unique
class Opcode(IntEnum):
    """Single-byte opcodes identifying MCAP record types.

    Opcodes in the range 0x01-0x7F are reserved for MCAP format usage.
    0x80-0xFF are reserved for private records. 0x00 is not a valid opcode.
    """

    ATTACHMENT = 0x09
    ATTACHMENT_INDEX = 0x0A
    CHANNEL = 0x04
    CHUNK = 0x06
    CHUNK_INDEX = 0x08
    DATA_END = 0x0F
    FOOTER = 0x02
    HEADER = 0x01
    MESSAGE = 0x05
    MESSAGE_INDEX = 0x07
    METADATA = 0x0C
    METADATA_INDEX = 0x0D
    SCHEMA = 0x03
    STATISTICS = 0x0B
    SUMMARY_OFFSET = 0x0E


def _read_string(data: bytes | memoryview, offset: int) -> tuple[str, int]:
    """Read a length-prefixed string from bytes.

    Strings are serialized using a uint32 byte length followed by UTF-8 encoded string data.

    Args:
        data: The byte buffer to read from
        offset: The starting position in the buffer

    Returns:
        A tuple of (decoded string, new offset after reading)
    """
    string_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    string_bytes = data[offset : offset + string_len]
    string_val = bytes(string_bytes).decode("utf-8")
    offset += string_len
    return string_val, offset


def _read_map(data: bytes | memoryview, offset: int) -> tuple[dict[str, str], int]:
    """Read a length-prefixed map from bytes.

    Maps are serialized using a uint32 byte length followed by key/value pairs,
    where both keys and values are length-prefixed strings.

    Args:
        data: The byte buffer to read from
        offset: The starting position in the buffer

    Returns:
        A tuple of (decoded map, new offset after reading)
    """
    map_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    map_end = offset + map_len
    result: dict[str, str] = {}
    while offset < map_end:
        key, offset = _read_string(data, offset)
        value, offset = _read_string(data, offset)
        result[key] = value
    return result, offset


def _write_string(value: str) -> bytes:
    """Write a length-prefixed string to bytes.

    Strings are serialized using a uint32 byte length followed by UTF-8 encoded string data.

    Args:
        value: The string to encode

    Returns:
        Serialized bytes with length prefix
    """
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _write_map(value: dict[str, str]) -> bytes:
    """Write a length-prefixed map to bytes.

    Maps are serialized using a uint32 byte length followed by key/value pairs,
    where both keys and values are length-prefixed strings.

    Args:
        value: The dictionary to encode

    Returns:
        Serialized bytes with length prefix
    """
    if not value:
        return struct.pack("<I", 0)

    parts = [_write_string(k) + _write_string(v) for k, v in value.items()]
    entries = b"".join(parts)
    return struct.pack("<I", len(entries)) + entries


T = TypeVar("T", bound="McapRecord")


class McapRecord(ABC):
    """Abstract base class for all MCAP record types.

    All MCAP records are serialized as: <opcode (1 byte)><length (8 bytes)><content>
    Opcodes are single-byte identifiers. Record content length is a uint64 value.
    """

    OPCODE: ClassVar[int]

    @abstractmethod
    def write_record_to(self, out: WritableBuffer) -> int:
        """Write the complete record (opcode + length + content) to a buffer.

        Each record handles its own serialization including opcode and length prefix.

        Args:
            out: The buffer to write to (any object with a write method)

        Returns:
            The number of bytes written
        """
        ...

    @classmethod
    @abstractmethod
    def read(cls, data: bytes | memoryview) -> "McapRecord":
        """Deserialize the record content without opcode and length prefix.

        Note: Always use `unpack_from` to also handle padding correctly.

        Args:
            data: The raw record content bytes

        Returns:
            The deserialized record instance
        """
        ...

    @classmethod
    def read_record(cls: type[T], data: IO[bytes] | io.BufferedIOBase) -> T:  # noqa: PYI019
        """Read and deserialize a complete record with opcode and length prefix.

        Args:
            data: A file-like object to read from

        Returns:
            The deserialized record instance

        Raises:
            EOFError: If not enough data is available to read the record
            ValueError: If the opcode doesn't match the expected record type
        """
        header = data.read(9)
        if len(header) < 9:
            raise EOFError("Not enough data to read record header")
        opcode, length = OPCODE_AND_LEN_STRUCT.unpack(header)
        content = data.read(length)
        if len(content) < length:
            raise EOFError("Not enough data to read record content")
        opcode = Opcode(opcode)
        if opcode != cls.OPCODE:
            raise ValueError(f"Expected opcode {cls.OPCODE}, got {opcode}")
        return cast("T", cls.read(content))


@dataclass(slots=True)
class Attachment(McapRecord):
    """Attachment record (op=0x09) containing auxiliary artifacts.

    Attachment records contain arbitrary data such as text, core dumps, calibration data,
    or other auxiliary files. Attachments must not appear within a chunk.

    Attributes:
        log_time: [8 bytes] Time at which the attachment was recorded (nanoseconds since epoch)
        create_time: [8 bytes] Time at which the attachment was created, or zero if not available
        name: [4 + N bytes] Name of the attachment (e.g., "scene1.jpg")
        media_type: [4 + N bytes] Media type (e.g., "text/plain")
        data: [8 + N bytes] The attachment data bytes (uint64 length-prefixed)
    """

    OPCODE: ClassVar[int] = Opcode.ATTACHMENT
    _TIMES_STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQ")
    _DATA_LEN_STRUCT: ClassVar[struct.Struct] = struct.Struct("<Q")
    _CRC_STRUCT: ClassVar[struct.Struct] = struct.Struct("<I")

    log_time: int
    create_time: int
    name: str
    media_type: str
    data: bytes | memoryview = field(repr=False)

    def write_record_to(self, out: WritableBuffer) -> int:
        content = b"".join(
            [
                self._TIMES_STRUCT.pack(self.log_time, self.create_time),
                _write_string(self.name),
                _write_string(self.media_type),
                self._DATA_LEN_STRUCT.pack(len(self.data)),
                self.data if isinstance(self.data, bytes) else bytes(self.data),
            ]
        )
        content_with_crc = content + self._CRC_STRUCT.pack(zlib.crc32(content))
        record = OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, len(content_with_crc)) + content_with_crc
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "Attachment":
        log_time, create_time = cls._TIMES_STRUCT.unpack_from(data, 0)
        name, offset = _read_string(data, 16)
        media_type, offset = _read_string(data, offset)
        data_len = cls._DATA_LEN_STRUCT.unpack_from(data, offset)[0]
        offset += 8
        return cls(log_time, create_time, name, media_type, data[offset : offset + data_len])


@dataclass(slots=True)
class AttachmentIndex(McapRecord):
    """Attachment Index record (op=0x0A) locating an attachment in the file.

    An Attachment Index record exists for every Attachment record in the file.
    These records appear in the summary section for fast lookup.

    Attributes:
        offset: [8 bytes] Byte offset from the start of the file to the attachment record
        length: [8 bytes] Byte length of the attachment record, including opcode and length prefix
        log_time: [8 bytes] Time at which the attachment was recorded (nanoseconds since epoch)
        create_time: [8 bytes] Time at which the attachment was created, or zero if not available
        data_size: [8 bytes] Size of the attachment data
        name: [4 + N bytes] Name of the attachment
        media_type: [4 + N bytes] Media type of the attachment (e.g., "text/plain")
    """

    OPCODE: ClassVar[int] = Opcode.ATTACHMENT_INDEX
    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQQQQ")

    offset: int
    length: int
    log_time: int
    create_time: int
    data_size: int
    name: str
    media_type: str

    def write_record_to(self, out: WritableBuffer) -> int:
        content = b"".join(
            [
                self._STRUCT.pack(
                    self.offset,
                    self.length,
                    self.log_time,
                    self.create_time,
                    self.data_size,
                ),
                _write_string(self.name),
                _write_string(self.media_type),
            ]
        )
        record = OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, len(content)) + content
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "AttachmentIndex":
        offset, length, log_time, create_time, data_size = cls._STRUCT.unpack_from(data, 0)
        name, off = _read_string(data, 40)
        media_type, _ = _read_string(data, off)
        return cls(offset, length, log_time, create_time, data_size, name, media_type)


@dataclass(slots=True)
class Channel(McapRecord):
    """Channel record (op=0x04) defining an encoded stream of messages on a topic.

    Channel records are uniquely identified within a file by their channel ID.
    A Channel record must occur at least once in the file prior to any message
    referring to its channel ID. Any two channel records sharing a common ID must be identical.

    Attributes:
        id: [2 bytes] A unique identifier for this channel within the file
        schema_id: [2 bytes] The schema for messages on this channel (0 indicates no schema)
        topic: [4 + N bytes] The channel topic
        message_encoding: [4 + N bytes] Encoding for messages on this channel
            (e.g., "json", "protobuf")
        metadata: [4 + N bytes] Arbitrary metadata about this channel as key-value pairs
    """

    OPCODE: ClassVar[int] = Opcode.CHANNEL
    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<HH")

    id: int
    schema_id: int
    topic: str
    message_encoding: str
    metadata: dict[str, str]

    def write_record_to(self, out: WritableBuffer) -> int:
        content = b"".join(
            [
                self._STRUCT.pack(self.id, self.schema_id),
                _write_string(self.topic),
                _write_string(self.message_encoding),
                _write_map(self.metadata),
            ]
        )
        record = OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, len(content)) + content
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "Channel":
        channel_id, schema_id = cls._STRUCT.unpack_from(data, 0)
        topic, offset = _read_string(data, 4)
        message_encoding, offset = _read_string(data, offset)
        metadata, _ = _read_map(data, offset)
        return cls(channel_id, schema_id, topic, message_encoding, metadata)


@dataclass(slots=True)
class Chunk(McapRecord):
    """Chunk record (op=0x06) containing a batch of compressed or uncompressed records.

    A Chunk contains Schema, Channel, and Message records. The batch of records
    may be compressed or uncompressed. All messages in the chunk must reference
    channels recorded earlier in the file.

    Attributes:
        message_start_time: [8 bytes] Earliest message log_time in the chunk (zero if no messages)
        message_end_time: [8 bytes] Latest message log_time in the chunk (zero if no messages)
        uncompressed_size: [8 bytes] Uncompressed size of the records field
        uncompressed_crc: [4 bytes] CRC32 checksum of uncompressed records
            (zero disables validation)
        compression: [4 + N bytes] Compression algorithm
            (e.g., "zstd", "lz4", or "" for no compression)
        data: [8 + N bytes] The chunk records, possibly compressed (uint64 length-prefixed)
    """

    OPCODE: ClassVar[int] = Opcode.CHUNK
    _HEADER_STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQQI")
    _DATA_LEN_STRUCT: ClassVar[struct.Struct] = struct.Struct("<Q")

    message_start_time: int
    message_end_time: int
    uncompressed_size: int
    uncompressed_crc: int
    compression: str
    data: bytes | memoryview = field(repr=False)

    def write_record_to(self, out: WritableBuffer) -> int:
        """Optimized write_record_to that avoids double copy of large data."""
        parts = (
            self._HEADER_STRUCT.pack(
                self.message_start_time,
                self.message_end_time,
                self.uncompressed_size,
                self.uncompressed_crc,
            ),
            _write_string(self.compression),
            self._DATA_LEN_STRUCT.pack(len(self.data)),
            self.data if isinstance(self.data, bytes) else bytes(self.data),
        )
        content_len = sum(len(p) for p in parts)
        record = b"".join((OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, content_len), *parts))
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "Chunk":
        message_start_time, message_end_time, uncompressed_size, uncompressed_crc = (
            cls._HEADER_STRUCT.unpack_from(data, 0)
        )
        compression, offset = _read_string(data, 28)
        data_len = cls._DATA_LEN_STRUCT.unpack_from(data, offset)[0]
        return cls(
            message_start_time,
            message_end_time,
            uncompressed_size,
            uncompressed_crc,
            compression,
            data[offset + 8 : offset + 8 + data_len],
        )


@dataclass(slots=True)
class ChunkIndex(McapRecord):
    """Chunk Index record (op=0x08) containing the location of a Chunk and its Message Indexes.

    A Chunk Index record exists for every Chunk in the file. These records appear
    in the summary section for fast lookup of chunks by time range.

    Attributes:
        message_start_time: [8 bytes] Earliest message log_time in the chunk (zero if no messages)
        message_end_time: [8 bytes] Latest message log_time in the chunk (zero if no messages)
        chunk_start_offset: [8 bytes] Byte offset from the start of the file to the chunk record
        chunk_length: [8 bytes] Byte length of the chunk record, including opcode and length prefix
        message_index_offsets: [4 + N bytes] Map from channel ID (uint16) to offset (uint64)
            of its message index record
        message_index_length: [8 bytes] Total byte length of all message index records
            after the chunk
        compression: [4 + N bytes] Compression used within the chunk
            (should match the Chunk record)
        compressed_size: [8 bytes] Size of the chunk records field
        uncompressed_size: [8 bytes] Uncompressed size of the chunk records
            (should match Chunk record)
    """

    OPCODE: ClassVar[int] = Opcode.CHUNK_INDEX
    _HEADER_STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQQQI")
    _OFFSET_ENTRY_STRUCT: ClassVar[struct.Struct] = struct.Struct("<HQ")
    _INDEX_LEN_STRUCT: ClassVar[struct.Struct] = struct.Struct("<Q")
    _SIZES_STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQ")
    _READ_HEADER_STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQQQ")
    _MAP_LEN_STRUCT: ClassVar[struct.Struct] = struct.Struct("<I")

    message_start_time: int
    message_end_time: int
    chunk_start_offset: int
    chunk_length: int
    message_index_offsets: dict[int, int]
    message_index_length: int
    compression: str
    compressed_size: int
    uncompressed_size: int

    def write_record_to(self, out: WritableBuffer) -> int:
        offsets_data = b"".join(
            self._OFFSET_ENTRY_STRUCT.pack(channel_id, offset)
            for channel_id, offset in self.message_index_offsets.items()
        )
        content = b"".join(
            [
                self._HEADER_STRUCT.pack(
                    self.message_start_time,
                    self.message_end_time,
                    self.chunk_start_offset,
                    self.chunk_length,
                    len(offsets_data),
                ),
                offsets_data,
                self._INDEX_LEN_STRUCT.pack(self.message_index_length),
                _write_string(self.compression),
                self._SIZES_STRUCT.pack(self.compressed_size, self.uncompressed_size),
            ]
        )
        record = OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, len(content)) + content
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "ChunkIndex":
        message_start_time, message_end_time, chunk_start_offset, chunk_length = (
            cls._READ_HEADER_STRUCT.unpack_from(data, 0)
        )
        map_len = cls._MAP_LEN_STRUCT.unpack_from(data, 32)[0]
        message_index_offsets = dict(
            struct.iter_unpack(cls._OFFSET_ENTRY_STRUCT.format, data[36 : 36 + map_len])
        )
        offset = 36 + map_len
        message_index_length = cls._INDEX_LEN_STRUCT.unpack_from(data, offset)[0]
        compression, offset = _read_string(data, offset + 8)
        compressed_size, uncompressed_size = cls._SIZES_STRUCT.unpack_from(data, offset)
        return cls(
            message_start_time,
            message_end_time,
            chunk_start_offset,
            chunk_length,
            message_index_offsets,
            message_index_length,
            compression,
            compressed_size,
            uncompressed_size,
        )


@dataclass(slots=True)
class DataEnd(McapRecord):
    """Data End record (op=0x0F) indicating the end of the data section.

    The Data End record provides a clear delineation that the data section has ended
    and the summary section starts. This must be the last record in the data section.

    Attributes:
        data_section_crc: [4 bytes] CRC32 of all bytes from the beginning of the file up to
            the DataEnd record (zero indicates CRC32 is not available)
    """

    OPCODE: ClassVar[int] = Opcode.DATA_END
    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<I")
    # Fixed content size: 4 bytes for data_section_crc
    _RECORD_STRUCT: ClassVar[struct.Struct] = struct.Struct("<BQI")

    data_section_crc: int

    def write_record_to(self, out: WritableBuffer) -> int:
        out.write(self._RECORD_STRUCT.pack(self.OPCODE, 4, self.data_section_crc))
        return 13  # 1 + 8 + 4

    @classmethod
    def read(cls, data: bytes | memoryview) -> "DataEnd":
        return cls(cls._STRUCT.unpack_from(data, 0)[0])


@dataclass(slots=True)
class Footer(McapRecord):
    """Footer record (op=0x02) containing end-of-file information.

    The Footer must be the last record in the file, before the trailing magic bytes.
    Readers using the index to read the file will begin by reading the footer.

    Attributes:
        summary_start: [8 bytes] Byte offset from start of file to the first record in the summary
            section (zero if no summary section)
        summary_offset_start: [8 bytes] Byte offset from start of file to the first record in the
            summary offset section (zero if no summary offset records)
        summary_crc: [4 bytes] CRC32 of all bytes from the start of the summary section up through
            and including the end of summary_offset_start field in the footer
            (zero indicates CRC32 is not available)
    """

    OPCODE: ClassVar[int] = Opcode.FOOTER
    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQI")
    # Fixed content size: 8 + 8 + 4 = 20 bytes
    _RECORD_STRUCT: ClassVar[struct.Struct] = struct.Struct("<BQQQI")

    summary_start: int
    summary_offset_start: int
    summary_crc: int

    def write_record_to(self, out: WritableBuffer) -> int:
        out.write(
            self._RECORD_STRUCT.pack(
                self.OPCODE, 20, self.summary_start, self.summary_offset_start, self.summary_crc
            )
        )
        return 29  # 1 + 8 + 20

    @classmethod
    def read(cls, data: bytes | memoryview) -> "Footer":
        return cls(*cls._STRUCT.unpack_from(data, 0))


@dataclass(slots=True)
class Header(McapRecord):
    """Header record (op=0x01) appearing first after the leading magic bytes.

    The Header is the first record in the file and provides information about
    the file profile and the library that wrote it.

    Attributes:
        profile: [4 + N bytes] Indicates requirements for fields throughout the file. If the value
            matches a well-known profile, the file should conform to that profile.
            May be empty or contain a framework that is not recognized.
        library: [4 + N bytes] Free-form string for writer to specify its name, version, or other
            information for use in debugging
    """

    OPCODE: ClassVar[int] = Opcode.HEADER

    profile: str
    library: str

    def write_record_to(self, out: WritableBuffer) -> int:
        content = _write_string(self.profile) + _write_string(self.library)
        record = OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, len(content)) + content
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "Header":
        profile, offset = _read_string(data, 0)
        library, _ = _read_string(data, offset)
        return cls(profile, library)


@dataclass(slots=True)
class Message(McapRecord):
    """Message record (op=0x05) encoding a single timestamped message on a channel.

    The message encoding and schema must match that of the Channel record
    corresponding to the message's channel ID.

    Attributes:
        channel_id: [2 bytes] The ID of the channel this message belongs to
        sequence: [4 bytes] Optional message counter to detect message gaps (zero if not used)
        log_time: [8 bytes] Time at which the message was recorded (nanoseconds since epoch)
        publish_time: [8 bytes] Time at which the message was published (nanoseconds since epoch),
            must be set to log_time if not available
        data: [N bytes] Message data, to be decoded according to the schema of the channel
    """

    OPCODE: ClassVar[int] = Opcode.MESSAGE
    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<HIQQ")
    # Combined struct for write_record_to: opcode + length + fields
    _RECORD_STRUCT: ClassVar[struct.Struct] = struct.Struct("<BQHIQQ")

    channel_id: int
    sequence: int
    log_time: int
    publish_time: int
    data: bytes | memoryview = field(repr=False)

    def write_record_to(self, out: WritableBuffer) -> int:
        """Optimized write_record_to for Message - the hot path."""
        data_len = len(self.data)
        out.write(
            self._RECORD_STRUCT.pack(
                self.OPCODE,
                22 + data_len,
                self.channel_id,
                self.sequence,
                self.log_time,
                self.publish_time,
            )
        )
        out.write(self.data)
        return 31 + data_len

    @classmethod
    def read(cls, data: bytes | memoryview) -> "Message":
        channel_id, sequence, log_time, publish_time = cls._STRUCT.unpack_from(data, 0)
        return cls(channel_id, sequence, log_time, publish_time, data[22:])


@dataclass(slots=True)
class MessageIndex(McapRecord):
    """Message Index record (op=0x07) locating individual messages within a chunk.

    A sequence of Message Index records occurs immediately after each chunk.
    Exactly one Message Index record must exist for every channel that has
    messages inside the chunk.

    Attributes:
        channel_id: [2 bytes] The channel ID for which this index applies
        records: [4 + N bytes] Array of (Timestamp, uint64) tuples for each message record.
            The offset is relative to the start of the uncompressed chunk data.
    """

    OPCODE: ClassVar[int] = Opcode.MESSAGE_INDEX
    _HEADER_STRUCT: ClassVar[struct.Struct] = struct.Struct("<HI")
    _ENTRY_STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQ")

    channel_id: int
    records: list[tuple[int, int]]

    def write_record_to(self, out: WritableBuffer) -> int:
        # Pre-allocate buffer for better performance
        num_records = len(self.records)
        records_size = num_records * 16  # 2 * 8 bytes per record
        content_size = 2 + 4 + records_size  # channel_id + length prefix + records

        # Build complete record: header (9 bytes) + content
        buffer = bytearray(9 + content_size)
        OPCODE_AND_LEN_STRUCT.pack_into(buffer, 0, self.OPCODE, content_size)
        self._HEADER_STRUCT.pack_into(buffer, 9, self.channel_id, records_size)
        offset = 15  # 9 + 6
        for timestamp, msg_offset in self.records:
            self._ENTRY_STRUCT.pack_into(buffer, offset, timestamp, msg_offset)
            offset += 16

        out.write(buffer)
        return len(buffer)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "MessageIndex":
        channel_id, records_len = cls._HEADER_STRUCT.unpack_from(data, 0)
        if records_len == 0:
            return cls(channel_id, [])
        records = list(struct.iter_unpack(cls._ENTRY_STRUCT.format, data[6 : 6 + records_len]))
        return cls(channel_id, records)


@dataclass(slots=True)
class Metadata(McapRecord):
    """Metadata record (op=0x0C) containing arbitrary user data in key-value pairs.

    Attributes:
        name: [4 + N bytes] Name of the metadata record (e.g., "my_company_name_hardware_info")
        metadata: [4 + N bytes] Arbitrary key-value pairs
            (e.g., "part_id", "serial", "board_revision")
    """

    OPCODE: ClassVar[int] = Opcode.METADATA

    name: str
    metadata: dict[str, str]

    def write_record_to(self, out: WritableBuffer) -> int:
        content = _write_string(self.name) + _write_map(self.metadata)
        record = OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, len(content)) + content
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "Metadata":
        name, offset = _read_string(data, 0)
        metadata, _ = _read_map(data, offset)
        return cls(name, metadata)


@dataclass(slots=True)
class MetadataIndex(McapRecord):
    """Metadata Index record (op=0x0D) containing the location of a metadata record.

    Attributes:
        offset: [8 bytes] Byte offset from the start of the file to the metadata record
        length: [8 bytes] Total byte length of the record, including opcode and length prefix
        name: [4 + N bytes] Name of the metadata record
    """

    OPCODE: ClassVar[int] = Opcode.METADATA_INDEX
    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQ")

    offset: int
    length: int
    name: str

    def write_record_to(self, out: WritableBuffer) -> int:
        content = self._STRUCT.pack(self.offset, self.length) + _write_string(self.name)
        record = OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, len(content)) + content
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "MetadataIndex":
        offset, length = cls._STRUCT.unpack_from(data, 0)
        name, _ = _read_string(data, 16)
        return cls(offset, length, name)


@dataclass(slots=True)
class Schema(McapRecord):
    """Schema record (op=0x03) defining an individual schema.

    Schema records are uniquely identified within a file by their schema ID.
    A Schema record must occur at least once in the file prior to any Channel
    referring to its ID. Any two schema records sharing a common ID must be identical.
    A schema ID of zero is invalid.

    Attributes:
        id: [2 bytes] A unique identifier for this schema within the file (must not be zero)
        name: [4 + N bytes] An identifier for the schema
        encoding: [4 + N bytes] Format for the schema (e.g., "protobuf", "jsonschema").
            An empty string indicates no schema is available.
        data: [4 + N bytes] Schema data conforming to the encoding (uint32 length-prefixed).
            If encoding is empty, data should be empty.
    """

    OPCODE: ClassVar[int] = Opcode.SCHEMA
    _ID_STRUCT: ClassVar[struct.Struct] = struct.Struct("<H")
    _DATA_LEN_STRUCT: ClassVar[struct.Struct] = struct.Struct("<I")

    id: int
    name: str
    encoding: str
    data: bytes = field(repr=False)

    def write_record_to(self, out: WritableBuffer) -> int:
        content = b"".join(
            [
                self._ID_STRUCT.pack(self.id),
                _write_string(self.name),
                _write_string(self.encoding),
                self._DATA_LEN_STRUCT.pack(len(self.data)),
                self.data,
            ]
        )
        record = OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, len(content)) + content
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "Schema":
        schema_id = cls._ID_STRUCT.unpack_from(data, 0)[0]
        name, offset = _read_string(data, 2)
        encoding, offset = _read_string(data, offset)
        data_len = cls._DATA_LEN_STRUCT.unpack_from(data, offset)[0]
        schema_data = data[offset + 4 : offset + 4 + data_len]
        return cls(
            schema_id,
            name,
            encoding,
            bytes(schema_data) if isinstance(schema_data, memoryview) else schema_data,
        )


@dataclass(slots=True)
class Statistics(McapRecord):
    """Statistics record (op=0x0B) containing summary information about the recorded data.

    The statistics record is optional, but the file should contain at most one.
    When using a Statistics record with non-empty channel_message_counts, the summary
    section must contain a copy of all Channel records prior to the statistics record.

    Attributes:
        message_count: [8 bytes] Total number of Message records in the file
        schema_count: [2 bytes] Number of unique schema IDs in the file (not including zero)
        channel_count: [4 bytes] Number of unique channel IDs in the file
        attachment_count: [4 bytes] Number of Attachment records in the file
        metadata_count: [4 bytes] Number of Metadata records in the file
        chunk_count: [4 bytes] Number of Chunk records in the file
        message_start_time: [8 bytes] Earliest message log_time in the file (zero if no messages)
        message_end_time: [8 bytes] Latest message log_time in the file (zero if no messages)
        channel_message_counts: [4 + N bytes] Map from channel ID (uint16) to total message
            count (uint64) for that channel (empty map indicates this statistic is not available)
    """

    OPCODE: ClassVar[int] = Opcode.STATISTICS
    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<QHIIIIQQI")
    _COUNT_ENTRY_STRUCT: ClassVar[struct.Struct] = struct.Struct("<HQ")
    # For reading - split into parts due to different offsets
    _MSG_COUNT_STRUCT: ClassVar[struct.Struct] = struct.Struct("<Q")
    _SCHEMA_COUNT_STRUCT: ClassVar[struct.Struct] = struct.Struct("<H")
    _COUNTS_STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIII")
    _TIMES_STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQ")
    _MAP_LEN_STRUCT: ClassVar[struct.Struct] = struct.Struct("<I")

    message_count: int
    schema_count: int
    channel_count: int
    attachment_count: int
    metadata_count: int
    chunk_count: int
    message_start_time: int
    message_end_time: int
    channel_message_counts: dict[int, int]

    def write_record_to(self, out: WritableBuffer) -> int:
        counts_data = b"".join(
            self._COUNT_ENTRY_STRUCT.pack(channel_id, count)
            for channel_id, count in self.channel_message_counts.items()
        )
        content = (
            self._STRUCT.pack(
                self.message_count,
                self.schema_count,
                self.channel_count,
                self.attachment_count,
                self.metadata_count,
                self.chunk_count,
                self.message_start_time,
                self.message_end_time,
                len(counts_data),
            )
            + counts_data
        )
        record = OPCODE_AND_LEN_STRUCT.pack(self.OPCODE, len(content)) + content
        out.write(record)
        return len(record)

    @classmethod
    def read(cls, data: bytes | memoryview) -> "Statistics":
        message_count = cls._MSG_COUNT_STRUCT.unpack_from(data, 0)[0]
        schema_count = cls._SCHEMA_COUNT_STRUCT.unpack_from(data, 8)[0]
        channel_count, attachment_count, metadata_count, chunk_count = (
            cls._COUNTS_STRUCT.unpack_from(data, 10)
        )
        message_start_time, message_end_time = cls._TIMES_STRUCT.unpack_from(data, 26)
        counts_len = cls._MAP_LEN_STRUCT.unpack_from(data, 42)[0]
        channel_message_counts = dict(
            struct.iter_unpack(cls._COUNT_ENTRY_STRUCT.format, data[46 : 46 + counts_len])
        )
        return cls(
            message_count,
            schema_count,
            channel_count,
            attachment_count,
            metadata_count,
            chunk_count,
            message_start_time,
            message_end_time,
            channel_message_counts,
        )


@dataclass(slots=True)
class SummaryOffset(McapRecord):
    """Summary Offset record (op=0x0E) containing the location of summary section records.

    Each Summary Offset record corresponds to a group of summary records with the same opcode.
    These records appear in the summary offset section for fast lookup.

    Attributes:
        group_opcode: [1 byte] The opcode of all records in the group
        group_start: [8 bytes] Byte offset from the start of the file of the first record
            in the group
        group_length: [8 bytes] Total byte length of all records in the group
    """

    OPCODE: ClassVar[int] = Opcode.SUMMARY_OFFSET
    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<BQQ")
    # Fixed content size: 1 + 8 + 8 = 17 bytes
    _RECORD_STRUCT: ClassVar[struct.Struct] = struct.Struct("<BQBQQ")

    group_opcode: int
    group_start: int
    group_length: int

    def write_record_to(self, out: WritableBuffer) -> int:
        out.write(
            self._RECORD_STRUCT.pack(
                self.OPCODE, 17, self.group_opcode, self.group_start, self.group_length
            )
        )
        return 26  # 1 + 8 + 17

    @classmethod
    def read(cls, data: bytes | memoryview) -> "SummaryOffset":
        return cls(*cls._STRUCT.unpack_from(data, 0))


@dataclass(slots=True)
class Summary:
    """Container for summary section data from an MCAP file.

    The summary section contains records for fast lookup of file information
    and data section records. All records in the summary section are grouped by opcode.

    Attributes:
        statistics: Optional statistics record with summary information about the file
        schemas: Map from schema ID to Schema records (duplicates from data section)
        channels: Map from channel ID to Channel records (duplicates from data section)
        chunk_indexes: List of Chunk Index records for locating chunks
        attachment_indexes: List of Attachment Index records for locating attachments
        metadata_indexes: List of Metadata Index records for locating metadata
    """

    statistics: Statistics | None = None
    schemas: dict[int, Schema] = field(default_factory=dict)
    channels: dict[int, Channel] = field(default_factory=dict)
    chunk_indexes: list[ChunkIndex] = field(default_factory=list)
    attachment_indexes: list[AttachmentIndex] = field(default_factory=list)
    metadata_indexes: list[MetadataIndex] = field(default_factory=list)


OPCODE_TO_RECORD: Final[dict[int, type[McapRecord]]] = {
    Opcode.ATTACHMENT: Attachment,
    Opcode.ATTACHMENT_INDEX: AttachmentIndex,
    Opcode.CHANNEL: Channel,
    Opcode.CHUNK: Chunk,
    Opcode.CHUNK_INDEX: ChunkIndex,
    Opcode.DATA_END: DataEnd,
    Opcode.FOOTER: Footer,
    Opcode.HEADER: Header,
    Opcode.MESSAGE: Message,
    Opcode.MESSAGE_INDEX: MessageIndex,
    Opcode.METADATA: Metadata,
    Opcode.METADATA_INDEX: MetadataIndex,
    Opcode.SCHEMA: Schema,
    Opcode.STATISTICS: Statistics,
    Opcode.SUMMARY_OFFSET: SummaryOffset,
}


@dataclass(slots=True)
class LazyChunk:
    """
    Lazy-loaded chunk that stores metadata without reading the compressed data.
    Used for efficient scanning of MCAP files when chunk data is not immediately needed.
    """

    message_start_time: int
    message_end_time: int
    uncompressed_size: int
    uncompressed_crc: int
    compression: str
    record_start: int  # Offset where the Chunk record begins
    data_len: int  # Length of the compressed data

    @classmethod
    def read_from_stream(cls, stream: IO[bytes], record_start: int) -> "LazyChunk":
        """Read chunk metadata from stream without loading the compressed data."""
        data = stream.read(8 + 8 + 8 + 4 + 4)
        message_start_time, message_end_time, uncompressed_size, uncompressed_crc, str_len = (
            struct.unpack("<QQQII", data)
        )
        compression = stream.read(str_len).decode("utf-8")
        data_len = struct.unpack("<Q", stream.read(8))[0]
        stream.seek(data_len, 1)  # Skip the data for now
        return cls(
            message_start_time,
            message_end_time,
            uncompressed_size,
            uncompressed_crc,
            compression,
            record_start,
            data_len,
        )

    def to_chunk(self, stream: IO[bytes]) -> Chunk:
        """Convert to a full Chunk by reading from the stream."""
        current_pos = stream.tell()
        stream.seek(self.record_start)
        chunk = Chunk.read_record(stream)
        stream.seek(current_pos)
        return chunk
