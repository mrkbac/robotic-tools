import io
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import IO, ClassVar, Final, TypeVar, cast

MAGIC = b"\x89MCAP0\r\n"
MAGIC_SIZE = len(MAGIC)


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


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
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
    string_val = data[offset : offset + string_len].decode("utf-8")
    offset += string_len
    return string_val, offset


def _read_map(data: bytes, offset: int) -> tuple[dict[str, str], int]:
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

    parts: list[bytes] = []
    for k, v in value.items():
        parts.append(_write_string(k))
        parts.append(_write_string(v))
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
    def write(self) -> bytes:
        """Serialize the record content without opcode and length prefix.

        Returns:
            The serialized record content as bytes
        """
        ...

    def write_record(self) -> bytes:
        """Serialize the complete record with opcode and length prefix.

        Returns:
            The complete serialized record: <opcode><length><content>
        """
        content = self.write()
        return struct.pack("<BQ", self.OPCODE, len(content)) + content

    @classmethod
    @abstractmethod
    def read(cls, data: bytes) -> "McapRecord":
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
        opcode, length = struct.unpack("<BQ", header)
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

    log_time: int
    create_time: int
    name: str
    media_type: str
    data: bytes

    def write(self) -> bytes:
        content = (
            struct.pack("<QQ", self.log_time, self.create_time)
            + _write_string(self.name)
            + _write_string(self.media_type)
            + struct.pack("<Q", len(self.data))
            + self.data
        )
        return content + struct.pack("<I", zlib.crc32(content))

    @classmethod
    def read(cls, data: bytes) -> "Attachment":
        log_time, create_time = struct.unpack_from("<QQ", data, 0)
        name, offset = _read_string(data, 16)
        media_type, offset = _read_string(data, offset)
        data_len = struct.unpack_from("<Q", data, offset)[0]
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

    offset: int
    length: int
    log_time: int
    create_time: int
    data_size: int
    name: str
    media_type: str

    def write(self) -> bytes:
        return (
            struct.pack(
                "<QQQQQ",
                self.offset,
                self.length,
                self.log_time,
                self.create_time,
                self.data_size,
            )
            + _write_string(self.name)
            + _write_string(self.media_type)
        )

    @classmethod
    def read(cls, data: bytes) -> "AttachmentIndex":
        offset, length, log_time, create_time, data_size = struct.unpack_from("<QQQQQ", data, 0)
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

    id: int
    schema_id: int
    topic: str
    message_encoding: str
    metadata: dict[str, str]

    def write(self) -> bytes:
        return (
            struct.pack("<HH", self.id, self.schema_id)
            + _write_string(self.topic)
            + _write_string(self.message_encoding)
            + _write_map(self.metadata)
        )

    @classmethod
    def read(cls, data: bytes) -> "Channel":
        channel_id, schema_id = struct.unpack_from("<HH", data, 0)
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

    message_start_time: int
    message_end_time: int
    uncompressed_size: int
    uncompressed_crc: int
    compression: str
    data: bytes = field(repr=False)

    def write(self) -> bytes:
        return (
            struct.pack(
                "<QQQI",
                self.message_start_time,
                self.message_end_time,
                self.uncompressed_size,
                self.uncompressed_crc,
            )
            + _write_string(self.compression)
            + struct.pack("<Q", len(self.data))
            + self.data
        )

    @classmethod
    def read(cls, data: bytes) -> "Chunk":
        message_start_time, message_end_time, uncompressed_size, uncompressed_crc = (
            struct.unpack_from("<QQQI", data, 0)
        )
        compression, offset = _read_string(data, 28)
        data_len = struct.unpack_from("<Q", data, offset)[0]
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

    message_start_time: int
    message_end_time: int
    chunk_start_offset: int
    chunk_length: int
    message_index_offsets: dict[int, int]
    message_index_length: int
    compression: str
    compressed_size: int
    uncompressed_size: int

    def write(self) -> bytes:
        offsets_data = b"".join(
            struct.pack("<HQ", channel_id, offset)
            for channel_id, offset in self.message_index_offsets.items()
        )
        return (
            struct.pack(
                "<QQQQ",
                self.message_start_time,
                self.message_end_time,
                self.chunk_start_offset,
                self.chunk_length,
            )
            + struct.pack("<I", len(offsets_data))
            + offsets_data
            + struct.pack("<Q", self.message_index_length)
            + _write_string(self.compression)
            + struct.pack("<QQ", self.compressed_size, self.uncompressed_size)
        )

    @classmethod
    def read(cls, data: bytes) -> "ChunkIndex":
        message_start_time, message_end_time, chunk_start_offset, chunk_length = struct.unpack_from(
            "<QQQQ", data, 0
        )
        map_len = struct.unpack_from("<I", data, 32)[0]
        offset = 36
        map_end = offset + map_len
        message_index_offsets: dict[int, int] = {}
        while offset < map_end:
            channel_id, channel_offset = struct.unpack_from("<HQ", data, offset)
            message_index_offsets[channel_id] = channel_offset
            offset += 10
        message_index_length = struct.unpack_from("<Q", data, offset)[0]
        compression, offset = _read_string(data, offset + 8)
        compressed_size, uncompressed_size = struct.unpack_from("<QQ", data, offset)
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

    data_section_crc: int

    def write(self) -> bytes:
        return struct.pack("<I", self.data_section_crc)

    @classmethod
    def read(cls, data: bytes) -> "DataEnd":
        return cls(struct.unpack_from("<I", data, 0)[0])


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

    summary_start: int
    summary_offset_start: int
    summary_crc: int

    def write(self) -> bytes:
        return struct.pack("<QQI", self.summary_start, self.summary_offset_start, self.summary_crc)

    @classmethod
    def read(cls, data: bytes) -> "Footer":
        return cls(*struct.unpack_from("<QQI", data, 0))


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

    def write(self) -> bytes:
        return _write_string(self.profile) + _write_string(self.library)

    @classmethod
    def read(cls, data: bytes) -> "Header":
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

    channel_id: int
    sequence: int
    log_time: int
    publish_time: int
    data: bytes

    def write(self) -> bytes:
        return (
            struct.pack(
                "<HIQQ",
                self.channel_id,
                self.sequence,
                self.log_time,
                self.publish_time,
            )
            + self.data
        )

    @classmethod
    def read(cls, data: bytes) -> "Message":
        channel_id, sequence, log_time, publish_time = struct.unpack_from("<HIQQ", data, 0)
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

    channel_id: int
    records: list[tuple[int, int]]

    def write(self) -> bytes:
        # Pre-allocate buffer for better performance
        num_records = len(self.records)
        records_size = num_records * 16  # 2 * 8 bytes per record
        buffer = bytearray(2 + 4 + records_size)  # channel_id + length + records

        struct.pack_into("<HI", buffer, 0, self.channel_id, records_size)
        offset = 6
        for timestamp, msg_offset in self.records:
            struct.pack_into("<QQ", buffer, offset, timestamp, msg_offset)
            offset += 16

        return bytes(buffer)

    @classmethod
    def read(cls, data: bytes) -> "MessageIndex":
        channel_id = struct.unpack_from("<H", data, 0)[0]
        records_len = struct.unpack_from("<I", data, 2)[0]
        if records_len == 0:
            return cls(channel_id, [])
        records = list(struct.iter_unpack("<QQ", data[6 : 6 + records_len]))
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

    def write(self) -> bytes:
        return _write_string(self.name) + _write_map(self.metadata)

    @classmethod
    def read(cls, data: bytes) -> "Metadata":
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

    offset: int
    length: int
    name: str

    def write(self) -> bytes:
        return struct.pack("<QQ", self.offset, self.length) + _write_string(self.name)

    @classmethod
    def read(cls, data: bytes) -> "MetadataIndex":
        offset, length = struct.unpack_from("<QQ", data, 0)
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

    id: int
    name: str
    encoding: str
    data: bytes

    def write(self) -> bytes:
        return (
            struct.pack("<H", self.id)
            + _write_string(self.name)
            + _write_string(self.encoding)
            + struct.pack("<I", len(self.data))
            + self.data
        )

    @classmethod
    def read(cls, data: bytes) -> "Schema":
        schema_id = struct.unpack_from("<H", data, 0)[0]
        name, offset = _read_string(data, 2)
        encoding, offset = _read_string(data, offset)
        data_len = struct.unpack_from("<I", data, offset)[0]
        return cls(schema_id, name, encoding, data[offset + 4 : offset + 4 + data_len])


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

    message_count: int
    schema_count: int
    channel_count: int
    attachment_count: int
    metadata_count: int
    chunk_count: int
    message_start_time: int
    message_end_time: int
    channel_message_counts: dict[int, int]

    def write(self) -> bytes:
        counts_data = b"".join(
            struct.pack("<HQ", channel_id, count)
            for channel_id, count in self.channel_message_counts.items()
        )
        return (
            struct.pack("<Q", self.message_count)
            + struct.pack("<H", self.schema_count)
            + struct.pack(
                "<IIII",
                self.channel_count,
                self.attachment_count,
                self.metadata_count,
                self.chunk_count,
            )
            + struct.pack("<QQ", self.message_start_time, self.message_end_time)
            + struct.pack("<I", len(counts_data))
            + counts_data
        )

    @classmethod
    def read(cls, data: bytes) -> "Statistics":
        message_count = struct.unpack_from("<Q", data, 0)[0]
        schema_count = struct.unpack_from("<H", data, 8)[0]
        channel_count, attachment_count, metadata_count, chunk_count = struct.unpack_from(
            "<IIII", data, 10
        )
        message_start_time, message_end_time = struct.unpack_from("<QQ", data, 26)
        counts_len = struct.unpack_from("<I", data, 42)[0]
        offset, counts_end = 46, 46 + counts_len
        channel_message_counts = {}
        while offset < counts_end:
            channel_id, count = struct.unpack_from("<HQ", data, offset)
            channel_message_counts[channel_id] = count
            offset += 10
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

    group_opcode: int
    group_start: int
    group_length: int

    def write(self) -> bytes:
        return struct.pack("<BQQ", self.group_opcode, self.group_start, self.group_length)

    @classmethod
    def read(cls, data: bytes) -> "SummaryOffset":
        return cls(*struct.unpack_from("<BQQ", data, 0))


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
