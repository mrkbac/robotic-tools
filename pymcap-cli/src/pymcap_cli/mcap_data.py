import io
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import IO, ClassVar, Final, TypeVar, cast


@unique
class Opcode(IntEnum):
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
    """Read a length-prefixed string from bytes. Returns (string, new_offset)."""
    string_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    string_val = data[offset : offset + string_len].decode("utf-8")
    offset += string_len
    return string_val, offset


def _read_map(data: bytes, offset: int) -> tuple[dict[str, str], int]:
    """Read a length-prefixed map from bytes. Returns (dict, new_offset)."""
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
    """Write a length-prefixed string to bytes."""
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _write_map(value: dict[str, str]) -> bytes:
    """Write a length-prefixed map to bytes."""
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
    OPCODE: ClassVar[int]

    @abstractmethod
    def write(self) -> bytes:
        """Write the record content (without opcode and length prefix)."""
        ...

    def write_record(self) -> bytes:
        """Write the complete record with opcode and length prefix."""
        content = self.write()
        return struct.pack("<BQ", self.OPCODE, len(content)) + content

    @classmethod
    @abstractmethod
    def read(cls, data: bytes) -> "McapRecord":
        """Read the record content (without opcode and length prefix)."""
        ...

    @classmethod
    def read_record(cls: type[T], data: IO[bytes] | io.BufferedIOBase) -> T:  # noqa: PYI019
        """Read a complete record with opcode and length prefix."""
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
                "<QQQQQ", self.offset, self.length, self.log_time, self.create_time, self.data_size
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
    OPCODE: ClassVar[int] = Opcode.DATA_END

    data_section_crc: int

    def write(self) -> bytes:
        return struct.pack("<I", self.data_section_crc)

    @classmethod
    def read(cls, data: bytes) -> "DataEnd":
        return cls(struct.unpack("<I", data)[0])


@dataclass(slots=True)
class Footer(McapRecord):
    OPCODE: ClassVar[int] = Opcode.FOOTER

    summary_start: int
    summary_offset_start: int
    summary_crc: int

    def write(self) -> bytes:
        return struct.pack("<QQI", self.summary_start, self.summary_offset_start, self.summary_crc)

    @classmethod
    def read(cls, data: bytes) -> "Footer":
        return cls(*struct.unpack("<QQI", data))


@dataclass(slots=True)
class Header(McapRecord):
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
    OPCODE: ClassVar[int] = Opcode.MESSAGE

    channel_id: int
    sequence: int
    log_time: int
    publish_time: int
    data: bytes

    def write(self) -> bytes:
        return (
            struct.pack("<HIQQ", self.channel_id, self.sequence, self.log_time, self.publish_time)
            + self.data
        )

    @classmethod
    def read(cls, data: bytes) -> "Message":
        channel_id, sequence, log_time, publish_time = struct.unpack_from("<HIQQ", data, 0)
        return cls(channel_id, sequence, log_time, publish_time, data[22:])


@dataclass(slots=True)
class MessageIndex(McapRecord):
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
        num_records = records_len // 16
        # Optimization: Use single struct.unpack call instead of list comprehension
        if num_records == 0:
            return cls(channel_id, [])
        # Unpack all records at once: each record is 2 Q values (8 bytes each = 16 bytes)
        format_str = f"<{num_records * 2}Q"
        unpacked = struct.unpack_from(format_str, data, 6)
        # Convert flat list to list of tuples
        records = [(unpacked[i * 2], unpacked[i * 2 + 1]) for i in range(num_records)]
        return cls(channel_id, records)


@dataclass(slots=True)
class Metadata(McapRecord):
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
    OPCODE: ClassVar[int] = Opcode.SUMMARY_OFFSET

    group_opcode: int
    group_start: int
    group_length: int

    def write(self) -> bytes:
        return struct.pack("<BQQ", self.group_opcode, self.group_start, self.group_length)

    @classmethod
    def read(cls, data: bytes) -> "SummaryOffset":
        return cls(*struct.unpack("<BQQ", data))


@dataclass(slots=True)
class Summary:
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
