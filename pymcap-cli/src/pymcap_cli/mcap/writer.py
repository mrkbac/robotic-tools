"""Custom MCAP writer optimized for pymcap-cli operations."""

import struct
import zlib
from collections import OrderedDict, defaultdict
from enum import Enum, Flag, auto
from typing import BinaryIO

from mcap.data_stream import RecordBuilder
from mcap.exceptions import UnsupportedCompressionError
from mcap.opcode import Opcode
from mcap.records import (
    Attachment,
    AttachmentIndex,
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Footer,
    Header,
    Message,
    MessageIndex,
    Metadata,
    MetadataIndex,
    Schema,
    Statistics,
    SummaryOffset,
)

try:
    from lz4.frame import compress as lz4_compress
except ImportError:
    lz4_compress = None


try:
    from zstandard import compress as zstd_compress
except ImportError:
    zstd_compress = None  # ty: ignore


class ChunkBuilder:
    def __init__(self) -> None:
        self.message_end_time = 0
        self.message_indices: dict[int, MessageIndex] = {}
        self.message_start_time = 0
        self.record_writer = RecordBuilder()
        self.num_messages = 0

    @property
    def count(self) -> int:
        return self.record_writer.count

    def end(self) -> bytes:
        return self.record_writer.end()

    def add_channel(self, channel: Channel) -> None:
        channel.write(self.record_writer)

    def add_schema(self, schema: Schema) -> None:
        schema.write(self.record_writer)

    def add_message(self, message: Message) -> None:
        if self.num_messages == 0:
            self.message_start_time = message.log_time
        else:
            self.message_start_time = min(self.message_start_time, message.log_time)
        self.message_end_time = max(self.message_end_time, message.log_time)

        if not self.message_indices.get(message.channel_id):
            self.message_indices[message.channel_id] = MessageIndex(
                channel_id=message.channel_id, records=[]
            )
        self.message_indices[message.channel_id].records.append(
            (message.log_time, self.record_writer.count)
        )

        self.num_messages += 1
        message.write(self.record_writer)

    def reset(self) -> None:
        self.message_end_time = 0
        self.message_indices.clear()
        self.message_start_time = 0
        self.record_writer.end()
        self.num_messages = 0


MCAP0_MAGIC = struct.pack("<8B", 137, 77, 67, 65, 80, 48, 13, 10)
LIBRARY_IDENTIFIER = "pymcap-cli"


class CompressionType(Enum):
    NONE = auto()
    LZ4 = auto()
    ZSTD = auto()


class IndexType(Flag):
    """Determines what indexes should be written to the MCAP file. If in doubt, choose ALL."""

    NONE = auto()
    ATTACHMENT = auto()
    CHUNK = auto()
    MESSAGE = auto()
    METADATA = auto()
    ALL = ATTACHMENT | CHUNK | MESSAGE | METADATA


class McapWriter:
    """
    Writes MCAP data.

    :param output: A filename or stream to write to.
    :param chunk_size: The maximum size of individual data chunks in a chunked file.
    :param compression: Compression to apply to chunk data, if any.
    :param index_types: Indexes to write to the file. See IndexType for possibilities.
    :param repeat_channels: Repeat channel information at the end of the file.
    :param repeat_schemas: Repeat schemas at the end of the file.
    :param use_chunking: Group data in chunks.
    :param use_statistics: Write statistics record.
    :param use_summary_offsets: Write summary offset records.
    """

    def __init__(
        self,
        output: BinaryIO,
        chunk_size: int = 1024 * 1024,
        compression: CompressionType = CompressionType.ZSTD,
        index_types: IndexType = IndexType.ALL,
        repeat_channels: bool = True,
        repeat_schemas: bool = True,
        use_chunking: bool = True,
        use_statistics: bool = True,
        use_summary_offsets: bool = True,
        enable_crcs: bool = True,
        enable_data_crcs: bool = False,
    ) -> None:
        self.__should_close = False
        self.__stream = output
        self.__record_builder = RecordBuilder()
        self.__attachment_indexes: list[AttachmentIndex] = []
        self.__metadata_indexes: list[MetadataIndex] = []
        self.__channels: OrderedDict[int, Channel] = OrderedDict()
        self.__chunk_builder = ChunkBuilder() if use_chunking else None
        self.__chunk_indices: list[ChunkIndex] = []
        self.__chunk_size = chunk_size
        self.__compression = compression
        self.__index_types = index_types
        self.__repeat_channels = repeat_channels
        self.__repeat_schemas = repeat_schemas
        self.__schemas: OrderedDict[int, Schema] = OrderedDict()
        self.__statistics = Statistics(
            attachment_count=0,
            channel_count=0,
            channel_message_counts=defaultdict(int),
            chunk_count=0,
            message_count=0,
            metadata_count=0,
            message_start_time=0,
            message_end_time=0,
            schema_count=0,
        )
        self.__summary_offsets: list[SummaryOffset] = []
        self.__use_statistics = use_statistics
        self.__use_summary_offsets = use_summary_offsets
        self.__enable_crcs = enable_crcs
        self.__enable_data_crcs = enable_data_crcs
        self.__data_section_crc = 0

        # validate compression
        if self.__compression == CompressionType.LZ4:
            if lz4_compress is None:
                raise UnsupportedCompressionError("lz4")
        elif self.__compression == CompressionType.ZSTD and zstd_compress is None:
            raise UnsupportedCompressionError("zstandard")

    def add_attachment(
        self, create_time: int, log_time: int, name: str, media_type: str, data: bytes
    ) -> None:
        """
        Adds an attachment to the file.

        :param log_time: Time at which the attachment was recorded.
        :param create_time: Time at which the attachment was created. If not available,
            must be set to zero.
        :param name: Name of the attachment, e.g "scene1.jpg".
        :param media_type: Media Type (e.g "text/plain").
        :param data: Attachment data.
        """
        self.__flush()
        offset = self.__stream.tell()
        self.__statistics.attachment_count += 1
        attachment = Attachment(
            create_time=create_time,
            log_time=log_time,
            name=name,
            media_type=media_type,
            data=data,
        )
        attachment.write(self.__record_builder)
        if self.__index_types & IndexType.ATTACHMENT:
            index = AttachmentIndex(
                offset=offset,
                length=self.__record_builder.count,
                create_time=attachment.create_time,
                log_time=attachment.log_time,
                data_size=len(attachment.data),
                name=attachment.name,
                media_type=attachment.media_type,
            )
            self.__attachment_indexes.append(index)
        self.__flush()

    def add_message(
        self,
        channel_id: int,
        log_time: int,
        data: bytes,
        publish_time: int,
        sequence: int = 0,
    ) -> None:
        """
        Adds a new message to the file. If chunking is enabled the message will be added to the
        current chunk.

        :param channel_id: The id of the channel to which the message should be added.
        :param sequence: Optional message counter assigned by publisher.
        :param log_time: Time at which the message was recorded as nanoseconds since a
            user-understood epoch (i.e unix epoch, robot boot time, etc.).
        :param publish_time: Time at which the message was published as nanoseconds since a
            user-understood epoch (i.e unix epoch, robot boot time, etc.).
        :param data: Message data, to be decoded according to the schema of the channel.
        """
        message = Message(
            channel_id=channel_id,
            log_time=log_time,
            data=data,
            publish_time=publish_time,
            sequence=sequence,
        )
        if self.__statistics.message_count == 0:
            self.__statistics.message_start_time = log_time
        else:
            self.__statistics.message_start_time = min(
                log_time, self.__statistics.message_start_time
            )
        self.__statistics.message_end_time = max(log_time, self.__statistics.message_end_time)
        self.__statistics.channel_message_counts[message.channel_id] += 1
        self.__statistics.message_count += 1
        if self.__chunk_builder:
            self.__chunk_builder.add_message(message)
            self.__maybe_finalize_chunk()
        else:
            message.write(self.__record_builder)
            self.__flush()

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        """
        Adds key-value metadata to the file.

        :param name: A name to associate with the metadata.
        :param data: Key-value metadata.
        """
        self.__flush()
        offset = self.__stream.tell()
        self.__statistics.metadata_count += 1
        metadata = Metadata(name=name, metadata=data)
        metadata.write(self.__record_builder)
        if self.__index_types & IndexType.METADATA:
            index = MetadataIndex(offset=offset, length=self.__record_builder.count, name=name)
            self.__metadata_indexes.append(index)
        self.__flush()

    def finish(self) -> None:
        """
        Writes any final indexes, summaries etc to the file. Note that it does
        not close the underlying output stream.
        """
        self.__finalize_chunk()

        DataEnd(self.__data_section_crc).write(self.__record_builder)
        self.__flush()

        summary_start = self.__stream.tell()
        summary_builder = RecordBuilder()

        if self.__repeat_schemas:
            group_start = summary_builder.count
            for schema in self.__schemas.values():
                schema.write(summary_builder)
            self.__summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.SCHEMA,
                    group_start=summary_start + group_start,
                    group_length=summary_builder.count - group_start,
                )
            )

        if self.__repeat_channels:
            group_start = summary_builder.count
            for channel in self.__channels.values():
                channel.write(summary_builder)
            self.__summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.CHANNEL,
                    group_start=summary_start + group_start,
                    group_length=summary_builder.count - group_start,
                )
            )

        if self.__use_statistics:
            group_start = summary_builder.count
            self.__statistics.write(summary_builder)
            self.__summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.STATISTICS,
                    group_start=summary_start + group_start,
                    group_length=summary_builder.count - group_start,
                )
            )

        if self.__index_types & IndexType.CHUNK:
            group_start = summary_builder.count
            for index in self.__chunk_indices:
                index.write(summary_builder)
            self.__summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.CHUNK_INDEX,
                    group_start=summary_start + group_start,
                    group_length=summary_builder.count - group_start,
                )
            )

        if self.__index_types & IndexType.ATTACHMENT:
            group_start = summary_builder.count
            for index in self.__attachment_indexes:
                index.write(summary_builder)
            self.__summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.ATTACHMENT_INDEX,
                    group_start=summary_start + group_start,
                    group_length=summary_builder.count - group_start,
                )
            )

        if self.__index_types & IndexType.METADATA:
            group_start = summary_builder.count
            for index in self.__metadata_indexes:
                index.write(summary_builder)
            self.__summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.METADATA_INDEX,
                    group_start=summary_start + group_start,
                    group_length=summary_builder.count - group_start,
                )
            )

        summary_offset_start = (
            summary_start + summary_builder.count if self.__use_summary_offsets else 0
        )
        if self.__use_summary_offsets:
            for offset in self.__summary_offsets:
                offset.write(summary_builder)

        summary_data = summary_builder.end()
        summary_length = len(summary_data)

        summary_crc = 0
        if self.__enable_crcs:
            summary_crc = zlib.crc32(summary_data)
            summary_crc = zlib.crc32(
                struct.pack(
                    "<BQQQ",  # cspell:disable-line
                    Opcode.FOOTER,
                    8 + 8 + 4,
                    0 if summary_length == 0 else summary_start,
                    summary_offset_start,
                ),
                summary_crc,
            )

        self.__stream.write(summary_data)

        Footer(
            summary_start=0 if summary_length == 0 else summary_start,
            summary_offset_start=summary_offset_start,
            summary_crc=summary_crc,
        ).write(self.__record_builder)

        self.__flush()
        self.__stream.write(MCAP0_MAGIC)
        if self.__should_close:
            self.__stream.close()

    def add_channel(
        self,
        channel_id: int,
        topic: str,
        message_encoding: str,
        schema_id: int,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """
        Registers a new message channel. Returns the numeric id of the new channel.

        :param schema_id: The schema for messages on this channel. A schema_id of 0 indicates there
            is no schema for this channel.
        :param topic: The channel topic.
        :param message_encoding: Encoding for messages on this channel. See the list of well-known
            message encodings for common values.
        :param metadata: Metadata about this channel.
        """
        channel = Channel(
            id=channel_id,
            topic=topic,
            message_encoding=message_encoding,
            schema_id=schema_id,
            metadata=metadata or {},
        )
        self.__channels[channel_id] = channel
        self.__statistics.channel_count += 1
        if self.__chunk_builder:
            self.__chunk_builder.add_channel(channel)
            self.__maybe_finalize_chunk()
        else:
            channel.write(self.__record_builder)

    def add_schema(
        self,
        schema_id: int,
        name: str,
        encoding: str,
        data: bytes,
    ) -> None:
        """
        Registers a new message schema. Returns the new integer schema id.

        :param name: An identifier for the schema.
        :param encoding: Format for the schema. See the list of well-known schema encodings for
            common values. An empty string indicates no schema is available.
        :param data: Schema data. Must conform to the schema encoding. If `encoding` is an empty
            string, `data` should be 0 length.
        """
        schema = Schema(id=schema_id, data=data, encoding=encoding, name=name)
        self.__schemas[schema_id] = schema
        self.__statistics.schema_count += 1
        if self.__chunk_builder:
            self.__chunk_builder.add_schema(schema)
            self.__maybe_finalize_chunk()
        else:
            schema.write(self.__record_builder)

    def start(self, profile: str = "", library: str = LIBRARY_IDENTIFIER) -> None:
        """
        Starts writing to the output stream.

        :param profile: The profile is used for indicating requirements for fields
            throughout the file (encoding, user_data, etc).
        :param library: Free-form string for writer to specify its name, version, or other
            information for use in debugging.
        """
        self.__stream.write(MCAP0_MAGIC)
        if self.__enable_data_crcs:
            self.__data_section_crc = zlib.crc32(MCAP0_MAGIC, self.__data_section_crc)
        Header(profile, library).write(self.__record_builder)
        self.__flush()

    def __flush(self) -> None:
        data = self.__record_builder.end()
        if self.__enable_data_crcs:
            self.__data_section_crc = zlib.crc32(data, self.__data_section_crc)
        self.__stream.write(data)

    def add_chunk_with_indexes(self, chunk: Chunk, indexes: list[MessageIndex]) -> None:
        self.__finalize_chunk()  # finish any in-progress chunks
        self._add_chunk_with_indexes_raw(chunk, indexes)

        if self.__statistics.message_count == 0:
            self.__statistics.message_start_time = chunk.message_start_time
        else:
            self.__statistics.message_start_time = min(
                chunk.message_start_time, self.__statistics.message_start_time
            )
        self.__statistics.message_end_time = max(
            chunk.message_end_time, self.__statistics.message_end_time
        )

        for idx in indexes:
            self.__statistics.channel_message_counts[idx.channel_id] += len(idx.records)
            self.__statistics.message_count += len(idx.records)

    def _add_chunk_with_indexes_raw(self, chunk: Chunk, indexes: list[MessageIndex]) -> None:
        self.__statistics.chunk_count += 1

        self.__flush()
        chunk_start_offset = self.__stream.tell()
        chunk.write(self.__record_builder)
        chunk_size = self.__record_builder.count

        chunk_index = ChunkIndex(
            message_start_time=chunk.message_start_time,
            message_end_time=chunk.message_end_time,
            chunk_start_offset=chunk_start_offset,
            chunk_length=chunk_size,
            message_index_offsets={},
            message_index_length=0,
            compression=chunk.compression,
            compressed_size=len(chunk.data),
            uncompressed_size=chunk.uncompressed_size,
        )

        self.__flush()
        message_index_start_offset = self.__stream.tell()

        if self.__index_types & IndexType.MESSAGE:
            for index in indexes:
                chunk_index.message_index_offsets[index.channel_id] = (
                    message_index_start_offset + self.__record_builder.count
                )
                index.write(self.__record_builder)

        chunk_index.message_index_length = self.__record_builder.count

        self.__flush()

        self.__chunk_indices.append(chunk_index)

    def __finalize_chunk(self) -> None:
        if not self.__chunk_builder:
            return

        if self.__chunk_builder.num_messages == 0:
            return

        chunk_data = self.__chunk_builder.end()
        if self.__compression == CompressionType.LZ4:
            compression = "lz4"
            compressed_data: bytes = lz4_compress(chunk_data)  # type: ignore[reportOptionalCall]
        elif self.__compression == CompressionType.ZSTD:
            compression = "zstd"
            compressed_data: bytes = zstd_compress(chunk_data)  # type: ignore[reportOptionalCall]
        else:
            compression = ""
            compressed_data: bytes = chunk_data
        chunk = Chunk(
            compression=compression,
            data=compressed_data,
            message_start_time=self.__chunk_builder.message_start_time,
            message_end_time=self.__chunk_builder.message_end_time,
            uncompressed_crc=zlib.crc32(chunk_data) if self.__enable_crcs else 0,
            uncompressed_size=len(chunk_data),
        )

        self._add_chunk_with_indexes_raw(chunk, list(self.__chunk_builder.message_indices.values()))
        self.__chunk_builder.reset()

    def __maybe_finalize_chunk(self) -> None:
        if self.__chunk_builder and self.__chunk_builder.count > self.__chunk_size:
            self.__finalize_chunk()
