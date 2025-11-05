import io
import struct
import zlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, Flag, auto
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol

from small_mcap.records import (
    MAGIC,
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
    Opcode,
    Schema,
    Statistics,
    SummaryOffset,
)

if TYPE_CHECKING:
    import zstandard
    from lz4.frame import compress as lz4_compress  # type: ignore[import-untyped]
else:
    try:
        import zstandard
    except ImportError:
        zstandard = None  # type: ignore[assignment]

    try:
        from lz4.frame import compress as lz4_compress
    except ImportError:
        lz4_compress = None  # type: ignore[assignment]


# Buffer allocation multiplier for chunk builder
BUFFER_SIZE_MULTIPLIER = 1.25


@dataclass(slots=True)
class PrebuiltChunk:
    """
    Special marker for pre-built chunks that should be written directly
    without going through the chunk builder.
    """

    chunk: Chunk
    indexes: list[MessageIndex]


class CompressionType(Enum):
    NONE = ""
    LZ4 = "lz4"
    ZSTD = "zstd"


class IndexType(Flag):
    """Determines what indexes should be written to the MCAP file. If in doubt, choose ALL."""

    NONE = auto()
    ATTACHMENT = auto()
    CHUNK = auto()
    MESSAGE = auto()
    METADATA = auto()
    ALL = ATTACHMENT | CHUNK | MESSAGE | METADATA


class _CRCWriter:
    """Wraps a BinaryIO and calculates CRC32 of all data written."""

    def __init__(self, f: BinaryIO, enable_crc: bool) -> None:
        self._f = f
        self.enable_crc = enable_crc
        self._crc = 0

    def write(self, data: bytes) -> int:
        if self.enable_crc:
            self._crc = zlib.crc32(data, self._crc)
        return self._f.write(data)

    @property
    def crc(self) -> int:
        return self._crc

    def flush(self) -> None:
        self._f.flush()

    def tell(self) -> int:
        return self._f.tell()

    def close(self) -> None:
        self._f.close()


def _compress_chunk_data(
    data: bytes, compression: CompressionType, min_ratio: float = 0.05
) -> tuple[bytes, str]:
    """
    Compress chunk data and return (compressed_data, compression_type_used).

    Falls back to uncompressed if compression doesn't save at least min_ratio (default 5%).

    :param data: Uncompressed chunk data
    :param compression: Desired compression type
    :param min_ratio: Minimum compression ratio to use compression (default 0.05 = 5%)
    :return: Tuple of (compressed_data, compression_type_string)
    """
    if compression == CompressionType.NONE:
        return data, ""

    # Try compression
    if compression == CompressionType.ZSTD:
        if zstandard is None:
            raise ImportError("zstandard module not available")
        cctx = zstandard.ZstdCompressor()
        compressed = cctx.compress(data)
    elif compression == CompressionType.LZ4:
        if lz4_compress is None:
            raise ImportError("lz4 module not available")
        compressed = lz4_compress(data)
    else:
        raise ValueError(f"Unsupported compression type: {compression}")

    # Check if compression is beneficial
    original_size = len(data)
    compressed_size = len(compressed)
    savings_ratio = (original_size - compressed_size) / original_size

    if savings_ratio < min_ratio:
        # Compression not beneficial, use uncompressed
        return data, ""

    return compressed, compression.value


def _write_summary_section(
    buffer: io.BytesIO,
    offsets: list[SummaryOffset],
    opcode: Opcode,
    items: list[Any],
    summary_start: int,
) -> None:
    """Write a group of records to summary and add corresponding offset."""
    if not items:
        return

    group_start = buffer.tell()
    buffer.writelines(item.write_record() for item in items)

    offsets.append(
        SummaryOffset(
            group_opcode=opcode,
            group_start=summary_start + group_start,
            group_length=buffer.tell() - group_start,
        )
    )


def _calculate_summary_offset_start(
    summary_start: int,
    summary_data: bytes,
    summary_offsets: list[SummaryOffset],
    use_summary_offsets: bool,
) -> int:
    """Calculate the summary offset start position."""
    if not use_summary_offsets or not summary_offsets:
        return 0
    return (
        summary_start
        + len(summary_data)
        - sum(len(offset.write_record()) for offset in summary_offsets)
    )


def _calculate_summary_crc(
    summary_data: bytes,
    summary_start: int,
    summary_offsets: list[SummaryOffset],
    use_summary_offsets: bool,
    enable_crcs: bool,
) -> int:
    """Calculate CRC for summary section including footer fields."""
    if not enable_crcs:
        return 0

    # CRC of summary data
    summary_crc = zlib.crc32(summary_data)

    # Include footer fields in CRC
    summary_offset_start = _calculate_summary_offset_start(
        summary_start, summary_data, summary_offsets, use_summary_offsets
    )
    footer_fields = struct.pack(
        "<BQQQ",
        Opcode.FOOTER,
        8 + 8 + 4,  # Footer record length
        0 if len(summary_data) == 0 else summary_start,
        summary_offset_start,
    )
    return zlib.crc32(footer_fields, summary_crc)


class _ChunkBuilder:
    """Builds chunks by accumulating messages until chunk size limit is reached."""

    def __init__(
        self,
        compression: CompressionType,
        enable_crcs: bool,
        chunk_size: int = 1024 * 1024,
        schemas: dict[int, Schema] | None = None,
        channels: dict[int, Channel] | None = None,
    ) -> None:
        self.compression = compression
        self.enable_crcs = enable_crcs
        self.chunk_size = chunk_size
        # Schema and channel registries (for auto-ensure functionality)
        # Store reference to provided dicts or create empty ones
        self.schemas = schemas if schemas is not None else {}
        self.channels = channels if channels is not None else {}

        # Pre-allocate buffer to avoid reallocations
        self.buffer_data = bytearray(int(BUFFER_SIZE_MULTIPLIER * chunk_size))

        self.reset()

    def reset(self) -> None:
        """Reset builder state for a new chunk."""
        self.buffer_pos = 0
        self.message_start_time = 0
        self.message_end_time = 0
        self.message_indices: dict[int, MessageIndex] = {}
        self.num_messages = 0
        self._written_schemas: set[int] = set()
        self._written_channels: set[int] = set()

    def _ensure_schema_in_chunk(self, schema_id: int) -> None:
        if schema_id == 0 or schema_id in self._written_schemas:
            return

        if schema := self.schemas.get(schema_id):
            self.add(schema)

    def _ensure_channel_in_chunk(self, channel_id: int) -> None:
        if channel_id in self._written_channels:
            return

        if channel := self.channels.get(channel_id):
            self._ensure_schema_in_chunk(channel.schema_id)
            self.add(channel)

    def add(self, record: Schema | Channel | Message) -> None:
        if isinstance(record, Message):
            self._ensure_channel_in_chunk(record.channel_id)

        # Add record to current chunk
        record_data = record.write_record()
        record_len = len(record_data)

        # Ensure buffer has space (grow if needed)
        if self.buffer_pos + record_len > len(self.buffer_data):
            self.buffer_data.extend(bytearray(max(record_len, self.chunk_size)))

        if isinstance(record, Schema):
            self._written_schemas.add(record.id)
        elif isinstance(record, Channel):
            self._written_channels.add(record.id)
        elif isinstance(record, Message):
            if self.num_messages == 0:
                self.message_start_time = record.log_time
            else:
                self.message_start_time = min(self.message_start_time, record.log_time)
            self.message_end_time = max(self.message_end_time, record.log_time)

            if record.channel_id not in self.message_indices:
                self.message_indices[record.channel_id] = MessageIndex(
                    channel_id=record.channel_id, records=[]
                )
            self.message_indices[record.channel_id].records.append(
                (record.log_time, self.buffer_pos)
            )

            self.num_messages += 1
        self.buffer_data[self.buffer_pos : self.buffer_pos + record_len] = record_data
        self.buffer_pos += record_len

    def maybe_finalize(self) -> tuple[Chunk, dict[int, MessageIndex]] | None:
        # Check if we need to finalize current chunk before adding this record
        if self.buffer_pos >= self.chunk_size and self.num_messages > 0:
            result = self.finalize()
            self.reset()
            return result
        return None

    def finalize(self) -> tuple[Chunk, dict[int, MessageIndex]] | None:
        """Build and return the final chunk from current buffer state."""
        if self.num_messages == 0:
            return None

        chunk_data = bytes(self.buffer_data[: self.buffer_pos])

        # Compress data (will fall back to uncompressed if < 5% savings)
        compressed_data, compression_used = _compress_chunk_data(chunk_data, self.compression)

        return Chunk(
            compression=compression_used,
            data=compressed_data,
            message_start_time=self.message_start_time,
            message_end_time=self.message_end_time,
            uncompressed_crc=zlib.crc32(chunk_data) if self.enable_crcs else 0,
            uncompressed_size=len(chunk_data),
        ), self.message_indices.copy()


class EncoderFactoryProtocol(Protocol):
    profile: str
    library: str
    encoding: str  # Schema encoding format
    message_encoding: str  # Message data encoding format

    def encoder_for(self, schema: Schema | None) -> Callable[[Any], bytes] | None: ...


class McapWriter:
    """
    MCAP file writer with simple imperative API.
    Handles chunking, compression, indexing, and CRC generation.
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
        encoder_factory: EncoderFactoryProtocol | None = None,
    ) -> None:
        # Validate compression type
        if not isinstance(compression, CompressionType):
            raise TypeError(
                f"compression must be a CompressionType enum, got {type(compression).__name__}. "
                f"Valid values: CompressionType.NONE, CompressionType.LZ4, CompressionType.ZSTD"
            )

        # Check if required compression libraries are available
        if compression == CompressionType.LZ4 and lz4_compress is None:
            raise ValueError(
                "LZ4 compression requested but lz4 module is not installed. "
                "Install it with: pip install lz4"
            )

        if compression == CompressionType.ZSTD and zstandard is None:
            raise ValueError(
                "ZSTD compression requested but zstandard module is not installed. "
                "Install it with: pip install zstandard"
            )

        # Configuration
        self.output = output
        self.chunk_size = chunk_size
        self.compression = compression
        self.index_types = index_types
        self.repeat_channels = repeat_channels
        self.repeat_schemas = repeat_schemas
        self.use_chunking = use_chunking
        self.use_statistics = use_statistics
        self.use_summary_offsets = use_summary_offsets
        self.enable_crcs = enable_crcs
        self.enable_data_crcs = enable_data_crcs
        self.encoder_factory = encoder_factory

        # Writer state
        self._started = False
        self._finished = False

        # Schemas and channels
        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}

        # Indexes
        self.chunk_indices: list[ChunkIndex] = []
        self.attachment_indexes: list[AttachmentIndex] = []
        self.metadata_indexes: list[MetadataIndex] = []

        # Encoder factory caching (for automatic schema/channel registration)
        self._schema_ids_by_name: dict[str, int] = {}
        self._channel_ids_by_topic: dict[str, int] = {}
        self._next_schema_id = 1
        self._next_channel_id = 1

        # Statistics tracking
        self.statistics = Statistics(
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

        # I/O components
        self.crc_writer = _CRCWriter(output, enable_crcs)
        self.chunk_builder: _ChunkBuilder | None = None
        if use_chunking:
            self.chunk_builder = _ChunkBuilder(
                compression, enable_crcs, chunk_size, self.schemas, self.channels
            )

    def start(self, profile: str = "", library: str = "pymcap-cli 0.1.0") -> None:
        """Start writing the MCAP file."""
        if self._started:
            raise RuntimeError("Writer already started")

        # Use factory profile/library if available
        if self.encoder_factory is not None:
            profile = self.encoder_factory.profile
            library = self.encoder_factory.library

        # Write header
        self.crc_writer.write(MAGIC)
        header_data = Header(profile=profile, library=library).write_record()
        self.crc_writer.write(header_data)

        self._started = True

    def add_schema(self, schema_id: int, name: str, encoding: str, data: bytes) -> None:
        """Add a schema to the file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

        schema = Schema(id=schema_id, name=name, encoding=encoding, data=data)
        self.schemas[schema.id] = schema
        self.statistics.schema_count += 1

        self._write_record(schema)

    def add_channel(
        self,
        channel_id: int,
        topic: str,
        message_encoding: str,
        schema_id: int,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Add a channel to the file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

        # Validate schema_id exists if non-zero
        if schema_id != 0 and schema_id not in self.schemas:
            raise ValueError(
                f"Schema ID {schema_id} does not exist. "
                f"Add the schema using add_schema() before adding this channel."
            )

        channel = Channel(
            id=channel_id,
            schema_id=schema_id,
            topic=topic,
            message_encoding=message_encoding,
            metadata=metadata or {},
        )
        self.channels[channel.id] = channel
        self.statistics.channel_count += 1

        self._write_record(channel)

    def add_message(
        self,
        channel_id: int,
        log_time: int,
        data: bytes,
        publish_time: int,
        sequence: int = 0,
    ) -> None:
        """Add a message to the file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

        # Validate channel_id exists
        if channel_id not in self.channels:
            raise ValueError(
                f"Channel ID {channel_id} does not exist. "
                f"Add the channel using add_channel() before adding messages to it."
            )

        message = Message(
            channel_id=channel_id,
            sequence=sequence,
            log_time=log_time,
            publish_time=publish_time,
            data=data,
        )

        self._write_record(message)

        # Update statistics for non-chunked mode
        # In chunked mode, statistics are updated when chunks are written
        if not self.use_chunking:
            if self.statistics.message_count == 0:
                self.statistics.message_start_time = log_time
            else:
                self.statistics.message_start_time = min(
                    log_time, self.statistics.message_start_time
                )
            self.statistics.message_end_time = max(log_time, self.statistics.message_end_time)
            self.statistics.channel_message_counts[channel_id] += 1
            self.statistics.message_count += 1

    def add_attachment(
        self,
        log_time: int,
        create_time: int,
        name: str,
        media_type: str,
        data: bytes,
    ) -> None:
        """Add an attachment to the file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

        attachment = Attachment(
            log_time=log_time,
            create_time=create_time,
            name=name,
            media_type=media_type,
            data=data,
        )
        offset = self.crc_writer.tell()
        data = attachment.write_record()
        self.crc_writer.write(data)

        self.attachment_indexes.append(
            AttachmentIndex(
                offset=offset,
                length=len(data),
                log_time=log_time,
                create_time=create_time,
                data_size=len(attachment.data),
                name=name,
                media_type=media_type,
            )
        )

        self.statistics.attachment_count += 1

    def add_metadata(self, name: str, metadata: dict[str, str]) -> None:
        """Add a metadata record to the file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

        record = Metadata(name=name, metadata=metadata)

        offset = self.crc_writer.tell()
        data = record.write_record()
        self.crc_writer.write(data)

        self.metadata_indexes.append(MetadataIndex(offset=offset, length=len(data), name=name))

        self.statistics.metadata_count += 1

    def register_schema(self, name: str, data: bytes) -> int:
        """
        Register a schema by name and return its ID. If already registered, returns existing ID.
        Uses encoder_factory.encoding if available, otherwise defaults to empty string.

        :param name: Schema name (e.g., message type)
        :param data: Schema definition data
        :return: Schema ID
        """
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

        # Check cache
        if name in self._schema_ids_by_name:
            return self._schema_ids_by_name[name]

        # Create new schema
        schema_id = self._next_schema_id
        self._next_schema_id += 1

        encoding = self.encoder_factory.encoding if self.encoder_factory else ""
        self.add_schema(schema_id, name, encoding, data)
        self._schema_ids_by_name[name] = schema_id

        return schema_id

    def add_message_object(
        self,
        topic: str,
        schema_name: str,
        schema_data: bytes,
        message_obj: Any,
        log_time: int,
        publish_time: int,
        sequence: int = 0,
    ) -> None:
        """
        Add a message object, automatically registering schema and channel as needed.
        Requires encoder_factory to be set.

        :param topic: The topic of the message
        :param schema_name: The schema name (e.g., message type)
        :param schema_data: The schema definition data
        :param message_obj: The message object to encode
        :param log_time: The time at which the message was logged (nanosecond UNIX timestamp)
        :param publish_time: The time at which the message was published (nanosecond UNIX timestamp)
        :param sequence: Optional sequence number
        """
        if self.encoder_factory is None:
            raise RuntimeError("encoder_factory must be set to use add_message_object()")

        # Register schema if needed
        schema_id = self.register_schema(schema_name, schema_data)

        # Register channel if needed
        if topic not in self._channel_ids_by_topic:
            channel_id = self._next_channel_id
            self._next_channel_id += 1
            self.add_channel(
                channel_id=channel_id,
                topic=topic,
                message_encoding=self.encoder_factory.message_encoding,
                schema_id=schema_id,
            )
            self._channel_ids_by_topic[topic] = channel_id

        channel_id = self._channel_ids_by_topic[topic]

        # Get schema and encoder
        schema = self.schemas[schema_id]
        encoder = self.encoder_factory.encoder_for(schema)
        if encoder is None:
            raise RuntimeError(f"encoder_factory returned None for schema {schema_name}")

        # Encode message
        data = encoder(message_obj)

        # Add encoded message
        self.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=publish_time,
            data=data,
            sequence=sequence,
        )

    def add_chunk_with_indexes(self, chunk: Chunk, indexes: list[MessageIndex]) -> None:
        """
        Add a pre-built chunk with its indexes directly to the file.
        This is used for fast chunk copying without decompression.
        """
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

        # Finalize any in-progress chunk first
        self._finalize_current_chunk()

        # Write the pre-built chunk
        self._write_chunk(chunk, {idx.channel_id: idx for idx in indexes})

    def _finalize_current_chunk(self) -> None:
        """Finalize and write any remaining chunk."""
        if self.chunk_builder is not None and (result := self.chunk_builder.finalize()):
            chunk, message_indices = result
            self._write_chunk(chunk, message_indices)
            self.chunk_builder.reset()

    def finish(self) -> None:
        """Finish writing the MCAP file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            return

        # Finalize any remaining chunk
        self._finalize_current_chunk()

        # Write DataEnd record
        data_end = DataEnd(data_section_crc=self.crc_writer.crc).write_record()
        self.crc_writer.enable_crc = False  # No need to include DataEnd in CRC
        self.crc_writer.write(data_end)

        # Build and write summary section
        summary_start = self.crc_writer.tell()
        summary_data, summary_offsets = self._build_summary(summary_start)
        summary_crc = _calculate_summary_crc(
            summary_data, summary_start, summary_offsets, self.use_summary_offsets, self.enable_crcs
        )
        self.crc_writer.write(summary_data)

        # Write footer
        summary_offset_start = _calculate_summary_offset_start(
            summary_start, summary_data, summary_offsets, self.use_summary_offsets
        )
        footer = Footer(
            summary_start=0 if len(summary_data) == 0 else summary_start,
            summary_offset_start=summary_offset_start,
            summary_crc=summary_crc,
        )
        self.crc_writer.write(footer.write_record())

        # Write closing magic
        self.crc_writer.write(MAGIC)

        self._finished = True

    def _write_record(self, record: Schema | Channel | Message) -> None:
        """Write a record to the output, either directly or via chunk builder."""
        if not self.use_chunking:
            self.crc_writer.write(record.write_record())
            return

        if self.chunk_builder is None:
            return

        self.chunk_builder.add(record)
        # Add to chunk builder
        if result := self.chunk_builder.maybe_finalize():
            # Chunk was completed, write it
            chunk, message_indices = result
            self._write_chunk(chunk, message_indices)

    def _write_chunk(self, chunk: Chunk, message_indices: dict[int, MessageIndex]) -> None:
        """Write a chunk and its indexes to output."""
        # Update statistics for messages in chunk
        if self.statistics.message_count == 0:
            self.statistics.message_start_time = chunk.message_start_time
        else:
            self.statistics.message_start_time = min(
                chunk.message_start_time, self.statistics.message_start_time
            )
        self.statistics.message_end_time = max(
            chunk.message_end_time, self.statistics.message_end_time
        )

        for idx in message_indices.values():
            self.statistics.channel_message_counts[idx.channel_id] += len(idx.records)
            self.statistics.message_count += len(idx.records)

        self.statistics.chunk_count += 1

        # Write chunk
        chunk_start_offset = self.crc_writer.tell()
        chunk_data = chunk.write_record()
        self.crc_writer.write(chunk_data)

        # Write message indexes
        message_index_start_offset = self.crc_writer.tell()
        message_index_offsets: dict[int, int] = {}
        index_buffer = io.BytesIO()

        if self.index_types & IndexType.MESSAGE:
            for idx in message_indices.values():
                message_index_offsets[idx.channel_id] = (
                    message_index_start_offset + index_buffer.tell()
                )
                index_buffer.write(idx.write_record())

        index_data = index_buffer.getvalue()
        self.crc_writer.write(index_data)

        # Store chunk index
        self.chunk_indices.append(
            ChunkIndex(
                message_start_time=chunk.message_start_time,
                message_end_time=chunk.message_end_time,
                chunk_start_offset=chunk_start_offset,
                chunk_length=len(chunk_data),
                message_index_offsets=message_index_offsets,
                message_index_length=len(index_data),
                compression=chunk.compression,
                compressed_size=len(chunk.data),
                uncompressed_size=chunk.uncompressed_size,
            )
        )

    def _build_summary(self, summary_start: int) -> tuple[bytes, list[SummaryOffset]]:
        """Build summary section and return (summary_data, summary_offsets)."""
        summary_buffer = io.BytesIO()
        summary_offsets: list[SummaryOffset] = []

        # Define sections to write (opcode, items, should_write)
        sections: list[tuple[Opcode, list[Any], bool]] = [
            (Opcode.SCHEMA, list(self.schemas.values()), self.repeat_schemas),
            (Opcode.CHANNEL, list(self.channels.values()), self.repeat_channels),
            (Opcode.STATISTICS, [self.statistics], self.use_statistics),
            (Opcode.CHUNK_INDEX, self.chunk_indices, bool(self.index_types & IndexType.CHUNK)),
            (
                Opcode.ATTACHMENT_INDEX,
                self.attachment_indexes,
                bool(self.index_types & IndexType.ATTACHMENT),
            ),
            (
                Opcode.METADATA_INDEX,
                self.metadata_indexes,
                bool(self.index_types & IndexType.METADATA),
            ),
        ]

        # Write all sections using data-driven approach
        for opcode, items, should_write in sections:
            if should_write and items:
                _write_summary_section(
                    summary_buffer, summary_offsets, opcode, items, summary_start
                )

        # Write summary offsets at the end
        if self.use_summary_offsets:
            for offset in summary_offsets:
                summary_buffer.write(offset.write_record())

        return summary_buffer.getvalue(), summary_offsets
