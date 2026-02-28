import io
import struct
import threading
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
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
    McapRecord,
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
    from lz4.frame import compress as lz4_compress
else:
    try:
        import zstandard
    except ImportError:
        zstandard = None

    try:
        from lz4.frame import compress as lz4_compress
    except ImportError:
        lz4_compress = None

# Module-level constants for hot-path message serialization (avoid per-call attribute lookups)
_MSG_RECORD_STRUCT = struct.Struct(
    "<BQHIQQ"
)  # opcode + length + channel_id + sequence + log_time + publish_time
_MSG_HEADER_SIZE = _MSG_RECORD_STRUCT.size  # 31
_MSG_OPCODE = Opcode.MESSAGE
_MSG_FIXED_LEN = 22  # channel_id(2) + sequence(4) + log_time(8) + publish_time(8)


class _ThreadLocal(threading.local):
    zstd_compressor: "zstandard.ZstdCompressor | None" = None


_thread_local = _ThreadLocal()

# Fixed size of SummaryOffset record (opcode 1 + length 8 + content 17 = 26 bytes)
SUMMARY_OFFSET_RECORD_SIZE = 26


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

    def write(self, data: bytes | bytearray | memoryview, /) -> int:
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


def _write_summary_section(
    buffer: io.BytesIO,
    offsets: list[SummaryOffset],
    opcode: Opcode,
    items: Iterable[McapRecord],
    summary_start: int,
) -> None:
    """Write a group of records to summary and add corresponding offset."""
    if not items:
        return

    group_start = buffer.tell()
    for item in items:
        item.write_record_to(buffer)

    offsets.append(
        SummaryOffset(
            group_opcode=opcode,
            group_start=summary_start + group_start,
            group_length=buffer.tell() - group_start,
        )
    )


def _calculate_summary_offset_start(
    summary_start: int,
    summary_data: bytes | memoryview,
    summary_offsets: list[SummaryOffset],
    use_summary_offsets: bool,
) -> int:
    """Calculate the summary offset start position."""
    if not use_summary_offsets or not summary_offsets:
        return 0
    return summary_start + len(summary_data) - len(summary_offsets) * SUMMARY_OFFSET_RECORD_SIZE


def _calculate_summary_crc(
    summary_data: bytes | memoryview,
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
        summary_start if summary_data else 0,
        summary_offset_start,
    )
    return zlib.crc32(footer_fields, summary_crc)


class McapWriterRaw:
    def __init__(
        self,
        output: BinaryIO,
        index_types: IndexType = IndexType.ALL,
        repeat_channels: bool = True,
        repeat_schemas: bool = True,
        use_statistics: bool = True,
        use_summary_offsets: bool = True,
        enable_crcs: bool = True,
    ) -> None:
        # Configuration
        self.output = output
        self.index_types = index_types
        self.repeat_channels = repeat_channels
        self.repeat_schemas = repeat_schemas
        self.use_statistics = use_statistics
        self.use_summary_offsets = use_summary_offsets
        self.enable_crcs = enable_crcs

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

    def start(self, profile: str = "", library: str = "small-mcap") -> None:
        """Start writing the MCAP file."""
        if self._started:
            raise RuntimeError("Writer already started")

        # Write header
        self.crc_writer.write(MAGIC)
        Header(profile=profile, library=library).write_record_to(self.crc_writer)

        self._started = True

    def add_schema(self, schema_id: int, name: str, encoding: str, data: bytes) -> None:
        """Add a schema to the file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")
        if schema_id == 0:
            raise ValueError("Schema ID cannot be 0")

        schema = Schema(id=schema_id, name=name, encoding=encoding, data=data)
        self.schemas[schema.id] = schema

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

        self._write_record(channel)

    def add_message(
        self,
        channel_id: int,
        log_time: int,
        data: bytes | memoryview,
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

        # Update statistics (needed for unchunked mode where add_chunk is never called)
        if self.statistics.message_count == 0:
            self.statistics.message_start_time = log_time
        else:
            self.statistics.message_start_time = min(log_time, self.statistics.message_start_time)
        self.statistics.message_end_time = max(log_time, self.statistics.message_end_time)
        self.statistics.message_count += 1
        self.statistics.channel_message_counts[channel_id] += 1

    def add_attachment(
        self,
        log_time: int,
        create_time: int,
        name: str,
        media_type: str,
        data: bytes | memoryview,
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
        record_len = attachment.write_record_to(self.crc_writer)

        self.attachment_indexes.append(
            AttachmentIndex(
                offset=offset,
                length=record_len,
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
        record_len = record.write_record_to(self.crc_writer)

        self.metadata_indexes.append(MetadataIndex(offset=offset, length=record_len, name=name))

        self.statistics.metadata_count += 1

    def add_chunk(self, chunk: Chunk, message_indices: dict[int, MessageIndex]) -> None:
        """Write a chunk and its indexes to output."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

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
            self.statistics.channel_message_counts[idx.channel_id] += len(idx.timestamps)
            self.statistics.message_count += len(idx.timestamps)

        self.statistics.chunk_count += 1

        # Write chunk
        chunk_start_offset = self.crc_writer.tell()
        chunk_len = chunk.write_record_to(self.crc_writer)

        # Write message indexes directly to crc_writer
        message_index_offsets: dict[int, int] = {}
        message_index_start_offset = self.crc_writer.tell()

        if self.index_types & IndexType.MESSAGE:
            for idx in message_indices.values():
                message_index_offsets[idx.channel_id] = self.crc_writer.tell()
                idx.write_record_to(self.crc_writer)

        message_index_length = self.crc_writer.tell() - message_index_start_offset

        # Store chunk index
        self.chunk_indices.append(
            ChunkIndex(
                message_start_time=chunk.message_start_time,
                message_end_time=chunk.message_end_time,
                chunk_start_offset=chunk_start_offset,
                chunk_length=chunk_len,
                message_index_offsets=message_index_offsets,
                message_index_length=message_index_length,
                compression=chunk.compression,
                compressed_size=len(chunk.data),
                uncompressed_size=chunk.uncompressed_size,
            )
        )

    def finish(self) -> None:
        """Finish writing the MCAP file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            return

        # Write DataEnd record
        self.crc_writer.enable_crc = False  # No need to include DataEnd in CRC
        DataEnd(data_section_crc=self.crc_writer.crc).write_record_to(self.crc_writer)

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
            summary_start=summary_start if summary_data else 0,
            summary_offset_start=summary_offset_start,
            summary_crc=summary_crc,
        )
        footer.write_record_to(self.crc_writer)

        # Write closing magic
        self.crc_writer.write(MAGIC)

        self._finished = True

    def _write_record(self, record: McapRecord) -> None:
        """Write a record to the output, either directly or via chunk builder."""
        record.write_record_to(self.crc_writer)

    def _build_summary(self, summary_start: int) -> tuple[bytes, list[SummaryOffset]]:
        """Build summary section and return (summary_data, summary_offsets)."""
        summary_buffer = io.BytesIO()
        summary_offsets: list[SummaryOffset] = []

        self.statistics.schema_count = len(self.schemas)
        self.statistics.channel_count = len(self.channels)

        # Define sections to write (opcode, items, should_write)
        sections: list[tuple[Opcode, Iterable[McapRecord], bool]] = [
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
                offset.write_record_to(summary_buffer)

        return summary_buffer.getvalue(), summary_offsets


# Chunked writer


def _compress_chunk_data(
    data: bytes | memoryview, compression: CompressionType, min_ratio: float = 0.05
) -> tuple[bytes | memoryview, str]:
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
        compressor = _thread_local.zstd_compressor
        if compressor is None:
            compressor = zstandard.ZstdCompressor()
            _thread_local.zstd_compressor = compressor
        compressed = compressor.compress(data)
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


class _ChunkBuilder:
    """Builds chunks by accumulating messages until chunk size limit is reached."""

    def __init__(
        self,
        compression: CompressionType,
        enable_crcs: bool,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        self.compression = compression
        self.enable_crcs = enable_crcs
        self.chunk_size = chunk_size
        self.buffer = io.BytesIO()
        self.reset()

    def reset(self) -> None:
        """Reset builder state for a new chunk."""
        self.buffer.seek(0)
        self.buffer.truncate(0)
        self.buf_pos = 0
        self.message_start_time = 0
        self.message_end_time = 0
        self.message_indices: dict[int, MessageIndex] = {}
        self.num_messages = 0

    def add(self, record: Message) -> None:
        # Update time tracking
        if self.num_messages == 0:
            self.message_start_time = record.log_time
            self.message_end_time = record.log_time
        else:
            self.message_start_time = min(self.message_start_time, record.log_time)
            self.message_end_time = max(self.message_end_time, record.log_time)

        # Update message index (get position before writing)
        msg_idx = self.message_indices.get(record.channel_id)
        if msg_idx is None:
            msg_idx = MessageIndex(channel_id=record.channel_id, timestamps=[], offsets=[])
            self.message_indices[record.channel_id] = msg_idx
        msg_idx.timestamps.append(record.log_time)
        msg_idx.offsets.append(self.buffer.tell())

        self.num_messages += 1

        # Write directly to BytesIO buffer
        record.write_record_to(self.buffer)

    def add_raw(
        self,
        channel_id: int,
        log_time: int,
        data: bytes | memoryview,
        publish_time: int,
        sequence: int,
    ) -> None:
        """Inline message serialization — no Message object created."""
        buf = self.buffer
        data_len = len(data)

        # Update time tracking (if/else instead of min/max)
        if self.num_messages == 0:
            self.message_start_time = log_time
            self.message_end_time = log_time
        else:
            self.message_start_time = min(self.message_start_time, log_time)
            self.message_end_time = max(self.message_end_time, log_time)

        # Update message index using tracked position
        msg_idx = self.message_indices.get(channel_id)
        if msg_idx is None:
            msg_idx = MessageIndex(channel_id=channel_id, timestamps=[], offsets=[])
            self.message_indices[channel_id] = msg_idx
        msg_idx.timestamps.append(log_time)
        msg_idx.offsets.append(self.buf_pos)

        self.num_messages += 1

        # Pack header + write data directly (no Message dataclass)
        buf.write(
            _MSG_RECORD_STRUCT.pack(
                _MSG_OPCODE,
                _MSG_FIXED_LEN + data_len,
                channel_id,
                sequence,
                log_time,
                publish_time,
            )
        )
        buf.write(data)
        self.buf_pos += _MSG_HEADER_SIZE + data_len

    def extract(
        self,
    ) -> tuple[bytes, int, int, int, int, dict[int, MessageIndex]] | None:
        """Extract raw chunk data and metadata without compressing.

        Returns (chunk_data, uncompressed_crc, uncompressed_size,
                 start_time, end_time, message_indices) or None if empty.
        """
        if self.num_messages == 0:
            return None

        chunk_data = self.buffer.getvalue()
        uncompressed_size = len(chunk_data)
        uncompressed_crc = zlib.crc32(chunk_data) if self.enable_crcs else 0

        return (
            chunk_data,
            uncompressed_crc,
            uncompressed_size,
            self.message_start_time,
            self.message_end_time,
            self.message_indices,
        )

    def finalize(self) -> tuple[Chunk, dict[int, MessageIndex]] | None:
        """Build and return the final chunk from current buffer state."""
        extracted = self.extract()
        if extracted is None:
            return None
        chunk_data, uncompressed_crc, uncompressed_size, start, end, indices = extracted
        compressed_data, compression_used = _compress_chunk_data(chunk_data, self.compression)
        return Chunk(
            compression=compression_used,
            data=compressed_data,
            message_start_time=start,
            message_end_time=end,
            uncompressed_crc=uncompressed_crc,
            uncompressed_size=uncompressed_size,
        ), indices


@dataclass(slots=True)
class _PendingChunk:
    """Holds metadata for a chunk whose compression is in flight."""

    future: Future[tuple[bytes | memoryview, str]]
    message_start_time: int
    message_end_time: int
    uncompressed_size: int
    uncompressed_crc: int
    message_indices: dict[int, MessageIndex]


class EncoderFactoryProtocol(Protocol):
    profile: str
    encoding: str  # Schema encoding format
    message_encoding: str  # Message data encoding format

    def encoder_for(self, schema: Schema | None) -> Callable[[Any], bytes | memoryview] | None: ...


class McapWriter(McapWriterRaw):
    def __init__(
        self,
        output: BinaryIO,
        index_types: IndexType = IndexType.ALL,
        repeat_channels: bool = True,
        repeat_schemas: bool = True,
        use_statistics: bool = True,
        use_summary_offsets: bool = True,
        enable_crcs: bool = True,
        use_chunking: bool = True,
        chunk_size: int = 1024 * 1024,
        compression: CompressionType = CompressionType.ZSTD,
        encoder_factory: EncoderFactoryProtocol | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            output=output,
            index_types=index_types,
            repeat_channels=repeat_channels,
            repeat_schemas=repeat_schemas,
            use_statistics=use_statistics,
            use_summary_offsets=use_summary_offsets,
            enable_crcs=enable_crcs,
        )
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.compression = compression
        self.encoder_factory = encoder_factory
        self.encoder_functions: dict[int, Callable[[Any], bytes | memoryview]] = {}
        self.chunk_builder: _ChunkBuilder | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._pending: list[_PendingChunk] = []
        if use_chunking:
            self.chunk_builder = _ChunkBuilder(compression, enable_crcs, chunk_size)
            if num_workers > 0:
                self._pool = ThreadPoolExecutor(max_workers=num_workers)
                self._max_pending = num_workers

    def add_message_encode(
        self,
        channel_id: int,
        log_time: int,
        data: Any,
        publish_time: int,
        sequence: int = 0,
    ) -> None:
        if self.encoder_factory is None:
            raise RuntimeError("encoder_factory must be set to use add_message_object()")

        channel = self.channels.get(channel_id)
        if channel is None:
            raise ValueError(f"Channel ID {channel_id} does not exist.")

        # Support schemaless channels (schema_id=0)
        schema = self.schemas.get(channel.schema_id) if channel.schema_id != 0 else None
        cache_key = channel.schema_id if schema else 0

        encoder = self.encoder_functions.get(cache_key)
        if encoder is None:
            encoder = self.encoder_factory.encoder_for(schema)
            if encoder is None:
                raise ValueError(f"No encoder found for schema ID {channel.schema_id}.")
            self.encoder_functions[cache_key] = encoder

        encoded_data = encoder(data)
        self.add_message(
            channel_id=channel_id,
            log_time=log_time,
            data=encoded_data,
            publish_time=publish_time,
            sequence=sequence,
        )

    def add_message(
        self,
        channel_id: int,
        log_time: int,
        data: bytes | memoryview,
        publish_time: int,
        sequence: int = 0,
    ) -> None:
        """Add a message to the file."""
        if self.use_chunking and self.chunk_builder is not None:
            # Check if chunk is ready to be written *before* adding the new message
            if (
                self.chunk_builder.buf_pos >= self.chunk_builder.chunk_size
                and self.chunk_builder.num_messages > 0
            ):
                self._submit_or_write_chunk()
                self.chunk_builder.reset()

            self.chunk_builder.add_raw(channel_id, log_time, data, publish_time, sequence)
        else:
            # No chunking - write directly
            super().add_message(channel_id, log_time, data, publish_time, sequence)

    def _write_chunk(
        self,
        compressed_data: bytes | memoryview,
        compression: str,
        message_start_time: int,
        message_end_time: int,
        uncompressed_size: int,
        uncompressed_crc: int,
        message_indices: dict[int, MessageIndex],
    ) -> None:
        """Construct a Chunk from fields and write it via the base class."""
        chunk = Chunk(
            compression=compression,
            data=compressed_data,
            message_start_time=message_start_time,
            message_end_time=message_end_time,
            uncompressed_crc=uncompressed_crc,
            uncompressed_size=uncompressed_size,
        )
        super().add_chunk(chunk, message_indices)

    def _submit_or_write_chunk(self) -> None:
        """Submit chunk for async compression or write synchronously.

        Caller must ensure ``self.chunk_builder`` is not None.
        """
        assert self.chunk_builder is not None
        extracted = self.chunk_builder.extract()
        if extracted is None:
            return
        chunk_data, uncompressed_crc, uncompressed_size, start_time, end_time, message_indices = (
            extracted
        )

        if self._pool is not None:
            future = self._pool.submit(_compress_chunk_data, chunk_data, self.compression)
            self._pending.append(
                _PendingChunk(
                    future=future,
                    message_start_time=start_time,
                    message_end_time=end_time,
                    uncompressed_size=uncompressed_size,
                    uncompressed_crc=uncompressed_crc,
                    message_indices=message_indices,
                )
            )
            if len(self._pending) > self._max_pending:
                self._drain_one()
        else:
            compressed_data, compression_used = _compress_chunk_data(chunk_data, self.compression)
            self._write_chunk(
                compressed_data,
                compression_used,
                start_time,
                end_time,
                uncompressed_size,
                uncompressed_crc,
                message_indices,
            )

    def _drain_one(self) -> None:
        """Wait on the oldest pending chunk and write it."""
        if not self._pending:
            return
        pending = self._pending.pop(0)
        compressed_data, compression_used = pending.future.result()
        self._write_chunk(
            compressed_data,
            compression_used,
            pending.message_start_time,
            pending.message_end_time,
            pending.uncompressed_size,
            pending.uncompressed_crc,
            pending.message_indices,
        )

    def _drain_all(self) -> None:
        """Drain all pending chunks in FIFO order."""
        while self._pending:
            self._drain_one()

    def add_chunk(self, chunk: Chunk, message_indices: dict[int, MessageIndex]) -> None:
        self._finalize_current_chunk()
        return super().add_chunk(chunk, message_indices)

    def add_attachment(
        self,
        log_time: int,
        create_time: int,
        name: str,
        media_type: str,
        data: bytes | memoryview,
    ) -> None:
        self._drain_all()
        return super().add_attachment(log_time, create_time, name, media_type, data)

    def add_metadata(self, name: str, metadata: dict[str, str]) -> None:
        self._drain_all()
        return super().add_metadata(name, metadata)

    def _finalize_current_chunk(self) -> None:
        """Finalize and write any remaining chunk."""
        if self.chunk_builder is not None and self.chunk_builder.num_messages > 0:
            self._submit_or_write_chunk()
            self.chunk_builder.reset()
        self._drain_all()

    def finish(self) -> None:
        self._finalize_current_chunk()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        return super().finish()
