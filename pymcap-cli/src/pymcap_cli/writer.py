import io
import struct
import zlib
from collections import defaultdict
from collections.abc import Generator
from enum import Enum, Flag, auto
from typing import BinaryIO

import zstandard

try:
    from lz4.frame import compress as lz4_compress
except ImportError:
    lz4_compress = None


from dataclasses import dataclass

from .mcap_data import (
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Footer,
    Header,
    McapRecord,
    Message,
    MessageIndex,
    Opcode,
    Schema,
    Statistics,
    SummaryOffset,
)

MCAP0_MAGIC = struct.pack("<8B", 137, 77, 67, 65, 80, 48, 13, 10)


@dataclass
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


def _chunk_builder(
    compression: CompressionType,
    enable_crcs: bool,
    chunk_size: int = 1024 * 1024,
) -> Generator[tuple[Chunk, dict[int, MessageIndex]] | None, McapRecord | None, None]:
    """
    Generator that builds chunks and yields them when full.
    Yields (Chunk, message_indices) when chunk is complete, or None for empty chunks.
    Send None to finalize and get the last chunk.
    """

    def build_chunk() -> tuple[Chunk, dict[int, MessageIndex]] | None:
        """Build a chunk from current buffer state."""
        if num_messages == 0:
            return None

        chunk_data = buffer.getvalue()

        if compression == CompressionType.ZSTD:
            cctx = zstandard.ZstdCompressor()
            compressed_data = cctx.compress(chunk_data)
        elif compression == CompressionType.LZ4:
            if lz4_compress is None:
                raise ImportError("lz4 module not available")
            compressed_data = lz4_compress(chunk_data)
        elif compression == CompressionType.NONE:
            compressed_data = chunk_data
        else:
            raise ValueError(f"Unsupported compression type: {compression}")

        return Chunk(
            compression=compression.value,
            data=compressed_data,
            message_start_time=message_start_time,
            message_end_time=message_end_time,
            uncompressed_crc=zlib.crc32(chunk_data) if enable_crcs else 0,
            uncompressed_size=len(chunk_data),
        ), message_indices.copy()

    message_end_time = 0
    message_start_time = 0
    message_indices: dict[int, MessageIndex] = {}
    num_messages = 0
    buffer = io.BytesIO()

    while True:
        record = yield
        if record is None:
            # Finalize - yield remaining chunk if any
            result = build_chunk()
            buffer.close()
            yield result
            return

        # Check if we need to yield current chunk before adding this record
        if buffer.tell() >= chunk_size and num_messages > 0:
            result = build_chunk()
            yield result
            # Reset for next chunk
            buffer = io.BytesIO()
            message_indices = {}
            num_messages = 0

        # Add record to current chunk
        if isinstance(record, (Schema, Channel)):
            buffer.write(record.write_record())
        elif isinstance(record, Message):
            if num_messages == 0:
                message_start_time = record.log_time
            else:
                message_start_time = min(message_start_time, record.log_time)
            message_end_time = max(message_end_time, record.log_time)

            if not message_indices.get(record.channel_id):
                message_indices[record.channel_id] = MessageIndex(
                    channel_id=record.channel_id, records=[]
                )
            message_indices[record.channel_id].records.append((record.log_time, buffer.tell()))

            num_messages += 1
            buffer.write(record.write_record())
        else:
            msg = f"Unexpected record type in chunk builder: {type(record)}"
            raise TypeError(msg)


def _build_summary(
    schemas: dict[int, Schema],
    channels: dict[int, Channel],
    statistics: Statistics,
    chunk_indices: list[ChunkIndex],
    index_types: IndexType,
    repeat_schemas: bool,
    repeat_channels: bool,
    use_statistics: bool,
    use_summary_offsets: bool,
) -> tuple[bytes, list[SummaryOffset]]:
    """Build summary section and return (summary_data, summary_offsets)."""
    summary_buffer = io.BytesIO()
    summary_offsets: list[SummaryOffset] = []

    # Note: summary_start will be calculated by caller after DataEnd is written
    summary_start = 0

    # Write schemas
    if repeat_schemas and schemas:
        group_start = summary_buffer.tell()
        for schema in schemas.values():
            summary_buffer.write(schema.write_record())
        summary_offsets.append(
            SummaryOffset(
                group_opcode=Opcode.SCHEMA,
                group_start=summary_start + group_start,
                group_length=summary_buffer.tell() - group_start,
            )
        )

    # Write channels
    if repeat_channels and channels:
        group_start = summary_buffer.tell()
        for channel in channels.values():
            summary_buffer.write(channel.write_record())
        summary_offsets.append(
            SummaryOffset(
                group_opcode=Opcode.CHANNEL,
                group_start=summary_start + group_start,
                group_length=summary_buffer.tell() - group_start,
            )
        )

    # Write statistics
    if use_statistics:
        group_start = summary_buffer.tell()
        summary_buffer.write(statistics.write_record())
        summary_offsets.append(
            SummaryOffset(
                group_opcode=Opcode.STATISTICS,
                group_start=summary_start + group_start,
                group_length=summary_buffer.tell() - group_start,
            )
        )

    # Write chunk indexes
    if (index_types & IndexType.CHUNK) and chunk_indices:
        group_start = summary_buffer.tell()
        for chunk_index in chunk_indices:
            summary_buffer.write(chunk_index.write_record())
        summary_offsets.append(
            SummaryOffset(
                group_opcode=Opcode.CHUNK_INDEX,
                group_start=summary_start + group_start,
                group_length=summary_buffer.tell() - group_start,
            )
        )

    # Write summary offsets
    if use_summary_offsets:
        for offset in summary_offsets:
            summary_buffer.write(offset.write_record())

    return summary_buffer.getvalue(), summary_offsets


class CRCWriter:
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


def write_mcap(
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
) -> Generator[None, McapRecord, None]:
    """
    Generator-based MCAP writer. Simpler than McapWriter class.

    Caller is responsible for:
    - Opening/closing the output stream
    - Flushing as needed

    Usage:
        writer = write_mcap(output_file)
        next(writer)  # Initialize
        writer.send(Schema(...))
        writer.send(Channel(...))
        writer.send(Message(...))
        ...
        writer.send(None)  # Finalize
    """
    # Write header
    crc_writer = CRCWriter(output, enable_crcs)
    crc_writer.write(MCAP0_MAGIC)
    header_data = Header(profile="", library="pymcap-cli 0.1.0").write_record()
    crc_writer.write(header_data)

    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    chunk_indices: list[ChunkIndex] = []

    statistics = Statistics(
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

    # Chunking state
    current_chunk: (
        Generator[tuple[Chunk, dict[int, MessageIndex]] | None, McapRecord | None, None] | None
    ) = None

    if use_chunking:
        current_chunk = _chunk_builder(compression, enable_crcs, chunk_size)
        next(current_chunk)  # Initialize the generator

    def write_chunk(chunk: Chunk, message_indices: dict[int, MessageIndex]) -> None:
        """Write a chunk and its indexes to output."""

        # Update statistics for messages in chunk
        if statistics.message_count == 0:
            statistics.message_start_time = chunk.message_start_time
        else:
            statistics.message_start_time = min(
                chunk.message_start_time, statistics.message_start_time
            )
        statistics.message_end_time = max(chunk.message_end_time, statistics.message_end_time)

        for idx in message_indices.values():
            statistics.channel_message_counts[idx.channel_id] += len(idx.records)
            statistics.message_count += len(idx.records)

        statistics.chunk_count += 1

        # Write chunk
        chunk_start_offset = crc_writer.tell()
        chunk_data = chunk.write_record()
        crc_writer.write(chunk_data)

        # Write message indexes
        message_index_start_offset = crc_writer.tell()
        message_index_offsets: dict[int, int] = {}
        index_buffer = io.BytesIO()

        if index_types & IndexType.MESSAGE:
            for idx in message_indices.values():
                message_index_offsets[idx.channel_id] = (
                    message_index_start_offset + index_buffer.tell()
                )
                index_buffer.write(idx.write_record())

        index_data = index_buffer.getvalue()
        crc_writer.write(index_data)

        # Store chunk index
        chunk_indices.append(
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

    def write_record(record: McapRecord) -> None:
        nonlocal current_chunk
        if not use_chunking:
            crc_writer.write(record.write_record())
            return
        if current_chunk is None:
            return

        if result := current_chunk.send(record):
            # Chunk was yielded, write it
            chunk, message_indices = result
            write_chunk(chunk, message_indices)

    while True:
        record = yield
        if record is None:
            break

        # Handle pre-built chunks
        if isinstance(record, PrebuiltChunk):
            # Finalize any in-progress chunk first
            if use_chunking and current_chunk is not None:
                if result := current_chunk.send(None):
                    chunk, message_indices = result
                    write_chunk(chunk, message_indices)
                # Restart chunk builder
                current_chunk = _chunk_builder(compression, enable_crcs, chunk_size)
                next(current_chunk)

            # Write the pre-built chunk directly
            write_chunk(record.chunk, {idx.channel_id: idx for idx in record.indexes})
            continue

        if not isinstance(record, McapRecord):
            raise TypeError(f"Expected McapRecord or PrebuiltChunk, got {type(record)}")

        if isinstance(record, Schema):
            write_record(record)
            schemas[record.id] = record
            statistics.schema_count += 1

        elif isinstance(record, Channel):
            write_record(record)
            channels[record.id] = record
            statistics.channel_count += 1

        elif isinstance(record, Message):
            write_record(record)

            # Non-chunked mode - update statistics and write directly
            # In chunked mode, statistics are updated when the chunk is written
            if not use_chunking:
                if statistics.message_count == 0:
                    statistics.message_start_time = record.log_time
                else:
                    statistics.message_start_time = min(
                        record.log_time, statistics.message_start_time
                    )
                statistics.message_end_time = max(record.log_time, statistics.message_end_time)
                statistics.channel_message_counts[record.channel_id] += 1
                statistics.message_count += 1
    # Finalize any remaining chunk
    if use_chunking and current_chunk is not None and (result := current_chunk.send(None)):
        chunk, message_indices = result
        write_chunk(chunk, message_indices)

    # Write DataEnd
    data_end = DataEnd(data_section_crc=crc_writer.crc).write_record()
    crc_writer.enable_crc = False  # No need to include DataEnd in CRC
    crc_writer.write(data_end)

    # Build and write summary section
    summary_start = crc_writer.tell()
    summary_data, summary_offsets = _build_summary(
        schemas=schemas,
        channels=channels,
        statistics=statistics,
        chunk_indices=chunk_indices,
        index_types=index_types,
        repeat_schemas=repeat_schemas,
        repeat_channels=repeat_channels,
        use_statistics=use_statistics,
        use_summary_offsets=use_summary_offsets,
    )

    # Fixup summary offsets with actual summary_start position
    for offset in summary_offsets:
        offset.group_start += summary_start

    summary_length = len(summary_data)
    summary_offset_start = (
        summary_start
        + summary_length
        - sum(len(offset.write_record()) for offset in summary_offsets)
        if use_summary_offsets and summary_offsets
        else 0
    )

    # Calculate summary CRC
    summary_crc = 0
    if enable_crcs:
        summary_crc = zlib.crc32(summary_data)
        # Include footer fields in CRC
        summary_crc = zlib.crc32(
            struct.pack(
                "<BQQQ",
                Opcode.FOOTER,
                8 + 8 + 4,  # Footer record length
                0 if summary_length == 0 else summary_start,
                summary_offset_start,
            ),
            summary_crc,
        )

    crc_writer.write(summary_data)

    # Write footer
    footer = Footer(
        summary_start=0 if summary_length == 0 else summary_start,
        summary_offset_start=summary_offset_start,
        summary_crc=summary_crc,
    )
    crc_writer.write(footer.write_record())

    # Write closing magic
    crc_writer.write(MCAP0_MAGIC)


class McapWriter:
    """
    Class-based MCAP writer API wrapping the generator-based write_mcap().
    Maintains compatibility with existing code using the old McapWriter API.
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
        self._generator = write_mcap(
            output=output,
            chunk_size=chunk_size,
            compression=compression,
            index_types=index_types,
            repeat_channels=repeat_channels,
            repeat_schemas=repeat_schemas,
            use_chunking=use_chunking,
            use_statistics=use_statistics,
            use_summary_offsets=use_summary_offsets,
            enable_crcs=enable_crcs,
            enable_data_crcs=enable_data_crcs,
        )
        self._started = False
        self._finished = False

    def start(self, profile: str = "", library: str = "pymcap-cli 0.1.0") -> None:
        """Start writing the MCAP file."""
        if self._started:
            raise RuntimeError("Writer already started")
        # Initialize the generator
        next(self._generator)
        self._started = True

    def add_schema(self, schema_id: int, name: str, encoding: str, data: bytes) -> None:
        """Add a schema to the file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

        schema = Schema(id=schema_id, name=name, encoding=encoding, data=data)
        self._generator.send(schema)

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

        channel = Channel(
            id=channel_id,
            schema_id=schema_id,
            topic=topic,
            message_encoding=message_encoding,
            metadata=metadata or {},
        )
        self._generator.send(channel)

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

        message = Message(
            channel_id=channel_id,
            sequence=sequence,
            log_time=log_time,
            publish_time=publish_time,
            data=data,
        )
        self._generator.send(message)

    def add_attachment(
        self,
        create_time: int,
        log_time: int,
        name: str,
        media_type: str,
        data: bytes,
    ) -> None:
        """Add an attachment to the file. (Not implemented in generator version)"""
        # The generator version doesn't support attachments yet
        # This is a no-op for compatibility

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        """Add metadata to the file. (Not implemented in generator version)"""
        # The generator version doesn't support metadata yet
        # This is a no-op for compatibility

    def add_chunk_with_indexes(self, chunk: Chunk, indexes: list[MessageIndex]) -> None:
        """
        Add a pre-built chunk with its indexes directly to the file.
        This is used for fast chunk copying without decompression.
        """
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            raise RuntimeError("Writer already finished")

        prebuilt = PrebuiltChunk(chunk=chunk, indexes=indexes)
        self._generator.send(prebuilt)  # type: ignore[arg-type]

    def finish(self) -> None:
        """Finish writing the MCAP file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            return

        try:
            self._generator.send(None)  # type: ignore[arg-type]
        except StopIteration:
            pass

        self._finished = True
