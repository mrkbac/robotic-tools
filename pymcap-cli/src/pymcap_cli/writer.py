import io
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, Flag, auto
from typing import BinaryIO

import zstandard

try:
    from lz4.frame import compress as lz4_compress
except ImportError:
    lz4_compress = None


from .mcap_data import (
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Footer,
    Header,
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
        # Pre-allocate buffer to avoid reallocations
        self.buffer_data = bytearray(2 * chunk_size)
        self.buffer_pos = 0
        self.reset()

    def reset(self) -> None:
        """Reset builder state for a new chunk."""
        self.buffer_pos = 0
        self.message_start_time = 0
        self.message_end_time = 0
        self.message_indices: dict[int, MessageIndex] = {}
        self.num_messages = 0

    def add(
        self, record: Schema | Channel | Message
    ) -> tuple[Chunk, dict[int, MessageIndex]] | None:
        """
        Add a record to the current chunk.
        Returns (chunk, message_indices) if chunk is full and ready to write, None otherwise.
        """
        # Check if we need to finalize current chunk before adding this record
        if self.buffer_pos >= self.chunk_size and self.num_messages > 0:
            result = self.finalize()
            self.reset()
            # After reset, fall through to add the record to the new chunk
        else:
            result = None

        # Add record to current chunk
        record_data = record.write_record()
        record_len = len(record_data)

        # Ensure buffer has space (grow if needed)
        if self.buffer_pos + record_len > len(self.buffer_data):
            self.buffer_data.extend(bytearray(max(record_len, self.chunk_size)))

        if isinstance(record, (Schema, Channel)):
            self.buffer_data[self.buffer_pos : self.buffer_pos + record_len] = record_data
            self.buffer_pos += record_len
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

        return result

    def finalize(self) -> tuple[Chunk, dict[int, MessageIndex]] | None:
        """Build and return the final chunk from current buffer state."""
        if self.num_messages == 0:
            return None

        chunk_data = bytes(self.buffer_data[: self.buffer_pos])

        if self.compression == CompressionType.ZSTD:
            cctx = zstandard.ZstdCompressor()
            compressed_data = cctx.compress(chunk_data)
        elif self.compression == CompressionType.LZ4:
            if lz4_compress is None:
                raise ImportError("lz4 module not available")
            compressed_data = lz4_compress(chunk_data)
        elif self.compression == CompressionType.NONE:
            compressed_data = chunk_data
        else:
            raise ValueError(f"Unsupported compression type: {self.compression}")

        return Chunk(
            compression=self.compression.value,
            data=compressed_data,
            message_start_time=self.message_start_time,
            message_end_time=self.message_end_time,
            uncompressed_crc=zlib.crc32(chunk_data) if self.enable_crcs else 0,
            uncompressed_size=len(chunk_data),
        ), self.message_indices.copy()


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
    ) -> None:
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

        self._started = False
        self._finished = False

        # State tracking
        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}
        self.chunk_indices: list[ChunkIndex] = []
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

        # CRC writer wraps output
        self.crc_writer = _CRCWriter(output, enable_crcs)

        # Chunk builder for accumulating messages
        self.chunk_builder: _ChunkBuilder | None = None
        if use_chunking:
            self.chunk_builder = _ChunkBuilder(compression, enable_crcs, chunk_size)

    def start(self, profile: str = "", library: str = "pymcap-cli 0.1.0") -> None:
        """Start writing the MCAP file."""
        if self._started:
            raise RuntimeError("Writer already started")

        # Write header
        self.crc_writer.write(MCAP0_MAGIC)
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
        create_time: int,
        log_time: int,
        name: str,
        media_type: str,
        data: bytes,
    ) -> None:
        """Add an attachment to the file. (Not implemented yet)"""
        # Not implemented in simplified version

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        """Add metadata to the file. (Not implemented yet)"""
        # Not implemented in simplified version

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
        if self.chunk_builder is not None and (result := self.chunk_builder.finalize()):
            chunk_obj, message_indices = result
            self._write_chunk(chunk_obj, message_indices)
            self.chunk_builder.reset()

        # Write the pre-built chunk
        self._write_chunk(chunk, {idx.channel_id: idx for idx in indexes})

    def finish(self) -> None:
        """Finish writing the MCAP file."""
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")
        if self._finished:
            return

        # Finalize any remaining chunk
        if self.chunk_builder is not None and (result := self.chunk_builder.finalize()):
            chunk, message_indices = result
            self._write_chunk(chunk, message_indices)

        # Write DataEnd
        data_end = DataEnd(data_section_crc=self.crc_writer.crc).write_record()
        self.crc_writer.enable_crc = False  # No need to include DataEnd in CRC
        self.crc_writer.write(data_end)

        # Build and write summary section
        summary_start = self.crc_writer.tell()
        summary_data, summary_offsets = self._build_summary(summary_start)

        # Calculate summary CRC
        summary_crc = 0
        if self.enable_crcs:
            summary_crc = zlib.crc32(summary_data)
            # Include footer fields in CRC
            summary_offset_start = (
                summary_start
                + len(summary_data)
                - sum(len(offset.write_record()) for offset in summary_offsets)
                if self.use_summary_offsets and summary_offsets
                else 0
            )
            summary_crc = zlib.crc32(
                struct.pack(
                    "<BQQQ",
                    Opcode.FOOTER,
                    8 + 8 + 4,  # Footer record length
                    0 if len(summary_data) == 0 else summary_start,
                    summary_offset_start,
                ),
                summary_crc,
            )

        self.crc_writer.write(summary_data)

        # Write footer
        summary_offset_start = (
            summary_start
            + len(summary_data)
            - sum(len(offset.write_record()) for offset in summary_offsets)
            if self.use_summary_offsets and summary_offsets
            else 0
        )
        footer = Footer(
            summary_start=0 if len(summary_data) == 0 else summary_start,
            summary_offset_start=summary_offset_start,
            summary_crc=summary_crc,
        )
        self.crc_writer.write(footer.write_record())

        # Write closing magic
        self.crc_writer.write(MCAP0_MAGIC)

        self._finished = True

    def _write_record(self, record: Schema | Channel | Message) -> None:
        """Write a record to the output, either directly or via chunk builder."""
        if not self.use_chunking:
            self.crc_writer.write(record.write_record())
            return

        if self.chunk_builder is None:
            return

        # Add to chunk builder
        if result := self.chunk_builder.add(record):
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

        # Write schemas
        if self.repeat_schemas and self.schemas:
            group_start = summary_buffer.tell()
            for schema in self.schemas.values():
                summary_buffer.write(schema.write_record())
            summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.SCHEMA,
                    group_start=summary_start + group_start,
                    group_length=summary_buffer.tell() - group_start,
                )
            )

        # Write channels
        if self.repeat_channels and self.channels:
            group_start = summary_buffer.tell()
            for channel in self.channels.values():
                summary_buffer.write(channel.write_record())
            summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.CHANNEL,
                    group_start=summary_start + group_start,
                    group_length=summary_buffer.tell() - group_start,
                )
            )

        # Write statistics
        if self.use_statistics:
            group_start = summary_buffer.tell()
            summary_buffer.write(self.statistics.write_record())
            summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.STATISTICS,
                    group_start=summary_start + group_start,
                    group_length=summary_buffer.tell() - group_start,
                )
            )

        # Write chunk indexes
        if (self.index_types & IndexType.CHUNK) and self.chunk_indices:
            group_start = summary_buffer.tell()
            for chunk_index in self.chunk_indices:
                summary_buffer.write(chunk_index.write_record())
            summary_offsets.append(
                SummaryOffset(
                    group_opcode=Opcode.CHUNK_INDEX,
                    group_start=summary_start + group_start,
                    group_length=summary_buffer.tell() - group_start,
                )
            )

        # Write summary offsets
        if self.use_summary_offsets:
            for offset in summary_offsets:
                summary_buffer.write(offset.write_record())

        return summary_buffer.getvalue(), summary_offsets
