"""Rebuild MCAP summary section from data section."""

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import IO

from small_mcap.reader import (
    McapError,
    breakup_chunk,
    stream_reader,
)
from small_mcap.records import (
    Attachment,
    AttachmentIndex,
    Channel,
    Chunk,
    ChunkIndex,
    Header,
    LazyChunk,
    Message,
    MessageIndex,
    Metadata,
    MetadataIndex,
    Schema,
    Statistics,
    Summary,
)


def _estimate_size_from_indexes(indexes: list[MessageIndex], chunk_size: int) -> dict[int, int]:
    """Estimate message sizes from MessageIndex offsets without decompressing.

    This calculates approximate message sizes by looking at the gaps between message
    offsets within a chunk. The size is estimated as the difference between consecutive
    offsets minus the message record header size.

    Args:
        indexes: List of MessageIndex records for a chunk
        chunk_size: Uncompressed size of the chunk

    Returns:
        Dict mapping channel_id to total estimated bytes for that channel
    """
    sizes_dd: dict[int, int] = defaultdict(int)

    for cur, (_, end_offset) in itertools.pairwise(
        heapq.merge(
            *(
                ((msg_idx.channel_id, record[1]) for record in msg_idx.records)
                for msg_idx in indexes
            ),
            ((None, chunk_size),),
            key=lambda x: x[1],
        )
    ):
        channel, start_offset = cur
        assert channel is not None, "Channel ID should not be None"
        # Estimate message data size from offset gap minus record overhead
        size = (end_offset - start_offset) - (2 + 4 + 8 + 8)
        assert size > 0, f"Invalid size for channel {channel}: {size}"
        sizes_dd[channel] += size

    return dict(sizes_dd)


@dataclass(slots=True)
class RebuildInfo:
    """Result of rebuilding MCAP summary from data section.

    Attributes:
        header: The MCAP file header
        summary: The rebuilt summary section
        channel_sizes: Optional dict mapping channel_id to total uncompressed message data bytes.
            This represents the sum of len(message.data) for all messages on each channel.
            None if calculate_channel_sizes was False.
        estimated_channel_sizes: Whether channel_sizes are estimated (True) or exact (False).
            Only meaningful when channel_sizes is not None.
        chunk_information: Optional dict mapping chunk offset to MessageIndex records.
        next_offset: Byte offset where the next read should start.
    """

    header: Header
    summary: Summary
    channel_sizes: dict[int, int] | None = None
    estimated_channel_sizes: bool = False
    chunk_information: dict[int, list[MessageIndex]] | None = None
    next_offset: int = 0


def rebuild_summary(
    stream: IO[bytes],
    *,
    validate_crc: bool,
    calculate_channel_sizes: bool,
    exact_sizes: bool,
    initial_state: RebuildInfo | None = None,
    skip_magic: bool = False,
) -> RebuildInfo:
    """Rebuild summary section from an MCAP file's data section.

    This function reconstructs the summary section by reading through the data section
    and collecting all necessary information. It reuses the same chunk processing logic
    as message reading to efficiently handle channel definitions.

    Args:
        stream: Input stream to read from (must be non-seekable or at start of file)
        validate_crc: Whether to validate CRC checksums when processing chunks
        calculate_channel_sizes: Whether to calculate per-channel message data sizes.
        exact_sizes: When True, decompresses all chunks for exact sizes. When False, estimates
            sizes from MessageIndex offsets (faster but approximate). Only relevant when
            calculate_channel_sizes=True.
        initial_state: Optional previous RebuildInfo to resume from. When used for resuming, first
            seek to initial_state.next_offset before calling and set skip_magic=True.

    Returns:
        RebuildInfo containing header, summary, and optionally channel_sizes.
        The estimated_channel_sizes flag indicates if sizes are estimated.
        The next_offset field contains the byte position where reading stopped.

    Raises:
        McapError: If the file is invalid or header is missing
    """
    # Initialize or resume from previous state
    if initial_state is not None:
        # Resume from previous state
        header = initial_state.header
        summary = initial_state.summary
        statistics = summary.statistics
        assert statistics is not None, "Initial state's summary must have statistics"
        channel_sizes = defaultdict(int, initial_state.channel_sizes or {})
        chunk_information = dict(initial_state.chunk_information or {})
    else:
        # Start fresh
        header = None
        summary = Summary()
        statistics = Statistics(
            attachment_count=0,
            channel_count=0,
            channel_message_counts=defaultdict(int),
            chunk_count=0,
            message_count=0,
            message_end_time=0,
            message_start_time=0,
            metadata_count=0,
            schema_count=0,
        )
        channel_sizes = defaultdict(int)
        chunk_information = {}

    # These always start fresh (handle chunk boundaries)
    pending_chunk: LazyChunk | None = None
    pending_chunk_start_offset: int = 0
    pending_indexes: list[MessageIndex] = []
    pending_message_index_offsets: dict[int, int] = {}

    def update_message(record: Message) -> None:
        # Update message time statistics
        if statistics.message_start_time == 0:
            statistics.message_start_time = record.log_time
        else:
            statistics.message_start_time = min(statistics.message_start_time, record.log_time)
        if statistics.message_end_time == 0:
            statistics.message_end_time = record.log_time
        else:
            statistics.message_end_time = max(statistics.message_end_time, record.log_time)

        # Update channel message count
        statistics.message_count += 1
        statistics.channel_message_counts[record.channel_id] += 1
        # Calculate channel sizes if requested
        if calculate_channel_sizes:
            channel_sizes[record.channel_id] += len(record.data)

    def finish_chunk(*, force: bool = False) -> None:
        nonlocal pending_chunk
        if pending_chunk is None:
            return

        if (
            force
            or exact_sizes
            or any(msg_idx.channel_id not in summary.channels for msg_idx in pending_indexes)
        ):
            for record in breakup_chunk(
                pending_chunk.to_chunk(stream),
                validate_crc=validate_crc,
            ):
                if isinstance(record, Channel):
                    summary.channels[record.id] = record
                elif isinstance(record, Schema):
                    summary.schemas[record.id] = record
                elif isinstance(record, Message):
                    update_message(record)
        else:
            if calculate_channel_sizes:
                estimated_sizes = _estimate_size_from_indexes(
                    pending_indexes, pending_chunk.uncompressed_size
                )
                for channel_id, size in estimated_sizes.items():
                    channel_sizes[channel_id] += size

            for idx in pending_indexes:
                statistics.message_count += len(idx.records)
                statistics.channel_message_counts[idx.channel_id] += len(idx.records)

        # Store MessageIndex records for this chunk before clearing
        if pending_indexes:
            chunk_information[pending_chunk_start_offset] = pending_indexes.copy()

        pending_indexes.clear()
        pending_message_index_offsets.clear()
        pending_chunk = None

    message_index_start_offset: int = 0
    prev_pos: int = 0
    last_message_index_end_offset: int = 0

    # Track position for resumable reads
    next_offset = stream.tell()

    try:
        for record in stream_reader(
            stream,
            skip_magic=skip_magic,
            emit_chunks=True,
            lazy_chunks=True,
            validate_crc=validate_crc,
        ):
            record_start_pos = prev_pos
            current_pos = stream.tell()
            next_offset = current_pos  # Track position after each successful record read
            prev_pos = current_pos

            if not isinstance(record, MessageIndex):
                # Finish previous chunk and add its ChunkIndex
                if pending_chunk is not None:
                    # Calculate message index section length
                    message_index_length = record_start_pos - message_index_start_offset

                    summary.chunk_indexes.append(
                        ChunkIndex(
                            chunk_length=message_index_start_offset - pending_chunk_start_offset,
                            chunk_start_offset=pending_chunk_start_offset,
                            compression=pending_chunk.compression,
                            compressed_size=pending_chunk.data_len,
                            message_end_time=pending_chunk.message_end_time,
                            message_index_length=message_index_length,
                            message_index_offsets=pending_message_index_offsets,
                            message_start_time=pending_chunk.message_start_time,
                            uncompressed_size=pending_chunk.uncompressed_size,
                        )
                    )
                finish_chunk()

            if isinstance(record, Header):
                header = record
            elif isinstance(record, Channel):
                summary.channels[record.id] = record
            elif isinstance(record, Schema):
                summary.schemas[record.id] = record
            elif isinstance(record, Message):
                update_message(record)
            elif isinstance(record, Chunk):
                raise RuntimeError("Unreachable")  # noqa: TRY004, TRY301
            elif isinstance(record, LazyChunk):
                pending_chunk_start_offset = record_start_pos
                message_index_start_offset = current_pos

                statistics.chunk_count += 1

                # Update time statistics from chunk
                if statistics.message_start_time == 0:
                    statistics.message_start_time = record.message_start_time
                else:
                    statistics.message_start_time = min(
                        statistics.message_start_time, record.message_start_time
                    )
                if statistics.message_end_time == 0:
                    statistics.message_end_time = record.message_end_time
                else:
                    statistics.message_end_time = max(
                        statistics.message_end_time, record.message_end_time
                    )
                pending_chunk = record
            elif isinstance(record, MessageIndex):
                if record.records:
                    pending_indexes.append(record)
                    # Track the position of this MessageIndex for this channel
                    pending_message_index_offsets.setdefault(record.channel_id, record_start_pos)
                # Track the end position of MessageIndex records
                last_message_index_end_offset = current_pos
            elif isinstance(record, Attachment):
                statistics.attachment_count += 1
            elif isinstance(record, AttachmentIndex):
                summary.attachment_indexes.append(record)
            elif isinstance(record, Metadata):
                statistics.metadata_count += 1
            elif isinstance(record, MetadataIndex):
                summary.metadata_indexes.append(record)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Error while rebuilding summary at offset {next_offset}: {e}")  # noqa: T201
    # TODO: figure out how to handle partial message indexes of broken mcaps
    # final ChunkIndex
    if pending_chunk is not None:
        message_index_length = last_message_index_end_offset - message_index_start_offset
        summary.chunk_indexes.append(
            ChunkIndex(
                chunk_length=message_index_start_offset - pending_chunk_start_offset,
                chunk_start_offset=pending_chunk_start_offset,
                compression=pending_chunk.compression,
                compressed_size=pending_chunk.data_len,
                message_end_time=pending_chunk.message_end_time,
                message_index_length=message_index_length,
                message_index_offsets=pending_message_index_offsets,
                message_start_time=pending_chunk.message_start_time,
                uncompressed_size=pending_chunk.uncompressed_size,
            )
        )
    finish_chunk()

    # Finalize statistics
    statistics.schema_count = len(summary.schemas)
    statistics.channel_count = len(summary.channels)
    summary.statistics = statistics

    if header is None:
        raise McapError("No header found in MCAP file")

    return RebuildInfo(
        header=header,
        summary=summary,
        channel_sizes=channel_sizes if calculate_channel_sizes else None,
        estimated_channel_sizes=calculate_channel_sizes and not exact_sizes,
        chunk_information=chunk_information if chunk_information else None,
        next_offset=next_offset,
    )
