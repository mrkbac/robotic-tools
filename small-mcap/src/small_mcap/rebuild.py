"""Rebuild MCAP summary section from data section."""

import itertools
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import IO

from small_mcap.reader import McapError, breakup_chunk, stream_reader
from small_mcap.records import (
    Attachment,
    AttachmentIndex,
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Header,
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
    idx_list: list[tuple[int | None, int]] = [
        (idx.channel_id, offset) for idx in indexes for _, offset in idx.records
    ]

    # Sort by offset (second element) using operator.itemgetter for better performance
    idx_list.sort(key=operator.itemgetter(1))
    idx_list.append((None, chunk_size))

    sizes_dd: dict[int, int] = defaultdict(int)

    for cur, (_, end_offset) in itertools.pairwise(idx_list):
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
    """

    header: Header
    summary: Summary
    channel_sizes: dict[int, int] | None = None
    estimated_channel_sizes: bool = False
    chunk_information: dict[int, list[MessageIndex]] | None = None


def rebuild_summary(
    stream: IO[bytes],
    *,
    validate_crc: bool,
    calculate_channel_sizes: bool,
    exact_sizes: bool,
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

    Returns:
        RebuildInfo containing header, summary, and optionally channel_sizes.
        The estimated_channel_sizes flag indicates if sizes are estimated.

    Raises:
        McapError: If the file is invalid or header is missing
    """
    header: Header | None = None
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
    seen_channels: set[int] = set()
    channel_sizes: defaultdict[int, int] = defaultdict(int)
    # Track chunks and their MessageIndex for estimation mode
    pending_chunk: Chunk | None = None
    pending_indexes: list[MessageIndex] = []

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

    def finish_chunk() -> None:
        nonlocal pending_chunk
        if pending_chunk is None:
            return

        if any(msg_idx.channel_id not in seen_channels for msg_idx in pending_indexes):
            for record in breakup_chunk(
                pending_chunk,
                validate_crc=validate_crc,
            ):
                if isinstance(record, Channel):
                    summary.channels[record.id] = record
                    seen_channels.add(record.id)
                elif isinstance(record, Schema):
                    summary.schemas[record.id] = record
                elif isinstance(record, Message):
                    update_message(record)
        elif calculate_channel_sizes:
            estimated_sizes = _estimate_size_from_indexes(
                pending_indexes, pending_chunk.uncompressed_size
            )
            for channel_id, size in estimated_sizes.items():
                channel_sizes[channel_id] = channel_sizes.get(channel_id, 0) + size

        pending_indexes.clear()
        pending_chunk = None

    for record in stream_reader(stream, emit_chunks=True, validate_crc=validate_crc):
        if not isinstance(record, MessageIndex):
            finish_chunk()

        if isinstance(record, Header):
            header = record
        elif isinstance(record, Channel):
            summary.channels[record.id] = record
            seen_channels.add(record.id)
        elif isinstance(record, Schema):
            summary.schemas[record.id] = record
        elif isinstance(record, Message):
            update_message(record)
        elif isinstance(record, Chunk):
            # Track chunk info
            summary.chunk_indexes.append(
                ChunkIndex(
                    chunk_length=0,  # Not available without file position tracking
                    chunk_start_offset=0,  # Not available without file position tracking
                    compression=record.compression,
                    compressed_size=len(record.data),
                    message_end_time=record.message_end_time,
                    message_index_length=0,  # Not available without file position tracking
                    message_index_offsets={},
                    message_start_time=record.message_start_time,
                    uncompressed_size=record.uncompressed_size,
                )
            )
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
            if len(record.records) > 0:
                pending_indexes.append(record)
        elif isinstance(record, Attachment):
            statistics.attachment_count += 1
        elif isinstance(record, AttachmentIndex):
            summary.attachment_indexes.append(record)
        elif isinstance(record, Metadata):
            statistics.metadata_count += 1
        elif isinstance(record, MetadataIndex):
            summary.metadata_indexes.append(record)
        elif isinstance(record, DataEnd):
            break

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
    )
