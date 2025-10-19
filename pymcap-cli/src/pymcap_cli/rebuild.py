import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import BinaryIO

from rich.console import Console

from pymcap_cli.mcap_data import (
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
from pymcap_cli.reader import (
    LazyChunk,
    McapError,
    breakup_chunk,
    get_header,
    get_summary,
    stream_reader,
)
from pymcap_cli.utils import file_progress


@dataclass
class Info:
    header: Header
    summary: Summary
    channel_sizes: dict[int, int] | None = None


console = Console()  # TODO improve


def _estimate_size_from_indexes(indexes: list[MessageIndex], chunk_size: int) -> dict[int, int]:
    idx_list: list[tuple[int | None, int]] = [
        (idx.channel_id, offset) for idx in indexes for _, offset in idx.records
    ]

    # Sort by offset (second element)
    idx_list.sort(key=lambda x: x[1])
    idx_list.append((None, chunk_size))

    sizes_dd: dict[int, int] = defaultdict(int)

    for cur, (_, end_offset) in itertools.pairwise(idx_list):
        channel, start_offset = cur
        size = (end_offset - start_offset) - (2 + 4 + 8 + 8)
        assert size > 0, f"Invalid size for channel {channel}: {size}"
        sizes_dd[channel] += size

    return dict(sizes_dd)


def rebuild_info(f: BinaryIO, file_size: int, *, exact_sizes: bool = False) -> Info:
    header: Header | None = None
    statistics: Statistics = Statistics(
        attachment_count=0,
        channel_count=0,
        channel_message_counts={},
        chunk_count=0,
        message_count=0,
        message_end_time=0,
        message_start_time=0,
        metadata_count=0,
        schema_count=0,
    )
    summary = Summary()
    channel_sizes: dict[int, int] = {}

    def handle_message(record: Message) -> None:
        statistics.message_start_time = (
            min(statistics.message_start_time, record.log_time)
            if statistics.message_start_time
            else record.log_time
        )
        statistics.message_end_time = (
            max(statistics.message_end_time, record.log_time)
            if statistics.message_end_time
            else record.log_time
        )
        channel_sizes[record.channel_id] = channel_sizes.get(record.channel_id, 0) + len(
            record.data
        )

    def update_from_chunk(chunk: Chunk) -> None:
        for record in breakup_chunk(chunk):
            if isinstance(record, Channel):
                summary.channels[record.id] = record
            elif isinstance(record, Schema):
                summary.schemas[record.id] = record
            elif isinstance(record, Message):
                handle_message(record)
            else:
                raise McapError(f"Unexpected record type: {type(record)}")

    last_chunk: Chunk | LazyChunk | None = None
    last_chunk_message_indexes: list[MessageIndex] = []

    with file_progress("[bold blue]Rebuilding MCAP info...", console) as progress:
        task = progress.add_task("Processing", total=file_size)

        for record in stream_reader(
            f, skip_magic=False, emit_chunks=not exact_sizes, lazy_chunks=not exact_sizes
        ):
            progress.update(task, completed=f.tell())

            if not isinstance(record, MessageIndex) and last_chunk:
                decode = False
                for idx in last_chunk_message_indexes:
                    if idx.channel_id not in summary.channels:
                        decode = True
                        break

                if decode:
                    if isinstance(last_chunk, LazyChunk):
                        # Convert LazyChunk to full Chunk
                        last_chunk = last_chunk.to_chunk(f)
                    update_from_chunk(last_chunk)
                else:
                    new_sizes = _estimate_size_from_indexes(
                        last_chunk_message_indexes, last_chunk.uncompressed_size
                    )
                    channel_sizes = {
                        k: channel_sizes.get(k, 0) + new_sizes.get(k, 0)
                        for k in channel_sizes.keys() | new_sizes.keys()
                    }

                last_chunk = None
                last_chunk_message_indexes = []

            if isinstance(record, Header):
                header = record
            elif isinstance(record, Channel):
                summary.channels[record.id] = record
            elif isinstance(record, Schema):
                summary.schemas[record.id] = record
            elif isinstance(record, Message):
                handle_message(record)
            elif isinstance(record, (Chunk, LazyChunk)):
                last_chunk = record

                summary.chunk_indexes.append(
                    ChunkIndex(
                        chunk_length=0,  # TODO
                        chunk_start_offset=0,  # TODO
                        compression=record.compression,
                        compressed_size=len(record.data)
                        if isinstance(record, Chunk)
                        else record.data_len,
                        message_end_time=record.message_end_time,
                        message_index_length=0,  # TODO
                        message_index_offsets={},
                        message_start_time=record.message_start_time,
                        uncompressed_size=record.uncompressed_size,
                    )
                )
                statistics.chunk_count += 1
                statistics.message_start_time = (
                    min(statistics.message_start_time, record.message_start_time)
                    if statistics.message_start_time
                    else record.message_start_time
                )
                statistics.message_end_time = max(
                    statistics.message_end_time, record.message_end_time
                )

            elif isinstance(record, MessageIndex):
                statistics.message_count += len(record.records)
                statistics.channel_message_counts[record.channel_id] = (
                    statistics.channel_message_counts.get(record.channel_id, 0)
                    + len(record.records)
                )
                last_chunk_message_indexes.append(record)

            elif isinstance(record, Statistics):
                pass  # We calculate statistics ourselves
            elif isinstance(record, DataEnd):
                break
            elif isinstance(record, Attachment):
                statistics.attachment_count += 1
            elif isinstance(record, AttachmentIndex):
                summary.attachment_indexes.append(record)
            elif isinstance(record, Metadata):
                statistics.metadata_count += 1
            elif isinstance(record, MetadataIndex):
                summary.metadata_indexes.append(record)
            else:
                raise McapError(f"Unexpected record type: {type(record)}")

        # Ensure progress completes
        progress.update(task, completed=file_size, visible=False)

    summary.statistics = statistics
    assert header is not None, "Header should not be None"
    return Info(header=header, summary=summary, channel_sizes=channel_sizes)


def read_info(f: BinaryIO) -> Info:
    header = get_header(f)
    summary = get_summary(f)
    assert summary is not None, "Summary should not be None"
    return Info(header=header, summary=summary)
