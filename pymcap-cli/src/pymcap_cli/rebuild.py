import itertools
from dataclasses import dataclass
from pathlib import Path

from mcap.exceptions import McapError
from mcap.reader import make_reader
from mcap.records import (
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Header,
    Message,
    MessageIndex,
    Schema,
    Statistics,
)
from mcap.stream_reader import breakup_chunk
from mcap.summary import Summary
from rich.console import Console

from pymcap_cli.mcap.reader import stream_reader
from pymcap_cli.utils import file_progress


@dataclass
class Info:
    header: Header
    summary: Summary
    channel_sizes: dict[int, int] | None = None


console = Console()  # TODO improve


def _estimate_size_from_indexes(indexes: list[MessageIndex], chunk_size: int) -> dict[int, int]:
    # channel, offset
    idx_list = []
    for idx in indexes:
        for _time, offset in idx.records:
            idx_list.append((idx.channel_id, offset))
    sorted_idx_by_off = sorted(idx_list, key=lambda x: x[1])
    sorted_idx_by_off.append((None, chunk_size))

    sizes: dict[int, int] = {}
    for cur, (_, end_offset) in itertools.pairwise(sorted_idx_by_off):
        channel, start_offset = cur
        size = (end_offset - start_offset) - (2 + 4 + 8 + 8)
        assert size > 0, f"Invalid size for channel {channel}: {size}"
        sizes[channel] = sizes.get(channel, 0) + size

    return sizes


def rebuild_info(file: Path, *, exact_sizes: bool = False) -> Info:
    file_size = file.stat().st_size

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

    last_chunk = None
    last_chunk_message_indexes: list[MessageIndex] = []

    with (
        file_progress("[bold blue]Rebuilding MCAP info...", console) as progress,
        file.open("rb") as f,
    ):
        task = progress.add_task("Processing", total=file_size)

        reader = stream_reader(f, skip_magic=False, emit_chunks=not exact_sizes)

        for record in reader:
            progress.update(task, completed=f.tell())

            if not isinstance(record, MessageIndex) and last_chunk:
                decode = False
                for idx in last_chunk_message_indexes:
                    if idx.channel_id not in summary.channels:
                        decode = True
                        break

                if decode:
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
            elif isinstance(record, Chunk):
                last_chunk = record

                summary.chunk_indexes.append(
                    ChunkIndex(
                        chunk_length=0,  # TODO
                        chunk_start_offset=0,  # TODO
                        compression=record.compression,
                        compressed_size=len(record.data),
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
            else:
                raise McapError(f"Unexpected record type: {type(record)}")

        # Ensure progress completes
        progress.update(task, completed=file_size, visible=False)

    summary.statistics = statistics
    assert header is not None, "Header should not be None"
    return Info(header=header, summary=summary, channel_sizes=channel_sizes)


def read_info(file: Path) -> Info:
    with file.open("rb") as f:
        reader = make_reader(f)
        header = reader.get_header()
        summary = reader.get_summary()
        assert summary is not None, "Summary should not be None"
        return Info(header=header, summary=summary)
