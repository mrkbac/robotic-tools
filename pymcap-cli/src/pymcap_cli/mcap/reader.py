import io
import itertools
import struct
from collections.abc import Callable, Generator, Iterator
from typing import IO

from mcap.data_stream import ReadDataStream
from mcap.exceptions import EndOfFile, InvalidMagic, RecordLengthLimitExceeded
from mcap.opcode import Opcode
from mcap.reader import FOOTER_SIZE
from mcap.records import (
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
    Schema,
    Statistics,
    SummaryOffset,
)
from mcap.stream_reader import MAGIC_SIZE, breakup_chunk
from mcap.summary import Summary


def _read_record(_stream: ReadDataStream, opcode: int, length: int) -> McapRecord | None:
    if opcode == Opcode.ATTACHMENT:
        return Attachment.read(_stream)
    if opcode == Opcode.ATTACHMENT_INDEX:
        return AttachmentIndex.read(_stream)
    if opcode == Opcode.CHANNEL:
        return Channel.read(_stream)
    if opcode == Opcode.CHUNK:
        return Chunk.read(_stream)
    if opcode == Opcode.CHUNK_INDEX:
        return ChunkIndex.read(_stream)
    if opcode == Opcode.DATA_END:
        return DataEnd.read(_stream)
    if opcode == Opcode.FOOTER:
        return Footer.read(_stream)
    if opcode == Opcode.HEADER:
        return Header.read(_stream)
    if opcode == Opcode.MESSAGE:
        return Message.read(_stream, length)
    if opcode == Opcode.MESSAGE_INDEX:
        return MessageIndex.read(_stream)
    if opcode == Opcode.METADATA:
        return Metadata.read(_stream)
    if opcode == Opcode.METADATA_INDEX:
        return MetadataIndex.read(_stream)
    if opcode == Opcode.SCHEMA:
        return Schema.read(_stream)
    if opcode == Opcode.STATISTICS:
        return Statistics.read(_stream)
    if opcode == Opcode.SUMMARY_OFFSET:
        return SummaryOffset.read(_stream)

    # Skip unknown record types
    _stream.read(length)
    return None


def _read_magic(stream: ReadDataStream) -> bool:
    magic = struct.unpack("<8B", stream.read(MAGIC_SIZE))
    if magic != (137, 77, 67, 65, 80, 48, 13, 10):
        raise InvalidMagic(magic)
    return True


def stream_reader(
    input: io.BytesIO | io.RawIOBase | io.BufferedReader | IO[bytes],
    skip_magic: bool = False,
    emit_chunks: bool = False,
    record_size_limit: int | None = (4 * 2**30),  # 4 Gib
) -> Generator[McapRecord, None, None]:
    if isinstance(input, io.RawIOBase):
        stream = ReadDataStream(io.BufferedReader(input))
    else:
        stream = ReadDataStream(input)
    footer: Footer | None = None

    if not skip_magic:
        _read_magic(stream)

    while footer is None:
        opcode = stream.read1()
        length = stream.read8()
        if record_size_limit is not None and length > record_size_limit:
            raise RecordLengthLimitExceeded(opcode, length, record_size_limit)
        count = stream.count
        record = _read_record(stream, opcode, length)

        padding = length - (stream.count - count)
        if padding > 0:
            stream.read(padding)
        if isinstance(record, Chunk) and not emit_chunks:
            chunk_records = breakup_chunk(record)
            yield from chunk_records
        elif record:
            yield record
        if isinstance(record, Footer):
            footer = record
            _read_magic(stream)


def _read_summary_from_stream_reader(
    stream_reader: Generator[McapRecord, None, None],
) -> Summary | None:
    """read summary records from an MCAP stream reader, collecting them into a Summary."""
    summary = Summary()
    for record in stream_reader:
        if isinstance(record, Statistics):
            summary.statistics = record
        elif isinstance(record, Schema):
            summary.schemas[record.id] = record
        elif isinstance(record, Channel):
            summary.channels[record.id] = record
        elif isinstance(record, AttachmentIndex):
            summary.attachment_indexes.append(record)
        elif isinstance(record, ChunkIndex):
            summary.chunk_indexes.append(record)
        elif isinstance(record, MetadataIndex):
            summary.metadata_indexes.append(record)
        elif isinstance(record, Footer):
            # There is no summary!
            if record.summary_start == 0:
                return None
            return summary
    return summary


def _get_summary(stream: IO[bytes]) -> Summary | None:
    """Get the start and end indexes of each chunk in the stream."""
    if not stream.seekable():
        return None
    try:
        stream.seek(-(FOOTER_SIZE + MAGIC_SIZE), io.SEEK_END)
        footer = next(stream_reader(stream, skip_magic=True, record_size_limit=None))
        if not isinstance(footer, Footer):
            return None
        if footer.summary_start == 0:
            return None
        stream.seek(footer.summary_start, io.SEEK_SET)
        return _read_summary_from_stream_reader(
            stream_reader(stream, skip_magic=True, record_size_limit=None)
        )
    except (OSError, StopIteration, EndOfFile):
        return None


def _chunks_matching_topics(
    summary: Summary,
    should_include: Callable[[Channel], bool] | None,
    start_time: float | None,
    end_time: float | None,
) -> list[ChunkIndex]:
    channel_set: set[int] | None = None
    if should_include:
        channel_set = {
            channel.id for channel in summary.channels.values() if should_include(channel)
        }

    return [
        chunk_index
        for chunk_index in summary.chunk_indexes
        if not (start_time is not None and chunk_index.message_end_time < start_time)
        and not (end_time is not None and chunk_index.message_start_time >= end_time)
        and any(
            channel_set is None or channel_id in channel_set
            for channel_id in chunk_index.message_index_offsets
        )
    ]


def _read_inner(
    reader: Iterator[McapRecord],
    should_include: Callable[[Channel], bool] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Generator[McapRecord, None, None]:
    include: set[int] = set()

    for rec in reader:
        if isinstance(rec, Channel) and should_include and should_include(rec):
            include.add(rec.id)
        elif isinstance(rec, Message) and should_include and rec.channel_id not in include:
            continue
        elif isinstance(rec, Chunk):
            if start_time and rec.message_start_time < start_time:
                continue
            if end_time and rec.message_end_time >= end_time:
                continue

        yield rec


def _read_message_seeking(
    stream: IO[bytes],
    should_include: Callable[[Channel], bool] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    emit_chunks: bool = False,
) -> Generator[McapRecord, None, None]:
    summary = _get_summary(stream)
    # No summary or chunk indexes exists
    if summary is None or not summary.chunk_indexes:
        # seek to start
        stream.seek(0, io.SEEK_SET)
        yield from _read_message_non_seeking(stream, should_include, start_time, end_time)
        return

    chunk_indexes = _chunks_matching_topics(summary, should_include, start_time, end_time)

    def reader() -> Generator[McapRecord, None, None]:
        # yield all schemas and channels
        yield from summary.schemas.values()
        yield from summary.channels.values()
        for cidx in chunk_indexes:
            if emit_chunks:
                stream.seek(cidx.chunk_start_offset, io.SEEK_SET)
                data = stream.read(cidx.chunk_length + cidx.message_index_length)
                reader = stream_reader(io.BytesIO(data), skip_magic=True, emit_chunks=emit_chunks)
                yield from itertools.islice(reader, len(cidx.message_index_offsets))
            else:
                stream.seek(cidx.chunk_start_offset + 1 + 8, io.SEEK_SET)
                yield from breakup_chunk(Chunk.read(ReadDataStream(stream)))

    yield from _read_inner(reader(), should_include, start_time, end_time)


def _read_message_non_seeking(
    stream: IO[bytes],
    should_include: Callable[[Channel], bool] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    emit_chunks: bool = False,
) -> Generator[McapRecord, None, None]:
    reader = stream_reader(stream, emit_chunks=emit_chunks)

    yield from _read_inner(reader, should_include, start_time, end_time)


def read_message(
    stream: IO[bytes],
    should_include: Callable[[Channel], bool] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    emit_chunks: bool = False,
) -> Generator[McapRecord, None, None]:
    if stream.seekable():
        yield from _read_message_seeking(stream, should_include, start_time, end_time, emit_chunks)
    else:
        yield from _read_message_non_seeking(
            stream, should_include, start_time, end_time, emit_chunks
        )
