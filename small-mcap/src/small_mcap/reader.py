import bisect
import contextlib
import heapq
import io
import itertools
import sys
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import IO, TYPE_CHECKING, Any, Literal, Protocol, cast, overload

from small_mcap.records import (
    MAGIC,
    MAGIC_SIZE,
    OPCODE_AND_LEN_STRUCT,
    OPCODE_TO_RECORD,
    Attachment,
    AttachmentIndex,
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Footer,
    Header,
    LazyChunk,
    McapRecord,
    Message,
    MessageIndex,
    Metadata,
    MetadataIndex,
    Opcode,
    Schema,
    Statistics,
    Summary,
    SummaryOffset,
)

if TYPE_CHECKING:
    from lz4.frame import decompress as lz4_decompress  # type: ignore[import-untyped]
    from zstandard import ZstdDecompressor
else:
    try:
        from zstandard import ZstdDecompressor
    except ImportError:
        ZstdDecompressor = None  # type: ignore[assignment,misc]

    try:
        from lz4.frame import decompress as lz4_decompress
    except ImportError:
        lz4_decompress = None  # type: ignore[assignment]

_zstd_decompressor: "ZstdDecompressor | None" = None

_OPCODE_SIZE = 1
_RECORD_LENGTH_SIZE = 8
_RECORD_HEADER_SIZE = _OPCODE_SIZE + _RECORD_LENGTH_SIZE
_FOOTER_SIZE = _RECORD_HEADER_SIZE + 8 + 8 + 4

# Limits and defaults
_RECORD_SIZE_LIMIT = 4 * 2**30  # 4 GiB - maximum size for a single record

# Type aliases for stream_reader return types
# Records that come from inside chunks when emit_chunks=False
ChunkContentRecord = Channel | Message | Schema

# Records that are never filtered (except Chunk and MessageIndex which are conditional)
NonChunkRecord = (
    Header
    | Attachment
    | AttachmentIndex
    | ChunkIndex
    | DataEnd
    | Footer
    | Metadata
    | MetadataIndex
    | Statistics
    | SummaryOffset
)


_ReaderReturnType = Iterable[tuple[Schema | None, Channel, Message]]
_ShouldIncludeType = Callable[[Channel, Schema | None], bool]


class McapError(Exception):
    pass


class InvalidMagicError(McapError):
    def __init__(self, bad_magic: bytes | memoryview) -> None:
        super().__init__(
            f"not a valid MCAP file, invalid magic: {bytes(bad_magic).decode('utf-8', 'replace')}"
        )


class EndOfFileError(McapError):
    pass


class CRCValidationError(McapError):
    def __init__(self, expected: int, actual: int, record: McapRecord) -> None:
        super().__init__(
            f"crc validation failed in {type(record).__name__}, "
            f"expected: {expected}, calculated: {actual}"
        )


class RecordLengthLimitExceededError(McapError):
    def __init__(self, opcode: int, length: int, limit: int) -> None:
        opcode_name = f"unknown (opcode {opcode})"
        with contextlib.suppress(ValueError):
            opcode_name = Opcode(opcode).name
        super().__init__(
            f"{opcode_name} record has length {length} that exceeds limit {limit}",
        )


class UnsupportedCompressionError(McapError):
    def __init__(self, compression: str) -> None:
        super().__init__(f"unsupported compression type {compression}")


def _get_chunk_data_stream(chunk: Chunk, validate_crc: bool = False) -> bytes | memoryview:
    # Validate compression string
    if not isinstance(chunk.compression, str):
        raise UnsupportedCompressionError(
            f"compression must be a string, got {type(chunk.compression).__name__}"
        )

    data: bytes | memoryview
    if chunk.compression == "zstd":
        if ZstdDecompressor is None:
            raise UnsupportedCompressionError(
                "zstd compression used but zstandard module is not installed. "
                "Install it with: pip install zstandard"
            )
        global _zstd_decompressor  # noqa: PLW0603
        if _zstd_decompressor is None:
            _zstd_decompressor = ZstdDecompressor()
        data = _zstd_decompressor.decompress(chunk.data, max_output_size=chunk.uncompressed_size)
    elif chunk.compression == "lz4":
        if lz4_decompress is None:
            raise UnsupportedCompressionError(
                "lz4 compression used but lz4 module is not installed. "
                "Install it with: pip install lz4"
            )
        data = lz4_decompress(chunk.data)
    elif chunk.compression == "":
        data = chunk.data
    else:
        raise UnsupportedCompressionError(
            f"Unknown compression type '{chunk.compression}'. "
            f"Supported types: 'zstd', 'lz4', '' (uncompressed)"
        )

    if validate_crc and chunk.uncompressed_crc != 0:
        calculated_crc = zlib.crc32(data)
        if calculated_crc != chunk.uncompressed_crc:
            raise CRCValidationError(
                expected=chunk.uncompressed_crc,
                actual=calculated_crc,
                record=chunk,
            )

    return data


def breakup_chunk(chunk: Chunk, validate_crc: bool = False) -> Iterable[McapRecord]:
    data = _get_chunk_data_stream(chunk, validate_crc=validate_crc)
    view = memoryview(data)
    pos = 0

    while pos < len(view):
        opcode, length = OPCODE_AND_LEN_STRUCT.unpack_from(view, pos)
        pos += _RECORD_HEADER_SIZE
        record_data_end = pos + length

        if opcode == Opcode.MESSAGE:
            yield Message.read(view[pos:record_data_end])
        elif opcode == Opcode.CHANNEL:
            yield Channel.read(view[pos:record_data_end])
        elif opcode == Opcode.SCHEMA:
            yield Schema.read(view[pos:record_data_end])
        # TODO: raise illegal opcode in chunk error

        pos = record_data_end


def _breakup_chunk_with_indexes(
    chunk: Chunk, message_indexes: Iterable[MessageIndex], validate_crc: bool = False
) -> Iterable[McapRecord]:
    # materialize for truthy emptiness check
    message_indexes = list(message_indexes)
    if not message_indexes:
        return

    data = _get_chunk_data_stream(chunk, validate_crc=validate_crc)
    view = memoryview(data)

    for _timestamp, offset in heapq.merge(
        # sort by time
        *(x.records for x in message_indexes),
        key=lambda mi: mi[0],
    ):
        pos = offset
        opcode, length = OPCODE_AND_LEN_STRUCT.unpack_from(view, pos)
        pos += _RECORD_HEADER_SIZE
        record_data_end = pos + length
        if opcode == Opcode.MESSAGE:
            yield Message.read(view[pos:record_data_end])
        # TODO: raise illegal opcode in chunk error


def _read_chunk_and_indexes(data: bytes) -> tuple[Chunk, list[MessageIndex]]:
    view = memoryview(data)
    # Read the chunk record header (opcode + length)
    opcode, chunk_length = OPCODE_AND_LEN_STRUCT.unpack_from(view, 0)
    pos = _RECORD_HEADER_SIZE

    chunk = Chunk.read(view[pos : pos + chunk_length])
    pos += chunk_length

    # Read message index records that follow the chunk
    message_indexes: list[MessageIndex] = []
    while pos < len(view):
        opcode, length = OPCODE_AND_LEN_STRUCT.unpack_from(view, pos)
        pos += _RECORD_HEADER_SIZE
        record_data_end = pos + length

        if opcode == Opcode.MESSAGE_INDEX:
            message_index = MessageIndex.read(view[pos:record_data_end])
            message_indexes.append(message_index)

        pos = record_data_end

    return chunk, message_indexes


@overload
def stream_reader(
    stream: IO[bytes],
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: Literal[True] = ...,
    lazy_chunks: Literal[True] = ...,
) -> Iterable[NonChunkRecord | LazyChunk | MessageIndex]: ...


@overload
def stream_reader(
    stream: IO[bytes],
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: Literal[True] = ...,
    lazy_chunks: Literal[False] = ...,
) -> Iterable[NonChunkRecord | Chunk | MessageIndex]: ...


@overload
def stream_reader(
    stream: IO[bytes],
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: Literal[False] = ...,
    lazy_chunks: Literal[False] = ...,
) -> Iterable[NonChunkRecord | ChunkContentRecord]: ...


def stream_reader(
    stream: IO[bytes],
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: bool = False,
    lazy_chunks: bool = False,
) -> Iterable[McapRecord] | Iterable[McapRecord | LazyChunk]:
    record_size_limit = _RECORD_SIZE_LIMIT
    checksum = 0

    def read(n: int) -> memoryview:
        data = stream.read(n)
        if len(data) < n:
            raise EndOfFileError
        if validate_crc and not skip_magic and not lazy_chunks:
            nonlocal checksum
            checksum = zlib.crc32(data, checksum)
        return memoryview(data)

    cached_pos = stream.tell()

    if not skip_magic:
        magic = read(MAGIC_SIZE)
        if magic != MAGIC:
            raise InvalidMagicError(magic)
        cached_pos += MAGIC_SIZE

    while True:
        checksum_before_read = checksum
        opcode, length = OPCODE_AND_LEN_STRUCT.unpack(read(_RECORD_HEADER_SIZE))
        if record_size_limit is not None and length > record_size_limit:
            raise RecordLengthLimitExceededError(opcode, length, record_size_limit)

        record_start = cached_pos
        cached_pos += _RECORD_HEADER_SIZE

        # Handle lazy chunk loading when requested
        record: McapRecord | LazyChunk | None
        if opcode == Opcode.CHUNK and emit_chunks and lazy_chunks:
            record = LazyChunk.read_from_stream(stream, record_start)
            cached_pos = stream.tell()
        else:
            cached_pos += length
            record_data = read(length)

            if record_cls := OPCODE_TO_RECORD.get(opcode):
                record = record_cls.read(record_data)
            else:
                record = None  # Unknown record type, skip it.

        if (
            validate_crc
            and not skip_magic
            and isinstance(record, DataEnd)
            and record.data_section_crc not in (0, checksum_before_read)
        ):
            raise CRCValidationError(
                expected=record.data_section_crc,
                actual=checksum_before_read,
                record=record,
            )

        # Handle padding (only needed when not using lazy chunks, as lazy read seeks past data)
        if not (opcode == Opcode.CHUNK and emit_chunks and lazy_chunks):
            padding = length - (cached_pos - record_start)
            if padding > 0:
                read(padding)
                cached_pos += padding

        if isinstance(record, Chunk) and not emit_chunks:
            chunk_records = breakup_chunk(record, validate_crc)
            yield from chunk_records
        elif record:
            # When breaking up chunks (emit_chunks=False), skip MessageIndex records
            # as they are metadata about messages in chunks that we're already yielding directly
            if not emit_chunks and isinstance(record, MessageIndex):
                continue
            yield record

        if isinstance(record, Footer):
            if not skip_magic:
                magic = read(MAGIC_SIZE)
                if magic != MAGIC:
                    raise InvalidMagicError(magic)
            break


def _read_summary_from_iterable(stream_reader: Iterable[McapRecord | LazyChunk]) -> Summary | None:
    """read summary records from an MCAP stream reader, collecting them into a Summary."""
    summary = Summary()
    for record in stream_reader:
        if isinstance(record, ChunkIndex):
            summary.chunk_indexes.append(record)
        elif isinstance(record, Channel):
            summary.channels[record.id] = record
        elif isinstance(record, Schema):
            summary.schemas[record.id] = record
        elif isinstance(record, AttachmentIndex):
            summary.attachment_indexes.append(record)
        elif isinstance(record, MetadataIndex):
            summary.metadata_indexes.append(record)
        elif isinstance(record, Statistics):
            summary.statistics = record
        elif isinstance(record, Footer):
            # There is no summary!
            if record.summary_start == 0:
                return None
            return summary
    return summary


def get_summary(stream: IO[bytes]) -> Summary | None:
    """Get the start and end indexes of each chunk in the stream."""
    if not stream.seekable():
        return None
    try:
        stream.seek(-MAGIC_SIZE, io.SEEK_END)
        magic = stream.read(MAGIC_SIZE)
        if magic != MAGIC:
            raise InvalidMagicError(magic)
        stream.seek(-(_FOOTER_SIZE + MAGIC_SIZE), io.SEEK_END)
        footer = next(iter(stream_reader(stream, skip_magic=True)))
        if not isinstance(footer, Footer):
            return None
        if footer.summary_start == 0:
            return None
        stream.seek(footer.summary_start, io.SEEK_SET)
        return _read_summary_from_iterable(stream_reader(stream, skip_magic=True))
    except (OSError, StopIteration, EndOfFileError):
        return None


def get_header(stream: IO[bytes]) -> Header:
    if stream.seekable():
        stream.seek(0, io.SEEK_SET)

    header = next(iter(stream_reader(stream, skip_magic=False)))
    if not isinstance(header, Header):
        raise McapError(f"expected header at beginning of MCAP file, found {type(header)}")
    return header


def _read_inner(
    reader: Iterable[McapRecord],
    should_include: _ShouldIncludeType,
    exclude_channels: set[int],
    start_time_ns: int,
    end_time_ns: int,
    schemas: dict[int, Schema] | None = None,
    channels: dict[int, Channel] | None = None,
) -> _ReaderReturnType:
    _schemas: dict[int, Schema] = schemas or {}
    _channels: dict[int, Channel] = channels or {}

    for record in reader:
        if isinstance(record, Message):
            if record.channel_id not in _channels:
                raise McapError(f"no channel record found with id {record.channel_id}")
            if (
                (record.channel_id in exclude_channels)
                or (record.log_time < start_time_ns)
                or (record.log_time >= end_time_ns)
            ):
                continue
            channel = _channels[record.channel_id]
            schema = _schemas.get(channel.schema_id)
            yield (schema, channel, record)
        elif isinstance(record, Schema):
            _schemas[record.id] = record
        elif isinstance(record, Channel) and record.id not in _channels:  # New channel
            if record.schema_id != 0 and record.schema_id not in _schemas:
                raise McapError(f"no schema record found with id {record.schema_id}")
            _channels[record.id] = record
            if not should_include(_channels[record.id], _schemas.get(record.schema_id)):
                exclude_channels.add(record.id)


def _filter_message_index_by_time(
    message_index: MessageIndex,
    start_time_ns: int,
    end_time_ns: int,
) -> MessageIndex:
    """Filter a MessageIndex to only include records within the time range using binary search.

    Args:
        message_index: The MessageIndex to filter
        start_time_ns: Start time (inclusive) in nanoseconds
        end_time_ns: End time (exclusive) in nanoseconds

    Returns:
        A new MessageIndex with filtered records, or the original if no filtering needed
    """
    if not message_index.records:
        return message_index

    # Check if we need to filter at all
    first_time = message_index.records[0][0]
    last_time = message_index.records[-1][0]

    if first_time >= start_time_ns and last_time < end_time_ns:
        # All records are within range, no filtering needed
        return message_index

    if last_time < start_time_ns or first_time >= end_time_ns:
        # No records are within range
        return MessageIndex(message_index.channel_id, [])

    # Binary search for start index (first record >= start_time_ns)
    start_idx = bisect.bisect_left(
        message_index.records, start_time_ns, key=lambda record: record[0]
    )

    # Binary search for end index (first record >= end_time_ns)
    end_idx = bisect.bisect_left(message_index.records, end_time_ns, key=lambda record: record[0])

    # Return filtered MessageIndex
    return MessageIndex(message_index.channel_id, message_index.records[start_idx:end_idx])


def _filter_message_indices_by_time(
    message_index: Iterable[MessageIndex],
    start_time_ns: int,
    end_time_ns: int,
) -> Iterable[MessageIndex]:
    if start_time_ns == 0 and end_time_ns == sys.maxsize:
        yield from message_index
        return

    for mi in message_index:
        filtered_mi = _filter_message_index_by_time(mi, start_time_ns, end_time_ns)
        if filtered_mi.records:
            yield filtered_mi


def _read_message_seeking(
    stream: IO[bytes],
    should_include: _ShouldIncludeType,
    start_time_ns: int,
    end_time_ns: int,
    validate_crc: bool,
    reverse: bool,
) -> _ReaderReturnType:
    summary = get_summary(stream)
    # No summary or chunk indexes exists
    if summary is None or not summary.chunk_indexes:
        # seek to start
        stream.seek(0, io.SEEK_SET)
        yield from _read_message_non_seeking(
            stream, should_include, start_time_ns, end_time_ns, validate_crc, reverse
        )
        return
    exclude_channels: set[int] = {
        channel.id
        for channel in summary.channels.values()
        if not should_include(channel, summary.schemas.get(channel.schema_id))
    }

    def _lazy_yield(stream: IO[bytes], index: ChunkIndex) -> Iterable[McapRecord | ChunkIndex]:
        # Emit the chunk index first to enable lazy loading - this allows heapq.merge
        # to initialize by looking at just metadata without loading all chunks into memory
        yield index

        # late check for mcap with empty channel summary if this chunks is excluded entirely
        if index.message_index_offsets and exclude_channels.issuperset(index.message_index_offsets):
            return

        stream.seek(index.chunk_start_offset)
        chunk_and_message_index = stream.read(index.chunk_length + index.message_index_length)
        chunk, message_indexes = _read_chunk_and_indexes(chunk_and_message_index)

        if message_indexes:
            # Filter message indexes to only include non-excluded channels
            filtered_message_indexes: Iterable[MessageIndex] = (
                mi for mi in message_indexes if mi.channel_id not in exclude_channels
            )
            filtered_message_indexes = _filter_message_indices_by_time(
                filtered_message_indexes, start_time_ns, end_time_ns
            )

            yield from _breakup_chunk_with_indexes(chunk, filtered_message_indexes, validate_crc)
        else:
            yield from breakup_chunk(chunk, validate_crc=validate_crc)

    # Filter and sort chunks that overlap with the time range
    sorted_chunks = sorted(
        (
            cidx
            for cidx in summary.chunk_indexes
            if not (
                cidx.message_index_offsets
                and exclude_channels.issuperset(cidx.message_index_offsets)
            )
            and cidx.message_end_time >= start_time_ns
            and cidx.message_start_time < end_time_ns
        ),
        key=lambda c: (-c.message_end_time if reverse else c.message_start_time),
    )

    # Check if chunks are non-overlapping
    # This is a common case for well-formed MCAP files
    if reverse:
        # In reverse: chunks sorted by end_time descending
        # Non-overlapping means prev starts >= current ends
        chunks_non_overlapping = all(
            prev.message_start_time >= current.message_end_time
            for prev, current in itertools.pairwise(sorted_chunks)
        )
    else:
        # In forward: chunks sorted by start_time ascending
        # Non-overlapping means prev ends <= current starts
        chunks_non_overlapping = all(
            prev.message_end_time <= current.message_start_time
            for prev, current in itertools.pairwise(sorted_chunks)
        )

    def _lazy_sort(item: McapRecord) -> int:
        if isinstance(item, Message):
            return -item.log_time if reverse else item.log_time
        if isinstance(item, ChunkIndex):
            # Use message_end_time for reverse (last message time)
            # Use message_start_time for forward (first message time)
            time = item.message_end_time if reverse else item.message_start_time
            return -time if reverse else time
        return 0  # Schema and Channel records should come before messages

    lazy_iterables = (_lazy_yield(stream, cidx) for cidx in sorted_chunks)

    # If chunks don't overlap, we can yield sequentially without heap merging
    # This is more efficient as it avoids the heap overhead
    reader: Iterable[McapRecord]
    if chunks_non_overlapping:
        # Chunks are ordered, no need for heap merge
        reader = itertools.chain.from_iterable(lazy_iterables)
    else:
        # Chunks overlap, use heap merge to maintain time order
        reader = heapq.merge(*lazy_iterables, key=_lazy_sort)

    yield from _read_inner(
        reader,
        should_include,
        exclude_channels,
        start_time_ns,
        end_time_ns,
        summary.schemas,
        summary.channels,
    )


def _read_message_non_seeking(
    stream: IO[bytes],
    should_include: _ShouldIncludeType,
    start_time_ns: int,
    end_time_ns: int,
    validate_crc: bool,
    reverse: bool,
) -> _ReaderReturnType:
    if reverse:
        raise McapError("reverse=True is not supported for non-seekable streams.")

    exclude_channels: set[int] = set()
    seen_channels: set[int] = set()

    def _inner() -> Iterable[McapRecord]:
        pending_chunk: Chunk | None = None
        pending_message_indexes: list[MessageIndex] = []
        for record in stream_reader(
            stream, emit_chunks=True, validate_crc=validate_crc, lazy_chunks=False
        ):
            if not isinstance(record, MessageIndex) and pending_chunk:
                in_time_range = (
                    pending_chunk.message_start_time < end_time_ns
                    and pending_chunk.message_end_time >= start_time_ns
                )
                if in_time_range:
                    if pending_message_indexes:
                        channel_ids = {msg_idx.channel_id for msg_idx in pending_message_indexes}
                        new_channels = bool(channel_ids - seen_channels)
                        all_excluded = channel_ids.issubset(exclude_channels)
                        if new_channels or not all_excluded:
                            # Filter message indexes to only include non-excluded channels
                            filtered_message_indexes: Iterable[MessageIndex] = (
                                mi
                                for mi in pending_message_indexes
                                if mi.channel_id not in exclude_channels
                            )

                            filtered_message_indexes = _filter_message_indices_by_time(
                                filtered_message_indexes, start_time_ns, end_time_ns
                            )

                            yield from _breakup_chunk_with_indexes(
                                pending_chunk, filtered_message_indexes, validate_crc
                            )
                    else:
                        yield from breakup_chunk(pending_chunk, validate_crc=validate_crc)
                pending_chunk = None
                pending_message_indexes.clear()

            if isinstance(record, Chunk):
                pending_chunk = record
            elif isinstance(record, MessageIndex):
                # Ignore empty MessageIndex records
                if record.records:
                    pending_message_indexes.append(record)
            else:
                yield record

        if pending_chunk:
            in_time_range = (
                pending_chunk.message_start_time < end_time_ns
                and pending_chunk.message_end_time >= start_time_ns
            )
            if in_time_range:
                # Final chunks must always be decompressed fully since we can't ensure we
                # got all message indexes
                yield from breakup_chunk(pending_chunk, validate_crc=validate_crc)

    def _should_include_wrapper(channel: Channel, schema: Schema | None) -> bool:
        seen_channels.add(channel.id)
        return should_include(channel, schema)

    yield from _read_inner(
        _inner(),
        _should_include_wrapper,
        exclude_channels,
        start_time_ns,
        end_time_ns,
    )


class Remapper:
    """Smart ID remapper that minimizes remapping by preserving original IDs when possible.

    Tracks schema and channel IDs across multiple streams, only remapping when conflicts occur.
    Deduplicates identical schemas/channels by content to avoid duplicates.
    """

    def __init__(self, summaries: Iterable[tuple[int, Summary]] | None = None) -> None:
        self._used_schema_ids: set[int] = set()
        self._used_channel_ids: set[int] = set()

        self._schema_lookup_fast: dict[tuple[int, int], Schema | None] = {}
        self._channel_lookup_fast: dict[tuple[int, int], Channel] = {}
        self._schema_lookup_slow: dict[tuple[str, bytes], Schema | None] = {}
        self._channel_lookup_slow: dict[tuple[str, str], Channel] = {}

        # Track which IDs were actually remapped for fast-copy optimization
        self._remapped_schemas: set[tuple[int, int]] = set()  # (stream_id, original_id)
        self._remapped_channels: set[tuple[int, int]] = set()  # (stream_id, original_id)

        if summaries is not None:
            for stream_id, summary in summaries:
                for schema in summary.schemas.values():
                    self.remap_schema(stream_id, schema)
                for channel in summary.channels.values():
                    self.remap_channel(stream_id, channel)

    def remap_schema(self, stream_id: int, schema: Schema | None) -> Schema | None:
        if schema is None:
            return None

        # Fast path: lookup by stream_id + schema.id
        fast_key = (stream_id, schema.id)
        mapped_schema = self._schema_lookup_fast.get(fast_key)
        if mapped_schema is not None:
            return mapped_schema

        # Slow path: lookup by schema content (deduplication)
        slow_key = (schema.name, schema.data)
        mapped_schema = self._schema_lookup_slow.get(slow_key)
        if mapped_schema:
            # Cache in fast lookup for future access
            self._schema_lookup_fast[fast_key] = mapped_schema
            # Track if ID changed
            if mapped_schema.id != schema.id:
                self._remapped_schemas.add(fast_key)
            return mapped_schema

        # Try to preserve original ID if not in use
        new_id = schema.id
        if new_id in self._used_schema_ids:
            # ID conflict - must remap
            while new_id in self._used_schema_ids:
                new_id += 1
            self._remapped_schemas.add(fast_key)

        self._used_schema_ids.add(new_id)
        mapped_schema = replace(schema, id=new_id)
        self._schema_lookup_slow[slow_key] = mapped_schema
        self._schema_lookup_fast[fast_key] = mapped_schema
        return mapped_schema

    def remap_channel(self, stream_id: int, channel: Channel) -> Channel:
        # Fast path: lookup by stream_id + channel.id
        fast_key = (stream_id, channel.id)
        mapped_channel = self._channel_lookup_fast.get(fast_key)
        if mapped_channel is not None:
            return mapped_channel

        # Slow path: lookup by channel content (deduplication)
        slow_key = (
            channel.topic,
            channel.message_encoding,
        )  # TODO: include metadata
        mapped_channel = self._channel_lookup_slow.get(slow_key)
        if mapped_channel is not None:
            # Cache in fast lookup for future access
            self._channel_lookup_fast[fast_key] = mapped_channel
            # Track if ID changed
            if mapped_channel.id != channel.id:
                self._remapped_channels.add(fast_key)
            return mapped_channel

        # Try to preserve original ID if not in use
        new_id = channel.id
        if new_id in self._used_channel_ids:
            # ID conflict - must remap
            while new_id in self._used_channel_ids:
                new_id += 1
            self._remapped_channels.add(fast_key)

        self._used_channel_ids.add(new_id)
        mapped_channel = replace(channel, id=new_id, schema_id=0)
        self._channel_lookup_slow[slow_key] = mapped_channel
        self._channel_lookup_fast[fast_key] = mapped_channel
        return mapped_channel

    def was_schema_remapped(self, stream_id: int, original_id: int) -> bool:
        """Check if a schema ID was changed during remapping."""
        return (stream_id, original_id) in self._remapped_schemas

    def was_channel_remapped(self, stream_id: int, original_id: int) -> bool:
        """Check if a channel ID was changed during remapping."""
        return (stream_id, original_id) in self._remapped_channels

    def has_channel(self, stream_id: int, original_id: int) -> bool:
        """Check if a channel has been seen and mapped."""
        return (stream_id, original_id) in self._channel_lookup_fast

    def get_remapped_channel(self, stream_id: int, original_id: int) -> Channel | None:
        """Get the remapped channel for a given stream and original ID."""
        return self._channel_lookup_fast.get((stream_id, original_id))

    def get_remapped_schema(self, stream_id: int, original_schema_id: int) -> int:
        """Get the remapped schema ID for a given stream and original schema ID."""
        mapped_schema = self._schema_lookup_fast.get((stream_id, original_schema_id))
        return mapped_schema.id if mapped_schema else 0


def _should_include_all(_channel: Channel, _schema: Schema | None) -> bool:
    return True


def include_topics(topics: str | Iterable[str]) -> Callable[[Channel, Schema | None], bool]:
    topic_set = {topics} if isinstance(topics, str) else set(topics)
    return lambda channel, _schema: channel.topic in topic_set


def read_message(
    stream: IO[bytes] | Iterable[IO[bytes] | _ReaderReturnType],
    should_include: _ShouldIncludeType = _should_include_all,
    start_time_ns: int = 0,
    end_time_ns: int = sys.maxsize,
    validate_crc: bool = False,
    reverse: bool = False,
) -> _ReaderReturnType:
    """Read messages from MCAP stream(s).

    Args:
        stream: Single IO[bytes] stream or iterable of streams/generators
        should_include: Filter function for channels
        start_time_ns: Start time (inclusive) in nanoseconds
        end_time_ns: End time (exclusive) in nanoseconds
        validate_crc: Whether to validate CRC checksums
        reverse: If True, yield messages in descending log_time order.
                Only supported for seekable streams.

    Returns:
        Iterable of (schema, channel, message) tuples

    Raises:
        McapError: If reverse=True on a non-seekable stream
    """
    if isinstance(stream, io.IOBase):
        if stream.seekable():
            return _read_message_seeking(
                stream,
                should_include,
                start_time_ns,
                end_time_ns,
                validate_crc,
                reverse,
            )
        return _read_message_non_seeking(
            stream,
            should_include,
            start_time_ns,
            end_time_ns,
            validate_crc,
            reverse,
        )

    if isinstance(stream, Iterable):
        remapper = Remapper()

        def remap_schema_channel(
            generator: _ReaderReturnType,
            stream_id: int,
        ) -> _ReaderReturnType:
            for schema, channel, message in generator:
                mapped_schema = remapper.remap_schema(stream_id, schema)
                mapped_channel = remapper.remap_channel(stream_id, channel)
                yield (
                    mapped_schema,
                    mapped_channel,
                    replace(message, channel_id=mapped_channel.id),
                )

        return heapq.merge(
            *(
                remap_schema_channel(
                    read_message(
                        s, should_include, start_time_ns, end_time_ns, validate_crc, reverse
                    )
                    if isinstance(s, io.IOBase)
                    else cast("_ReaderReturnType", s),
                    stream_id=i,
                )
                for i, s in enumerate(stream)
            ),
            key=lambda x: (-x[2].log_time if reverse else x[2].log_time),
        )
    return None


class _SchemaProtocol(Protocol):
    id: int
    name: str
    encoding: str
    data: bytes


class DecoderFactoryProtocol(Protocol):
    def decoder_for(
        self, message_encoding: str, schema: _SchemaProtocol | None
    ) -> Callable[[bytes | memoryview], Any] | None: ...


@dataclass(frozen=True)
class DecodedMessage:
    schema: Schema | None
    channel: Channel
    message: Message
    decoder_function: Callable[[Schema | None, Channel, Message], Any]

    @cached_property
    def decoded_message(self) -> Any:
        return self.decoder_function(self.schema, self.channel, self.message)


def read_message_decoded(
    stream: IO[bytes] | Iterable[IO[bytes] | _ReaderReturnType],
    should_include: _ShouldIncludeType = _should_include_all,
    start_time_ns: int = 0,
    end_time_ns: int = sys.maxsize,
    decoder_factories: Iterable[DecoderFactoryProtocol] = (),
    reverse: bool = False,
    validate_crc: bool = False,
) -> Iterable[DecodedMessage]:
    decoders: dict[int, Callable[[bytes | memoryview], Any]] = {}

    def decoded_message(schema: Schema | None, channel: Channel, message: Message) -> Any:
        if schema is None:
            return message.data
        if decoder := decoders.get(schema.id):
            return decoder(bytes(message.data))
        for factory in decoder_factories:
            if decoder := factory.decoder_for(channel.message_encoding, schema):
                decoders[schema.id] = decoder
                return decoder(bytes(message.data))

        raise ValueError(
            f"no decoder factory supplied for message encoding {channel.message_encoding}, "
            f"schema {schema}"
        )

    for schema, channel, message in read_message(
        stream,
        should_include,
        start_time_ns,
        end_time_ns,
        validate_crc=validate_crc,
        reverse=reverse,
    ):
        yield DecodedMessage(schema, channel, message, decoded_message)
