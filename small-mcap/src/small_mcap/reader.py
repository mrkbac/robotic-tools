import bisect
import heapq
import io
import itertools
import sys
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property
from operator import attrgetter, itemgetter
from typing import IO, TYPE_CHECKING, Any, Literal, Protocol, cast, overload

from small_mcap.exceptions import (
    ChannelNotFoundError,
    CRCValidationError,
    EndOfFileError,
    IllegalOpcodeInChunkError,
    InvalidHeaderError,
    InvalidMagicError,
    RecordLengthLimitExceededError,
    SchemaNotFoundError,
    SeekRequiredError,
    UnsupportedCompressionError,
)
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
from small_mcap.remapper import Remapper

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
    """Decompress a chunk and yield its individual records.

    Chunks may only contain Schema, Channel, and Message records per the MCAP spec.

    Args:
        chunk: The chunk to decompress and iterate over.
        validate_crc: Whether to validate the chunk's CRC32 checksum.

    Yields:
        Schema, Channel, and Message records from the chunk.

    Raises:
        IllegalOpcodeInChunkError: If a record with an illegal opcode is found in the chunk.
        CRCValidationError: If validate_crc is True and the CRC doesn't match.
        UnsupportedCompressionError: If the chunk uses an unsupported compression type.
    """
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
        else:
            raise IllegalOpcodeInChunkError(opcode)

        pos = record_data_end


def _breakup_chunk_with_indexes(
    chunk: Chunk,
    message_indexes: Iterable[MessageIndex],
    validate_crc: bool = False,
    reverse: bool = False,
) -> Iterable[McapRecord]:
    # materialize for truthy emptiness check
    message_indexes = list(message_indexes)
    if not message_indexes:
        return

    data = _get_chunk_data_stream(chunk, validate_crc=validate_crc)
    view = memoryview(data)

    records: Iterable[tuple[int, int]]
    if len(message_indexes) == 1:
        # Fast path: single channel - direct iteration, no heap needed
        records = message_indexes[0].records
        if reverse:
            records = reversed(records)
    elif reverse:
        records = heapq.merge(
            *(reversed(x.records) for x in message_indexes),
            key=itemgetter(0),
            reverse=True,
        )
    else:
        records = heapq.merge(
            *(x.records for x in message_indexes),
            key=itemgetter(0),
        )

    for _timestamp, offset in records:
        pos = offset
        opcode, length = OPCODE_AND_LEN_STRUCT.unpack_from(view, pos)
        pos += _RECORD_HEADER_SIZE
        record_data_end = pos + length
        if opcode == Opcode.MESSAGE:
            yield Message.read(view[pos:record_data_end])
        else:
            raise IllegalOpcodeInChunkError(opcode)


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
    allow_incomplete: bool = False,
) -> Iterable[NonChunkRecord | LazyChunk | MessageIndex]: ...


@overload
def stream_reader(
    stream: IO[bytes],
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: Literal[True] = ...,
    lazy_chunks: Literal[False] = ...,
    allow_incomplete: bool = False,
) -> Iterable[NonChunkRecord | Chunk | MessageIndex]: ...


@overload
def stream_reader(
    stream: IO[bytes],
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: Literal[False] = ...,
    lazy_chunks: Literal[False] = ...,
    allow_incomplete: bool = False,
) -> Iterable[NonChunkRecord | ChunkContentRecord]: ...


def stream_reader(
    stream: IO[bytes],
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: bool = False,
    lazy_chunks: bool = False,
    allow_incomplete: bool = False,
) -> Iterable[McapRecord] | Iterable[McapRecord | LazyChunk]:
    """Low-level iterator that yields every record from an MCAP byte stream.

    Records are yielded in file order. When ``emit_chunks`` is False (default),
    Chunk records are automatically decompressed and their inner Schema, Channel,
    and Message records are yielded instead.

    Args:
        stream: A readable binary stream positioned at the start of the MCAP data
            (or just after the magic bytes if ``skip_magic`` is True).
        skip_magic: If True, skip validation of the leading magic bytes.
        validate_crc: If True, validate CRC checksums on chunks and the data section.
        emit_chunks: If True, yield Chunk (or LazyChunk) and MessageIndex records
            directly instead of breaking them up into their contents.
        lazy_chunks: If True (requires ``emit_chunks=True``), yield LazyChunk records
            that defer decompression until explicitly requested.
        allow_incomplete: If True, gracefully stop iteration when the stream is
            truncated instead of raising EndOfFileError.

    Yields:
        McapRecord instances in file order.
    """
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
        try:
            checksum_before_read = checksum
            opcode, length = OPCODE_AND_LEN_STRUCT.unpack(read(_RECORD_HEADER_SIZE))
        except EndOfFileError:
            if allow_incomplete:
                return
            raise

        if length > _RECORD_SIZE_LIMIT:
            raise RecordLengthLimitExceededError(opcode, length, _RECORD_SIZE_LIMIT)

        record_start = cached_pos
        cached_pos += _RECORD_HEADER_SIZE

        # Handle lazy chunk loading when requested
        record: McapRecord | LazyChunk | None
        if opcode == Opcode.CHUNK and emit_chunks and lazy_chunks:
            record = LazyChunk.read_from_stream(stream, record_start, length)
            cached_pos = stream.tell()
        else:
            try:
                cached_pos += length
                record_data = read(length)
            except EndOfFileError:
                if allow_incomplete:
                    return
                raise

            if record_cls := OPCODE_TO_RECORD.get(opcode):
                record = record_cls.read(record_data)
            else:
                record = None  # Unknown record type, skip it.

        if record is None:
            continue

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

        if isinstance(record, Chunk) and not emit_chunks:
            yield from breakup_chunk(record, validate_crc)
        elif not emit_chunks and isinstance(record, MessageIndex):
            pass  # skip when breaking up chunks
        elif isinstance(record, Footer):
            yield record
            if not skip_magic:
                magic = read(MAGIC_SIZE)
                if magic != MAGIC:
                    raise InvalidMagicError(magic)
            break
        else:
            yield record


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
    except (OSError, StopIteration, EndOfFileError, InvalidMagicError):
        return None


def get_header(stream: IO[bytes]) -> Header:
    if stream.seekable():
        stream.seek(0, io.SEEK_SET)

    header = next(iter(stream_reader(stream, skip_magic=False)))
    if not isinstance(header, Header):
        raise InvalidHeaderError(type(header))
    return header


def read_attachment(stream: IO[bytes], index: AttachmentIndex) -> Attachment:
    """Read a full Attachment record given its index.

    Args:
        stream: A seekable binary stream containing the MCAP data.
        index: The AttachmentIndex obtained from :func:`get_summary`.

    Returns:
        The deserialized Attachment record.
    """
    stream.seek(index.offset)
    return Attachment.read_record(stream)


def read_metadata(stream: IO[bytes], index: MetadataIndex) -> Metadata:
    """Read a full Metadata record given its index.

    Args:
        stream: A seekable binary stream containing the MCAP data.
        index: The MetadataIndex obtained from :func:`get_summary`.

    Returns:
        The deserialized Metadata record.
    """
    stream.seek(index.offset)
    return Metadata.read_record(stream)


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
                raise ChannelNotFoundError(record.channel_id)
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
                raise SchemaNotFoundError(record.schema_id)
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
    start_idx = bisect.bisect_left(message_index.records, start_time_ns, key=itemgetter(0))

    # Binary search for end index (first record >= end_time_ns)
    end_idx = bisect.bisect_left(message_index.records, end_time_ns, key=itemgetter(0))

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

            yield from _breakup_chunk_with_indexes(
                chunk, filtered_message_indexes, validate_crc, reverse=reverse
            )
        else:
            yield from breakup_chunk(chunk, validate_crc=validate_crc)

    def in_time_range(chunk_start: int, chunk_end: int) -> bool:
        return chunk_start < end_time_ns and chunk_end >= start_time_ns

    # Filter and sort chunks that overlap with the time range
    sorted_chunks = sorted(
        (
            cidx
            for cidx in summary.chunk_indexes
            if not (
                cidx.message_index_offsets
                and exclude_channels.issuperset(cidx.message_index_offsets)
            )
            and in_time_range(cidx.message_start_time, cidx.message_end_time)
        ),
        key=attrgetter("message_end_time") if reverse else attrgetter("message_start_time"),
        reverse=reverse,
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
            return item.log_time
        if isinstance(item, ChunkIndex):
            return item.message_end_time if reverse else item.message_start_time
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
        reader = heapq.merge(*lazy_iterables, key=_lazy_sort, reverse=reverse)

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
        raise SeekRequiredError("reverse=True")

    exclude_channels: set[int] = set()
    seen_channels: set[int] = set()

    def in_time_range(chunk_start: int, chunk_end: int) -> bool:
        return chunk_start < end_time_ns and chunk_end >= start_time_ns

    def _flush_pending_chunk(
        chunk: Chunk,
        message_indexes: list[MessageIndex],
        *,
        force_full: bool = False,
    ) -> Iterable[McapRecord]:
        """Decompress and yield records from a pending chunk.

        Args:
            chunk: The chunk to flush.
            message_indexes: Associated message indexes (may be empty).
            force_full: If True, always use full decompression (for final chunks
                where we can't guarantee all indexes were received).
        """
        if not in_time_range(chunk.message_start_time, chunk.message_end_time):
            return

        if force_full or not message_indexes:
            yield from breakup_chunk(chunk, validate_crc=validate_crc)
            return

        channel_ids = {mi.channel_id for mi in message_indexes}
        new_channels = bool(channel_ids - seen_channels)

        if new_channels:
            # Chunk may contain Schema/Channel records for unseen channels.
            # Use full decompression to yield them.
            yield from breakup_chunk(chunk, validate_crc=validate_crc)
        elif not channel_ids.issubset(exclude_channels):
            # All channels known — safe to use indexed message lookup.
            filtered = _filter_message_indices_by_time(
                (mi for mi in message_indexes if mi.channel_id not in exclude_channels),
                start_time_ns,
                end_time_ns,
            )
            yield from _breakup_chunk_with_indexes(chunk, filtered, validate_crc)

    def _inner() -> Iterable[McapRecord]:
        pending_chunk: Chunk | None = None
        pending_message_indexes: list[MessageIndex] = []
        for record in stream_reader(
            stream,
            emit_chunks=True,
            validate_crc=validate_crc,
            lazy_chunks=False,
            allow_incomplete=True,
        ):
            if not isinstance(record, MessageIndex) and pending_chunk:
                yield from _flush_pending_chunk(pending_chunk, pending_message_indexes)
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
            # Final chunk: always decompress fully since we can't ensure we
            # got all message indexes
            yield from _flush_pending_chunk(pending_chunk, pending_message_indexes, force_full=True)

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


def _should_include_all(_channel: Channel, _schema: Schema | None) -> bool:
    return True


def include_topics(topics: str | Iterable[str]) -> Callable[[Channel, Schema | None], bool]:
    """Create a filter function that accepts only channels matching the given topic(s).

    Intended for use as the ``should_include`` argument to :func:`read_message`.

    Args:
        topics: A single topic string or an iterable of topic strings to include.

    Returns:
        A callable ``(Channel, Schema | None) -> bool`` that returns True for matching channels.
    """
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
        io_stream = cast("IO[bytes]", stream)
        if stream.seekable():
            return _read_message_seeking(
                io_stream,
                should_include,
                start_time_ns,
                end_time_ns,
                validate_crc,
                reverse,
            )
        return _read_message_non_seeking(
            io_stream,
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
                mapped_message = remapper.remap_message(stream_id, message)
                yield (
                    mapped_schema,
                    mapped_channel,
                    mapped_message,
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
            key=lambda x: x[2].log_time,
            reverse=reverse,
        )
    raise TypeError(f"Unsupported stream type: {type(stream)}")


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
        # Use schema.id as cache key, or 0 for schemaless
        cache_key = schema.id if schema else 0

        if decoder := decoders.get(cache_key):
            return decoder(bytes(message.data))

        for factory in decoder_factories:
            if decoder := factory.decoder_for(channel.message_encoding, schema):
                decoders[cache_key] = decoder
                return decoder(bytes(message.data))

        # No decoder found - return raw data for schemaless, raise for schema-based
        if schema is None:
            return message.data

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
