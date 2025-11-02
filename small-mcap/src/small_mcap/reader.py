import contextlib
import heapq
import io
import struct
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import IO, TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, overload

if TYPE_CHECKING:
    from lz4.frame import decompress as lz4_decompress  # type: ignore[import-untyped]
    from zstandard import decompress as zstd_decompress
else:
    try:
        from zstandard import decompress as zstd_decompress
    except ImportError:
        zstd_decompress = None  # type: ignore[assignment]

    try:
        from lz4.frame import decompress as lz4_decompress
    except ImportError:
        lz4_decompress = None  # type: ignore[assignment]

from small_mcap.records import (
    MAGIC,
    MAGIC_SIZE,
    OPCODE_TO_RECORD,
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
    Summary,
    SummaryOffset,
)

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


_ReaderReturnType = Iterator[tuple[Schema | None, Channel, Message]]
_ShouldIncludeType = Callable[[Channel, Schema | None], bool]


class McapError(Exception):
    pass


class InvalidMagicError(McapError):
    def __init__(self, bad_magic: bytes) -> None:
        super().__init__(
            f"not a valid MCAP file, invalid magic: {bad_magic.decode('utf-8', 'replace')}"
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


@dataclass(slots=True)
class LazyChunk:
    """
    Lazy-loaded chunk that stores metadata without reading the compressed data.
    Used for efficient scanning of MCAP files when chunk data is not immediately needed.
    """

    message_start_time: int
    message_end_time: int
    uncompressed_size: int
    uncompressed_crc: int
    compression: str
    record_start: int  # Offset where the Chunk record begins
    data_len: int  # Length of the compressed data

    @classmethod
    def read_from_stream(
        cls, stream: IO[bytes] | io.BufferedIOBase, record_start: int
    ) -> "LazyChunk":
        """Read chunk metadata from stream without loading the compressed data."""
        data = stream.read(8 + 8 + 8 + 4 + 4)
        message_start_time, message_end_time, uncompressed_size, uncompressed_crc, str_len = (
            struct.unpack("<QQQII", data)
        )
        compression = stream.read(str_len).decode("utf-8")
        data_len = struct.unpack("<Q", stream.read(8))[0]
        stream.seek(data_len, 1)  # Skip the data for now
        return cls(
            message_start_time,
            message_end_time,
            uncompressed_size,
            uncompressed_crc,
            compression,
            record_start,
            data_len,
        )

    def to_chunk(self, stream: IO[bytes]) -> Chunk:
        """Convert to a full Chunk by reading from the stream."""
        stream.seek(self.record_start)
        return Chunk.read_record(stream)


def _get_chunk_data_stream(chunk: Chunk, validate_crc: bool = False) -> bytes:
    # Validate compression string
    if not isinstance(chunk.compression, str):
        raise UnsupportedCompressionError(
            f"compression must be a string, got {type(chunk.compression).__name__}"
        )

    if chunk.compression == "zstd":
        if zstd_decompress is None:
            raise UnsupportedCompressionError(
                "zstd compression used but zstandard module is not installed. "
                "Install it with: pip install zstandard"
            )
        data = zstd_decompress(chunk.data, chunk.uncompressed_size)
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


def breakup_chunk(chunk: Chunk, validate_crc: bool = False) -> list[McapRecord]:
    data = _get_chunk_data_stream(chunk, validate_crc=validate_crc)
    records: list[McapRecord] = []
    pos = 0

    while pos < len(data):
        opcode, length = struct.unpack_from("<BQ", data, pos)
        pos += _RECORD_HEADER_SIZE
        record_data_end = pos + length

        if opcode == Opcode.CHANNEL:
            records.append(Channel.read(data[pos:record_data_end]))
        elif opcode == Opcode.MESSAGE:
            records.append(Message.read(data[pos:record_data_end]))
        elif opcode == Opcode.SCHEMA:
            records.append(Schema.read(data[pos:record_data_end]))

        pos = record_data_end

    return records


@overload
def stream_reader(
    stream: IO[bytes] | io.BufferedIOBase,
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: Literal[True] = ...,
    lazy_chunks: Literal[True] = ...,
) -> Iterator[NonChunkRecord | LazyChunk | MessageIndex]: ...


@overload
def stream_reader(
    stream: IO[bytes] | io.BufferedIOBase,
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: Literal[True] = ...,
    lazy_chunks: Literal[False] = ...,
) -> Iterator[NonChunkRecord | Chunk | MessageIndex]: ...


@overload
def stream_reader(
    stream: IO[bytes] | io.BufferedIOBase,
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: Literal[False] = ...,
    lazy_chunks: Literal[False] = ...,
) -> Iterator[NonChunkRecord | ChunkContentRecord]: ...


def stream_reader(
    stream: IO[bytes] | io.BufferedIOBase,
    *,
    skip_magic: bool = False,
    validate_crc: bool = False,
    emit_chunks: bool = False,
    lazy_chunks: bool = False,
) -> Iterator[McapRecord] | Iterator[McapRecord | LazyChunk]:
    record_size_limit = _RECORD_SIZE_LIMIT
    checksum = 0

    def read(n: int) -> bytes:
        data = stream.read(n)
        if len(data) < n:
            raise EndOfFileError
        if validate_crc and not skip_magic and not lazy_chunks:
            nonlocal checksum
            checksum = zlib.crc32(data, checksum)
        return data

    cached_pos = stream.tell()

    if not skip_magic:
        magic = read(MAGIC_SIZE)
        if magic != MAGIC:
            raise InvalidMagicError(magic)
        cached_pos += MAGIC_SIZE

    while True:
        checksum_before_read = checksum
        opcode, length = struct.unpack("<BQ", read(_RECORD_HEADER_SIZE))
        if record_size_limit is not None and length > record_size_limit:
            raise RecordLengthLimitExceededError(opcode, length, record_size_limit)

        cached_pos += _RECORD_HEADER_SIZE
        record_start = cached_pos

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


def _read_summary_from_iterator(stream_reader: Iterator[McapRecord | LazyChunk]) -> Summary | None:
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


def get_summary(stream: IO[bytes] | io.BufferedIOBase) -> Summary | None:
    """Get the start and end indexes of each chunk in the stream."""
    if not stream.seekable():
        return None
    try:
        stream.seek(-MAGIC_SIZE, io.SEEK_END)
        magic = stream.read(MAGIC_SIZE)
        if magic != MAGIC:
            raise InvalidMagicError(magic)
        stream.seek(-(_FOOTER_SIZE + MAGIC_SIZE), io.SEEK_END)
        footer = next(stream_reader(stream, skip_magic=True))
        if not isinstance(footer, Footer):
            return None
        if footer.summary_start == 0:
            return None
        stream.seek(footer.summary_start, io.SEEK_SET)
        return _read_summary_from_iterator(stream_reader(stream, skip_magic=True))
    except (OSError, StopIteration, EndOfFileError):
        return None


def get_header(stream: IO[bytes]) -> Header:
    if stream.seekable():
        stream.seek(0, io.SEEK_SET)

    header = next(stream_reader(stream, skip_magic=False))
    if not isinstance(header, Header):
        raise McapError(f"expected header at beginning of MCAP file, found {type(header)}")
    return header


def _chunks_matching_topics(
    summary: Summary,
    exclude_cache: set[int],
    start_time_ns: int | None,
    end_time_ns: int | None,
) -> list[ChunkIndex]:
    return sorted(
        (
            cidx
            for cidx in summary.chunk_indexes
            if not all(channel_id in exclude_cache for channel_id in cidx.message_index_offsets)
            and (start_time_ns is None or cidx.message_end_time >= start_time_ns)
            and (end_time_ns is None or cidx.message_start_time < end_time_ns)
        ),
        key=lambda c: c.message_start_time,
    )


def _read_inner(
    reader: Iterator[McapRecord],
    should_include: _ShouldIncludeType,
    exclude_channels: set[int],
    start_time_ns: int | None,
    end_time_ns: int | None,
    schemas: dict[int, Schema] | None = None,
    channels: dict[int, Channel] | None = None,
) -> _ReaderReturnType:
    _schemas: dict[int, Schema] = schemas or {}
    _channels: dict[int, Channel] = channels or {}

    for record in reader:
        if isinstance(record, Schema):
            _schemas[record.id] = record
        if isinstance(record, Channel):
            if record.schema_id != 0 and record.schema_id not in _schemas:
                raise McapError(f"no schema record found with id {record.schema_id}")
            _channels[record.id] = record
            if not should_include(_channels[record.id], _schemas.get(record.schema_id)):
                exclude_channels.add(record.id)
        if isinstance(record, Message):
            if record.channel_id not in _channels:
                raise McapError(f"no channel record found with id {record.channel_id}")
            if (
                (record.channel_id in exclude_channels)
                or (start_time_ns is not None and record.log_time < start_time_ns)
                or (end_time_ns is not None and record.log_time >= end_time_ns)
            ):
                continue
            channel = _channels[record.channel_id]
            schema = _schemas.get(channel.schema_id)
            yield (schema, channel, record)


def _read_message_seeking(
    stream: IO[bytes] | io.BufferedIOBase,
    should_include: _ShouldIncludeType,
    start_time_ns: int | None,
    end_time_ns: int | None,
    validate_crc: bool,
) -> _ReaderReturnType:
    summary = get_summary(stream)
    # No summary or chunk indexes exists
    if summary is None or not summary.chunk_indexes:
        # seek to start
        stream.seek(0, io.SEEK_SET)
        yield from _read_message_non_seeking(
            stream, should_include, start_time_ns, end_time_ns, validate_crc
        )
        return
    exclude_channels: set[int] = {
        channel.id
        for channel in summary.channels.values()
        if not should_include(channel, summary.schemas.get(channel.schema_id))
    }

    chunk_indexes = _chunks_matching_topics(summary, exclude_channels, start_time_ns, end_time_ns)

    def _lazy_yield(
        stream: IO[bytes] | io.BufferedIOBase, index: ChunkIndex
    ) -> Iterator[McapRecord | ChunkIndex]:
        # Emit the chunk index to prevent early loading and decompression of chunk
        yield index
        # late check for mcap with empty channel summary if this chunks is excluded entirely
        if all(cid in exclude_channels for cid in index.message_index_offsets):
            return

        stream.seek(index.chunk_start_offset)
        chunk = Chunk.read_record(stream)

        yield from breakup_chunk(chunk)

    def _lazy_sort(item: McapRecord | ChunkIndex) -> int:
        if isinstance(item, ChunkIndex):
            return item.message_start_time
        if isinstance(item, Message):
            return item.log_time
        return 0  # Other records are not yielded

    lazy_iterators = [_lazy_yield(stream, cidx) for cidx in chunk_indexes]
    reader = (
        item
        for item in heapq.merge(*lazy_iterators, key=_lazy_sort)
        if not isinstance(item, ChunkIndex)  # Filter out sentinels
    )

    yield from _read_inner(
        reader,
        should_include,
        exclude_channels,
        start_time_ns,
        end_time_ns,
        summary.schemas,
        summary.channels,
    )


def _process_chunks_with_buffering(
    stream: IO[bytes] | io.BufferedIOBase,
    should_process_chunk: Callable[[Chunk, list[MessageIndex]], bool],
    on_chunk_processed: Callable[[list[MessageIndex]], None],
    validate_crc: bool,
) -> Iterator[McapRecord]:
    """Process records from stream, breaking up chunks based on a predicate.

    This is the shared chunk buffering logic used by both message reading and summary rebuilding.

    Args:
        stream: The input stream to read from
        should_process_chunk: Predicate to decide whether to decompress a chunk
        on_chunk_processed: Callback invoked after chunk is processed with its MessageIndex list
        validate_crc: Whether to validate CRC checksums

    Yields:
        Records from the stream, with chunks broken up based on the predicate
    """
    buffered_chunk: Chunk | None = None
    chunk_message_indexes: list[MessageIndex] = []

    for record in stream_reader(stream, emit_chunks=True, validate_crc=validate_crc):
        if isinstance(record, MessageIndex):
            # Ignore empty MessageIndex records
            if len(record.records) > 0:
                chunk_message_indexes.append(record)
        elif isinstance(record, Chunk):
            # Process any previously buffered chunk before storing the new one
            if buffered_chunk is not None and should_process_chunk(
                buffered_chunk, chunk_message_indexes
            ):
                yield from breakup_chunk(buffered_chunk, validate_crc=validate_crc)
                on_chunk_processed(chunk_message_indexes)
            buffered_chunk = record
            chunk_message_indexes.clear()
        else:
            # Process buffered chunk before yielding other records
            if buffered_chunk is not None and should_process_chunk(
                buffered_chunk, chunk_message_indexes
            ):
                yield from breakup_chunk(buffered_chunk, validate_crc=validate_crc)
                on_chunk_processed(chunk_message_indexes)
            buffered_chunk = None
            chunk_message_indexes.clear()
            # At this point, record must be NonChunkRecord (not MessageIndex or Chunk)
            assert not isinstance(record, (MessageIndex, Chunk, LazyChunk))
            yield record

    # Process final buffered chunk
    if buffered_chunk is not None and should_process_chunk(buffered_chunk, chunk_message_indexes):
        yield from breakup_chunk(buffered_chunk, validate_crc=validate_crc)
        on_chunk_processed(chunk_message_indexes)


def _read_message_non_seeking(
    stream: IO[bytes] | io.BufferedIOBase,
    should_include: _ShouldIncludeType,
    start_time_ns: int | None,
    end_time_ns: int | None,
    validate_crc: bool,
) -> _ReaderReturnType:
    exclude_channels: set[int] = set()
    seen_channels: set[int] = set()

    def should_process_chunk(chunk: Chunk, message_indexes: list[MessageIndex]) -> bool:
        """Check if chunk should be processed based on time range and channel filtering."""
        # Fast checks first: filter by time range (simple integer comparisons)
        if start_time_ns is not None and chunk.message_end_time < start_time_ns:
            return False
        if end_time_ns is not None and chunk.message_start_time >= end_time_ns:
            return False
        # More expensive check: process if chunk has unseen channels
        # (need Channel/Schema definitions)
        if message_indexes and any(mi.channel_id not in seen_channels for mi in message_indexes):
            return True
        # Filter by channels - skip if all channels are excluded
        return not (
            message_indexes and all(mi.channel_id in exclude_channels for mi in message_indexes)
        )

    def on_chunk_processed(message_indexes: list[MessageIndex]) -> None:
        """Mark channels as seen after processing."""
        for mi in message_indexes:
            seen_channels.add(mi.channel_id)

    yield from _read_inner(
        _process_chunks_with_buffering(
            stream, should_process_chunk, on_chunk_processed, validate_crc
        ),
        should_include,
        exclude_channels,
        start_time_ns,
        end_time_ns,
    )


class _Remapper:
    def __init__(self, summaries: list[tuple[int, Summary]] | None = None) -> None:
        self._used_schema_ids: set[int] = set()
        self._used_channel_ids: set[int] = set()

        self._schema_lookup_fast: dict[tuple[int, int], Schema | None] = {}
        self._channel_lookup_fast: dict[tuple[int, int], Channel] = {}
        self._schema_lookup_slow: dict[tuple[str, bytes], Schema | None] = {}
        self._channel_lookup_slow: dict[tuple[str, str], Channel] = {}

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

        # Slow path: lookup by schema content
        slow_key = (schema.name, schema.data)
        mapped_schema = self._schema_lookup_slow.get(slow_key)
        if mapped_schema:
            # Cache in fast lookup for future access
            self._schema_lookup_fast[fast_key] = mapped_schema
            return mapped_schema

        new_id = schema.id
        while new_id in self._used_schema_ids:
            new_id += 1
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

        # Slow path: lookup by channel content
        slow_key = (
            channel.topic,
            channel.message_encoding,
        )  # TODO: include metadata
        mapped_channel = self._channel_lookup_slow.get(slow_key)
        if mapped_channel is not None:
            # Cache in fast lookup for future access
            self._channel_lookup_fast[fast_key] = mapped_channel
            return mapped_channel

        new_id = channel.id
        while new_id in self._used_channel_ids:
            new_id += 1
        self._used_channel_ids.add(new_id)
        mapped_channel = replace(channel, id=new_id, schema_id=0)
        self._channel_lookup_slow[slow_key] = mapped_channel
        self._channel_lookup_fast[fast_key] = mapped_channel
        return mapped_channel


def _should_include_all(_channel: Channel, _schema: Schema | None) -> bool:
    return True


def include_topics(topics: Sequence[str]) -> Callable[[Channel, Schema | None], bool]:
    return lambda channel, _schema: channel.topic in topics


def read_message(
    stream: IO[bytes] | io.BufferedIOBase | list[IO[bytes] | _ReaderReturnType],
    should_include: _ShouldIncludeType = _should_include_all,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
    validate_crc: bool = False,
) -> _ReaderReturnType:
    if isinstance(stream, list):
        remapper = _Remapper()

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
            *[
                remap_schema_channel(
                    read_message(s, should_include, start_time_ns, end_time_ns, validate_crc)
                    if isinstance(s, IO)
                    else s,
                    stream_id=i,
                )
                for i, s in enumerate(stream)
            ],
            key=lambda x: x[2].log_time,
        )

    # if not buffer io bufferedreader
    if not isinstance(stream, io.BufferedIOBase):
        stream = io.BufferedReader(stream)  # type: ignore[type-var]

    if stream.seekable():
        return _read_message_seeking(
            stream,
            should_include,
            start_time_ns,
            end_time_ns,
            validate_crc,
        )
    return _read_message_non_seeking(
        stream,
        should_include,
        start_time_ns,
        end_time_ns,
        validate_crc,
    )


class DecoderFactoryProtocol(Protocol):
    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes], Any] | None: ...


class DecodedMessageTuple(NamedTuple):
    schema: Schema | None
    channel: Channel
    message: Message
    decoded_message: Any


def read_message_decoded(
    stream: IO[bytes] | io.BufferedIOBase | list[IO[bytes] | _ReaderReturnType],
    should_include: _ShouldIncludeType = _should_include_all,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
    decoder_factories: Iterable[DecoderFactoryProtocol] = (),
) -> Iterator[DecodedMessageTuple]:
    decoders: dict[int, Callable[[bytes], Any]] = {}

    def decoded_message(schema: Schema | None, channel: Channel, message: Message) -> Any:
        if schema is None:
            return message.data
        decoder = decoders.get(hash(schema.data))
        if decoder is not None:
            return decoder(message.data)
        for factory in decoder_factories:
            if decoder := factory.decoder_for(channel.message_encoding, schema):
                decoders[message.channel_id] = decoder
                return decoder(message.data)

        raise ValueError(
            f"no decoder factory supplied for message encoding {channel.message_encoding}, "
            f"schema {schema}"
        )

    for schema, channel, message in read_message(
        stream, should_include, start_time_ns, end_time_ns
    ):
        yield DecodedMessageTuple(
            schema, channel, message, decoded_message(schema, channel, message)
        )
