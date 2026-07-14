"""Unified MCAP processor combining recovery and filtering capabilities."""

import heapq
import logging
import os
import stat
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from itertools import product
from pathlib import Path
from string import Formatter
from typing import IO, BinaryIO, cast

from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.text import Text
from small_mcap import (
    MAGIC,
    MAGIC_SIZE,
    OPCODE_TO_RECORD,
    Attachment,
    Channel,
    Chunk,
    ChunkIndex,
    CompressionType,
    DataEnd,
    EndOfFileError,
    Footer,
    Header,
    InvalidMagicError,
    LazyChunk,
    McapError,
    McapRecord,
    McapWriter,
    Message,
    MessageIndex,
    Metadata,
    Opcode,
    RecordLengthLimitExceededError,
    Remapper,
    Schema,
    Statistics,
    Summary,
    breakup_chunk,
    get_header,
    get_summary,
)

# Private helpers — small-mcap does not re-export these at the top level.
from small_mcap.reader import _MESSAGE_STRUCT, _RECORD_SIZE_LIMIT, _predecompress_chunk
from small_mcap.records import OPCODE_AND_LEN_STRUCT
from small_mcap.writer import _ChunkBuilder, _compress_chunk_data

from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.input_options import InputOptions
from pymcap_cli.core.input_processor_chain import build_input_processors
from pymcap_cli.core.processors.base import (
    SPLIT_REQUIRED,
    Action,
    ChannelContext,
    ChunkContext,
    ChunkDecision,
    InputContext,
    InputProcessor,
    MessageContext,
    MessageHeader,
    MessageHeaderDecision,
    MessageScopeKind,
    MessageWithContext,
    OutputKey,
    OutputProcessor,
    OutputRouter,
    OutputSegmentInfo,
    PipelineContext,
    RouteKey,
    SegmentContext,
    TemplateValue,
)
from pymcap_cli.log_setup import ERR
from pymcap_cli.types.types_manual import (
    CompressionName,
    str_to_compression_type,
)
from pymcap_cli.utils import (
    McapWriterOptions,
    ProgressTrackingIO,
    confirm_output_overwrite,
    create_mcap_writer,
    file_progress,
    output_overwrites_input,
)

logger = logging.getLogger(__name__)
console = ERR
OUTPUT_LIBRARY = "pymcap-cli"
OutputStreamOpener = Callable[[OutputKey, int, int, int], tuple[str, BinaryIO]]
TemplateFieldProvider = Callable[[OutputKey], dict[str, TemplateValue]]


def _decode_chunk_records(chunk: Chunk) -> list[McapRecord]:
    """Worker-side: decompress a chunk and return its records as a list.

    Runs on a ThreadPoolExecutor worker. zstd/lz4 decompression releases the
    GIL so multiple chunks genuinely decompress in parallel.
    """
    return list(breakup_chunk(chunk, validate_crc=True))


def _chunk_records_match_writer_view(
    records: list[McapRecord],
    schemas: dict[int, Schema],
    channels: dict[int, Channel],
) -> bool:
    """Check whether a chunk's in-chunk Schema/Channel records are still valid.

    Returns True when every embedded Schema/Channel record in the chunk
    matches the writer's current view — i.e. the chunk's compressed bytes
    can be fast-copied without leaking stale records into the output.

    Uses ``type(record) is X`` dispatch (faster than isinstance) and exits
    on the first non-metadata record. MCAP writers in practice place
    Schema and Channel records as a small prefix before any Message that
    references them, so the first Message marks the end of metadata.
    """
    for record in records:
        rtype = type(record)
        if rtype is Schema:
            schema = cast("Schema", record)
            existing_schema = schemas.get(schema.id)
            if existing_schema is None or existing_schema != schema:
                return False
        elif rtype is Channel:
            channel = cast("Channel", record)
            existing_channel = channels.get(channel.id)
            if existing_channel is None or existing_channel != channel:
                return False
        else:
            return True
    return True


def _recompress_chunk(
    chunk: Chunk, target: CompressionType, zstd_level: int | None = None
) -> Chunk:
    """Worker-side: decompress + re-compress a chunk's data with a new codec.

    Avoids parsing/re-emitting every record when only the chunk compression
    changes. Message indexes remain valid because they reference offsets into
    the uncompressed chunk payload, which is unchanged.
    """
    decompressed = _predecompress_chunk(chunk, validate_crc=True)
    new_data, new_compression = _compress_chunk_data(
        decompressed.data, target, zstd_level=zstd_level
    )
    return Chunk(
        message_start_time=chunk.message_start_time,
        message_end_time=chunk.message_end_time,
        uncompressed_size=decompressed.uncompressed_size,
        uncompressed_crc=decompressed.uncompressed_crc,
        compression=new_compression,
        data=new_data,
    )


def _pread_exact(fd: int, length: int, offset: int) -> bytes:
    """``os.pread`` that retries until ``length`` bytes are read. Positional and
    thread-safe — it never touches the fd's own offset, so workers can read in
    parallel while the main thread scans the same descriptor sequentially.
    """
    data = os.pread(fd, length, offset)
    if len(data) == length:
        return data
    parts = [data]
    got = len(data)
    while got < length:
        more = os.pread(fd, length - got, offset + got)
        if not more:
            raise EOFError(f"short read at offset {offset}: {got}/{length} bytes")
        parts.append(more)
        got += len(more)
    return b"".join(parts)


def _read_exact(stream: IO[bytes], length: int) -> bytes:
    if length < 0:
        raise EndOfFileError
    data = stream.read(length)
    if len(data) < length:
        raise EndOfFileError
    return data


def _read_and_recompress_chunk(
    fd: int,
    body_offset: int,
    body_length: int,
    target: CompressionType,
    zstd_level: int | None = None,
) -> Chunk:
    """Worker-side: read a Chunk record body by absolute offset and recompress it.

    Moving the (large) data read onto the worker — instead of the main thread —
    is what keeps the pool saturated on big files, where the serial main-thread
    read was the throughput ceiling.
    """
    chunk = Chunk.read(_pread_exact(fd, body_length, body_offset))
    return _recompress_chunk(chunk, target, zstd_level)


def _read_and_decode_chunk_records(fd: int, body_offset: int, body_length: int) -> list[McapRecord]:
    """Worker-side: read a Chunk record body by absolute offset and decode records."""
    chunk = Chunk.read(_pread_exact(fd, body_length, body_offset))
    return _decode_chunk_records(chunk)


@dataclass(slots=True)
class PendingChunk:
    chunk: Chunk | LazyChunk
    indexes: list[MessageIndex]
    stream_id: int
    stream: IO[bytes]  # For converting LazyChunk to Chunk
    timestamp: int  # message_start_time for ordering

    def __lt__(self, other: "PendingChunk") -> bool:
        return self.timestamp < other.timestamp

    def get_chunk(self) -> Chunk:
        """Get full Chunk, converting from LazyChunk if needed."""
        if isinstance(self.chunk, LazyChunk):
            self.chunk = self.chunk.to_chunk(self.stream)
        return self.chunk


class OverwriteCollisionPolicy(str, Enum):
    """How split outputs handle collisions with existing files."""

    ASK = "ask"
    OVERWRITE = "overwrite"
    ERROR = "error"


@dataclass(slots=True)
class InputFile:
    """Input file stream with its size and options."""

    stream: IO[bytes]
    size: int
    options: InputOptions


@dataclass(slots=True)
class OutputOptions:
    """Options for output file format."""

    compression: CompressionName = DEFAULT_COMPRESSION
    chunk_size: int = DEFAULT_CHUNK_SIZE
    enable_crcs: bool = True
    use_chunking: bool = True
    # zstd compression level; None uses the library default (3). Negative levels
    # select the fast modes (much higher throughput, slightly larger output).
    zstd_level: int | None = None
    async_output_buffer_bytes: int = 0

    # Output processors (chunk grouping). When non-empty, each surviving
    # message is routed through a per-segment MessageGroup keyed by the
    # composite of every processor's ``chunk_group_key``. When empty, messages
    # are written directly through the writer (fast-copy eligible).
    output_processors: list[OutputProcessor] = field(default_factory=list)
    # Hard cap on concurrent chunk groups per output segment. When the cap is
    # hit, additional groups share the most-recently-created group (overflow).
    max_chunk_groups: int | None = None
    # Total uncompressed bytes buffered across all chunk groups in a segment.
    # When exceeded, the largest in-flight chunk is flushed prematurely.
    max_chunk_memory_bytes: int | None = None
    # Max log-time span of a single chunk. When set, a chunk is also flushed once
    # its buffered messages span this many nanoseconds, not only at chunk_size —
    # keeps low-byte-rate groups (already-compressed payloads) time-local.
    max_chunk_span_ns: int | None = None

    # Output routers (split routing, etc.)
    routers: list[OutputRouter] = field(default_factory=list)
    # Template for multi-output file naming (e.g., "output_{index:03d}.mcap")
    output_template: str = ""
    overwrite_policy: OverwriteCollisionPolicy = OverwriteCollisionPolicy.ASK
    # Resolved input paths, used to refuse opening a segment that would truncate
    # a file currently being read. Populated by the multi-output runner.
    input_paths: tuple[str, ...] = ()

    @property
    def compression_type(self) -> CompressionType:
        return str_to_compression_type(self.compression)

    def to_writer_options(self) -> McapWriterOptions:
        return McapWriterOptions(
            chunk_size=self.chunk_size,
            compression=self.compression,
            enable_crcs=self.enable_crcs,
            use_chunking=self.use_chunking,
            zstd_level=self.zstd_level,
        )

    @property
    def has_chunk_grouping(self) -> bool:
        return bool(self.output_processors)

    @property
    def is_splitting(self) -> bool:
        return bool(self.routers)


class ProcessingOptions:
    """Complete processing configuration."""

    def __init__(
        self,
        inputs: list[InputFile],
        input_options: InputOptions,
        output_options: OutputOptions,
    ) -> None:
        self.output_options = output_options

        self.inputs: list[InputFile] = []
        for input_file in inputs:
            merged_opts = input_options | input_file.options
            self.inputs.append(InputFile(input_file.stream, input_file.size, merged_opts))

        self.total_size = sum(input_file.size for input_file in inputs)


@dataclass(slots=True)
class ProcessingStats:
    # Input processing counts
    messages_processed: int = 0
    attachments_processed: int = 0
    metadata_processed: int = 0

    # Chunk processing strategy tracking
    chunks_processed: int = 0
    chunks_copied: int = 0  # Fast copied chunks (includes verified chunks)
    chunks_decoded: int = 0  # Decoded chunks
    chunks_verified: int = 0  # Decoded for verification, then fast-copied

    # Error and filtering tracking
    errors_encountered: int = 0
    validation_errors: int = 0
    filter_rejections: int = 0

    # Output statistics from McapWriter
    writer_statistics: Statistics = field(
        default_factory=lambda: Statistics(
            message_count=0,
            schema_count=0,
            channel_count=0,
            attachment_count=0,
            metadata_count=0,
            chunk_count=0,
            message_start_time=0,
            message_end_time=0,
            channel_message_counts={},
        )
    )

    def __rich_console__(self, _console: Console, _options: ConsoleOptions) -> RenderResult:
        lines = Text()
        ws = self.writer_statistics

        # Messages (show decoded count only if different from written)
        decoded = (
            f" ({self.messages_processed:,} decoded)"
            if (self.messages_processed > 0 and self.messages_processed != ws.message_count)
            else ""
        )
        lines.append(f"Messages:     {ws.message_count:,} written{decoded}\n")

        if self.attachments_processed > 0:
            lines.append(
                f"Attachments:  {ws.attachment_count} written "
                f"({self.attachments_processed} processed)\n"
            )
        if self.metadata_processed > 0:
            lines.append(
                f"Metadata:     {ws.metadata_count} written ({self.metadata_processed} processed)\n"
            )

        lines.append(f"Schemas:      {ws.schema_count} written\n")
        lines.append(f"Channels:     {ws.channel_count} written\n")

        if self.chunks_processed > 0:
            verified_note = f", {self.chunks_verified} verified" if self.chunks_verified else ""
            lines.append(
                f"Chunks:       {self.chunks_processed} "
                f"({self.chunks_copied} fast copied{verified_note}, "
                f"{self.chunks_decoded} decoded)\n"
            )
        if self.errors_encountered > 0:
            lines.append(f"Errors:       {self.errors_encountered}\n", style="yellow")
        if self.validation_errors > 0:
            lines.append(f"Validation:   {self.validation_errors} errors\n", style="yellow")
        if self.filter_rejections > 0:
            lines.append(f"Filtered:     {self.filter_rejections} records\n")

        lines.rstrip()
        yield Panel(lines, title="Processing Statistics", border_style="dim")


class MessageGroup:
    """Manages a group of messages that will be chunked together independently."""

    def __init__(
        self,
        writer: McapWriter,
        chunk_size: int,
        compression_type: CompressionType,
        max_chunk_span_ns: int | None = None,
    ) -> None:
        self.writer = writer
        self.chunk_size = chunk_size
        self.max_chunk_span_ns = max_chunk_span_ns
        self.message_count = 0
        self.compress_fail_counter = 0
        # Each group has its own chunk builder for independent chunking
        # Pass schemas/channels for auto-ensure
        self.chunk_builder = _ChunkBuilder(
            chunk_size=chunk_size,
            compression=compression_type,
            enable_crcs=writer.enable_crcs,
            zstd_level=writer.zstd_level,
        )

    def add_message(self, message: Message) -> None:
        self._flush_if_full()
        self.chunk_builder.add(message)
        self.message_count += 1

    def buffered_bytes(self) -> int:
        """Bytes currently held by the chunk builder (uncompressed)."""
        return self.chunk_builder.buffer.tell()

    def _flush_if_full(self) -> None:
        """Finalize and write the current chunk if it is full by size or time span.

        Runs before each append, so a span-capped chunk closes once its already
        buffered messages span ``max_chunk_span_ns``; the message that would push
        it over starts the next chunk.
        """
        if self.chunk_builder.num_messages == 0:
            return
        if self.chunk_builder.buffer.tell() >= self.chunk_builder.chunk_size:
            self._finalize_and_reset()
            return
        if self.max_chunk_span_ns is not None:
            span = self.chunk_builder.message_end_time - self.chunk_builder.message_start_time
            if span >= self.max_chunk_span_ns:
                self._finalize_and_reset()

    def flush_premature(self) -> None:
        """Finalize the current chunk now (before chunk_size) and reset the builder.

        Used by the segment-level memory cap to evict the largest in-flight
        chunk when total buffered bytes exceed ``max_chunk_memory_bytes``.
        """
        if self.chunk_builder.num_messages == 0:
            return
        self._finalize_and_reset()

    def _finalize_and_reset(self) -> None:
        if result := self.chunk_builder.finalize():
            chunk, message_indexes = result
            if chunk.compression != self.chunk_builder.compression.value:
                self.compress_fail_counter += 1
                if self.compress_fail_counter > 2:
                    console.print(
                        "[yellow]Multiple compression failures, switching to uncompressed.[/yellow]"
                    )
                    self.chunk_builder.compression = CompressionType.NONE
            self.writer.add_chunk(chunk, message_indexes)
        self.chunk_builder.reset()

    def flush(self) -> None:
        result = self.chunk_builder.finalize()
        if result is not None:
            chunk, message_indexes = result
            self.writer.add_chunk(chunk, message_indexes)


def _ns_to_iso(ns: int) -> str:
    """Convert nanosecond timestamp to ISO 8601 string."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


@dataclass(slots=True)
class OutputSegment:
    """One output file in a multi-output split."""

    key: OutputKey
    index: int
    stream: BinaryIO
    writer: McapWriter
    path: str
    written_schemas: set[int] = field(default_factory=set)
    written_channels: set[int] = field(default_factory=set)
    # Ordered list of MessageGroups for this segment (used for flush ordering).
    chunk_groups: list[MessageGroup] = field(default_factory=list)
    # Per-channel cache: channel id → group it currently writes into.
    channel_to_group: dict[int, MessageGroup] = field(default_factory=dict)
    start_time: int = 0
    end_time: int = 0
    # Composite group key (from OutputProcessor chain) → MessageGroup.
    groups: dict[Hashable, MessageGroup] = field(default_factory=dict)
    # Overflow group is lazily picked once ``max_chunk_groups`` is reached;
    # all further channels join it so the per-segment group count stops growing.
    overflow_group: MessageGroup | None = None
    # on_segment_open fires lazily per processor as streams reach the segment.
    replayed_processor_ids: set[int] = field(default_factory=set)


class OutputManager:
    """Manages writer pool for multi-output splitting.

    Writers are lazily created on first write to a segment key. Each segment
    gets its own McapWriter, schema/channel tracking, and optional rechunk groups.

    Pending attachments and metadata are buffered until segments are created,
    then flushed to each new segment to ensure records are not lost during
    lazy segment creation.
    """

    def __init__(
        self,
        output_options: "OutputOptions",
        schemas: dict[int, Schema],
        channels: dict[int, Channel],
        header: Header,
        open_output: OutputStreamOpener | None = None,
        template_fields: TemplateFieldProvider | None = None,
    ) -> None:
        self.output_options: OutputOptions = output_options
        self.schemas = schemas
        self.channels = channels
        self.header = header
        self._open_output = open_output or self._open_template_output
        self._template_fields = template_fields or (lambda _key: {})
        self.segments: dict[OutputKey, OutputSegment] = {}
        self._path_keys: dict[Path, OutputKey] = {}
        # Reverse lookup so rechunking can find a segment from its writer in O(1).
        self._segment_by_writer: dict[int, OutputSegment] = {}
        self._next_index: int = 0
        # Buffer records that arrive before any segments exist
        self._pending_attachments: list[Attachment] = []
        self._pending_metadata: list[tuple[str, dict[str, str]]] = []

    def handle_existing_output(self, path: Path) -> None:
        """Apply the configured collision policy for an existing output path."""
        if not path.exists():
            return

        policy = self.output_options.overwrite_policy
        if policy == OverwriteCollisionPolicy.OVERWRITE:
            return
        if policy == OverwriteCollisionPolicy.ERROR:
            console.print(f"[red]Error: Output file '{path}' already exists.[/red]")
            raise SystemExit(1)

        confirm_output_overwrite(path, force=False)

    def _open_template_output(
        self, key: OutputKey, index: int, start_time: int, end_time: int
    ) -> tuple[str, BinaryIO]:
        """Open a template-derived output path for a segment."""
        fields: dict[str, TemplateValue | OutputKey] = {
            "index": index,
            "index1": index + 1,
            "key": key,
            "start_time": start_time,
            "start_time_iso": _ns_to_iso(start_time) if start_time else "",
            "end_time": end_time,
        }
        extra_fields = self._template_fields(key)
        duplicate_fields = fields.keys() & extra_fields.keys()
        if duplicate_fields:
            names = ", ".join(sorted(duplicate_fields))
            raise ValueError(f"Output template fields conflict with built-ins: {names}")
        fields.update(extra_fields)
        for _, field_name, _, _ in Formatter().parse(self.output_options.output_template):
            if field_name is not None and field_name not in fields:
                raise ValueError(
                    f"Unknown output template field {field_name!r}; available fields: "
                    f"{', '.join(sorted(fields))}"
                )
        try:
            path = self.output_options.output_template.format(**fields)
        except (AttributeError, IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"Invalid output template {self.output_options.output_template!r}: {exc}"
            ) from exc

        path_obj = Path(path)
        if any(output_overwrites_input(src, path_obj) for src in self.output_options.input_paths):
            raise ValueError(
                f"Output segment '{path}' is the same file as an input; "
                "choose an output template that does not collide with the input."
            )
        resolved_path = path_obj.resolve(strict=False)
        previous_key = self._path_keys.get(resolved_path)
        if previous_key is not None and previous_key != key:
            raise ValueError(
                f"Segments {previous_key!r} and {key!r} resolve to the same output path "
                f"'{path}'; add '{{index}}' to the output template."
            )
        self.handle_existing_output(path_obj)
        stream = path_obj.open("wb")
        self._path_keys[resolved_path] = key
        return path, stream

    def _flush_pending_to_segment(self, segment: OutputSegment) -> None:
        """Write buffered attachments/metadata to a newly created segment."""
        for attachment in self._pending_attachments:
            segment.writer.add_attachment(
                log_time=attachment.log_time,
                create_time=attachment.create_time,
                name=attachment.name,
                media_type=attachment.media_type,
                data=attachment.data,
            )
        for name, metadata in self._pending_metadata:
            segment.writer.add_metadata(name=name, metadata=metadata)

    def get_or_create_segment(
        self, key: OutputKey, start_time: int = 0, end_time: int = 0
    ) -> OutputSegment:
        """Get or lazily create an output segment for the given key."""
        if key in self.segments:
            return self.segments[key]

        index = self._next_index
        self._next_index += 1

        path, stream = self._open_output(key, index, start_time, end_time)
        # Parallel compression: only useful when actually compressing — zstd/lz4
        # release the GIL, so a small worker pool meaningfully speeds up writes.
        # For uncompressed output the pool is pure overhead.
        compression_type = self.output_options.compression_type
        if compression_type == CompressionType.NONE:
            num_workers = 0
        else:
            num_workers = min(4, os.cpu_count() or 1)
            env = os.environ.get("MCAP_COMPRESS_WORKERS")
            if env:
                try:
                    num_workers = max(1, int(env))
                except ValueError:
                    console.print(
                        f"[yellow]Ignoring non-integer MCAP_COMPRESS_WORKERS={env!r}[/yellow]"
                    )
        writer = create_mcap_writer(
            stream,
            self.output_options.to_writer_options(),
            num_workers=num_workers,
        )
        # Schemas/channels are written lazily via ``ensure_channel_written`` when
        # a message (or fast-copied chunk) first references them — never eagerly
        # seeded from ``self.schemas``/``self.channels``. Eager seeding would emit
        # a channel record for every *input* channel even when a transform fully
        # consumes it (e.g. PointCloud2 → CompressedPointCloud2 on the same
        # topic), leaving an orphaned empty channel on the old schema. Lazy
        # writing keeps the output clean and matches standalone roscompress.
        writer.start(profile=self.header.profile, library=self.header.library)

        segment = OutputSegment(
            key=key,
            index=index,
            stream=stream,
            writer=writer,
            path=path,
            start_time=start_time,
            end_time=end_time,
        )
        self.segments[key] = segment
        self._segment_by_writer[id(writer)] = segment

        # Flush any buffered records to this new segment
        self._flush_pending_to_segment(segment)

        return segment

    def get_writer(self, key: OutputKey) -> McapWriter:
        """Get writer for output key, creating segment if needed."""
        return self.get_or_create_segment(key).writer

    def segment_for_writer(self, writer: McapWriter) -> OutputSegment | None:
        """Return the segment that owns ``writer``, or ``None`` if unknown."""
        return self._segment_by_writer.get(id(writer))

    def ensure_channel_written(self, channel_id: int, key: OutputKey) -> None:
        """Write schema and channel to a segment if not already written."""
        segment = self.get_or_create_segment(key)
        if channel_id in segment.written_channels:
            return

        channel = self.channels.get(channel_id)
        if not channel:
            return

        # Write schema first if needed
        if channel.schema_id != 0 and channel.schema_id not in segment.written_schemas:
            schema = self.schemas.get(channel.schema_id)
            if schema:
                segment.writer.add_schema(schema.id, schema.name, schema.encoding, schema.data)
                segment.written_schemas.add(schema.id)

        segment.writer.add_channel(
            channel.id,
            schema_id=channel.schema_id,
            topic=channel.topic,
            message_encoding=channel.message_encoding,
            metadata=channel.metadata,
        )
        segment.written_channels.add(channel_id)

    def finish_all(self) -> dict[OutputKey, Statistics]:
        """Finish and close all segment writers. Returns per-segment statistics."""
        stats: dict[OutputKey, Statistics] = {}
        for key, segment in self.segments.items():
            # Flush chunk groups
            for group in segment.chunk_groups:
                group.flush()
            segment.writer.finish()
            stats[key] = segment.writer.statistics
            segment.stream.close()
        # Clear buffers (records were flushed to all segments during creation)
        self._pending_attachments.clear()
        self._pending_metadata.clear()
        self._segment_by_writer.clear()
        return stats

    def add_attachment(self, attachment: Attachment) -> None:
        """Buffer attachment for later flushing to all segments, or write to existing ones."""
        self._pending_attachments.append(attachment)
        if self.segments:
            for segment in self.segments.values():
                segment.writer.add_attachment(
                    log_time=attachment.log_time,
                    create_time=attachment.create_time,
                    name=attachment.name,
                    media_type=attachment.media_type,
                    data=attachment.data,
                )

    def add_metadata(self, name: str, metadata: dict[str, str]) -> None:
        """Buffer metadata for later flushing to all segments, or write to existing ones."""
        self._pending_metadata.append((name, metadata))
        if self.segments:
            for segment in self.segments.values():
                segment.writer.add_metadata(name=name, metadata=metadata)


class McapProcessor:
    """Unified MCAP processor combining recovery and filtering capabilities.

    Supports processing single or multiple MCAP files with smart schema/channel ID
    remapping to minimize the need for chunk decoding. Each input can have its own
    filtering options (time range, topic filters, etc.).
    """

    def __init__(self, options: ProcessingOptions) -> None:
        self.options = options
        self.stats = ProcessingStats()

        # ID remapper for handling multiple files (zero overhead for single file)
        self.remapper = Remapper()

        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}

        # Cache filtering decisions per (stream_id, channel_id) for per-input filtering
        # Key: (stream_id, channel_id), Value: True if included
        self.channel_filter_cache: dict[tuple[int, int], bool] = {}

        # Cache the per-message context by (stream_id, input_channel_id). Its
        # fields (the stream's InputContext + the channel id) are constant for
        # every message on a channel, so this frozen dataclass is built once per
        # channel instead of ~3x per message on the hot dispatch path.
        self._message_context_cache: dict[tuple[int, int | None], MessageContext] = {}

        # Track which channels we've already seen to optimize metadata extraction
        self.known_channels: set[int] = set()

        # Content-dedupe cache for processor-registered output-only schemas:
        # (name, encoding, data) -> assigned schema id. Lets a processor call
        # register_schema repeatedly (once per message) without allocating a
        # new id each time.
        self._extra_schema_ids: dict[tuple[str, str, bytes], int] = {}

        # Frozen base for deterministic output-channel ids (see
        # _register_extra_channel). None until the first source-keyed
        # registration, then fixed above all input channel ids.
        self._extra_channel_base: int | None = None

        # Unified output management for both single-output and split-output modes.
        self.output_manager: OutputManager | None = None

        # Per-stream processor chains, built once from the merged InputOptions
        # so processor-implementation imports stay out of the options module.
        self._input_processors: list[list[InputProcessor]] = [
            build_input_processors(input_file.options) for input_file in self.options.inputs
        ]
        self._has_payload_skipping_filters = tuple(
            input_file.options.start_time_ns is not None
            or input_file.options.end_time_ns is not None
            or bool(input_file.options.include_topics)
            or bool(input_file.options.exclude_topics)
            for input_file in self.options.inputs
        )
        self._input_contexts: list[InputContext | None] = [None for _ in self.options.inputs]
        self._validate_processor_roles()

    def _validate_processor_roles(self) -> None:
        for stream_id, chain in enumerate(self._input_processors):
            for processor in chain:
                if not isinstance(processor, InputProcessor):
                    msg = (
                        f"input stream {stream_id} has {type(processor).__name__}; "
                        "input chains require InputProcessor instances"
                    )
                    raise TypeError(msg)
        for router in self.options.output_options.routers:
            if not isinstance(router, OutputRouter):
                msg = (
                    f"output router list has {type(router).__name__}; "
                    "output routing requires OutputRouter instances"
                )
                raise TypeError(msg)
        for processor in self.options.output_options.output_processors:
            if not isinstance(processor, OutputProcessor):
                msg = (
                    f"output processor list has {type(processor).__name__}; "
                    "output processors require OutputProcessor instances"
                )
                raise TypeError(msg)

    def _get_processors(self, stream_id: int) -> list[InputProcessor]:
        return self._input_processors[stream_id]

    def _iter_unique_input_processors(self) -> Iterator[InputProcessor]:
        seen: set[int] = set()
        for chain in self._input_processors:
            for processor in chain:
                if id(processor) in seen:
                    continue
                seen.add(id(processor))
                yield processor

    def _get_input_context(self, stream_id: int) -> InputContext:
        context = self._input_contexts[stream_id]
        if context is None:
            msg = f"input context for stream {stream_id} has not been prepared"
            raise RuntimeError(msg)
        return context

    def _message_context(
        self,
        stream_id: int,
        input_channel_id: int | None,
    ) -> MessageContext:
        key = (stream_id, input_channel_id)
        context = self._message_context_cache.get(key)
        if context is None:
            context = MessageContext(
                input=self._get_input_context(stream_id),
                input_channel_id=input_channel_id,
            )
            self._message_context_cache[key] = context
        return context

    def _chunk_context(self, stream_id: int, indexes: list[MessageIndex]) -> ChunkContext:
        return ChunkContext(
            input=self._get_input_context(stream_id),
            message_indexes=tuple(indexes) if indexes else None,
        )

    def _is_channel_included(self, stream_id: int, channel_id: int) -> bool:
        cache_key = (stream_id, channel_id)
        if cache_key in self.channel_filter_cache:
            return self.channel_filter_cache[cache_key]
        channel = self.channels.get(channel_id)
        if not channel:
            return False
        processors = self._get_processors(stream_id)
        if not processors:
            return True
        skip = False
        schema = self.schemas.get(channel.schema_id)
        context = ChannelContext(
            input=self._get_input_context(stream_id),
            input_channel_id=channel_id,
        )
        for p in processors:
            action = p.on_channel(context, channel, schema)
            if action & Action.SKIP:
                skip = True
        included = not skip
        self.channel_filter_cache[cache_key] = included
        return included

    def _composite_group_key(self, segment_key: OutputKey, channel: Channel) -> Hashable:
        """Compose every output processor's chunk_group_key into one key.

        Returns the (possibly empty) tuple of non-``None`` per-processor keys.
        Two channels yielding equal composite keys share a chunk group.
        """
        schema = self.schemas.get(channel.schema_id)
        parts: list[Hashable] = []
        for processor in self.options.output_options.output_processors:
            key = processor.chunk_group_key(segment_key, channel, schema)
            if key is not None:
                parts.append(key)
        return tuple(parts)

    def _get_or_create_group_for_channel(
        self, channel_id: int, channel: Channel, writer: McapWriter
    ) -> MessageGroup:
        """Resolve the chunk MessageGroup for ``channel_id`` on ``writer``'s segment.

        Each segment carries its own chunk groups so split outputs stay
        independent. The composite key returned by the configured
        ``OutputProcessor`` chain determines which channels share a group.
        """
        assert self.output_manager is not None
        segment = self.output_manager.segment_for_writer(writer)
        if segment is None:
            segment = self.output_manager.get_or_create_segment(0)

        if channel_id in segment.channel_to_group:
            return segment.channel_to_group[channel_id]

        composite_key = self._composite_group_key(segment.key, channel)
        group = segment.groups.get(composite_key)
        if group is None:
            group = self._create_or_overflow_group(segment, channel)
            segment.groups[composite_key] = group

        segment.channel_to_group[channel_id] = group
        return group

    def _create_or_overflow_group(self, segment: "OutputSegment", channel: Channel) -> MessageGroup:
        """Create a new group, or route into the segment's overflow when the cap is hit.

        With ``max_chunk_groups=N``, the first N callers each get their own
        new group. When the (N+1)th caller arrives, the most-recently-created
        group is promoted to the segment's overflow pool and all further
        callers join it. Total group count therefore never exceeds N.
        """
        max_groups = self.options.output_options.max_chunk_groups
        if max_groups is not None and len(segment.chunk_groups) >= max_groups:
            if segment.overflow_group is None:
                assert segment.chunk_groups, "max_chunk_groups must be >= 1 if set"
                segment.overflow_group = segment.chunk_groups[-1]
            return segment.overflow_group
        return self._create_segment_message_group(segment, channel)

    def _create_segment_message_group(
        self, segment: "OutputSegment", channel: Channel
    ) -> MessageGroup:
        """Create a MessageGroup attached to a specific segment."""
        opts = self.options.output_options
        compression = self._group_compression(segment.key, channel)
        group = MessageGroup(segment.writer, opts.chunk_size, compression, opts.max_chunk_span_ns)
        segment.chunk_groups.append(group)
        return group

    def _group_compression(self, segment_key: OutputKey, channel: Channel) -> CompressionType:
        """Compression for a new group, honoring the first processor override.

        Mirrors ``_composite_group_key``'s processor-chain walk, but for the
        compression choice: the first non-``None`` answer wins, else the
        run's default compression applies.
        """
        opts = self.options.output_options
        schema = self.schemas.get(channel.schema_id)
        for processor in opts.output_processors:
            override = processor.chunk_compression(segment_key, channel, schema)
            if override is not None:
                return override
        return opts.compression_type

    def _enforce_segment_memory_cap(self, writer: McapWriter) -> None:
        """Flush the largest builder in this writer's segment while total buffered > cap.

        Iterates because flushing the largest may still leave us over budget
        when several groups are near-full simultaneously.
        """
        cap = self.options.output_options.max_chunk_memory_bytes
        if cap is None or self.output_manager is None:
            return
        segment = self.output_manager.segment_for_writer(writer)
        if segment is None or not segment.chunk_groups:
            return
        while True:
            total = sum(g.buffered_bytes() for g in segment.chunk_groups)
            if total <= cap:
                return
            largest = max(segment.chunk_groups, key=MessageGroup.buffered_bytes)
            if largest.buffered_bytes() == 0:
                # No further progress possible — every builder is empty.
                return
            largest.flush_premature()

    def _replay_segment_open(
        self,
        route_key: OutputKey,
        stream_id: int,
        observed_time: int | None = None,
        observed_message: Message | None = None,
    ) -> None:
        """Fire ``on_segment_open`` once per processor as a stream reaches a segment.

        This keeps processors from needing stream seeks for common latch-style
        state: if stream B opens a segment before stream A has observed its
        latch, stream A's processors still get their own replay opportunity
        when stream A later writes to that segment.

        ``observed_time`` is the ``log_time`` of the record that triggered
        the open — used to backfill ``segment.start_time`` for dynamic
        (streaming-anchor) splits where the framework didn't know the
        segment's start upfront.
        """
        assert self.output_manager is not None
        segment = self.output_manager.get_or_create_segment(route_key)
        if segment.start_time == 0 and observed_time is not None:
            segment.start_time = observed_time

        for processor in self._get_processors(stream_id):
            processor_id = id(processor)
            if processor_id in segment.replayed_processor_ids:
                continue
            segment.replayed_processor_ids.add(processor_id)
            self._replay_from_processor(processor, route_key, segment, observed_message)

    def _replay_from_processor(
        self,
        processor: InputProcessor,
        route_key: OutputKey,
        segment: OutputSegment,
        observed_message: Message | None,
    ) -> None:
        context = SegmentContext(
            key=route_key,
            start_time=segment.start_time,
            observed_message=observed_message,
        )
        for channel_id, injected in processor.on_segment_open(context):
            assert self.output_manager is not None
            if channel_id not in self.channels:
                continue
            self.output_manager.ensure_channel_written(channel_id, route_key)
            segment.writer.add_message(
                channel_id=channel_id,
                log_time=injected.log_time,
                data=injected.data,
                publish_time=injected.publish_time,
                sequence=injected.sequence,
            )

    def process_message(
        self,
        message: Message,
        stream_id: int,
        *,
        input_channel_id: int | None = None,
        chain_channels: frozenset[int] | None = None,
    ) -> None:
        self.stats.messages_processed += 1
        assert self.output_manager is not None
        if input_channel_id is None:
            input_channel_id = message.channel_id

        # Topic filtering using cached decision (avoid repeated regex matching)
        if not self._is_channel_included(stream_id, message.channel_id):
            self.stats.filter_rejections += 1
            return

        # Run the message through the processor chain BEFORE routing. Each
        # processor yields zero or more messages and every yielded message
        # continues at the next processor. There is no primary/branch identity
        # rule: replacing a message and fan-out are both explicit in the
        # iterable length.
        processors = self._get_processors(stream_id)
        # Chain bypass: with a transcode, a chunk is decoded so a few channels
        # (camera/lidar) can be transformed, but it also carries the bulk of the
        # file's telemetry that no processor's ``message_scope`` covers. Those
        # messages pass through every processor unchanged, so skip the whole
        # chain and route them directly. ``chain_channels`` is the union of
        # processor scopes for this chunk (None ⇒ some processor needs every
        # message ⇒ no bypass).
        if not processors or (
            chain_channels is not None and message.channel_id not in chain_channels
        ):
            self._write_survivor(message, stream_id, input_channel_id)
            return

        pending: deque[tuple[Message, int, int, int | None]] = deque()
        pending.append((message, 0, stream_id, input_channel_id))
        survivors: list[tuple[Message, int, int | None]] = []
        dropped = False

        while pending:
            current, start_idx, current_stream_id, current_input_channel_id = pending.popleft()
            if start_idx >= len(processors):
                survivors.append((current, current_stream_id, current_input_channel_id))
                continue

            proc = processors[start_idx]
            context = self._message_context(current_stream_id, current_input_channel_id)
            produced = False
            for out in proc.on_message(context, current):
                out_message, out_stream_id, out_input_channel_id = self._processor_output_context(
                    proc,
                    out,
                    default_stream_id=current_stream_id,
                    default_input_channel_id=current_input_channel_id,
                    method="on_message",
                )
                produced = True
                pending.append((out_message, start_idx + 1, out_stream_id, out_input_channel_id))
            if not produced:
                dropped = True

        if not survivors:
            if dropped:
                self.stats.filter_rejections += 1
            return

        # Each survivor gets its own routing decision and writes — fan-out
        # messages may land in different segments and chunk grouping applies
        # per survivor.
        for survivor_msg, survivor_stream_id, survivor_input_channel_id in survivors:
            self._write_survivor(survivor_msg, survivor_stream_id, survivor_input_channel_id)

    def _processor_output_context(
        self,
        proc: InputProcessor,
        out: Message | MessageWithContext,
        *,
        default_stream_id: int,
        default_input_channel_id: int | None,
        method: str,
    ) -> tuple[Message, int, int | None]:
        if isinstance(out, MessageWithContext):
            if not isinstance(out.message, Message):
                msg = (
                    f"{type(proc).__name__}.{method} yielded MessageWithContext "
                    f"with {type(out.message).__name__}, expected Message"
                )
                raise TypeError(msg)
            return out.message, out.stream_id, out.input_channel_id
        if not isinstance(out, Message):
            msg = (
                f"{type(proc).__name__}.{method} yielded {type(out).__name__}, "
                "expected Message or MessageWithContext"
            )
            raise TypeError(msg)
        return out, default_stream_id, default_input_channel_id

    def _finalize_input_processors(self) -> None:
        """Flush end-of-stream output buffered by input processors.

        Runs once after every input chunk is consumed. Each unique processor's
        ``finalize()`` output is a fully-formed output record — routed and
        written directly, not re-fed through the chain (see the ``finalize``
        contract in ``processors/base.py``). Output channels registered via
        ``register_channel`` are marked included for every stream, but routers
        and segment-open replay may still depend on the producing input context.
        Buffered processors should yield ``MessageWithContext`` so the original
        stream/channel is preserved; plain ``Message`` yields retain the historic
        stream-0 fallback for compatibility.
        """
        assert self.output_manager is not None
        for proc in self._iter_unique_input_processors():
            for out in proc.finalize():
                message, stream_id, input_channel_id = self._processor_output_context(
                    proc,
                    out,
                    default_stream_id=0,
                    default_input_channel_id=None,
                    method="finalize",
                )
                if input_channel_id is None:
                    input_channel_id = message.channel_id
                self._write_survivor(
                    message, stream_id=stream_id, input_channel_id=input_channel_id
                )

    def _abort_input_processors(self) -> None:
        for proc in self._iter_unique_input_processors():
            try:
                proc.abort()
            except Exception:
                logger.exception("Failed to abort %s", type(proc).__name__)

    def _write_survivor(
        self,
        message: Message,
        stream_id: int,
        input_channel_id: int | None,
    ) -> None:
        """Route a post-chain message to its segment(s) and write it."""
        assert self.output_manager is not None

        for route_key in self._get_message_routes(stream_id, input_channel_id, message):
            self._replay_segment_open(
                route_key,
                stream_id,
                observed_time=message.log_time,
                observed_message=message,
            )
            self.output_manager.ensure_channel_written(message.channel_id, route_key)
            target_writer = self.output_manager.get_writer(route_key)

            if self.options.output_options.has_chunk_grouping:
                channel = self.channels.get(message.channel_id)
                if channel is None:
                    return
                group = self._get_or_create_group_for_channel(
                    message.channel_id, channel, target_writer
                )
                group.add_message(message)
                self._enforce_segment_memory_cap(target_writer)
            else:
                target_writer.add_message(
                    channel_id=message.channel_id,
                    log_time=message.log_time,
                    data=message.data,
                    publish_time=message.publish_time,
                )

    def _handle_schema_record(self, schema: Schema, stream_id: int) -> None:
        remapped_schema = self.remapper.remap_schema(stream_id, schema)
        if remapped_schema:
            self.schemas[remapped_schema.id] = remapped_schema

    def _handle_channel_record(self, channel: Channel, stream_id: int) -> None:
        remapped_channel = self.remapper.remap_channel(stream_id, channel)
        is_new = remapped_channel.id not in self.known_channels
        if is_new:
            self.known_channels.add(remapped_channel.id)
            self.channels[remapped_channel.id] = remapped_channel

        # Compute and cache this stream's include/exclude decision, under both
        # the original id (chunk-index lookups in _should_decode_chunk) and the
        # remapped id (post-remap message lookups in _is_channel_included).
        #
        # This must run for every stream, not only the one that first defines an
        # output id: on a merge a later stream's channel dedups to an
        # already-known id, and without its own cache entry every one of its
        # chunks is needlessly decoded (the slow path this avoids). The decision
        # is recomputed per stream — not shared by output id — because each input
        # can carry its own filter chain; sharing one stream's decision would
        # drop another stream's messages that its own filters would keep.
        if (stream_id, channel.id) not in self.channel_filter_cache:
            processors = self._get_processors(stream_id)
            should_include = True
            schema = self.schemas.get(remapped_channel.schema_id)
            context = ChannelContext(
                input=self._get_input_context(stream_id),
                input_channel_id=channel.id,
            )
            if processors:
                skip = any(
                    p.on_channel(context, remapped_channel, schema) & Action.SKIP
                    for p in processors
                )
                should_include = not skip
            for router in self.options.output_options.routers:
                router.on_channel(context, remapped_channel, schema)
            self.channel_filter_cache[(stream_id, channel.id)] = should_include
            self.channel_filter_cache[(stream_id, remapped_channel.id)] = should_include

            if is_new and not should_include:
                del self.channels[remapped_channel.id]

    def _register_extra_channel(
        self, stream_id: int, channel: Channel, source_channel_id: int | None = None
    ) -> Channel:
        """Register an output-only Channel allocated by a processor.

        Picks a fresh id above the writer registry and *reserves* it in the
        Remapper so a later input-channel allocation can't accidentally
        collide with it. Marks the channel as included in the filter cache
        for every input stream — synthetic channels bypass per-stream filters.

        When ``source_channel_id`` is given, the new id is derived
        deterministically from it (``base + source_channel_id``, where ``base``
        is fixed above all input channel ids). This makes a transcode's output
        channel ids depend only on the (stable) input channel — not on the order
        messages happen to arrive — so the same input compressed in independent
        time windows yields byte-compatible channel tables that ``merge`` can
        fast-copy instead of decoding to renumber.

        ``stream_id`` is retained in the signature for symmetry with
        ``remap_channel`` / ``remap_message`` closures even though the
        synthetic channel is global.
        """
        _ = stream_id  # synthetic channels are not stream-scoped
        if source_channel_id is not None:
            if self._extra_channel_base is None:
                self._extra_channel_base = max(self.known_channels, default=0) + 1
            new_id = self.remapper.reserve_channel_id(self._extra_channel_base + source_channel_id)
        else:
            new_id = self.remapper.reserve_channel_id(max(self.known_channels, default=0) + 1)
        new_channel = Channel(
            id=new_id,
            schema_id=channel.schema_id,
            topic=channel.topic,
            message_encoding=channel.message_encoding,
            metadata=dict(channel.metadata),
        )
        self.known_channels.add(new_id)
        self.channels[new_id] = new_channel
        for sid in range(len(self.options.inputs)):
            self.channel_filter_cache[(sid, new_id)] = True
        return new_channel

    def _register_extra_schema(self, name: str, encoding: str, data: bytes) -> int:
        """Register an output-only Schema allocated by a processor; return its id.

        Deduped by ``(name, encoding, data)`` so a processor calling this once
        per message reuses the same id. The Schema is stored in the shared
        ``self.schemas`` registry, so ``OutputManager.ensure_channel_written``
        writes it automatically when the first channel referencing it is
        written. Needed for transcode processors whose *output* schema
        (CompressedVideo, CompressedPointCloud2) has no input counterpart.
        """
        key = (name, encoding, data)
        existing = self._extra_schema_ids.get(key)
        if existing is not None:
            return existing
        preferred = max(self.schemas, default=0) + 1
        new_id = self.remapper.reserve_schema_id(preferred)
        self.schemas[new_id] = Schema(id=new_id, name=name, encoding=encoding, data=data)
        self._extra_schema_ids[key] = new_id
        return new_id

    def _handle_message_record(self, message: Message, stream_id: int) -> None:
        message_to_process = self.remapper.remap_message(stream_id, message)
        self.process_message(
            message_to_process,
            stream_id,
            input_channel_id=message.channel_id,
        )

    def _handle_attachment_record(self, attachment: Attachment, stream_id: int) -> None:
        self.stats.attachments_processed += 1
        assert self.output_manager is not None

        # Check all processors (includes AttachmentFilterProcessor and TimeFilterProcessor)
        context = self._get_input_context(stream_id)
        if any(
            p.on_attachment(context, attachment) == Action.SKIP
            for p in self._get_processors(stream_id)
        ):
            return

        self.output_manager.add_attachment(attachment)

    def _message_header_decision(
        self, header: MessageHeader, stream_id: int
    ) -> MessageHeaderDecision:
        if not self._is_channel_included(stream_id, header.channel_id):
            return MessageHeaderDecision.SKIP

        context = self._message_context(stream_id, header.channel_id)
        for proc in self._get_processors(stream_id):
            decision = proc.on_message_header(context, header)
            if decision is MessageHeaderDecision.SKIP:
                return decision
            if decision is MessageHeaderDecision.READ:
                return decision
        return MessageHeaderDecision.READ

    def _should_skip_message_payload(self, header: MessageHeader, stream_id: int) -> bool:
        if self._message_header_decision(header, stream_id) is not MessageHeaderDecision.SKIP:
            return False
        self.stats.messages_processed += 1
        self.stats.filter_rejections += 1
        return True

    def _handle_metadata_record(self, metadata: Metadata, stream_id: int) -> None:
        self.stats.metadata_processed += 1
        processors = self._get_processors(stream_id)
        context = self._get_input_context(stream_id)
        if not processors or all(
            p.on_metadata(context, metadata) != Action.SKIP for p in processors
        ):
            assert self.output_manager is not None
            self.output_manager.add_metadata(name=metadata.name, metadata=metadata.metadata)

    def _lazy_chunk_from_index(self, chunk_index: ChunkIndex) -> LazyChunk:
        return LazyChunk(
            message_start_time=chunk_index.message_start_time,
            message_end_time=chunk_index.message_end_time,
            uncompressed_size=chunk_index.uncompressed_size,
            uncompressed_crc=0,
            compression=chunk_index.compression,
            record_start=chunk_index.chunk_start_offset,
            data_len=chunk_index.compressed_size,
        )

    def _chunk_context_from_index(
        self,
        stream_id: int,
        chunk_index: ChunkIndex,
    ) -> ChunkContext:
        indexes = tuple(
            MessageIndex(channel_id=channel_id, timestamps=[], offsets=[])
            for channel_id in chunk_index.message_index_offsets
        )
        return ChunkContext(
            input=self._get_input_context(stream_id),
            message_indexes=indexes or None,
            chunk_start_time=chunk_index.message_start_time,
            chunk_end_time=chunk_index.message_end_time,
        )

    def _should_skip_indexed_chunk_before_indexes(
        self,
        chunk: LazyChunk,
        chunk_index: ChunkIndex,
        stream_id: int,
    ) -> bool:
        context = self._chunk_context_from_index(stream_id, chunk_index)
        for proc in self._get_processors(stream_id):
            decision = proc.on_chunk(context, chunk)
            if decision == ChunkDecision.SKIP:
                return True
            if decision in (ChunkDecision.DECODE, ChunkDecision.DECODE_VERIFY):
                return False

        channel_ids = chunk_index.message_index_offsets
        return bool(
            channel_ids
            and all(
                not self._is_channel_included(stream_id, channel_id) for channel_id in channel_ids
            )
        )

    def _indexed_chunk_scan_plan(
        self,
        stream_id: int,
        summary: Summary | None,
    ) -> list[ChunkIndex] | None:
        if summary is None or not summary.chunk_indexes:
            return None
        input_options = self.options.inputs[stream_id].options
        if input_options.include_metadata or input_options.include_attachments:
            return None

        selected: list[ChunkIndex] = []
        skipped = 0
        for chunk_index in sorted(summary.chunk_indexes, key=lambda item: item.message_start_time):
            chunk = self._lazy_chunk_from_index(chunk_index)
            if self._should_skip_indexed_chunk_before_indexes(chunk, chunk_index, stream_id):
                skipped += 1
                continue
            selected.append(chunk_index)

        return selected if skipped else None

    def _read_message_indexes_for_chunk(
        self,
        input_stream: IO[bytes],
        chunk_index: ChunkIndex,
    ) -> list[MessageIndex]:
        if not chunk_index.message_index_offsets:
            return []
        indexes: list[MessageIndex] = []
        for offset in sorted(chunk_index.message_index_offsets.values()):
            input_stream.seek(offset)
            indexes.append(MessageIndex.read_record(input_stream))
        return indexes

    def _generate_indexed_chunks_from_plan(
        self,
        input_stream: IO[bytes],
        stream_id: int,
        chunk_indexes: list[ChunkIndex],
    ) -> Iterator[PendingChunk]:
        for chunk_index in chunk_indexes:
            chunk = self._lazy_chunk_from_index(chunk_index)
            indexes = self._read_message_indexes_for_chunk(input_stream, chunk_index)
            self.stats.chunks_processed += 1
            yield PendingChunk(chunk, indexes, stream_id, input_stream, chunk.message_start_time)

    def _early_bail_end_time_ns(self, stream_id: int) -> int | None:
        input_options = self.options.inputs[stream_id].options
        if not input_options.is_early_bail_enabled:
            return None
        if input_options.invert_time:
            return None
        if input_options.include_metadata or input_options.include_attachments:
            return None
        if input_options.always_decode_chunk or input_options.extra_processors:
            return None
        end_time_ns = input_options.end_time_ns
        if not isinstance(end_time_ns, int):
            return None
        return end_time_ns

    def _generate_chunks_from_stream(
        self, input_stream: IO[bytes], stream_id: int, summary: Summary | None = None
    ) -> Iterator[PendingChunk]:
        """Generate chunks from a single stream in file order.

        Yields PendingChunk objects with timestamp for ordered merging.
        Non-chunk records (Schema, Channel, Message, Attachment, Metadata) are processed directly.
        Uses lazy_chunks=True for efficiency - chunk data is only read when needed.
        """
        indexed_plan = self._indexed_chunk_scan_plan(stream_id, summary)
        if indexed_plan is not None:
            yield from self._generate_indexed_chunks_from_plan(
                input_stream, stream_id, indexed_plan
            )
            return

        pending: PendingChunk | None = None
        early_bail_end_time_ns = self._early_bail_end_time_ns(stream_id)

        try:
            indexes: list[MessageIndex] = []
            magic = _read_exact(input_stream, MAGIC_SIZE)
            if magic != MAGIC:
                raise InvalidMagicError(magic)

            while True:
                record_start = input_stream.tell()
                header_bytes = input_stream.read(OPCODE_AND_LEN_STRUCT.size)
                if len(header_bytes) == 0:
                    break
                if len(header_bytes) < OPCODE_AND_LEN_STRUCT.size:
                    raise EndOfFileError
                opcode, length = OPCODE_AND_LEN_STRUCT.unpack(header_bytes)
                if length > _RECORD_SIZE_LIMIT:
                    raise RecordLengthLimitExceededError(opcode, length, _RECORD_SIZE_LIMIT)

                if opcode == Opcode.MESSAGE:
                    if pending:
                        yield pending
                        pending = None
                    if length < _MESSAGE_STRUCT.size:
                        raise EndOfFileError
                    if self._has_payload_skipping_filters[stream_id]:
                        message_header_bytes = _read_exact(input_stream, _MESSAGE_STRUCT.size)
                        channel_id, sequence, log_time, publish_time = _MESSAGE_STRUCT.unpack(
                            message_header_bytes
                        )
                        data_length = length - _MESSAGE_STRUCT.size
                        message_header = MessageHeader(
                            channel_id=channel_id,
                            sequence=sequence,
                            log_time=log_time,
                            publish_time=publish_time,
                            data_length=data_length,
                        )
                        if (
                            early_bail_end_time_ns is not None
                            and log_time >= early_bail_end_time_ns
                        ):
                            self.stats.messages_processed += 1
                            self.stats.filter_rejections += 1
                            break
                        if self._should_skip_message_payload(message_header, stream_id):
                            input_stream.seek(data_length, os.SEEK_CUR)
                            continue
                        data: bytes | memoryview = _read_exact(input_stream, data_length)
                    else:
                        message_body = _read_exact(input_stream, length)
                        channel_id, sequence, log_time, publish_time = _MESSAGE_STRUCT.unpack_from(
                            message_body
                        )
                        data_length = length - _MESSAGE_STRUCT.size
                        message_header = MessageHeader(
                            channel_id=channel_id,
                            sequence=sequence,
                            log_time=log_time,
                            publish_time=publish_time,
                            data_length=data_length,
                        )
                        if self._should_skip_message_payload(message_header, stream_id):
                            continue
                        data = memoryview(message_body)[_MESSAGE_STRUCT.size :]
                    self._handle_message_record(
                        Message(
                            channel_id=channel_id,
                            sequence=sequence,
                            log_time=log_time,
                            publish_time=publish_time,
                            data=data,
                        ),
                        stream_id,
                    )
                    continue

                if opcode == Opcode.CHUNK:
                    if pending:
                        yield pending
                    self.stats.chunks_processed += 1
                    chunk = LazyChunk.read_from_stream(input_stream, record_start, length)
                    pending = PendingChunk(
                        chunk, indexes := [], stream_id, input_stream, chunk.message_start_time
                    )
                    continue

                record: McapRecord | None
                if opcode == Opcode.MESSAGE_INDEX:
                    data = _read_exact(input_stream, length)
                    record = MessageIndex.read(data)
                elif record_cls := OPCODE_TO_RECORD.get(opcode):
                    data = _read_exact(input_stream, length)
                    record = record_cls.read(data)
                else:
                    input_stream.seek(length, os.SEEK_CUR)
                    continue

                # MessageIndex is by far the most common record on index-heavy
                # files (one per channel per chunk). isinstance against the
                # McapRecord ABC hits ABCMeta.__instancecheck__ and dominated the
                # main thread; a `type(...) is` identity check sidesteps it. The
                # cold branches keep isinstance — negligible volume, and it lets
                # the type checker narrow.
                if type(record) is MessageIndex:
                    indexes.append(record)
                    continue

                if pending:
                    yield pending
                    pending = None

                if isinstance(record, Header):
                    pass
                elif isinstance(record, Schema):
                    self._handle_schema_record(record, stream_id)
                elif isinstance(record, Channel):
                    self._handle_channel_record(record, stream_id)
                elif isinstance(record, Attachment):
                    self._handle_attachment_record(record, stream_id)
                elif isinstance(record, Metadata):
                    self._handle_metadata_record(record, stream_id)
                elif isinstance(record, (DataEnd, Footer)):
                    break

        except McapError as e:
            console.print(f"[yellow]Warning (stream {stream_id}): {e}[/yellow]")
            self.stats.errors_encountered += 1

        # Yield final pending chunk if any
        if pending:
            yield pending

    def _build_input_contexts(self, summaries: list[Summary | None]) -> list[InputContext]:
        """Build per-stream processor context from currently available resources."""
        contexts: list[InputContext] = []
        for stream_id, _input_file in enumerate(self.options.inputs):
            summary = summaries[stream_id] if stream_id < len(summaries) else None

            def remap_channel(channel: Channel, sid: int = stream_id) -> Channel:
                return self.remapper.remap_channel(sid, channel)

            def remap_message(message: Message, sid: int = stream_id) -> Message:
                return self.remapper.remap_message(sid, message)

            def register_channel(
                channel: Channel,
                source_channel_id: int | None = None,
                sid: int = stream_id,
            ) -> Channel:
                return self._register_extra_channel(sid, channel, source_channel_id)

            def register_schema(name: str, encoding: str, data: bytes) -> int:
                return self._register_extra_schema(name, encoding, data)

            contexts.append(
                InputContext(
                    stream_id=stream_id,
                    summary=summary,
                    statistics=summary.statistics if summary is not None else None,
                    chunk_indexes=(
                        tuple(summary.chunk_indexes)
                        if summary is not None and summary.chunk_indexes
                        else None
                    ),
                    remap_channel=remap_channel,
                    remap_message=remap_message,
                    register_channel=register_channel,
                    register_schema=register_schema,
                )
            )
        return contexts

    def _pipeline_context(
        self,
        output_segments: list[tuple[OutputKey, int, int]] | None = None,
    ) -> PipelineContext:
        segment_infos = tuple(
            OutputSegmentInfo(key=key, start_time=start_time, end_time=end_time)
            for key, start_time, end_time in output_segments or []
        )
        return PipelineContext(
            inputs=tuple(self._get_input_context(i) for i in range(len(self.options.inputs))),
            output_segments=segment_infos,
        )

    def _prepare_input_processors(self) -> None:
        """Give input processors per-stream context before record processing starts."""
        for stream_id in range(len(self.options.inputs)):
            context = self._get_input_context(stream_id)
            for processor in self._get_processors(stream_id):
                processor.prepare_input(context)

    def process(
        self,
        output_stream: BinaryIO | None = None,
    ) -> ProcessingStats:
        """Main processing function."""
        output_opts = self.options.output_options
        header = self._resolve_output_header()

        # Pass 1 — collect summaries from every input file (rewinding each).
        summaries: list[Summary | None] = []
        for input_opt in self.options.inputs:
            input_stream = input_opt.stream
            try:
                summary = get_summary(input_stream)
            except McapError:
                # In recovery mode, if we can't get summary (e.g., truncated file),
                # continue without it - we'll discover schemas/channels during chunk processing
                summary = None
            summaries.append(summary)
            input_stream.seek(0)

        for stream_id, context in enumerate(self._build_input_contexts(summaries)):
            self._input_contexts[stream_id] = context

        # Output routers initialize first because they define statically-known
        # output segments. Input processors then initialize with those segments
        # included in PipelineContext.
        for router in output_opts.routers:
            router.initialize(self._pipeline_context())

        known_segments: list[tuple[OutputKey, int, int]] = list(self._iter_known_output_segments())
        if not known_segments and not output_opts.is_splitting:
            known_segments = [(0, 0, 0)]

        pipeline_context = self._pipeline_context(known_segments)
        for proc in self._iter_unique_input_processors():
            proc.initialize(pipeline_context)
        for output_processor in output_opts.output_processors:
            output_processor.initialize(pipeline_context)

        # Pass 2 — pre-load schemas / channels via the standard handlers, which
        # also populate the channel filter cache.
        for stream_id, summary in enumerate(summaries):
            if summary is None:
                continue
            for schema in summary.schemas.values():
                self._handle_schema_record(schema, stream_id)
            for channel in summary.channels.values():
                self._handle_channel_record(channel, stream_id)

        self._prepare_input_processors()

        if not output_opts.is_splitting and output_stream is None:
            raise ValueError("output_stream is required when not splitting")

        if output_opts.is_splitting:
            open_output = None
        else:
            assert output_stream is not None
            open_output = self._build_single_output_opener(output_stream)
        self.output_manager = OutputManager(
            output_opts,
            self.schemas,
            self.channels,
            header,
            open_output=open_output,
            template_fields=self._template_fields_for_output_key,
        )
        # Pre-create statically-known segments. In single-output mode this creates segment 0.
        for key, start_time, end_time in known_segments:
            self.output_manager.get_or_create_segment(key, start_time=start_time, end_time=end_time)

        try:
            total_size = self.options.total_size
            with file_progress("[bold blue]Processing MCAP...", console) as progress:
                task = progress.add_task("Processing", total=total_size)

                # Wrap streams to track progress incrementally
                wrapped_streams = [
                    ProgressTrackingIO(inp.stream, task, progress, inp.stream.tell())
                    for inp in self.options.inputs
                ]

                # Create chunk generators for each wrapped stream
                chunk_generators = [
                    self._generate_chunks_from_stream(
                        wrapped_stream, stream_id, summaries[stream_id]
                    )
                    for stream_id, wrapped_stream in enumerate(wrapped_streams)
                ]

                # Process chunks in timestamp order using heapq.merge.
                # For DECODE-bound chunks the (slow) zstd/lz4 decompression is
                # offloaded to a small worker pool so that the main thread can
                # keep writing the previous chunk's records while the next
                # chunk decompresses in parallel. zstd/lz4 release the GIL so
                # the speedup is genuine.
                self._run_chunk_pipeline(heapq.merge(*chunk_generators))

                # Flush any output buffered past end-of-stream by input
                # processors (e.g. an async encoder draining trailing frames)
                # before the writers are finished.
                self._finalize_input_processors()

                # Complete progress
                progress.update(task, completed=total_size)

        except BaseException:
            self._abort_input_processors()
            raise
        finally:
            assert self.output_manager is not None
            segment_stats = self.output_manager.finish_all()
            # Aggregate writer statistics from all segments
            if segment_stats:
                first = next(iter(segment_stats.values()))
                times_with_messages = [s for s in segment_stats.values() if s.message_count > 0]
                total = Statistics(
                    message_count=sum(s.message_count for s in segment_stats.values()),
                    schema_count=first.schema_count,
                    channel_count=first.channel_count,
                    attachment_count=sum(s.attachment_count for s in segment_stats.values()),
                    metadata_count=sum(s.metadata_count for s in segment_stats.values()),
                    chunk_count=sum(s.chunk_count for s in segment_stats.values()),
                    message_start_time=(
                        min(s.message_start_time for s in times_with_messages)
                        if times_with_messages
                        else 0
                    ),
                    message_end_time=(
                        max(s.message_end_time for s in times_with_messages)
                        if times_with_messages
                        else 0
                    ),
                    channel_message_counts={},
                )
                self.stats.writer_statistics = total

        return self.stats

    def _should_decode_chunk(
        self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex], stream_id: int
    ) -> ChunkDecision:
        """Determine if chunk should be decoded or can be fast-copied.

        Returns:
            ChunkDecision:
            - SKIP: Chunk should be skipped entirely (filtered out)
            - CONTINUE: Chunk can be fast-copied without decoding
            - RECOMPRESS: Chunk needs new compression but no per-message work
            - DECODE_VERIFY: Stream had a remap; decode and verify in-chunk
              records, then fast-copy if clean else fall through to DECODE
            - DECODE: Chunk must be re-emitted (filter, rechunk, channel remap)
        """
        output_opts = self.options.output_options
        context = self._chunk_context(stream_id, indexes)

        # Ask input processors for chunk-level decision (time filtering, always_decode, etc.).
        # A processor that mutates Channel/Schema records (e.g. TopicRewrite) may
        # return DECODE_VERIFY to ask the dispatcher to re-verify that the chunk's
        # embedded Channel/Schema records still match the writer's view.
        decode_verify_requested = False
        for proc in self._get_processors(stream_id):
            decision = proc.on_chunk(context, chunk)
            if decision == ChunkDecision.SKIP:
                return ChunkDecision.SKIP
            if decision == ChunkDecision.DECODE:
                return ChunkDecision.DECODE
            if decision == ChunkDecision.DECODE_VERIFY:
                decode_verify_requested = True

        # Ask output routers (split routing forces DECODE on boundary chunks)
        for proc in output_opts.routers:
            decision = proc.on_chunk(context, chunk)
            if decision == ChunkDecision.DECODE:
                return ChunkDecision.DECODE
            if decision == ChunkDecision.DECODE_VERIFY:
                decode_verify_requested = True

        # Force decode if chunk grouping is active (must reorganize messages)
        if output_opts.has_chunk_grouping:
            return ChunkDecision.DECODE

        if not indexes:
            return ChunkDecision.DECODE

        output_compression = output_opts.compression_type
        compression_mismatch = chunk.compression != output_compression.value
        zstd_level_requested = (
            output_compression == CompressionType.ZSTD and output_opts.zstd_level is not None
        )

        # Single pass: check channel availability, per-channel remap, and filtering.
        has_include = False
        has_exclude = False
        for idx in indexes:
            ch_id = idx.channel_id
            if not self.remapper.has_channel(stream_id, ch_id):
                return ChunkDecision.DECODE
            # Messages reference the original channel_id; if it was reassigned
            # the chunk's message bytes are wrong without record-level rewrite.
            if self.remapper.was_channel_remapped(stream_id, ch_id):
                return ChunkDecision.DECODE
            cache_key = (stream_id, ch_id)
            if cache_key not in self.channel_filter_cache:
                return ChunkDecision.DECODE
            if self.channel_filter_cache[cache_key]:
                has_include = True
            else:
                has_exclude = True
            if has_include and has_exclude:
                return ChunkDecision.DECODE

        if has_exclude and not has_include:
            return ChunkDecision.SKIP

        # Stream-level remap: messages are fine (no channel was reassigned
        # above), but in-chunk Schema/Channel records may be stale because
        # some writers eagerly embed every channel in every chunk. Defer
        # to a verify pass on the decoded records. Same logic applies when
        # an input/output processor explicitly requested DECODE_VERIFY (e.g.
        # TopicRewrite mutating Channel.topic).
        if self.remapper.stream_had_remap(stream_id) or decode_verify_requested:
            if compression_mismatch or zstd_level_requested:
                # Verifying then fast-copying would still leave the wrong
                # codec/level; fall back to full DECODE for this rare combo.
                return ChunkDecision.DECODE
            return ChunkDecision.DECODE_VERIFY

        if compression_mismatch or zstd_level_requested:
            return ChunkDecision.RECOMPRESS

        return ChunkDecision.CONTINUE

    @staticmethod
    def _compose_route_groups(route_groups: list[list[RouteKey]]) -> list[OutputKey]:
        if not route_groups:
            return [0]
        output_keys: list[OutputKey] = []
        for route_tuple in product(*route_groups):
            key: OutputKey = route_tuple[0] if len(route_tuple) == 1 else route_tuple
            if key not in output_keys:
                output_keys.append(key)
        return output_keys

    def _template_fields_for_output_key(self, key: OutputKey) -> dict[str, TemplateValue]:
        routers = self.options.output_options.routers
        if not routers:
            return {}
        route_keys: tuple[int | str, ...]
        if len(routers) == 1:
            if not isinstance(key, (int, str)):
                raise ValueError(f"Expected one route key, got {key!r}")
            route_keys = (key,)
        else:
            if not isinstance(key, tuple) or len(key) != len(routers):
                raise ValueError(f"Expected {len(routers)} route keys, got {key!r}")
            route_keys = key

        fields: dict[str, TemplateValue] = {}
        for router, route_key in zip(routers, route_keys, strict=True):
            router_fields = router.template_fields(route_key)
            duplicate_fields = fields.keys() & router_fields.keys()
            if duplicate_fields:
                names = ", ".join(sorted(duplicate_fields))
                raise ValueError(f"Multiple output routers provide template fields: {names}")
            fields.update(router_fields)
        return fields

    def _get_chunk_routes(
        self,
        stream_id: int,
        chunk: Chunk | LazyChunk,
        indexes: list[MessageIndex],
    ) -> list[OutputKey]:
        """Get output keys for a chunk from output routers."""
        route_groups: list[list[RouteKey]] = []
        context = self._chunk_context(stream_id, indexes)
        for proc in self.options.output_options.routers:
            route_result = proc.route_chunk(context, chunk)
            if route_result is SPLIT_REQUIRED:
                msg = (
                    f"{type(proc).__name__}.route_chunk requested a split after "
                    "the chunk was classified as fast-copy"
                )
                raise RuntimeError(msg)
            routes = list(cast("Iterable[RouteKey]", route_result))
            if not routes:
                return []
            route_groups.append(routes)
        return self._compose_route_groups(route_groups)

    def _iter_known_output_segments(self) -> Iterator[tuple[OutputKey, int, int]]:
        """Yield statically known output routes for pre-creating segments."""
        known: list[list[tuple[int | str, int, int]]] = []
        for proc in self.options.output_options.routers:
            segments = proc.output_segments()
            if segments is None:
                return
            proc_known: list[tuple[int | str, int, int]] = []
            for segment in segments:
                assert isinstance(segment.key, (int, str))
                proc_known.append((segment.key, segment.start_time, segment.end_time))
            known.append(proc_known)

        if not known:
            return

        routes: list[tuple[tuple[int | str, ...], int, int]] = [((), 0, 0)]
        for proc_known in known:
            next_routes: list[tuple[tuple[int | str, ...], int, int]] = []
            for base_key, base_start, base_end in routes:
                for key, start_time, end_time in proc_known:
                    flattened_key = (*base_key, key)
                    combined_start = start_time if base_start == 0 else max(base_start, start_time)
                    if end_time == 0:
                        combined_end = base_end
                    elif base_end == 0:
                        combined_end = end_time
                    else:
                        combined_end = min(base_end, end_time)
                    if combined_start >= combined_end and combined_end != 0:
                        continue
                    next_routes.append((flattened_key, combined_start, combined_end))
            routes = next_routes

        for route_key, start_time, end_time in routes:
            key: OutputKey = route_key[0] if len(route_key) == 1 else route_key
            yield key, start_time, end_time

    def _build_single_output_opener(self, output_stream: BinaryIO) -> OutputStreamOpener:
        opened = False

        def open_output(
            key: OutputKey, index: int, start_time: int, end_time: int
        ) -> tuple[str, BinaryIO]:
            nonlocal opened
            _ = key, index, start_time, end_time
            if opened:
                raise ValueError("single-output mode only supports one output segment")
            opened = True
            path = self.options.output_options.output_template or "output.mcap"
            return path, output_stream

        return open_output

    def _get_message_routes(
        self,
        stream_id: int,
        input_channel_id: int | None,
        message: Message,
    ) -> list[OutputKey]:
        """Get output keys for a post-processor-chain message."""
        route_groups: list[list[RouteKey]] = []
        context = self._message_context(stream_id, input_channel_id)
        for proc in self.options.output_options.routers:
            routes = list(proc.route_message(context, message))
            if not routes:
                return []
            route_groups.append(routes)
        return self._compose_route_groups(route_groups)

    def _run_chunk_pipeline(self, chunks: Iterator[PendingChunk]) -> None:
        """Consume chunks from the merged iterator, decompressing DECODE chunks ahead.

        Maintains a short queue of in-flight decompression/recompression futures
        so the main thread can be writing chunk N's records while workers
        process N+1..N+W in parallel. For fast-copy and skip chunks the queue
        entry has no future and is handled directly.

        The inflight limit is kept modest (≤8): zstd/lz4 work is CPU-bound and
        more than ~8 workers on a typical MCAP saturates memory bandwidth
        rather than helping throughput.
        """
        max_inflight = min(8, os.cpu_count() or 1)
        queue: deque[
            tuple[PendingChunk, ChunkDecision, Future[list[McapRecord]] | Future[Chunk] | None]
        ] = deque()

        # Per-stream input fd for offloaded chunk reads (regular files only).
        # None means "not a seekable regular file" — fall back to main-thread reads.
        stream_fds: dict[int, int | None] = {}

        def regular_file_fd(stream_id: int, stream: IO[bytes]) -> int | None:
            if stream_id not in stream_fds:
                fd: int | None = None
                try:
                    candidate = stream.fileno()
                    if stat.S_ISREG(os.fstat(candidate).st_mode):
                        fd = candidate
                except (OSError, ValueError, AttributeError):
                    fd = None
                stream_fds[stream_id] = fd
            return stream_fds[stream_id]

        with ThreadPoolExecutor(max_workers=max_inflight) as pool:
            target_compression = self.options.output_options.compression_type
            target_zstd_level = self.options.output_options.zstd_level

            def enqueue_next() -> bool:
                try:
                    pending = next(chunks)
                except StopIteration:
                    return False
                decision = self._should_decode_chunk(
                    pending.chunk, pending.indexes, pending.stream_id
                )
                future: Future[list[McapRecord]] | Future[Chunk] | None = None

                # Fast path: recompressing a still-lazy chunk from a regular
                # file. Offload the (large) data read to the worker via os.pread
                # so the main thread isn't the serial read bottleneck. The chunk
                # stays a LazyChunk here — routing only needs its metadata.
                if decision == ChunkDecision.RECOMPRESS and type(pending.chunk) is LazyChunk:
                    fd = regular_file_fd(pending.stream_id, pending.stream)
                    if fd is not None:
                        record_start = pending.chunk.record_start
                        try:
                            opcode_and_len = _pread_exact(fd, 9, record_start)
                        except EOFError as e:
                            console.print(
                                f"[yellow]Warning (stream {pending.stream_id}): "
                                f"Failed to read chunk: {e}[/yellow]"
                            )
                            self.stats.errors_encountered += 1
                            return True
                        _opcode, record_length = OPCODE_AND_LEN_STRUCT.unpack(opcode_and_len)
                        body_offset = record_start + 9
                        # Keep the progress bar honest: the body bytes are read
                        # out-of-band on a worker, bypassing the progress wrapper.
                        if isinstance(pending.stream, ProgressTrackingIO):
                            pending.stream.mark_read_to(body_offset + record_length)
                        future = pool.submit(
                            _read_and_recompress_chunk,
                            fd,
                            body_offset,
                            record_length,
                            target_compression,
                            target_zstd_level,
                        )
                        queue.append((pending, decision, future))
                        return True

                if decision == ChunkDecision.DECODE and type(pending.chunk) is LazyChunk:
                    fd = regular_file_fd(pending.stream_id, pending.stream)
                    if fd is not None:
                        record_start = pending.chunk.record_start
                        try:
                            opcode_and_len = _pread_exact(fd, 9, record_start)
                        except EOFError as e:
                            console.print(
                                f"[yellow]Warning (stream {pending.stream_id}): "
                                f"Failed to read chunk: {e}[/yellow]"
                            )
                            self.stats.errors_encountered += 1
                            return True
                        _opcode, record_length = OPCODE_AND_LEN_STRUCT.unpack(opcode_and_len)
                        body_offset = record_start + 9
                        if isinstance(pending.stream, ProgressTrackingIO):
                            pending.stream.mark_read_to(body_offset + record_length)
                        future = pool.submit(
                            _read_and_decode_chunk_records,
                            fd,
                            body_offset,
                            record_length,
                        )
                        queue.append((pending, decision, future))
                        return True

                if decision in (
                    ChunkDecision.DECODE,
                    ChunkDecision.DECODE_VERIFY,
                    ChunkDecision.RECOMPRESS,
                ):
                    try:
                        # Seek+read happens on the shared input stream (main thread).
                        # Only the CPU-bound codec work is offloaded to workers.
                        materialized = pending.get_chunk()
                    except (EOFError, McapError) as e:
                        console.print(
                            f"[yellow]Warning (stream {pending.stream_id}): "
                            f"Failed to read chunk: {e}[/yellow]"
                        )
                        self.stats.errors_encountered += 1
                        return True
                    if decision == ChunkDecision.RECOMPRESS:
                        future = pool.submit(
                            _recompress_chunk, materialized, target_compression, target_zstd_level
                        )
                    else:
                        future = pool.submit(_decode_chunk_records, materialized)
                queue.append((pending, decision, future))
                return True

            # Prime the pipeline
            while len(queue) < max_inflight and enqueue_next():
                pass

            # Drain in order, refilling as we go
            while queue:
                pending, decision, future = queue.popleft()
                enqueue_next()
                self._process_chunk_smart(pending, decision, future)

    def _prepare_chunk_writes(
        self, pending: PendingChunk
    ) -> list[tuple[OutputKey, McapWriter, dict[int, MessageIndex]]]:
        """Resolve routes, fire segment-open, and ensure all referenced channels.

        Returns ``(route_key, writer, indices_by_channel)`` entries so each branch in
        ``_process_chunk_smart`` can hand the chunk to the writer without
        duplicating the boilerplate.
        """
        assert self.output_manager is not None
        writes: list[tuple[OutputKey, McapWriter, dict[int, MessageIndex]]] = []
        for route_key in self._get_chunk_routes(
            pending.stream_id,
            pending.chunk,
            pending.indexes,
        ):
            self._replay_segment_open(
                route_key,
                pending.stream_id,
                observed_time=pending.chunk.message_start_time,
            )
            writer = self.output_manager.get_writer(route_key)
            for idx in pending.indexes:
                if self._is_channel_included(pending.stream_id, idx.channel_id):
                    self.output_manager.ensure_channel_written(idx.channel_id, route_key)
            writes.append((route_key, writer, {idx.channel_id: idx for idx in pending.indexes}))
        return writes

    def _process_chunk_smart(
        self,
        pending: PendingChunk,
        decision: ChunkDecision,
        future: Future[list[McapRecord]] | Future[Chunk] | None,
    ) -> None:
        """Dispatch one pre-classified chunk."""
        assert self.output_manager is not None

        if decision == ChunkDecision.SKIP:
            return

        if decision == ChunkDecision.DECODE:
            assert future is not None
            try:
                records = cast("Future[list[McapRecord]]", future).result()
            except (EOFError, McapError) as e:
                console.print(
                    f"[yellow]Warning (stream {pending.stream_id}): "
                    f"Failed to decode chunk: {e}[/yellow]"
                )
                self.stats.errors_encountered += 1
                return
            self._process_decoded_records(
                records,
                pending.stream_id,
                self._chain_channels_for_chunk(pending.stream_id, pending),
            )
            return

        if decision == ChunkDecision.DECODE_VERIFY:
            assert future is not None
            try:
                records = cast("Future[list[McapRecord]]", future).result()
            except (EOFError, McapError) as e:
                console.print(
                    f"[yellow]Warning (stream {pending.stream_id}): "
                    f"Failed to decode chunk: {e}[/yellow]"
                )
                self.stats.errors_encountered += 1
                return
            if _chunk_records_match_writer_view(records, self.schemas, self.channels):
                # Clean: emit the materialized chunk as-is, skipping re-emit.
                assert isinstance(pending.chunk, Chunk)
                for _route_key, target_writer, indices_by_channel in self._prepare_chunk_writes(
                    pending
                ):
                    target_writer.add_chunk(pending.chunk, indices_by_channel)
                    self.stats.chunks_copied += 1
                self.stats.chunks_verified += 1
                return
            # Stale records: fall through to the existing re-emit path.
            self._process_decoded_records(
                records,
                pending.stream_id,
                self._chain_channels_for_chunk(pending.stream_id, pending),
            )
            return

        if decision == ChunkDecision.RECOMPRESS:
            assert future is not None
            try:
                new_chunk = cast("Future[Chunk]", future).result()
            except (EOFError, McapError) as e:
                console.print(
                    f"[yellow]Warning (stream {pending.stream_id}): "
                    f"Failed to recompress chunk: {e}[/yellow]"
                )
                self.stats.errors_encountered += 1
                return
            for _route_key, target_writer, indices_by_channel in self._prepare_chunk_writes(
                pending
            ):
                target_writer.add_chunk(new_chunk, indices_by_channel)
                self.stats.chunks_copied += 1
            return

        # Fast-copy path (CONTINUE) -> nothing about the chunk must be changed.
        # We pipe raw bytes from input to output (no Chunk materialization).
        for _route_key, target_writer, indices_by_channel in self._prepare_chunk_writes(pending):
            if isinstance(pending.chunk, LazyChunk):
                try:
                    target_writer.add_chunk_raw(pending.stream, pending.chunk, indices_by_channel)
                except (EOFError, McapError) as e:
                    console.print(
                        f"[yellow]Warning (stream {pending.stream_id}): Failed to copy chunk: "
                        f"{e}[/yellow]"
                    )
                    self.stats.errors_encountered += 1
                    return
            else:
                target_writer.add_chunk(pending.chunk, indices_by_channel)
            self.stats.chunks_copied += 1

    def _process_decoded_records(
        self,
        records: list[McapRecord],
        stream_id: int,
        chain_channels: frozenset[int] | None = None,
    ) -> None:
        """Process records that were already decoded from a chunk (by a worker)."""
        self.stats.chunks_decoded += 1
        for chunk_record in records:
            if isinstance(chunk_record, Message):
                message_to_write = self.remapper.remap_message(stream_id, chunk_record)
                self.process_message(
                    message_to_write,
                    stream_id,
                    input_channel_id=chunk_record.channel_id,
                    chain_channels=chain_channels,
                )
            elif isinstance(chunk_record, Schema):
                self._handle_schema_record(chunk_record, stream_id)
            elif isinstance(chunk_record, Channel):
                self._handle_channel_record(chunk_record, stream_id)

    def _chain_channels_for_chunk(
        self, stream_id: int, pending: PendingChunk
    ) -> frozenset[int] | None:
        """Union of input-processor message scopes for one chunk.

        Returns the set of channel ids whose messages must traverse the
        processor chain, or ``None`` when some processor needs *every* message
        (scope ``ALL``) so no bypass is safe. Cheap: a handful of processors,
        evaluated once per decoded chunk.
        """
        processors = self._get_processors(stream_id)
        if not processors:
            return frozenset()
        chunk = pending.chunk
        context = ChunkContext(
            input=self._get_input_context(stream_id),
            message_indexes=tuple(pending.indexes) if pending.indexes else None,
            chunk_start_time=chunk.message_start_time,
            chunk_end_time=chunk.message_end_time,
        )
        wanted: set[int] = set()
        for proc in processors:
            scope = proc.message_scope(context)
            if scope.kind is MessageScopeKind.ALL:
                return None
            if scope.kind is MessageScopeKind.CHANNELS:
                wanted.update(scope.channel_ids)
        return frozenset(wanted)

    def _resolve_output_header(self) -> Header:
        """Choose output header metadata from readable inputs.

        The output library should identify this tool. The profile is preserved only when
        all readable input headers agree, which covers the common merge/recover case.
        """
        profiles: set[str] = set()

        for input_file in self.options.inputs:
            input_stream = input_file.stream
            original_position = input_stream.tell()
            try:
                header = get_header(input_stream)
            except McapError:
                continue
            finally:
                input_stream.seek(original_position)
            profiles.add(header.profile)

        profile = profiles.pop() if len(profiles) == 1 else ""
        return Header(profile=profile, library=OUTPUT_LIBRARY)
