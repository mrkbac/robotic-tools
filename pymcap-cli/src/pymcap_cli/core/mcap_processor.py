"""Unified MCAP processor combining recovery and filtering capabilities."""

import heapq
import os
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import cached_property
from pathlib import Path
from re import Pattern
from typing import IO, BinaryIO, cast

from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.text import Text
from small_mcap import (
    Attachment,
    Channel,
    Chunk,
    CompressionType,
    DataEnd,
    Footer,
    Header,
    LazyChunk,
    McapError,
    McapRecord,
    McapWriter,
    Message,
    MessageIndex,
    Metadata,
    Remapper,
    Schema,
    Statistics,
    Summary,
    breakup_chunk,
    get_header,
    get_summary,
    rebuild_summary,
    stream_reader,
)

# Private helpers — small-mcap does not re-export these at the top level.
from small_mcap.reader import _predecompress_chunk
from small_mcap.writer import _ChunkBuilder, _compress_chunk_data

from pymcap_cli.core.processors.always_decode import AlwaysDecodeProcessor
from pymcap_cli.core.processors.attachment_filter import AttachmentFilterProcessor
from pymcap_cli.core.processors.base import SPLIT_REQUIRED, Action, ChunkDecision, Processor
from pymcap_cli.core.processors.metadata_filter import MetadataFilterProcessor
from pymcap_cli.core.processors.time_filter import TimeFilterProcessor
from pymcap_cli.core.processors.topic_filter import TopicFilterProcessor
from pymcap_cli.types.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    str_to_compression_type,
)
from pymcap_cli.utils import (
    ProgressTrackingIO,
    confirm_output_overwrite,
    file_progress,
    parse_timestamp_args,
)

console = Console()
OUTPUT_LIBRARY = "pymcap-cli"
OutputKey = int | str | tuple[int | str, ...]
OutputStreamOpener = Callable[[OutputKey, int, int, int], tuple[str, BinaryIO]]


def _decode_chunk_records(chunk: Chunk) -> list[McapRecord]:
    """Worker-side: decompress a chunk and return its records as a list.

    Runs on a ThreadPoolExecutor worker. zstd/lz4 decompression releases the
    GIL so multiple chunks genuinely decompress in parallel.
    """
    return list(breakup_chunk(chunk, validate_crc=True))


def _recompress_chunk(chunk: Chunk, target: CompressionType) -> Chunk:
    """Worker-side: decompress + re-compress a chunk's data with a new codec.

    Avoids parsing/re-emitting every record when only the chunk compression
    changes. Message indexes remain valid because they reference offsets into
    the uncompressed chunk payload, which is unchanged.
    """
    decompressed = _predecompress_chunk(chunk, validate_crc=True)
    new_data, new_compression = _compress_chunk_data(decompressed.data, target)
    return Chunk(
        message_start_time=chunk.message_start_time,
        message_end_time=chunk.message_end_time,
        uncompressed_size=decompressed.uncompressed_size,
        uncompressed_crc=decompressed.uncompressed_crc,
        compression=new_compression,
        data=new_data,
    )


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


# Rechunking strategy enum
class RechunkStrategy(str, Enum):
    """Rechunking strategy for organizing messages into chunks."""

    NONE = "none"  # No rechunking - use fast-copy optimization when possible
    PATTERN = "pattern"  # Group by regex patterns
    ALL = "all"  # Each topic in its own chunk group
    AUTO = "auto"  # Auto-group based on size (>15% threshold)


class OverwriteCollisionPolicy(str, Enum):
    """How split outputs handle collisions with existing files."""

    ASK = "ask"
    OVERWRITE = "overwrite"
    ERROR = "error"


@dataclass
class InputOptions:
    """Input file filtering options.

    Stores explicit fields for proper merging. Processors are built lazily via build_processors().
    """

    always_decode_chunk: bool
    start_time_ns: int | None
    end_time_ns: int | None
    include_topics: list[str]
    exclude_topics: list[str]
    include_metadata: bool
    include_attachments: bool

    @classmethod
    def from_args(
        cls,
        always_decode_chunk: bool = False,
        # Raw CLI args for time (accept any combination)
        start: str = "",
        start_nsecs: int = 0,
        start_secs: int = 0,
        end: str = "",
        end_nsecs: int = 0,
        end_secs: int = 0,
        # Raw CLI args for topics (regex strings, not compiled)
        include_topic_regex: list[str] | None = None,
        exclude_topic_regex: list[str] | None = None,
        # Content filtering
        include_metadata: bool = True,
        include_attachments: bool = True,
    ) -> "InputOptions":
        return cls(
            always_decode_chunk=always_decode_chunk,
            start_time_ns=parse_timestamp_args(start, start_secs, start_nsecs),
            end_time_ns=parse_timestamp_args(end, end_secs, end_nsecs),
            include_topics=include_topic_regex or [],
            exclude_topics=exclude_topic_regex or [],
            include_metadata=include_metadata,
            include_attachments=include_attachments,
        )

    @cached_property
    def processors(self) -> list[Processor]:
        """Build processor list from options. Cached on first access."""
        procs: list[Processor] = []

        if self.always_decode_chunk:
            procs.append(AlwaysDecodeProcessor())

        if self.start_time_ns is not None or self.end_time_ns is not None:
            procs.append(TimeFilterProcessor(self.start_time_ns, self.end_time_ns))

        if self.include_topics or self.exclude_topics:
            procs.append(TopicFilterProcessor(self.include_topics, self.exclude_topics))

        if not self.include_metadata:
            procs.append(MetadataFilterProcessor(include=False))

        if not self.include_attachments:
            procs.append(AttachmentFilterProcessor(include=False))

        return procs

    def __or__(self, other: "InputOptions") -> "InputOptions":
        """Merge options - other (per-file) overrides self (global) for non-default values."""
        return InputOptions(
            always_decode_chunk=self.always_decode_chunk or other.always_decode_chunk,
            start_time_ns=other.start_time_ns
            if other.start_time_ns is not None
            else self.start_time_ns,
            end_time_ns=other.end_time_ns if other.end_time_ns is not None else self.end_time_ns,
            include_topics=other.include_topics or self.include_topics,
            exclude_topics=other.exclude_topics or self.exclude_topics,
            include_metadata=self.include_metadata and other.include_metadata,
            include_attachments=self.include_attachments and other.include_attachments,
        )


@dataclass(slots=True)
class InputFile:
    """Input file stream with its size and options."""

    stream: IO[bytes]
    size: int
    options: InputOptions


@dataclass(slots=True)
class OutputOptions:
    """Options for output file format."""

    compression: str = DEFAULT_COMPRESSION
    chunk_size: int = DEFAULT_CHUNK_SIZE

    # Rechunking options
    rechunk_strategy: RechunkStrategy = RechunkStrategy.NONE
    rechunk_patterns: list[Pattern[str]] = field(default_factory=list)

    # Output processors (split routing, etc.)
    processors: list[Processor] = field(default_factory=list)
    # Template for multi-output file naming (e.g., "output_{index:03d}.mcap")
    output_template: str = ""
    overwrite_policy: OverwriteCollisionPolicy = OverwriteCollisionPolicy.ASK

    @property
    def compression_type(self) -> CompressionType:
        return str_to_compression_type(self.compression)

    @property
    def is_rechunking(self) -> bool:
        return self.rechunk_strategy != RechunkStrategy.NONE

    @property
    def is_splitting(self) -> bool:
        return bool(self.processors)


class ProcessingOptions:
    """Complete processing configuration."""

    def __init__(
        self,
        # stream, size, input_options
        inputs: list[InputFile],
        input_options: InputOptions,
        output_options: OutputOptions,
    ) -> None:
        # self.input_options = input_options
        self.output_options = output_options

        # merge input_options with local options
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
    chunks_copied: int = 0  # Fast copied chunks
    chunks_decoded: int = 0  # Decoded chunks

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
            lines.append(
                f"Chunks:       {self.chunks_processed} "
                f"({self.chunks_copied} fast copied, {self.chunks_decoded} decoded)\n"
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
    ) -> None:
        self.writer = writer
        self.chunk_size = chunk_size
        self.message_count = 0
        self.compress_fail_counter = 0
        # Each group has its own chunk builder for independent chunking
        # Pass schemas/channels for auto-ensure
        self.chunk_builder = _ChunkBuilder(
            chunk_size=chunk_size,
            compression=compression_type,
            enable_crcs=writer.enable_crcs,
        )

    def add_message(self, message: Message) -> None:
        self._flush_if_full()
        self.chunk_builder.add(message)
        self.message_count += 1

    def _flush_if_full(self) -> None:
        """Finalize and write the current chunk if it has reached the target size."""
        if (
            self.chunk_builder.buffer.tell() < self.chunk_builder.chunk_size
            or self.chunk_builder.num_messages == 0
        ):
            return
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
    rechunk_groups: list[MessageGroup] = field(default_factory=list)
    channel_to_group: dict[int, MessageGroup] = field(default_factory=dict)
    start_time: int = 0
    end_time: int = 0
    pattern_groups: dict[int, MessageGroup] = field(default_factory=dict)


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
    ) -> None:
        self.output_options = output_options
        self.schemas = schemas
        self.channels = channels
        self.header = header
        self._open_output = open_output or self._open_template_output
        self.segments: dict[OutputKey, OutputSegment] = {}
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
        path = self.output_options.output_template.format(
            index=index,
            index1=index + 1,
            key=key,
            start_time=start_time,
            start_time_iso=_ns_to_iso(start_time) if start_time else "",
            end_time=end_time,
        )

        path_obj = Path(path)
        self.handle_existing_output(path_obj)
        return path, path_obj.open("wb")

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
        num_workers = 0 if compression_type == CompressionType.NONE else min(4, os.cpu_count() or 1)
        writer = McapWriter(
            stream,
            chunk_size=self.output_options.chunk_size,
            compression=compression_type,
            num_workers=num_workers,
        )
        writer.schemas = dict(self.schemas)
        writer.channels = dict(self.channels)
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

        # Flush any buffered records to this new segment
        self._flush_pending_to_segment(segment)

        return segment

    def get_writer(self, key: OutputKey) -> McapWriter:
        """Get writer for output key, creating segment if needed."""
        return self.get_or_create_segment(key).writer

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
            # Flush rechunk groups
            for group in segment.rechunk_groups:
                group.flush()
            segment.writer.finish()
            stats[key] = segment.writer.statistics
            segment.stream.close()
        # Clear buffers (records were flushed to all segments during creation)
        self._pending_attachments.clear()
        self._pending_metadata.clear()
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

        # Track which channels we've already seen to optimize metadata extraction
        self.known_channels: set[int] = set()

        # Rechunking state
        self.large_channels: set[int] = set()  # For AUTO mode

        # Unified output management for both single-output and split-output modes.
        self.output_manager: OutputManager | None = None

    def _get_input(self, stream_id: int) -> InputOptions:
        return self.options.inputs[stream_id].options

    def _is_channel_included(self, stream_id: int, channel_id: int) -> bool:
        cache_key = (stream_id, channel_id)
        if cache_key in self.channel_filter_cache:
            return self.channel_filter_cache[cache_key]
        channel = self.channels.get(channel_id)
        if not channel:
            return False
        input_opts = self._get_input(stream_id)
        if not input_opts.processors:
            return True
        for p in input_opts.processors:
            if p.on_channel(channel, self.schemas.get(channel.schema_id)) == Action.SKIP:
                self.channel_filter_cache[cache_key] = False
                return False
        self.channel_filter_cache[cache_key] = True
        return True

    def _analyze_for_auto_grouping(self, input_streams: Sequence[IO[bytes]]) -> None:
        """Pre-analyze files to identify large channels (>15% of total uncompressed size)."""
        console.print("[dim]Analyzing files for auto-grouping...[/dim]")

        # Aggregate channel sizes across all files (accounting for remapped IDs)
        # Map: remapped_channel_id -> total_size
        channel_sizes: dict[int, int] = {}

        for stream_id, input_stream in enumerate(input_streams):
            try:
                rebuild_info = rebuild_summary(
                    input_stream,
                    validate_crc=False,
                    calculate_channel_sizes=True,
                    exact_sizes=False,  # Use fast estimation
                )

                if rebuild_info.channel_sizes:
                    # Map original channel IDs to remapped IDs and aggregate
                    for original_ch_id, size in rebuild_info.channel_sizes.items():
                        # Get remapped channel ID
                        remapped_channel = self.remapper.get_remapped_channel(
                            stream_id, original_ch_id
                        )
                        # Channel not yet remapped will keep original ID
                        remapped_id = remapped_channel.id if remapped_channel else original_ch_id

                        channel_sizes[remapped_id] = channel_sizes.get(remapped_id, 0) + size

                # Seek back to start for processing
                input_stream.seek(0)

            except (McapError, OSError) as e:
                console.print(
                    f"[yellow]Warning: Could not analyze stream {stream_id}: {e}[/yellow]"
                )
                input_stream.seek(0)

        if not channel_sizes:
            console.print("[yellow]Warning: Could not determine channel sizes[/yellow]")
            return

        # Calculate 15% threshold
        total_size = sum(channel_sizes.values())
        threshold = total_size * 0.15

        # Identify large channels (using remapped IDs)
        self.large_channels = {ch_id for ch_id, size in channel_sizes.items() if size > threshold}

        if self.large_channels:
            console.print(
                f"[dim]Found {len(self.large_channels)} large channel(s) "
                f"(>{threshold / 1024 / 1024:.1f}MB each)[/dim]"
            )

    def _find_matching_pattern_index(self, topic: str) -> int | None:
        """Find first pattern that matches topic. Returns pattern index or None."""
        for i, pattern in enumerate(self.options.output_options.rechunk_patterns):
            if pattern.search(topic):
                return i
        return None

    def _get_or_create_group_for_channel(
        self, channel_id: int, channel: Channel, writer: McapWriter
    ) -> MessageGroup:
        """Get or create appropriate MessageGroup for a channel based on rechunk strategy."""
        return self._get_or_create_group_for_channel_split(channel_id, channel, writer)

    def _get_or_create_group_for_channel_split(
        self, channel_id: int, channel: Channel, writer: McapWriter
    ) -> MessageGroup:
        """Get or create MessageGroup when splitting is active.

        Uses per-segment group tracking since each segment needs independent rechunk groups.
        """
        assert self.output_manager is not None
        # Find which segment this writer belongs to
        segment: OutputSegment | None = None
        for seg in self.output_manager.segments.values():
            if seg.writer is writer:
                segment = seg
                break
        if segment is None:
            # Writer not yet in any segment, create one
            segment = self.output_manager.get_or_create_segment(0)

        if channel_id in segment.channel_to_group:
            return segment.channel_to_group[channel_id]

        strategy = self.options.output_options.rechunk_strategy
        group: MessageGroup | None = None

        if strategy == RechunkStrategy.ALL:
            group = self._create_segment_message_group(segment)

        elif strategy == RechunkStrategy.AUTO:
            if channel_id in self.large_channels:
                group = self._create_segment_message_group(segment)
            else:
                for ch_id, existing in segment.channel_to_group.items():
                    if ch_id not in self.large_channels:
                        group = existing
                        break

        elif strategy == RechunkStrategy.PATTERN:
            pattern_idx = self._find_matching_pattern_index(channel.topic)
            group_key = pattern_idx if pattern_idx is not None else -1
            group = segment.pattern_groups.get(group_key)
            if group is None:
                group = self._create_segment_message_group(segment)
                segment.pattern_groups[group_key] = group

        if group is None:
            group = self._create_segment_message_group(segment)

        segment.channel_to_group[channel_id] = group
        return group

    def _create_segment_message_group(self, segment: "OutputSegment") -> MessageGroup:
        """Create a MessageGroup attached to a specific segment."""
        opts = self.options.output_options
        group = MessageGroup(segment.writer, opts.chunk_size, opts.compression_type)
        segment.rechunk_groups.append(group)
        return group

    def process_message(self, message: Message, stream_id: int) -> None:
        self.stats.messages_processed += 1
        assert self.output_manager is not None

        # Topic filtering using cached decision (avoid repeated regex matching)
        if not self._is_channel_included(stream_id, message.channel_id):
            self.stats.filter_rejections += 1
            return

        # Time filtering and other message processors
        input_opts = self._get_input(stream_id)
        if any(p.on_message(message) == Action.SKIP for p in input_opts.processors):
            self.stats.filter_rejections += 1
            return

        route_key = self._get_message_route(message)
        if route_key is None:
            route_key = 0
        target_writer = self.output_manager.get_writer(route_key)
        self.output_manager.ensure_channel_written(message.channel_id, route_key)

        # Route to appropriate destination based on rechunking mode
        if self.options.output_options.is_rechunking:
            # Get channel for this message
            channel = self.channels.get(message.channel_id)
            if not channel:
                # Channel not yet seen - skip message
                return

            # Get or create the MessageGroup for this channel
            group = self._get_or_create_group_for_channel(
                message.channel_id, channel, target_writer
            )

            # Add message to its group (chunk builder auto-ensures within chunks)
            group.add_message(message)
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

        if remapped_channel.id not in self.known_channels:
            self.known_channels.add(remapped_channel.id)
            self.channels[remapped_channel.id] = remapped_channel

            # Pre-compute filtering decision for this stream (cache it)
            input_opts = self._get_input(stream_id)
            should_include = True
            if input_opts.processors:
                schema = self.schemas.get(remapped_channel.schema_id)
                for p in input_opts.processors:
                    if p.on_channel(remapped_channel, schema) == Action.SKIP:
                        should_include = False
                        break
            self.channel_filter_cache[(stream_id, remapped_channel.id)] = should_include

            if not should_include:
                del self.channels[remapped_channel.id]

    def _handle_message_record(self, message: Message, stream_id: int) -> None:
        message_to_process = self.remapper.remap_message(stream_id, message)
        self.process_message(message_to_process, stream_id)

    def _handle_attachment_record(self, attachment: Attachment, stream_id: int) -> None:
        self.stats.attachments_processed += 1
        assert self.output_manager is not None

        # Check all processors (includes AttachmentFilterProcessor and TimeFilterProcessor)
        input_opts = self._get_input(stream_id)
        if any(p.on_attachment(attachment) == Action.SKIP for p in input_opts.processors):
            return

        self.output_manager.add_attachment(attachment)

    def _generate_chunks_from_stream(
        self, input_stream: IO[bytes], stream_id: int
    ) -> Iterator[PendingChunk]:
        """Generate chunks from a single stream in file order.

        Yields PendingChunk objects with timestamp for ordered merging.
        Non-chunk records (Schema, Channel, Message, Attachment, Metadata) are processed directly.
        Uses lazy_chunks=True for efficiency - chunk data is only read when needed.
        """
        pending: PendingChunk | None = None

        try:
            records = stream_reader(input_stream, emit_chunks=True, lazy_chunks=True)
            indexes: list[MessageIndex] = []

            for record in records:
                # Yield pending chunk when we see a non-MessageIndex record
                if not isinstance(record, MessageIndex) and pending:
                    yield pending
                    pending = None

                if isinstance(record, Header):
                    pass  # Header handled separately
                elif isinstance(record, (Chunk, LazyChunk)):
                    self.stats.chunks_processed += 1
                    pending = PendingChunk(
                        record, indexes := [], stream_id, input_stream, record.message_start_time
                    )
                elif isinstance(record, MessageIndex):
                    indexes.append(record)
                elif isinstance(record, Schema):
                    self._handle_schema_record(record, stream_id)
                elif isinstance(record, Channel):
                    self._handle_channel_record(record, stream_id)
                elif isinstance(record, Message):
                    self._handle_message_record(record, stream_id)
                elif isinstance(record, Attachment):
                    self._handle_attachment_record(record, stream_id)
                elif isinstance(record, Metadata):
                    self.stats.metadata_processed += 1
                    input_opts = self._get_input(stream_id)
                    if not input_opts.processors or all(
                        p.on_metadata(record) != Action.SKIP for p in input_opts.processors
                    ):
                        assert self.output_manager is not None
                        self.output_manager.add_metadata(name=record.name, metadata=record.metadata)
                elif isinstance(record, (DataEnd, Footer)):
                    break

        except McapError as e:
            console.print(f"[yellow]Warning (stream {stream_id}): {e}[/yellow]")
            self.stats.errors_encountered += 1

        # Yield final pending chunk if any
        if pending:
            yield pending

    def process(
        self,
        output_stream: BinaryIO | None = None,
    ) -> ProcessingStats:
        """Main processing function."""
        output_opts = self.options.output_options
        header = self._resolve_output_header()

        # Pre-load schemas and channels from all files' summaries
        # This ensures we have all metadata before processing any chunks
        summaries: list[Summary | None] = []
        for stream_id, input_opt in enumerate(self.options.inputs):
            input_stream = input_opt.stream
            try:
                summary = get_summary(input_stream)
            except McapError:
                # In recovery mode, if we can't get summary (e.g., truncated file),
                # continue without it - we'll discover schemas/channels during chunk processing
                summary = None

            summaries.append(summary)

            if summary:
                # Remap and store all schemas
                for schema in summary.schemas.values():
                    self._handle_schema_record(schema, stream_id)

                # Remap and store all channels
                for channel in summary.channels.values():
                    self._handle_channel_record(channel, stream_id)

            # Seek back to start for processing
            input_stream.seek(0)

        # Initialize output processors with summaries (for split boundary computation, etc.)
        for proc in output_opts.processors:
            proc.initialize(summaries)

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
        )
        # Pre-create statically-known segments. In single-output mode this creates segment 0.
        known_segments = list(self._iter_known_output_segments())
        if not known_segments:
            known_segments = [(0, 0, 0)]
        for key, start_time, end_time in known_segments:
            self.output_manager.get_or_create_segment(key, start_time=start_time, end_time=end_time)

        try:
            # For AUTO rechunking mode, pre-analyze files to identify large channels
            if output_opts.rechunk_strategy == RechunkStrategy.AUTO:
                input_streams = [inp.stream for inp in self.options.inputs]
                self._analyze_for_auto_grouping(input_streams)

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
                    self._generate_chunks_from_stream(wrapped_stream, stream_id)
                    for stream_id, wrapped_stream in enumerate(wrapped_streams)
                ]

                # Process chunks in timestamp order using heapq.merge.
                # For DECODE-bound chunks the (slow) zstd/lz4 decompression is
                # offloaded to a small worker pool so that the main thread can
                # keep writing the previous chunk's records while the next
                # chunk decompresses in parallel. zstd/lz4 release the GIL so
                # the speedup is genuine.
                self._run_chunk_pipeline(heapq.merge(*chunk_generators))

                # Complete progress
                progress.update(task, completed=total_size)

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
            - DECODE: Chunk must be decoded to change channels or filter messages
        """
        input_opts = self._get_input(stream_id)
        output_opts = self.options.output_options

        # Ask input processors for chunk-level decision (time filtering, always_decode, etc.)
        for proc in input_opts.processors:
            decision = proc.on_chunk(chunk, indexes)
            if decision == ChunkDecision.SKIP:
                return ChunkDecision.SKIP
            if decision == ChunkDecision.DECODE:
                # Track that we need to decode, but keep checking for SKIP
                return ChunkDecision.DECODE

        # Ask output processors (split routing forces DECODE on boundary chunks)
        for proc in output_opts.processors:
            decision = proc.on_chunk(chunk, indexes)
            if decision == ChunkDecision.DECODE:
                return ChunkDecision.DECODE

        # Force decode if rechunking is active (must reorganize messages)
        if output_opts.is_rechunking:
            return ChunkDecision.DECODE

        # Compression mismatch alone doesn't need a full decode — if nothing
        # else forces per-message work, RECOMPRESS (chunk-level) is enough.
        compression_mismatch = chunk.compression != output_opts.compression_type.value

        # Single pass: check remapping, channel availability, and filtering
        has_include = False
        has_exclude = False

        for idx in indexes:
            ch_id = idx.channel_id
            # Check if channel metadata is available
            if not self.remapper.has_channel(stream_id, ch_id):
                return ChunkDecision.DECODE
            # If channel id was remapped, must decode
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

        # If chunk has ONLY excluded channels, skip it entirely
        if has_exclude and not has_include:
            return ChunkDecision.SKIP

        if compression_mismatch:
            return ChunkDecision.RECOMPRESS

        # No reason to decode - can fast-copy
        return ChunkDecision.CONTINUE

    def _get_chunk_route(self, chunk: Chunk | LazyChunk) -> OutputKey | None:
        """Get output key for a chunk from output processors."""
        routes: list[int | str] = []
        for proc in self.options.output_options.processors:
            route = proc.route_chunk(chunk)
            if route is SPLIT_REQUIRED:
                return None
            if route is not None:
                assert isinstance(route, (int, str))
                routes.append(route)
        if not routes:
            return None
        if len(routes) == 1:
            return routes[0]
        return tuple(routes)

    def _iter_known_output_segments(self) -> Iterator[tuple[OutputKey, int, int]]:
        """Yield statically known output routes for pre-creating segments."""
        known: list[list[tuple[int | str, int, int]]] = []
        for proc in self.options.output_options.processors:
            keys = proc.output_keys()
            if keys is None:
                return
            boundaries = getattr(proc, "boundaries", None)
            proc_known: list[tuple[int | str, int, int]] = []
            for key in keys:
                start_time = 0
                end_time = 0
                if boundaries and isinstance(key, int):
                    start_time = boundaries[key]
                    end_time = boundaries[key + 1]
                proc_known.append((key, start_time, end_time))
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
        """Create an opener for single-output processing."""
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

    def _get_message_route(self, message: Message) -> OutputKey | None:
        """Get output key for a message from output processors."""
        routes: list[int | str] = []
        for proc in self.options.output_options.processors:
            route = proc.route_message(message)
            if route is not None:
                routes.append(route)
        if not routes:
            return None
        if len(routes) == 1:
            return routes[0]
        return tuple(routes)

    def _run_chunk_pipeline(self, chunks: Iterator[PendingChunk]) -> None:
        """Consume chunks from the merged iterator, decompressing DECODE chunks ahead.

        Maintains a short queue of in-flight decompression futures so the main
        thread can be writing chunk N's records while workers decompress chunks
        N+1..N+W in parallel. For fast-copy and skip chunks the queue entry has
        no future and is handled directly.
        """
        max_inflight = min(4, os.cpu_count() or 1)
        queue: deque[
            tuple[PendingChunk, ChunkDecision, Future[list[McapRecord]] | Future[Chunk] | None]
        ] = deque()

        with ThreadPoolExecutor(max_workers=max_inflight) as pool:
            target_compression = self.options.output_options.compression_type

            def enqueue_next() -> bool:
                try:
                    pending = next(chunks)
                except StopIteration:
                    return False
                decision = self._should_decode_chunk(
                    pending.chunk, pending.indexes, pending.stream_id
                )
                future: Future[list[McapRecord]] | Future[Chunk] | None = None
                if decision in (ChunkDecision.DECODE, ChunkDecision.RECOMPRESS):
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
                    if decision == ChunkDecision.DECODE:
                        future = pool.submit(_decode_chunk_records, materialized)
                    else:
                        future = pool.submit(_recompress_chunk, materialized, target_compression)
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
            except McapError as e:
                console.print(
                    f"[yellow]Warning (stream {pending.stream_id}): "
                    f"Failed to decode chunk: {e}[/yellow]"
                )
                self.stats.errors_encountered += 1
                return
            self._process_decoded_records(records, pending.stream_id)
            return

        if decision == ChunkDecision.RECOMPRESS:
            assert future is not None
            try:
                new_chunk = cast("Future[Chunk]", future).result()
            except McapError as e:
                console.print(
                    f"[yellow]Warning (stream {pending.stream_id}): "
                    f"Failed to recompress chunk: {e}[/yellow]"
                )
                self.stats.errors_encountered += 1
                return
            route_key = self._get_chunk_route(pending.chunk)
            if route_key is None:
                route_key = 0
            target_writer = self.output_manager.get_writer(route_key)
            for idx in pending.indexes:
                if self._is_channel_included(pending.stream_id, idx.channel_id):
                    self.output_manager.ensure_channel_written(idx.channel_id, route_key)
            indices_by_channel = {idx.channel_id: idx for idx in pending.indexes}
            target_writer.add_chunk(new_chunk, indices_by_channel)
            self.stats.chunks_copied += 1
            return

        # Fast-copy path (CONTINUE) -> nothing about the chunk must be changed.
        # We pipe raw bytes from input to output (no Chunk materialization).
        route_key = self._get_chunk_route(pending.chunk)
        if route_key is None:
            route_key = 0
        target_writer = self.output_manager.get_writer(route_key)
        for idx in pending.indexes:
            if self._is_channel_included(pending.stream_id, idx.channel_id):
                self.output_manager.ensure_channel_written(idx.channel_id, route_key)

        indices_by_channel = {idx.channel_id: idx for idx in pending.indexes}
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

    def _process_decoded_records(self, records: list[McapRecord], stream_id: int) -> None:
        """Process records that were already decoded from a chunk (by a worker)."""
        self.stats.chunks_decoded += 1
        for chunk_record in records:
            if isinstance(chunk_record, Message):
                message_to_write = self.remapper.remap_message(stream_id, chunk_record)
                self.process_message(message_to_write, stream_id)
            elif isinstance(chunk_record, Schema):
                self._handle_schema_record(chunk_record, stream_id)
            elif isinstance(chunk_record, Channel):
                self._handle_channel_record(chunk_record, stream_id)

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
