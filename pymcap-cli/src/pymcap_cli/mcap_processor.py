"""Unified MCAP processor combining recovery and filtering capabilities."""

import heapq
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from re import Pattern
from typing import IO, BinaryIO

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
    McapError,
    McapWriter,
    Message,
    MessageIndex,
    Metadata,
    Remapper,
    Schema,
    Statistics,
    breakup_chunk,
    get_summary,
    stream_reader,
)
from small_mcap.rebuild import rebuild_summary
from small_mcap.writer import _ChunkBuilder

from pymcap_cli.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    str_to_compression_type,
)
from pymcap_cli.utils import (
    MAX_INT64,
    ProgressTrackingIO,
    compile_topic_patterns,
    file_progress,
    parse_timestamp_args,
)

console = Console()


@dataclass(slots=True)
class PendingChunk:
    chunk: Chunk
    indexes: list[MessageIndex]
    stream_id: int
    timestamp: int  # message_start_time for ordering

    def __lt__(self, other: "PendingChunk") -> bool:
        return self.timestamp < other.timestamp


# Rechunking strategy enum
class RechunkStrategy(str, Enum):
    """Rechunking strategy for organizing messages into chunks."""

    NONE = "none"  # No rechunking - use fast-copy optimization when possible
    PATTERN = "pattern"  # Group by regex patterns
    ALL = "all"  # Each topic in its own chunk group
    AUTO = "auto"  # Auto-group based on size (>15% threshold)


class ShouldDecode(Enum):
    SKIP = auto()
    """Chunk does not contain anything new, everything is filtered out"""
    COPY = auto()
    """Chunk does not contain anything new, everything is copied"""
    DECODE = auto()
    """Chunk must be decoded to change channels or filter on message level"""


@dataclass  # No slots=True - needed for @cached_property
class InputOptions:
    """Input file with its filtering options.

    Accepts raw CLI arguments and lazily computes derived values via cached_property.
    """

    # The input stream
    stream: IO[bytes]
    file_size: int  # For progress tracking

    always_decode_chunk: bool = False

    # Raw CLI args for time (accept any combination)
    start: str = ""
    start_nsecs: int = 0
    start_secs: int = 0
    end: str = ""
    end_nsecs: int = 0
    end_secs: int = 0

    # Raw CLI args for topics (regex strings, not compiled)
    include_topic_regex: list[str] | None = None
    exclude_topic_regex: list[str] | None = None

    # Content filtering
    include_metadata: bool = True
    include_attachments: bool = True

    def __post_init__(self) -> None:
        """Normalize None to empty list and validate mutually exclusive options."""
        if self.include_topic_regex and self.exclude_topic_regex:
            raise ValueError("Cannot use both include and exclude topic filters")

    @cached_property
    def start_time(self) -> int:
        """Compute start time in nanoseconds from CLI args."""
        return parse_timestamp_args(self.start, self.start_nsecs, self.start_secs)

    @cached_property
    def end_time(self) -> int:
        """Compute end time in nanoseconds from CLI args."""
        parsed = parse_timestamp_args(self.end, self.end_nsecs, self.end_secs)
        result = MAX_INT64 if parsed == 0 else parsed
        if result < self.start_time:
            raise ValueError("End time cannot be before start time")
        return result

    @cached_property
    def include_topics(self) -> list[Pattern[str]]:
        """Compile include topic regex patterns."""
        return compile_topic_patterns(self.include_topic_regex or [])

    @cached_property
    def exclude_topics(self) -> list[Pattern[str]]:
        """Compile exclude topic regex patterns."""
        return compile_topic_patterns(self.exclude_topic_regex or [])

    @property
    def has_time_filter(self) -> bool:
        """Check if time filtering is active."""
        return self.start_time > 0 or self.end_time < MAX_INT64

    @property
    def has_topic_filter(self) -> bool:
        """Check if topic filtering is active."""
        return bool(self.include_topics or self.exclude_topics)


@dataclass(slots=True)
class OutputOptions:
    """Options for output file format."""

    compression: str = DEFAULT_COMPRESSION
    chunk_size: int = DEFAULT_CHUNK_SIZE

    # Rechunking options
    rechunk_strategy: RechunkStrategy = RechunkStrategy.NONE
    rechunk_patterns: list[Pattern[str]] = field(default_factory=list)

    @property
    def compression_type(self) -> CompressionType:
        """Get the CompressionType enum value for the compression string."""
        return str_to_compression_type(self.compression)

    @property
    def is_rechunking(self) -> bool:
        """Check if rechunking is active."""
        return self.rechunk_strategy != RechunkStrategy.NONE


@dataclass(slots=True)
class ProcessingOptions:
    """Complete processing configuration."""

    inputs: list[InputOptions]
    output: OutputOptions = field(default_factory=OutputOptions)

    @property
    def total_size(self) -> int:
        """Total size of all input files for progress tracking."""
        return sum(inp.file_size for inp in self.inputs)


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

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render processing statistics as a styled Rich Panel."""
        lines = Text()
        ws = self.writer_statistics

        # Messages line - show written count, and decoded count only if different
        lines.append(f"Messages:     {ws.message_count:,} written")
        if self.messages_processed > 0 and self.messages_processed != ws.message_count:
            lines.append(f" ({self.messages_processed:,} decoded)")
        lines.append("\n")

        # Attachments (only if any were processed)
        if self.attachments_processed > 0:
            lines.append(f"Attachments:  {ws.attachment_count} written")
            lines.append(f" ({self.attachments_processed} processed)\n")

        # Metadata (only if any were processed)
        if self.metadata_processed > 0:
            lines.append(f"Metadata:     {ws.metadata_count} written")
            lines.append(f" ({self.metadata_processed} processed)\n")

        # Schemas and channels
        lines.append(f"Schemas:      {ws.schema_count} written\n")
        lines.append(f"Channels:     {ws.channel_count} written\n")

        # Chunks (only if any were processed)
        if self.chunks_processed > 0:
            lines.append(f"Chunks:       {self.chunks_processed} ")
            lines.append(f"({self.chunks_copied} fast copied, ")
            lines.append(f"{self.chunks_decoded} decoded)\n")

        # Errors (yellow if any)
        if self.errors_encountered > 0:
            lines.append("Errors:       ", style="yellow")
            lines.append(f"{self.errors_encountered}\n", style="yellow")

        # Validation errors (yellow if any)
        if self.validation_errors > 0:
            lines.append("Validation:   ", style="yellow")
            lines.append(f"{self.validation_errors} errors\n", style="yellow")

        # Filter rejections (if any)
        if self.filter_rejections > 0:
            lines.append(f"Filtered:     {self.filter_rejections} records\n")

        # Remove trailing newline
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
        """Add message to this group's chunk builder."""
        # ChunkBuilder auto-ensures channel and schema
        self.chunk_builder.add(message)

        # If chunk builder returns a completed chunk, write it immediately
        if result := self.chunk_builder.maybe_finalize():
            chunk, message_indexes = result
            # TODO: also check requested chunk size?
            if chunk.compression != self.chunk_builder.compression.value:
                self.compress_fail_counter += 1
                if self.compress_fail_counter > 2:
                    console.print(
                        "[yellow]Multiple compression failures, switching to uncompressed.[/yellow]"
                    )
                    self.chunk_builder.compression = CompressionType.NONE
            self.writer.add_chunk(chunk, message_indexes)

        self.message_count += 1

    def flush(self) -> None:
        """Flush any remaining messages in this group's chunk builder."""
        result = self.chunk_builder.finalize()
        if result is not None:
            chunk, message_indexes = result
            self.writer.add_chunk(chunk, message_indexes)


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

        # Rechunking state (only used when options.output.is_rechunking)
        self.channel_to_group: dict[int, MessageGroup] = {}
        self.large_channels: set[int] = set()  # For AUTO mode
        self.rechunk_groups: list[MessageGroup] = []  # Track unique groups
        self.pattern_index_to_group: dict[int, MessageGroup] = {}  # For PATTERN mode cache

        # Track which schemas/channels we've written to the main file (not in chunks)
        self.written_schemas: set[int] = set()
        self.written_channels: set[int] = set()

    def _get_input(self, stream_id: int) -> InputOptions:
        """Get InputOptions for a stream."""
        return self.options.inputs[stream_id]

    def _compute_channel_filter_decision(self, stream_id: int, topic: str) -> bool:
        """Compute whether a channel with given topic should be included for a stream.

        Returns True if the channel should be included, False otherwise.
        """
        input_opts = self._get_input(stream_id)
        if input_opts.include_topics:
            return any(p.search(topic) for p in input_opts.include_topics)
        if input_opts.exclude_topics:
            return not any(p.search(topic) for p in input_opts.exclude_topics)
        return True

    def _is_channel_included(self, stream_id: int, channel_id: int) -> bool:
        """Check if a channel is included for a specific stream, using cache."""
        cache_key = (stream_id, channel_id)
        if cache_key in self.channel_filter_cache:
            return self.channel_filter_cache[cache_key]

        # Compute and cache
        channel = self.channels.get(channel_id)
        topic = channel.topic if channel else ""
        result = self._compute_channel_filter_decision(stream_id, topic)
        self.channel_filter_cache[cache_key] = result
        return result

    def _ensure_channel_written(self, channel_id: int, writer: McapWriter) -> None:
        """Ensure channel and its schema are written to the main file (not in chunks)."""
        if channel_id in self.written_channels:
            return

        channel = self.channels.get(channel_id)
        if not channel:
            return

        # Write schema first if needed
        if channel.schema_id != 0 and channel.schema_id not in self.written_schemas:
            schema = self.schemas.get(channel.schema_id)
            if schema:
                writer.add_schema(schema.id, schema.name, schema.encoding, schema.data)
                self.written_schemas.add(schema.id)

        # Write channel
        writer.add_channel(
            channel.id, channel.topic, channel.message_encoding, channel.schema_id, channel.metadata
        )
        self.written_channels.add(channel_id)

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
        for i, pattern in enumerate(self.options.output.rechunk_patterns):
            if pattern.search(topic):
                return i
        return None

    def _create_message_group(self, writer: McapWriter) -> MessageGroup:
        """Create a new MessageGroup with output options."""
        opts = self.options.output
        group = MessageGroup(writer, opts.chunk_size, opts.compression_type)
        self.rechunk_groups.append(group)
        return group

    def _get_or_create_group_for_channel(
        self, channel_id: int, channel: Channel, writer: McapWriter
    ) -> MessageGroup:
        """Get or create appropriate MessageGroup for a channel based on rechunk strategy."""
        if channel_id in self.channel_to_group:
            return self.channel_to_group[channel_id]

        strategy = self.options.output.rechunk_strategy
        group: MessageGroup | None = None

        if strategy == RechunkStrategy.ALL:
            # Each channel gets its own unique group
            group = self._create_message_group(writer)

        elif strategy == RechunkStrategy.AUTO:
            # Large channels get their own group, small channels share one
            if channel_id in self.large_channels:
                group = self._create_message_group(writer)
            else:
                # Try to find existing shared group for small channels
                for ch_id, existing in self.channel_to_group.items():
                    if ch_id not in self.large_channels:
                        group = existing
                        break

        elif strategy == RechunkStrategy.PATTERN:
            # Find which pattern matches this channel's topic
            pattern_idx = self._find_matching_pattern_index(channel.topic)
            group_key = pattern_idx if pattern_idx is not None else -1
            group = self.pattern_index_to_group.get(group_key)
            if group is None:
                group = self._create_message_group(writer)
                self.pattern_index_to_group[group_key] = group

        # Fallback: create new group if none assigned
        if group is None:
            group = self._create_message_group(writer)

        self.channel_to_group[channel_id] = group
        return group

    def process_message(
        self,
        message: Message,
        writer: McapWriter,
        stream_id: int,
    ) -> None:
        """Process a message record."""
        self.stats.messages_processed += 1
        input_opts = self._get_input(stream_id)

        # Time filtering (per-input)
        if not (input_opts.start_time <= message.log_time < input_opts.end_time):
            return

        # Topic filtering using cached decision (avoid repeated regex matching)
        if not self._is_channel_included(stream_id, message.channel_id):
            self.stats.filter_rejections += 1
            return

        # Route to appropriate destination based on rechunking mode
        if self.options.output.is_rechunking:
            # Get channel for this message
            channel = self.channels.get(message.channel_id)
            if not channel:
                # Channel not yet seen - skip message
                return

            # Get or create the MessageGroup for this channel
            group = self._get_or_create_group_for_channel(message.channel_id, channel, writer)

            # Ensure channel is written to main file (not in chunks)
            self._ensure_channel_written(message.channel_id, writer)

            # Add message to its group (chunk builder auto-ensures within chunks)
            group.add_message(message)
        else:
            # Normal mode: ensure channel is written before writing message
            self._ensure_channel_written(message.channel_id, writer)
            writer.add_message(
                channel_id=message.channel_id,
                log_time=message.log_time,
                data=message.data,
                publish_time=message.publish_time,
            )

    def _handle_schema_record(self, schema: Schema, stream_id: int) -> None:
        """Handle a schema record from the stream."""
        remapped_schema = self.remapper.remap_schema(stream_id, schema)
        if remapped_schema:
            self.schemas[remapped_schema.id] = remapped_schema

    def _handle_channel_record(self, channel: Channel, stream_id: int) -> None:
        """Handle a channel record from the stream."""
        remapped_channel = self.remapper.remap_channel(stream_id, channel)

        if remapped_channel.id not in self.known_channels:
            self.known_channels.add(remapped_channel.id)

            # Pre-compute filtering decision for this stream (cache it)
            should_include = self._compute_channel_filter_decision(
                stream_id, remapped_channel.topic
            )
            self.channel_filter_cache[(stream_id, remapped_channel.id)] = should_include

            # Only add channel to output if it passes the filter
            if should_include:
                self.channels[remapped_channel.id] = remapped_channel

    def _handle_message_record(self, message: Message, writer: McapWriter, stream_id: int) -> None:
        """Handle a message record from the stream."""
        message_to_process = self.remapper.remap_message(stream_id, message)
        self.process_message(message_to_process, writer, stream_id)

    def _handle_attachment_record(
        self, attachment: Attachment, writer: McapWriter, stream_id: int
    ) -> None:
        """Handle an attachment record from the stream."""
        self.stats.attachments_processed += 1
        input_opts = self._get_input(stream_id)
        if input_opts.include_attachments and (
            input_opts.start_time <= attachment.log_time < input_opts.end_time
        ):
            writer.add_attachment(
                log_time=attachment.log_time,
                create_time=attachment.create_time,
                name=attachment.name,
                media_type=attachment.media_type,
                data=attachment.data,
            )

    def _generate_chunks_from_stream(
        self, input_stream: IO[bytes], stream_id: int, writer: McapWriter
    ) -> Iterator[PendingChunk]:
        """Generate chunks from a single stream in file order.

        Yields PendingChunk objects with timestamp for ordered merging.
        Non-chunk records (Schema, Channel, Message, Attachment, Metadata) are processed directly.
        """
        pending: PendingChunk | None = None

        def make_pending(chunk: Chunk, indexes: list[MessageIndex]) -> PendingChunk:
            return PendingChunk(chunk, indexes, stream_id, chunk.message_start_time)

        try:
            records = stream_reader(input_stream, emit_chunks=True)
            indexes: list[MessageIndex] = []

            for record in records:
                # Yield pending chunk when we see a non-MessageIndex record
                if not isinstance(record, MessageIndex) and pending:
                    yield pending
                    pending = None

                if isinstance(record, Header):
                    pass  # Header handled separately
                elif isinstance(record, Chunk):
                    self.stats.chunks_processed += 1
                    pending = make_pending(record, indexes := [])
                elif isinstance(record, MessageIndex):
                    indexes.append(record)
                elif isinstance(record, Schema):
                    self._handle_schema_record(record, stream_id)
                elif isinstance(record, Channel):
                    self._handle_channel_record(record, stream_id)
                elif isinstance(record, Message):
                    self._handle_message_record(record, writer, stream_id)
                elif isinstance(record, Attachment):
                    self._handle_attachment_record(record, writer, stream_id)
                elif isinstance(record, Metadata):
                    self.stats.metadata_processed += 1
                    if self._get_input(stream_id).include_metadata:
                        writer.add_metadata(name=record.name, metadata=record.metadata)
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
        output_stream: BinaryIO,
    ) -> ProcessingStats:
        """Main processing function supporting single or multiple input files.

        Args:
            output_stream: Output MCAP file stream

        Returns:
            Processing statistics
        """
        output_opts = self.options.output

        # Initialize writer and share schema/channel dicts for auto-ensure
        writer = McapWriter(
            output_stream,
            chunk_size=output_opts.chunk_size,
            compression=output_opts.compression_type,
        )
        writer.schemas = self.schemas
        writer.channels = self.channels

        # Start writer (use default profile/library for merged files)
        writer.start()

        try:
            # Pre-load schemas and channels from all files' summaries
            # This ensures we have all metadata before processing any chunks
            for stream_id, input_opt in enumerate(self.options.inputs):
                input_stream = input_opt.stream
                try:
                    summary = get_summary(input_stream)
                except McapError:
                    # In recovery mode, if we can't get summary (e.g., truncated file),
                    # continue without it - we'll discover schemas/channels during chunk processing
                    summary = None

                if summary:
                    # Remap and store all schemas
                    for schema in summary.schemas.values():
                        self._handle_schema_record(schema, stream_id)

                    # Remap and store all channels
                    for channel in summary.channels.values():
                        self._handle_channel_record(channel, stream_id)

                # Seek back to start for processing
                input_stream.seek(0)

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
                    self._generate_chunks_from_stream(wrapped_stream, stream_id, writer)
                    for stream_id, wrapped_stream in enumerate(wrapped_streams)
                ]

                # Process chunks in timestamp order using heapq.merge
                # PendingChunk is ordered by timestamp, so no key function needed
                for pending_chunk in heapq.merge(*chunk_generators):
                    self._process_chunk_smart(
                        pending_chunk.chunk,
                        pending_chunk.indexes,
                        writer,
                        pending_chunk.stream_id,
                    )

                # Complete progress
                progress.update(task, completed=total_size)

        finally:
            # Flush all rechunk groups before finishing
            if output_opts.is_rechunking:
                for group in self.rechunk_groups:
                    group.flush()

            writer.finish()

        # Save writer statistics (single source of truth for output counts)
        self.stats.writer_statistics = writer.statistics

        return self.stats

    def _should_decode_chunk(
        self, chunk: Chunk, indexes: list[MessageIndex], stream_id: int
    ) -> ShouldDecode:
        """Determine if chunk should be decoded or can be fast-copied.

        Returns:
            ShouldDecode enum:
            - SKIP: Chunk should be skipped entirely (filtered out)
            - COPY: Chunk can be fast-copied without decoding
            - DECODE: Chunk must be decoded to change channels or filter messages
        """
        input_opts = self._get_input(stream_id)
        output_opts = self.options.output

        if input_opts.always_decode_chunk:
            return ShouldDecode.DECODE

        # Chunk time outside of limits - skip entirely (per-input time filter)
        if (
            chunk.message_end_time < input_opts.start_time
            or chunk.message_start_time >= input_opts.end_time
        ):
            return ShouldDecode.SKIP

        # Force decode if rechunking is active (must reorganize messages)
        if output_opts.is_rechunking:
            return ShouldDecode.DECODE

        # Check if compression matches - must decode to re-compress
        if chunk.compression != output_opts.compression_type.value:
            return ShouldDecode.DECODE

        # Check if time filtering requires per-message filtering
        if input_opts.has_time_filter and not (
            chunk.message_start_time >= input_opts.start_time
            and chunk.message_end_time < input_opts.end_time
        ):
            return ShouldDecode.DECODE

        # Check if any channel was remapped - must decode to fix channel IDs
        # Also check if we have metadata for all channels (they might not be loaded yet)
        for idx in indexes:
            # Check if channel metadata is available
            if not self.remapper.has_channel(stream_id, idx.channel_id):
                # Channel not yet seen - must decode to discover it
                return ShouldDecode.DECODE
            # If channel id was remapped, must decode
            if self.remapper.was_channel_remapped(stream_id, idx.channel_id):
                return ShouldDecode.DECODE

        # Check topic filtering in a single pass over indexes (per-input topic filter)
        if input_opts.has_topic_filter:
            has_include = False
            has_exclude = False

            for idx in indexes:
                cache_key = (stream_id, idx.channel_id)
                # Check if channel filter decision is cached
                if cache_key not in self.channel_filter_cache:
                    # Unknown channel - must decode to discover it
                    return ShouldDecode.DECODE
                if self.channel_filter_cache[cache_key]:
                    has_include = True
                else:
                    has_exclude = True
                # Early exit if we have both - must decode
                if has_include and has_exclude:
                    return ShouldDecode.DECODE

            # If chunk has ONLY excluded channels, skip it entirely
            if has_exclude and not has_include:
                return ShouldDecode.SKIP

        # No reason to decode - can fast-copy
        return ShouldDecode.COPY

    def _process_chunk_smart(
        self, chunk: Chunk, indexes: list[MessageIndex], writer: McapWriter, stream_id: int
    ) -> None:
        """Smart chunk processing with fast copying when possible."""
        should_decode = self._should_decode_chunk(chunk, indexes, stream_id)

        if should_decode == ShouldDecode.SKIP:
            return

        if should_decode == ShouldDecode.DECODE:
            self._process_chunk_fallback(chunk, writer, stream_id)
        else:
            # Fast-copy path -> nothing about the chunks must be changed
            # ensure all channels are written before copying chunk
            for idx in indexes:
                # Check if channel should be included (per-input)
                if self._is_channel_included(stream_id, idx.channel_id):
                    self._ensure_channel_written(idx.channel_id, writer)
            # Fast-copy the chunk with its indexes
            writer.add_chunk(chunk, {idx.channel_id: idx for idx in indexes})
            self.stats.chunks_copied += 1

    def _process_chunk_fallback(
        self, chunk: Chunk, writer: McapWriter | None, stream_id: int
    ) -> None:
        """Fallback to decode chunk into individual records."""
        try:
            chunk_records = breakup_chunk(chunk, validate_crc=True)
            # Only count as decoded if we're actually writing messages
            if writer is not None:
                self.stats.chunks_decoded += 1

            for chunk_record in chunk_records:
                if isinstance(chunk_record, Schema):
                    self._handle_schema_record(chunk_record, stream_id)
                elif isinstance(chunk_record, Channel):
                    self._handle_channel_record(chunk_record, stream_id)
                elif isinstance(chunk_record, Message):
                    if not writer:
                        continue
                    message_to_write = self.remapper.remap_message(stream_id, chunk_record)
                    self.process_message(message_to_write, writer, stream_id)

        except McapError as e:
            console.print(
                f"[yellow]Warning (stream {stream_id}): Failed to decode chunk: {e}[/yellow]"
            )
            self.stats.errors_encountered += 1
