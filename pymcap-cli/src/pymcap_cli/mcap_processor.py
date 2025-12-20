"""Unified MCAP processor combining recovery and filtering capabilities."""

import heapq
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
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
    LazyChunk,
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

from pymcap_cli.processors import (
    Action,
    AlwaysDecodeProcessor,
    AttachmentFilterProcessor,
    ChunkDecision,
    Context,
    MetadataFilterProcessor,
    Processor,
    TimeFilterProcessor,
    TopicFilterProcessor,
)
from pymcap_cli.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    str_to_compression_type,
)
from pymcap_cli.utils import (
    ProgressTrackingIO,
    file_progress,
    parse_timestamp_args,
)

console = Console()


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
            include_topics=other.include_topics if other.include_topics else self.include_topics,
            exclude_topics=other.exclude_topics if other.exclude_topics else self.exclude_topics,
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

    @property
    def compression_type(self) -> CompressionType:
        """Get the CompressionType enum value for the compression string."""
        return str_to_compression_type(self.compression)

    @property
    def is_rechunking(self) -> bool:
        """Check if rechunking is active."""
        return self.rechunk_strategy != RechunkStrategy.NONE


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
        ctx = Context(stream_id=stream_id)
        for p in input_opts.processors:
            if p.on_channel(ctx, channel, self.schemas.get(channel.schema_id)) == Action.SKIP:
                self.channel_filter_cache[cache_key] = False
                return False
        self.channel_filter_cache[cache_key] = True
        return True

    def _ensure_channel_written(self, channel_id: int, writer: McapWriter) -> None:
        """Ensure channel and its schema are written to the main file (not in chunks)."""
        if channel_id in self.written_channels:
            return
        if not (channel := self.channels.get(channel_id)):
            return
        if (
            channel.schema_id != 0
            and channel.schema_id not in self.written_schemas
            and (schema := self.schemas.get(channel.schema_id))
        ):
            writer.add_schema(schema.id, schema.name, schema.encoding, schema.data)
            self.written_schemas.add(schema.id)
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
        for i, pattern in enumerate(self.options.output_options.rechunk_patterns):
            if pattern.search(topic):
                return i
        return None

    def _create_message_group(self, writer: McapWriter) -> MessageGroup:
        """Create a new MessageGroup with output options."""
        opts = self.options.output_options
        group = MessageGroup(writer, opts.chunk_size, opts.compression_type)
        self.rechunk_groups.append(group)
        return group

    def _get_or_create_group_for_channel(
        self, channel_id: int, channel: Channel, writer: McapWriter
    ) -> MessageGroup:
        """Get or create appropriate MessageGroup for a channel based on rechunk strategy."""
        if channel_id in self.channel_to_group:
            return self.channel_to_group[channel_id]

        strategy = self.options.output_options.rechunk_strategy
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

        # Topic filtering using cached decision (avoid repeated regex matching)
        if not self._is_channel_included(stream_id, message.channel_id):
            self.stats.filter_rejections += 1
            return

        # Time filtering and other message processors
        input_opts = self._get_input(stream_id)
        if any(
            p.on_message(Context(stream_id=stream_id), message) == Action.SKIP
            for p in input_opts.processors
        ):
            self.stats.filter_rejections += 1
            return

        # Route to appropriate destination based on rechunking mode
        if self.options.output_options.is_rechunking:
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
            self.channels[remapped_channel.id] = remapped_channel

            # Pre-compute filtering decision for this stream (cache it)
            input_opts = self._get_input(stream_id)
            should_include = True
            if input_opts.processors:
                ctx = Context(stream_id=stream_id)
                for p in input_opts.processors:
                    if p.on_channel(ctx, remapped_channel, self.schemas.get(remapped_channel.schema_id)) == Action.SKIP:
                        should_include = False
                        break
            self.channel_filter_cache[(stream_id, remapped_channel.id)] = should_include

            if not should_include:
                del self.channels[remapped_channel.id]

    def _handle_message_record(self, message: Message, writer: McapWriter, stream_id: int) -> None:
        """Handle a message record from the stream."""
        message_to_process = self.remapper.remap_message(stream_id, message)
        self.process_message(message_to_process, writer, stream_id)

    def _handle_attachment_record(
        self, attachment: Attachment, writer: McapWriter, stream_id: int
    ) -> None:
        """Handle an attachment record from the stream."""
        self.stats.attachments_processed += 1

        # Check all processors (includes AttachmentFilterProcessor and TimeFilterProcessor)
        input_opts = self._get_input(stream_id)
        if any(
            p.on_attachment(Context(stream_id=stream_id), attachment) == Action.SKIP
            for p in input_opts.processors
        ):
            return

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
        Uses lazy_chunks=True for efficiency - chunk data is only read when needed.
        """
        pending: PendingChunk | None = None

        try:
            records = stream_reader(input_stream, emit_chunks=True)  # lazy_chunks=True
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
                    self._handle_message_record(record, writer, stream_id)
                elif isinstance(record, Attachment):
                    self._handle_attachment_record(record, writer, stream_id)
                elif isinstance(record, Metadata):
                    self.stats.metadata_processed += 1
                    input_opts = self._get_input(stream_id)
                    if not input_opts.processors or all(
                        p.on_metadata(Context(stream_id=stream_id), record) != Action.SKIP
                        for p in input_opts.processors
                    ):
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
        output_opts = self.options.output_options

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
                    self._process_chunk_smart(pending_chunk, writer)

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

        # Ask processors for chunk-level decision (time filtering, always_decode, etc.)
        ctx = Context(stream_id=stream_id)

        for proc in input_opts.processors:
            decision = proc.on_chunk(ctx, chunk, indexes)
            if decision == ChunkDecision.SKIP:
                return ChunkDecision.SKIP
            if decision == ChunkDecision.DECODE:
                # Track that we need to decode, but keep checking for SKIP
                return ChunkDecision.DECODE

        # Force decode if rechunking is active (must reorganize messages)
        if output_opts.is_rechunking:
            return ChunkDecision.DECODE

        # Check if compression matches - must decode to re-compress
        if chunk.compression != output_opts.compression_type.value:
            return ChunkDecision.DECODE

        # Check if any channel was remapped - must decode to fix channel IDs
        # Also check if we have metadata for all channels (they might not be loaded yet)
        for idx in indexes:
            # Check if channel metadata is available
            if not self.remapper.has_channel(stream_id, idx.channel_id):
                # Channel not yet seen - must decode to discover it
                return ChunkDecision.DECODE
            # If channel id was remapped, must decode
            if self.remapper.was_channel_remapped(stream_id, idx.channel_id):
                return ChunkDecision.DECODE

        # Check channel filtering in a single pass over indexes
        if input_opts.processors:
            has_include = False
            has_exclude = False

            for idx in indexes:
                cache_key = (stream_id, idx.channel_id)
                # Check if channel filter decision is cached
                if cache_key not in self.channel_filter_cache:
                    # Unknown channel - must decode to discover it
                    return ChunkDecision.DECODE
                if self.channel_filter_cache[cache_key]:
                    has_include = True
                else:
                    has_exclude = True
                # Early exit if we have both - must decode
                if has_include and has_exclude:
                    return ChunkDecision.DECODE

            # If chunk has ONLY excluded channels, skip it entirely
            if has_exclude and not has_include:
                return ChunkDecision.SKIP

        # No reason to decode - can fast-copy
        return ChunkDecision.CONTINUE

    def _process_chunk_smart(self, pending: PendingChunk, writer: McapWriter) -> None:
        """Smart chunk processing with fast copying when possible."""
        decision = self._should_decode_chunk(pending.chunk, pending.indexes, pending.stream_id)

        if decision == ChunkDecision.SKIP:
            return

        # Now we need the full chunk data
        try:
            chunk = pending.get_chunk()
        except (EOFError, McapError) as e:
            console.print(
                f"[yellow]Warning (stream {pending.stream_id}): Failed to read chunk: {e}[/yellow]"
            )
            self.stats.errors_encountered += 1
            return

        if decision == ChunkDecision.DECODE:
            self._process_chunk_fallback(chunk, writer, pending.stream_id)
        else:
            # Fast-copy path (CONTINUE) -> nothing about the chunk must be changed
            # ensure all channels are written before copying chunk
            for idx in pending.indexes:
                # Check if channel should be included (per-input)
                if self._is_channel_included(pending.stream_id, idx.channel_id):
                    self._ensure_channel_written(idx.channel_id, writer)
            # Fast-copy the chunk with its indexes
            writer.add_chunk(chunk, {idx.channel_id: idx for idx in pending.indexes})
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
