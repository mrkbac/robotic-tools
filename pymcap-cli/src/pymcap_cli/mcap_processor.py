"""Unified MCAP processor combining recovery and filtering capabilities."""

import heapq
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
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
)
from pymcap_cli.utils import ProgressTrackingIO, file_progress

console = Console()

# Maximum value for a signed 64-bit integer (used for unbounded time range)
MAX_INT64 = 2**63 - 1


@dataclass(slots=True)
class PendingChunk:
    chunk: Chunk
    indexes: list[MessageIndex]
    stream_id: int
    timestamp: int  # message_start_time for ordering

    def __lt__(self, other: "PendingChunk") -> bool:
        return self.timestamp < other.timestamp


def str_to_compression_type(compression: str) -> CompressionType:
    """Convert compression string to CompressionType enum."""
    compression_lower = compression.lower()
    if compression_lower in ("none", "", "off"):
        return CompressionType.NONE
    if compression_lower == "lz4":
        return CompressionType.LZ4
    if compression_lower == "zstd":
        return CompressionType.ZSTD
    raise ValueError(f"Unknown compression type: {compression}")


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


@dataclass(slots=True)
class ProcessingOptions:
    """Unified options for MCAP processing (recovery + filtering)."""

    # Recovery options
    recovery_mode: bool = True  # Always handle errors gracefully
    always_decode_chunk: bool = False  # Force individual record processing

    # Filter options - Topic filtering
    include_topics: list[Pattern[str]] = field(default_factory=list)
    exclude_topics: list[Pattern[str]] = field(default_factory=list)

    # Filter options - Time filtering (nanoseconds)
    start_time: int = 0
    end_time: int = MAX_INT64

    # Filter options - Content filtering
    include_metadata: bool = True
    include_attachments: bool = True

    # Rechunking options
    rechunk_strategy: RechunkStrategy = RechunkStrategy.NONE
    rechunk_patterns: list[Pattern[str]] = field(default_factory=list)

    # Output options
    compression: str = DEFAULT_COMPRESSION
    chunk_size: int = DEFAULT_CHUNK_SIZE

    @property
    def compression_type(self) -> CompressionType:
        """Get the CompressionType enum value for the compression string."""
        return str_to_compression_type(self.compression)

    @property
    def has_time_filter(self) -> bool:
        """Check if time filtering is active."""
        return self.start_time > 0 or self.end_time < MAX_INT64

    @property
    def has_topic_filter(self) -> bool:
        """Check if topic filtering is active."""
        return bool(self.include_topics or self.exclude_topics)

    @property
    def is_rechunking(self) -> bool:
        """Check if rechunking is active."""
        return self.rechunk_strategy != RechunkStrategy.NONE


@dataclass
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
    # TODO: improve this?
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


def parse_time_arg(time_str: str) -> int:
    """Parse time argument that can be nanoseconds or RFC3339 date."""
    if not time_str:
        return 0

    # Try parsing as integer nanoseconds first
    try:
        return int(time_str)
    except ValueError:
        pass

    # Try parsing as RFC3339 date
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000_000)
    except ValueError:
        raise ValueError(
            f"Invalid time format: {time_str}. Use nanoseconds or RFC3339 format"
        ) from None


def compile_topic_patterns(patterns: list[str]) -> list[Pattern[str]]:
    """Compile topic regex patterns"""
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    return compiled


# Shared enums for command-line interfaces
class MetadataMode(str, Enum):
    """Metadata inclusion mode."""

    INCLUDE = "include"
    EXCLUDE = "exclude"


class AttachmentsMode(str, Enum):
    """Attachments inclusion mode."""

    INCLUDE = "include"
    EXCLUDE = "exclude"


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


# Shared helper functions for command-line interfaces
def parse_timestamp_args(date_or_nanos: str, nanoseconds: int, seconds: int) -> int:
    """Parse timestamp with precedence: date_or_nanos > nanoseconds > seconds."""
    if date_or_nanos:
        return parse_time_arg(date_or_nanos)
    if nanoseconds != 0:
        return nanoseconds
    return seconds * 1_000_000_000


def confirm_output_overwrite(output: Path, force: bool) -> None:
    """Confirm overwrite if output exists and force=False.

    Args:
        output: Output file path
        force: If True, skip confirmation

    Raises:
        SystemExit: If user declines to overwrite
    """
    if output.exists() and not force:
        response = input(f"Output file '{output}' already exists. Overwrite? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            print("Aborted.")  # noqa: T201
            raise SystemExit(1)


def build_processing_options(
    include_topic_regex: list[str] | None = None,
    exclude_topic_regex: list[str] | None = None,
    start: str = "",
    start_nsecs: int = 0,
    start_secs: int = 0,
    end: str = "",
    end_nsecs: int = 0,
    end_secs: int = 0,
    metadata_mode: MetadataMode = MetadataMode.INCLUDE,
    attachments_mode: AttachmentsMode = AttachmentsMode.INCLUDE,
    compression: str = DEFAULT_COMPRESSION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    recovery_mode: bool = True,
    always_decode_chunk: bool = False,
) -> ProcessingOptions:
    """Build ProcessingOptions from common command-line arguments.

    Args:
        include_topic_regex: List of regex patterns for topics to include
        exclude_topic_regex: List of regex patterns for topics to exclude
        start: Start time (nanoseconds or RFC3339 date)
        start_nsecs: Start time in nanoseconds (deprecated)
        start_secs: Start time in seconds
        end: End time (nanoseconds or RFC3339 date)
        end_nsecs: End time in nanoseconds (deprecated)
        end_secs: End time in seconds
        metadata_mode: Whether to include or exclude metadata
        attachments_mode: Whether to include or exclude attachments
        compression: Compression algorithm (zstd/lz4/none)
        chunk_size: Chunk size in bytes
        recovery_mode: Enable recovery mode for error handling
        always_decode_chunk: Force chunk decoding (disable fast copying)

    Returns:
        ProcessingOptions configured from the arguments

    Raises:
        ValueError: If arguments are invalid
    """
    # Handle None defaults for list parameters
    include_topic_regex = include_topic_regex or []
    exclude_topic_regex = exclude_topic_regex or []

    # Validate mutually exclusive options
    if include_topic_regex and exclude_topic_regex:
        raise ValueError("Cannot use both include and exclude topic filters")

    # Parse time arguments
    try:
        start_time = parse_timestamp_args(start, start_nsecs, start_secs)
        end_time = parse_timestamp_args(end, end_nsecs, end_secs)
    except ValueError as e:
        raise ValueError(f"Time parsing error: {e}") from e

    # Default end time to max if not specified
    if end_time == 0:
        end_time = MAX_INT64

    # Validate time range
    if end_time < start_time:
        raise ValueError("End time cannot be before start time")

    # Compile topic patterns
    include_topics = compile_topic_patterns(include_topic_regex)
    exclude_topics = compile_topic_patterns(exclude_topic_regex)

    return ProcessingOptions(
        recovery_mode=recovery_mode,
        always_decode_chunk=always_decode_chunk,
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        start_time=start_time,
        end_time=end_time,
        include_metadata=metadata_mode == MetadataMode.INCLUDE,
        include_attachments=attachments_mode == AttachmentsMode.INCLUDE,
        compression=compression,
        chunk_size=chunk_size,
    )


class McapProcessor:
    """Unified MCAP processor combining recovery and filtering capabilities.

    Supports processing single or multiple MCAP files with smart schema/channel ID
    remapping to minimize the need for chunk decoding.
    """

    def __init__(self, options: ProcessingOptions) -> None:
        self.options = options
        self.stats = ProcessingStats()

        # ID remapper for handling multiple files (zero overhead for single file)
        self.remapper = Remapper()

        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}

        # Cache filtering decisions per channel ID (set-based for efficiency)
        self.included_channels: set[int] = set()
        self.excluded_channels: set[int] = set()

        # Track which channels we've already seen to optimize metadata extraction
        self.known_channels: set[int] = set()

        # Rechunking state (only used when options.is_rechunking)
        self.channel_to_group: dict[int, MessageGroup] = {}
        self.large_channels: set[int] = set()  # For AUTO mode
        self.rechunk_groups: list[MessageGroup] = []  # Track unique groups
        self.pattern_index_to_group: dict[int, MessageGroup] = {}  # For PATTERN mode cache

        # Track which schemas/channels we've written to the main file (not in chunks)
        self.written_schemas: set[int] = set()
        self.written_channels: set[int] = set()

    def _compute_channel_filter_decision(self, topic: str) -> bool:
        """Compute whether a channel with given topic should be included.

        Returns True if the channel should be included, False otherwise.
        """
        if self.options.include_topics:
            return any(p.search(topic) for p in self.options.include_topics)
        if self.options.exclude_topics:
            return not any(p.search(topic) for p in self.options.exclude_topics)
        return True

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

    def _get_remapped_schema_id(self, stream_id: int, original_schema_id: int) -> int:
        """Get the remapped schema ID for an original schema ID.

        Args:
            stream_id: The stream ID
            original_schema_id: The original schema ID from the input file

        Returns:
            The remapped schema ID, or 0 if schema not found
        """
        if original_schema_id == 0:
            return 0

        # Use Remapper's built-in lookup
        return self.remapper.get_remapped_schema(stream_id, original_schema_id)

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
        for i, pattern in enumerate(self.options.rechunk_patterns):
            if pattern.search(topic):
                return i
        return None

    def _get_or_create_group_for_channel(
        self, channel_id: int, channel: Channel, writer: McapWriter
    ) -> MessageGroup:
        """Get or create appropriate MessageGroup for a channel based on rechunk strategy."""
        # Check if we already have a group for this channel
        if channel_id in self.channel_to_group:
            return self.channel_to_group[channel_id]

        # Determine which group to use based on strategy
        if self.options.rechunk_strategy == RechunkStrategy.ALL:
            # Each channel gets its own unique group
            group = MessageGroup(
                writer,
                self.options.chunk_size,
                self.options.compression_type,
            )
            self.channel_to_group[channel_id] = group
            self.rechunk_groups.append(group)
            return group

        if self.options.rechunk_strategy == RechunkStrategy.AUTO:
            # Large channels get their own group, small channels share one
            if channel_id in self.large_channels:
                # Create unique group for large channel
                group = MessageGroup(
                    writer,
                    self.options.chunk_size,
                    self.options.compression_type,
                )
                self.channel_to_group[channel_id] = group
                self.rechunk_groups.append(group)
                return group

            # Look for any existing small-channel group
            for ch_id, group in self.channel_to_group.items():
                if ch_id not in self.large_channels:
                    # Reuse this shared group
                    self.channel_to_group[channel_id] = group
                    return group

            # No shared group yet, create one
            group = MessageGroup(
                writer,
                self.options.chunk_size,
                self.options.compression_type,
            )
            self.channel_to_group[channel_id] = group
            self.rechunk_groups.append(group)
            return group

        # RechunkStrategy.PATTERN
        # Find which pattern matches this channel's topic
        pattern_idx = self._find_matching_pattern_index(channel.topic)
        group_key = pattern_idx if pattern_idx is not None else -1  # -1 for unmatched

        if group_key in self.pattern_index_to_group:
            group = self.pattern_index_to_group[group_key]
            self.channel_to_group[channel_id] = group
            return group

        # No existing group for this pattern, create new one
        group = MessageGroup(
            writer,
            self.options.chunk_size,
            self.options.compression_type,
        )
        self.channel_to_group[channel_id] = group
        self.pattern_index_to_group[group_key] = group
        self.rechunk_groups.append(group)
        return group

    def process_message(
        self,
        message: Message,
        writer: McapWriter,
    ) -> None:
        """Process a message record."""
        self.stats.messages_processed += 1

        # Time filtering
        if not (self.options.start_time <= message.log_time < self.options.end_time):
            return

        # Topic filtering using cached decision (avoid repeated regex matching)
        # Fast path: check if already determined
        if message.channel_id in self.excluded_channels:
            self.stats.filter_rejections += 1
            return

        if message.channel_id not in self.included_channels:
            # Channel not yet seen - compute and cache decision
            channel = self.channels.get(message.channel_id)
            topic = channel.topic if channel else ""
            should_include = self._compute_channel_filter_decision(topic)
            if should_include:
                self.included_channels.add(message.channel_id)
            else:
                self.excluded_channels.add(message.channel_id)
                self.stats.filter_rejections += 1
                return

        # Route to appropriate destination based on rechunking mode
        if self.options.is_rechunking:
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

    def _handle_channel_record(self, channel: Channel, stream_id: int) -> None:
        """Handle a channel record from the stream."""
        remapped_channel = self.remapper.remap_channel(stream_id, channel)

        # Restore schema_id (Remapper zeros it out to avoid stale refs)
        if channel.schema_id != 0:
            remapped_schema_id = self._get_remapped_schema_id(stream_id, channel.schema_id)
            if remapped_schema_id != 0:
                remapped_channel = replace(remapped_channel, schema_id=remapped_schema_id)

        if remapped_channel.id not in self.known_channels:
            # Pre-compute filtering decision for this channel
            should_include = self._compute_channel_filter_decision(remapped_channel.topic)
            if should_include:
                self.included_channels.add(remapped_channel.id)
            else:
                self.excluded_channels.add(remapped_channel.id)
            self.known_channels.add(remapped_channel.id)

            # Only add channel to output if it passes the filter
            if should_include:
                self.channels[remapped_channel.id] = remapped_channel

    def _handle_message_record(self, message: Message, writer: McapWriter, stream_id: int) -> None:
        """Handle a message record from the stream."""
        # Use cached channel ID lookup (avoids creating temporary Channel objects)
        remapped_channel_id = self.remapper.get_remapped_channel_id(stream_id, message.channel_id)

        if remapped_channel_id != message.channel_id:
            # Channel was remapped, create new message with remapped ID
            message_to_process = Message(
                channel_id=remapped_channel_id,
                sequence=message.sequence,
                log_time=message.log_time,
                publish_time=message.publish_time,
                data=message.data,
            )
        else:
            message_to_process = message

        self.process_message(message_to_process, writer)

    def _handle_attachment_record(self, attachment: Attachment, writer: McapWriter) -> None:
        """Handle an attachment record from the stream."""
        self.stats.attachments_processed += 1
        if self.options.include_attachments and (
            self.options.start_time <= attachment.log_time < self.options.end_time
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
        last_chunk: Chunk | None = None
        last_chunk_message_indexes: list[MessageIndex] = []

        try:
            records = stream_reader(input_stream, emit_chunks=True)

            for record in records:
                # Yield pending chunk when we see a non-MessageIndex record
                if not isinstance(record, MessageIndex) and last_chunk:
                    yield PendingChunk(
                        timestamp=last_chunk.message_start_time,
                        chunk=last_chunk,
                        indexes=last_chunk_message_indexes,
                        stream_id=stream_id,
                    )
                    last_chunk = None
                    last_chunk_message_indexes = []

                # Process current record
                if isinstance(record, Header):
                    pass  # Header handled separately
                elif isinstance(record, Chunk):
                    self.stats.chunks_processed += 1
                    last_chunk = record
                    last_chunk_message_indexes = []
                elif isinstance(record, MessageIndex):
                    last_chunk_message_indexes.append(record)
                elif isinstance(record, Schema):
                    # Remap schema ID and store
                    remapped_schema = self.remapper.remap_schema(stream_id, record)
                    if remapped_schema:
                        self.schemas[remapped_schema.id] = remapped_schema
                elif isinstance(record, Channel):
                    self._handle_channel_record(record, stream_id)
                elif isinstance(record, Message):
                    self._handle_message_record(record, writer, stream_id)
                elif isinstance(record, Attachment):
                    self._handle_attachment_record(record, writer)
                elif isinstance(record, Metadata):
                    self.stats.metadata_processed += 1
                    if self.options.include_metadata:
                        writer.add_metadata(name=record.name, metadata=record.metadata)
                elif isinstance(record, (DataEnd, Footer)):
                    break

        except McapError as e:
            # Catch errors from iteration itself (e.g., truncated file)
            if self.options.recovery_mode:
                console.print(f"[yellow]Warning (stream {stream_id}): {e}[/yellow]")
                self.stats.errors_encountered += 1
                # If we have a pending chunk, yield it for processing
                if last_chunk:
                    yield PendingChunk(
                        timestamp=last_chunk.message_start_time,
                        chunk=last_chunk,
                        indexes=last_chunk_message_indexes,
                        stream_id=stream_id,
                    )
                    last_chunk = None
            else:
                raise

        # Yield the final pending chunk if any
        if last_chunk:
            yield PendingChunk(
                timestamp=last_chunk.message_start_time,
                chunk=last_chunk,
                indexes=last_chunk_message_indexes,
                stream_id=stream_id,
            )

    def process(
        self,
        input_streams: Sequence[IO[bytes]],
        output_stream: BinaryIO,
        file_sizes: Sequence[int],
    ) -> ProcessingStats:
        """Main processing function supporting single or multiple input files.

        Args:
            input_streams: List of input MCAP file streams
            output_stream: Output MCAP file stream
            file_sizes: List of file sizes for progress tracking

        Returns:
            Processing statistics
        """
        # Initialize writer and share schema/channel dicts for auto-ensure
        writer = McapWriter(
            output_stream,
            chunk_size=self.options.chunk_size,
            compression=self.options.compression_type,
        )
        writer.schemas = self.schemas
        writer.channels = self.channels

        # Start writer (use default profile/library for merged files)
        writer.start()

        try:
            # Pre-load schemas and channels from all files' summaries
            # This ensures we have all metadata before processing any chunks
            for stream_id, input_stream in enumerate(input_streams):
                try:
                    summary = get_summary(input_stream)
                except McapError:
                    # In recovery mode, if we can't get summary (e.g., truncated file),
                    # continue without it - we'll discover schemas/channels during chunk processing
                    summary = None

                if summary:
                    # Remap and store all schemas
                    for schema in summary.schemas.values():
                        remapped_schema = self.remapper.remap_schema(stream_id, schema)
                        if remapped_schema:
                            self.schemas[remapped_schema.id] = remapped_schema

                    # Remap and store all channels
                    for channel in summary.channels.values():
                        remapped_channel = self.remapper.remap_channel(stream_id, channel)

                        # Restore schema_id (Remapper zeros it out to avoid stale refs)
                        if channel.schema_id != 0:
                            remapped_schema_id = self._get_remapped_schema_id(
                                stream_id, channel.schema_id
                            )
                            if remapped_schema_id != 0:
                                remapped_channel = replace(
                                    remapped_channel, schema_id=remapped_schema_id
                                )

                        if remapped_channel.id not in self.known_channels:
                            # Pre-compute filtering decision
                            should_include = self._compute_channel_filter_decision(
                                remapped_channel.topic
                            )
                            if should_include:
                                self.included_channels.add(remapped_channel.id)
                            else:
                                self.excluded_channels.add(remapped_channel.id)
                            self.known_channels.add(remapped_channel.id)

                            # Only add channel to output if it passes the filter
                            if should_include:
                                self.channels[remapped_channel.id] = remapped_channel

                # Seek back to start for processing
                input_stream.seek(0)

            # For AUTO rechunking mode, pre-analyze files to identify large channels
            if self.options.rechunk_strategy == RechunkStrategy.AUTO:
                self._analyze_for_auto_grouping(input_streams)

            total_size = sum(file_sizes)
            with file_progress("[bold blue]Processing MCAP...", console) as progress:
                task = progress.add_task("Processing", total=total_size)

                # Wrap streams to track progress incrementally
                wrapped_streams = [
                    ProgressTrackingIO(input_stream, task, progress, input_stream.tell())
                    for input_stream in input_streams
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
            if self.options.is_rechunking:
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
        if self.options.always_decode_chunk:
            return ShouldDecode.DECODE

        # Chunk time outside of limits - skip entirely
        if (
            chunk.message_end_time < self.options.start_time
            or chunk.message_start_time >= self.options.end_time
        ):
            return ShouldDecode.SKIP

        # Force decode if rechunking is active (must reorganize messages)
        if self.options.is_rechunking:
            return ShouldDecode.DECODE

        # Check if compression matches - must decode to re-compress
        if chunk.compression != self.options.compression_type.value:
            return ShouldDecode.DECODE

        # Check if time filtering requires per-message filtering
        if self.options.has_time_filter and not (
            chunk.message_start_time >= self.options.start_time
            and chunk.message_end_time < self.options.end_time
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

        # Check topic filtering in a single pass over indexes
        if self.options.has_topic_filter:
            has_include = False
            has_exclude = False

            for idx in indexes:
                # Check if channel filter decision is cached
                if (
                    idx.channel_id not in self.included_channels
                    and idx.channel_id not in self.excluded_channels
                ):
                    # Unknown channel - must decode to discover it
                    return ShouldDecode.DECODE
                if idx.channel_id in self.included_channels:
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
                # Check if channel should be included
                if idx.channel_id not in self.excluded_channels:
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
                    # Remap schema and store
                    remapped_schema = self.remapper.remap_schema(stream_id, chunk_record)
                    if remapped_schema:
                        self.schemas[remapped_schema.id] = remapped_schema
                elif isinstance(chunk_record, Channel):
                    # Remap channel and store
                    remapped_channel = self.remapper.remap_channel(stream_id, chunk_record)

                    # Restore schema_id (Remapper zeros it out to avoid stale refs)
                    if chunk_record.schema_id != 0:
                        remapped_schema_id = self._get_remapped_schema_id(
                            stream_id, chunk_record.schema_id
                        )
                        if remapped_schema_id != 0:
                            remapped_channel = replace(
                                remapped_channel, schema_id=remapped_schema_id
                            )

                    if remapped_channel.id not in self.known_channels:
                        # Pre-compute filtering decision for this channel
                        should_include = self._compute_channel_filter_decision(
                            remapped_channel.topic
                        )
                        if should_include:
                            self.included_channels.add(remapped_channel.id)
                        else:
                            self.excluded_channels.add(remapped_channel.id)
                        self.known_channels.add(remapped_channel.id)

                        # Only add channel to output if it passes the filter
                        if should_include:
                            self.channels[remapped_channel.id] = remapped_channel
                elif isinstance(chunk_record, Message):
                    if not writer:
                        continue
                    # Remap message channel ID if needed (using cache to avoid temporary objects)
                    remapped_channel_id = self.remapper.get_remapped_channel_id(
                        stream_id, chunk_record.channel_id
                    )
                    if remapped_channel_id != chunk_record.channel_id:
                        message_to_write = replace(
                            chunk_record,
                            channel_id=remapped_channel_id,
                        )
                    else:
                        message_to_write = chunk_record
                    self.process_message(message_to_write, writer)

        except McapError as e:
            if self.options.recovery_mode:
                console.print(
                    f"[yellow]Warning (stream {stream_id}): Failed to decode chunk: {e}[/yellow]"
                )
                self.stats.errors_encountered += 1
            else:
                raise
