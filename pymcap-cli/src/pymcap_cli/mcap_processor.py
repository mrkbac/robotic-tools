"""Unified MCAP processor combining recovery and filtering capabilities."""

import heapq
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import IO, BinaryIO

import typer
from rich.console import Console
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

from pymcap_cli.types import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
)
from pymcap_cli.utils import ProgressTrackingIO, file_progress

console = Console()


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


# Rechunking strategy enum (must be defined before ProcessingOptions)
class RechunkStrategy(str, Enum):
    """Rechunking strategy for organizing messages into chunks."""

    NONE = "none"  # No rechunking - use fast-copy optimization when possible
    PATTERN = "pattern"  # Group by regex patterns
    ALL = "all"  # Each topic in its own chunk group
    AUTO = "auto"  # Auto-group based on size (>15% threshold)


@dataclass
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
    end_time: int = 2**63 - 1  # Max int64

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
        return self.start_time > 0 or self.end_time < 2**63 - 1

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
        schemas: dict[int, Schema],
        channels: dict[int, Channel],
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
            schemas=schemas,
            channels=channels,
        )

    def add_message(self, message: Message) -> None:
        """Add message to this group's chunk builder."""
        # ChunkBuilder auto-ensures channel and schema
        self.chunk_builder.add(message)

        # If chunk builder returns a completed chunk, write it immediately
        if result := self.chunk_builder.maybe_finalize():
            chunk, message_indexes = result
            if chunk.compression != self.chunk_builder.compression.value:
                self.compress_fail_counter += 1
                if self.compress_fail_counter > 2:
                    console.print(
                        "[yellow]Multiple compression failures, switching to uncompressed.[/yellow]"
                    )
                    self.chunk_builder.compression = CompressionType.NONE
            self.writer.add_chunk_with_indexes(chunk, list(message_indexes.values()))

        self.message_count += 1

    def flush(self) -> None:
        """Flush any remaining messages in this group's chunk builder."""
        result = self.chunk_builder.finalize()
        if result is not None:
            chunk, message_indexes = result
            self.writer.add_chunk_with_indexes(chunk, list(message_indexes.values()))


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
        typer.Abort: If user declines to overwrite
    """
    if output.exists() and not force:
        typer.confirm(
            f"Output file '{output}' already exists. Overwrite?",
            abort=True,
        )


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
        end_time = 2**63 - 1

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


def report_processing_stats(
    stats: ProcessingStats,
    console_out: Console,
    num_files: int = 1,
    command_context: str = "process",
) -> None:
    """Report processing statistics with appropriate messaging.

    Args:
        stats: Processing statistics to report
        console_out: Rich console for output
        num_files: Number of input files processed
        command_context: Context string (merge/filter/process) for success message
    """
    # Success message
    if command_context == "merge" and num_files > 1:
        console_out.print(f"[green]✓ Successfully merged {num_files} files![/green]")
    elif command_context == "filter":
        console_out.print("[green]✓ Filter completed successfully![/green]")
    elif num_files > 1:
        console_out.print(f"[green]✓ Merged {num_files} files successfully![/green]")
    else:
        console_out.print("[green]✓ Processing completed successfully![/green]")

    # Basic stats
    console_out.print(
        f"Processed {stats.messages_processed:,} messages, "
        f"wrote {stats.writer_statistics.message_count:,} messages"
    )

    # Content stats
    if stats.attachments_processed > 0 and stats.writer_statistics:
        console_out.print(
            f"Processed {stats.attachments_processed} attachments, "
            f"wrote {stats.writer_statistics.attachment_count}"
        )
    if stats.metadata_processed > 0 and stats.writer_statistics:
        console_out.print(
            f"Processed {stats.metadata_processed} metadata records, "
            f"wrote {stats.writer_statistics.metadata_count}"
        )

    # Schema/channel stats
    console_out.print(
        f"Wrote {stats.writer_statistics.schema_count} schemas and "
        f"{stats.writer_statistics.channel_count} channels"
    )

    # Performance stats
    if stats.chunks_processed > 0:
        console_out.print(
            f"Processed {stats.chunks_processed} chunks "
            f"({stats.chunks_copied} fast copied, {stats.chunks_decoded} decoded)"
        )

    # Error stats
    if stats.errors_encountered > 0:
        console_out.print(f"[yellow]Encountered {stats.errors_encountered} errors[/yellow]")
    if stats.validation_errors > 0:
        console_out.print(f"[yellow]Found {stats.validation_errors} validation errors[/yellow]")
    if stats.filter_rejections > 0:
        console_out.print(f"Filtered out {stats.filter_rejections} records")


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

        # Track schemas and channels (shared with writer for auto-ensure)
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

    def _compute_channel_filter_decision(self, topic: str) -> bool:
        """Compute whether a channel with given topic should be included.

        Returns True if the channel should be included, False otherwise.
        """
        if self.options.include_topics:
            return any(p.search(topic) for p in self.options.include_topics)
        if self.options.exclude_topics:
            return not any(p.search(topic) for p in self.options.exclude_topics)
        return True

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

    def _get_remapped_channel_id(self, stream_id: int, original_channel_id: int) -> int:
        # Check if this channel was remapped
        if self.remapper.was_channel_remapped(stream_id, original_channel_id):
            # Use get_remapped_channel to get the remapped ID
            remapped_channel = self.remapper.get_remapped_channel(stream_id, original_channel_id)
            return remapped_channel.id if remapped_channel else original_channel_id
        return original_channel_id

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
                self.schemas,
                self.channels,
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
                    self.schemas,
                    self.channels,
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
                self.schemas,
                self.channels,
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
            self.schemas,
            self.channels,
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
            writer.ensure_channel_written(message.channel_id)

            # Add message to its group (chunk builder auto-ensures within chunks)
            group.add_message(message)
        else:
            # Normal mode: writer auto-ensures channel and schema are written
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
        remapped_channel_id = self._get_remapped_channel_id(stream_id, message.channel_id)

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

    def _get_remapped_channel_ids(
        self, indexes: list[MessageIndex], stream_id: int
    ) -> tuple[set[int], bool]:
        """Get remapped channel IDs for indexes and check if metadata extraction is needed.

        Returns:
            (remapped_channel_ids, needs_metadata_extraction)
        """
        remapped_channel_ids = set()
        needs_metadata_extraction = False

        for idx in indexes:
            original_id = idx.channel_id
            remapped_channel = self.remapper.get_remapped_channel(stream_id, original_id)
            if remapped_channel:
                remapped_id = remapped_channel.id
            else:
                # Not yet in remapper - will keep original ID, but need to extract metadata
                remapped_id = original_id
                needs_metadata_extraction = True

            remapped_channel_ids.add(remapped_id)

        # Bulk set operation: check if any IDs are unknown (more efficient than individual checks)
        if not needs_metadata_extraction:
            unknown_channels = remapped_channel_ids - self.known_channels
            if unknown_channels:
                needs_metadata_extraction = True

        return remapped_channel_ids, needs_metadata_extraction

    def _should_decode_chunk(
        self, chunk: Chunk, indexes: list[MessageIndex], stream_id: int
    ) -> tuple[bool, bool]:
        """Determine if chunk should be decoded or can be fast-copied.

        Returns:
            (should_skip, should_decode): tuple of booleans
            - should_skip: True if chunk should be skipped entirely
            - should_decode: True if chunk must be decoded, False if it can be fast-copied
        """
        # Chunk time outside of limits - skip entirely
        if (
            chunk.message_end_time < self.options.start_time
            or chunk.message_start_time >= self.options.end_time
        ):
            return (True, False)

        # Force decode if rechunking is active (must reorganize messages)
        if self.options.is_rechunking:
            return (False, True)

        # Force decode if requested
        if self.options.always_decode_chunk:
            return (False, True)

        # Check if compression matches - must decode to re-compress
        if chunk.compression != self.options.compression_type.value:
            return (False, True)

        # Check if time filtering requires per-message filtering
        if self.options.has_time_filter and not (
            chunk.message_start_time >= self.options.start_time
            and chunk.message_end_time < self.options.end_time
        ):
            return (False, True)

        # Check if any channel was remapped - must decode to fix channel IDs
        # Also check if we have metadata for all channels (they might not be loaded yet)
        for idx in indexes:
            # Check if channel metadata is available
            if not self.remapper.has_channel(stream_id, idx.channel_id):
                # Channel not yet seen - must decode to discover it
                return (False, True)
            if self.remapper.was_channel_remapped(stream_id, idx.channel_id):
                return (False, True)

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
                    return (False, True)
                if idx.channel_id in self.included_channels:
                    has_include = True
                else:
                    has_exclude = True
                # Early exit if we have both - must decode
                if has_include and has_exclude:
                    return (False, True)

            # If chunk has ONLY excluded channels, skip it entirely
            if has_exclude and not has_include:
                return (True, False)

        # No reason to decode - can fast-copy
        return (False, False)

    def _process_chunk_smart(
        self, chunk: Chunk, indexes: list[MessageIndex], writer: McapWriter, stream_id: int
    ) -> None:
        """Smart chunk processing with fast copying when possible."""
        should_skip, should_decode = self._should_decode_chunk(chunk, indexes, stream_id)

        if should_skip:
            return

        if should_decode:
            self._process_chunk_fallback(chunk, writer, stream_id)
        else:
            # Fast-copy path: extract schemas/channels first (only if needed)
            remapped_channel_ids, needs_metadata_extraction = self._get_remapped_channel_ids(
                indexes, stream_id
            )

            if needs_metadata_extraction:
                # Need to decode chunk to discover schemas/channels
                # Decode and write messages instead of metadata extraction + fast-copy
                self._process_chunk_fallback(chunk, writer, stream_id)
            else:
                # Ensure all channels (and their schemas) are written before fast-copying chunk
                # Only write channels that pass the filter
                for channel_id in remapped_channel_ids:
                    # Check if channel should be included
                    if channel_id not in self.excluded_channels:
                        writer.ensure_channel_written(channel_id)

                # Fast-copy the chunk with its indexes
                writer.add_chunk_with_indexes(chunk, indexes)
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
                    remapped_channel_id = self._get_remapped_channel_id(
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
