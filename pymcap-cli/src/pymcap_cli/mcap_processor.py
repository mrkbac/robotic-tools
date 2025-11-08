"""Unified MCAP processor combining recovery and filtering capabilities."""

import heapq
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from re import Pattern
from typing import BinaryIO, NamedTuple

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

from pymcap_cli.utils import file_progress

console = Console()


class PendingChunk(NamedTuple):
    """Chunk with its indexes and metadata for ordered processing."""

    chunk: Chunk
    indexes: list[MessageIndex]
    stream_id: int
    timestamp: int  # message_start_time for ordering


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

    # Output options
    compression: str = "zstd"
    chunk_size: int = 4 * 1024 * 1024  # 4MB

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

        # Channel topic cache for filtering
        self.channel_topics: dict[int, str] = {}

        # Cache filtering decisions per channel ID to avoid repeated regex matching
        self.channel_filter_cache: dict[int, bool] = {}

        # Track which channels we've already seen to optimize metadata extraction
        self.known_channels: set[int] = set()

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
        should_include = self.channel_filter_cache.get(message.channel_id)
        if should_include is None:
            # Channel not yet seen - this shouldn't happen in well-formed MCAP
            # but handle it gracefully
            topic = self.channel_topics.get(message.channel_id, "")
            should_include = self._compute_channel_filter_decision(topic)
            self.channel_filter_cache[message.channel_id] = should_include

        if not should_include:
            self.stats.filter_rejections += 1
            return

        # Writer auto-ensures channel and schema are written
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

        if remapped_channel.id not in self.channels:
            self.channels[remapped_channel.id] = remapped_channel
            self.channel_topics[remapped_channel.id] = remapped_channel.topic
            self.known_channels.add(remapped_channel.id)
            # Pre-compute filtering decision for this channel
            self.channel_filter_cache[remapped_channel.id] = self._compute_channel_filter_decision(
                remapped_channel.topic
            )

    def _handle_message_record(self, message: Message, writer: McapWriter, stream_id: int) -> None:
        """Handle a message record from the stream."""
        original_channel_id = message.channel_id
        message_to_process: Message = message
        if self.remapper.was_channel_remapped(stream_id, original_channel_id):
            # Look up the remapped channel
            original_channel = Channel(
                id=original_channel_id,
                schema_id=0,
                topic="",
                message_encoding="",
                metadata={},
            )
            remapped_channel = self.remapper.remap_channel(stream_id, original_channel)
            message_to_process = Message(
                channel_id=remapped_channel.id,
                sequence=message.sequence,
                log_time=message.log_time,
                publish_time=message.publish_time,
                data=message.data,
            )
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
        self, input_stream: BinaryIO, stream_id: int, writer: McapWriter
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
                        chunk=last_chunk,
                        indexes=last_chunk_message_indexes,
                        stream_id=stream_id,
                        timestamp=last_chunk.message_start_time,
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
                        chunk=last_chunk,
                        indexes=last_chunk_message_indexes,
                        stream_id=stream_id,
                        timestamp=last_chunk.message_start_time,
                    )
                    last_chunk = None
            else:
                raise

        # Yield the final pending chunk if any
        if last_chunk:
            yield PendingChunk(
                chunk=last_chunk,
                indexes=last_chunk_message_indexes,
                stream_id=stream_id,
                timestamp=last_chunk.message_start_time,
            )

    def process(
        self,
        input_streams: Sequence[BinaryIO],
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
                if summary := get_summary(input_stream):
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

                        if remapped_channel.id not in self.channels:
                            self.channels[remapped_channel.id] = remapped_channel
                            self.channel_topics[remapped_channel.id] = remapped_channel.topic
                            self.known_channels.add(remapped_channel.id)
                            # Pre-compute filtering decision
                            self.channel_filter_cache[remapped_channel.id] = (
                                self._compute_channel_filter_decision(remapped_channel.topic)
                            )

                # Seek back to start for processing
                input_stream.seek(0)

            total_size = sum(file_sizes)
            with file_progress("[bold blue]Processing MCAP...", console) as progress:
                task = progress.add_task("Processing", total=total_size)

                # Create chunk generators for each stream
                chunk_generators = [
                    self._generate_chunks_from_stream(input_stream, stream_id, writer)
                    for stream_id, input_stream in enumerate(input_streams)
                ]

                # Process chunks in timestamp order using heapq.merge
                # This maintains global ordering while allowing fast-copy optimization
                for pending_chunk in heapq.merge(*chunk_generators, key=lambda x: x.timestamp):
                    self._process_chunk_smart(
                        pending_chunk.chunk,
                        pending_chunk.indexes,
                        writer,
                        pending_chunk.stream_id,
                    )

                # Complete progress
                progress.update(task, completed=total_size)

        finally:
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

            # Check if this remapped ID is in known_channels
            if remapped_id not in self.known_channels:
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
                should_include = self.channel_filter_cache.get(idx.channel_id)
                if should_include is None:
                    # Unknown channel - must decode to discover it
                    return (False, True)
                if should_include:
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
                # Extract schemas/channels metadata without writing messages
                self._process_chunk_fallback(chunk, None, stream_id)
                # Re-build the remapped channel IDs after extraction
                remapped_channel_ids, _ = self._get_remapped_channel_ids(indexes, stream_id)

            # Ensure all channels (and their schemas) are written before fast-copying chunk
            for channel_id in remapped_channel_ids:
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

                    if remapped_channel.id not in self.channels:
                        self.channels[remapped_channel.id] = remapped_channel
                        self.channel_topics[remapped_channel.id] = remapped_channel.topic
                        self.known_channels.add(remapped_channel.id)
                        # Pre-compute filtering decision for this channel
                        self.channel_filter_cache[remapped_channel.id] = (
                            self._compute_channel_filter_decision(remapped_channel.topic)
                        )
                elif isinstance(chunk_record, Message):
                    if not writer:
                        continue
                    # Remap message channel ID if needed
                    original_channel_id = chunk_record.channel_id
                    message_to_write = chunk_record
                    if self.remapper.was_channel_remapped(stream_id, original_channel_id):
                        # Look up the remapped channel
                        original_channel = Channel(
                            id=original_channel_id,
                            schema_id=0,
                            topic="",
                            message_encoding="",
                            metadata={},
                        )
                        remapped_channel = self.remapper.remap_channel(stream_id, original_channel)
                        message_to_write = replace(
                            chunk_record,
                            channel_id=remapped_channel.id,
                        )
                    self.process_message(message_to_write, writer)

        except McapError as e:
            if self.options.recovery_mode:
                console.print(
                    f"[yellow]Warning (stream {stream_id}): Failed to decode chunk: {e}[/yellow]"
                )
                self.stats.errors_encountered += 1
            else:
                raise
