"""Unified MCAP processor combining recovery and filtering capabilities."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from re import Pattern
from typing import BinaryIO

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
    Schema,
    Statistics,
    breakup_chunk,
    stream_reader,
)

from pymcap_cli.utils import file_progress

console = Console()


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
    """Unified MCAP processor combining recovery and filtering capabilities."""

    def __init__(self, options: ProcessingOptions) -> None:
        self.options = options
        self.stats = ProcessingStats()

        # Track schemas and channels (shared with writer for auto-ensure)
        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}

        # Channel topic cache for filtering
        self.channel_topics: dict[int, str] = {}

        # Cache filtering decisions per channel ID to avoid repeated regex matching
        self.channel_filter_cache: dict[int, bool] = {}

        # Track which channels we've already seen to optimize metadata extraction
        self.known_channels: set[int] = set()

    def process_message(self, message: Message, writer: McapWriter) -> None:
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
            if self.options.include_topics:
                should_include = any(p.search(topic) for p in self.options.include_topics)
            elif self.options.exclude_topics:
                should_include = not any(p.search(topic) for p in self.options.exclude_topics)
            else:
                should_include = True
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

    def process(
        self,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        file_size: int,
    ) -> ProcessingStats:
        """Main processing function combining recovery and filtering."""
        # Initialize writer and share schema/channel dicts for auto-ensure
        writer = McapWriter(
            output_stream,
            chunk_size=self.options.chunk_size,
            compression=self.options.compression_type,
        )
        writer.schemas = self.schemas
        writer.channels = self.channels
        writer_started = False

        try:
            with file_progress("[bold blue]Processing MCAP...", console) as progress:
                task = progress.add_task("Processing", total=file_size)
                # Always emit chunks so we can track statistics and handle them properly
                records = stream_reader(input_stream, emit_chunks=True)

                last_chunk: Chunk | None = None
                last_chunk_message_indexes: list[MessageIndex] = []

                try:
                    for record in records:
                        # Update progress
                        progress.update(task, completed=input_stream.tell())

                        # Ensure writer is started before processing any records
                        if not writer_started:
                            if isinstance(record, Header):
                                writer.start(profile=record.profile, library=record.library)
                            else:
                                writer.start()
                            writer_started = True

                        # Finalize pending chunk when we see a non-MessageIndex record
                        if not isinstance(record, MessageIndex) and last_chunk:
                            self._process_chunk_smart(
                                last_chunk, last_chunk_message_indexes, writer
                            )
                            last_chunk = None
                            last_chunk_message_indexes = []

                        # Process current record
                        if isinstance(record, Header):
                            pass  # Already processed above
                        elif isinstance(record, Chunk):
                            self.stats.chunks_processed += 1
                            last_chunk = record
                            last_chunk_message_indexes = []
                        elif isinstance(record, MessageIndex):
                            last_chunk_message_indexes.append(record)
                        elif isinstance(record, Schema):
                            self.schemas[record.id] = record
                        elif isinstance(record, Channel):
                            if record.id not in self.channels:
                                self.channels[record.id] = record
                                self.channel_topics[record.id] = record.topic
                                self.known_channels.add(record.id)
                                # Pre-compute filtering decision for this channel
                                if self.options.include_topics:
                                    self.channel_filter_cache[record.id] = any(
                                        p.search(record.topic) for p in self.options.include_topics
                                    )
                                elif self.options.exclude_topics:
                                    self.channel_filter_cache[record.id] = not any(
                                        p.search(record.topic) for p in self.options.exclude_topics
                                    )
                                else:
                                    self.channel_filter_cache[record.id] = True
                        elif isinstance(record, Message):
                            self.process_message(record, writer)
                        elif isinstance(record, Attachment):
                            self.stats.attachments_processed += 1
                            if self.options.include_attachments and (
                                self.options.start_time <= record.log_time < self.options.end_time
                            ):
                                writer.add_attachment(
                                    log_time=record.log_time,
                                    create_time=record.create_time,
                                    name=record.name,
                                    media_type=record.media_type,
                                    data=record.data,
                                )
                        elif isinstance(record, Metadata):
                            self.stats.metadata_processed += 1
                            if self.options.include_metadata:
                                writer.add_metadata(name=record.name, metadata=record.metadata)
                        elif isinstance(record, (DataEnd, Footer)):
                            break

                except McapError as e:
                    # Catch errors from iteration itself (e.g., truncated file)
                    if self.options.recovery_mode:
                        console.print(f"[yellow]Warning: {e}[/yellow]")
                        self.stats.errors_encountered += 1
                        # If we have a pending chunk, decode it to recover what we can
                        if last_chunk:
                            self._process_chunk_fallback(last_chunk, writer)
                            last_chunk = None
                    else:
                        raise

                # Process final pending chunk
                if last_chunk:
                    self._process_chunk_smart(last_chunk, last_chunk_message_indexes, writer)

                # Complete progress
                if task and file_size:
                    progress.update(task, completed=file_size)

        finally:
            writer.finish()

        # Save writer statistics (single source of truth for output counts)
        self.stats.writer_statistics = writer.statistics

        return self.stats

    def _should_decode_chunk(self, chunk: Chunk, indexes: list[MessageIndex]) -> tuple[bool, bool]:
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
        self, chunk: Chunk, indexes: list[MessageIndex], writer: McapWriter
    ) -> None:
        """Smart chunk processing with fast copying when possible."""
        should_skip, should_decode = self._should_decode_chunk(chunk, indexes)

        if should_skip:
            return

        if should_decode:
            self._process_chunk_fallback(chunk, writer)
        else:
            # Fast-copy path: extract schemas and channels first (only if needed)
            all_channels_known = all(idx.channel_id in self.known_channels for idx in indexes)
            if not all_channels_known:
                # Extract schemas/channels metadata without writing messages
                self._process_chunk_fallback(chunk, None)

            # Ensure all channels (and their schemas) are written before fast-copying chunk
            for idx in indexes:
                writer.ensure_channel_written(idx.channel_id)

            # Fast-copy the chunk with its indexes
            writer.add_chunk_with_indexes(chunk, indexes)
            self.stats.chunks_copied += 1

    def _process_chunk_fallback(self, chunk: Chunk, writer: McapWriter | None) -> None:
        """Fallback to decode chunk into individual records."""
        try:
            chunk_records = breakup_chunk(chunk, validate_crc=True)
            # Only count as decoded if we're actually writing messages
            if writer is not None:
                self.stats.chunks_decoded += 1

            for chunk_record in chunk_records:
                if isinstance(chunk_record, Schema):
                    self.schemas[chunk_record.id] = chunk_record
                elif isinstance(chunk_record, Channel):
                    if chunk_record.id not in self.channels:
                        self.channels[chunk_record.id] = chunk_record
                        self.channel_topics[chunk_record.id] = chunk_record.topic
                        self.known_channels.add(chunk_record.id)
                        # Pre-compute filtering decision for this channel
                        if self.options.include_topics:
                            self.channel_filter_cache[chunk_record.id] = any(
                                p.search(chunk_record.topic) for p in self.options.include_topics
                            )
                        elif self.options.exclude_topics:
                            self.channel_filter_cache[chunk_record.id] = not any(
                                p.search(chunk_record.topic) for p in self.options.exclude_topics
                            )
                        else:
                            self.channel_filter_cache[chunk_record.id] = True
                elif isinstance(chunk_record, Message) and writer:
                    self.process_message(chunk_record, writer)

        except McapError as e:
            if self.options.recovery_mode:
                console.print(f"[yellow]Warning: Failed to decode chunk: {e}[/yellow]")
                self.stats.errors_encountered += 1
            else:
                raise
