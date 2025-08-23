"""Unified MCAP processor combining recovery and filtering capabilities."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from re import Pattern
from typing import BinaryIO

from mcap.exceptions import McapError
from mcap.records import (
    Attachment,
    Channel,
    Chunk,
    DataEnd,
    Footer,
    Header,
    Message,
    MessageIndex,
    Metadata,
    Schema,
)
from mcap.stream_reader import breakup_chunk
from rich.console import Console

from pymcap_cli.mcap.reader import read_message
from pymcap_cli.utils import file_progress

from .mcap.writer import CompressionType, McapWriter

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


@dataclass
class ProcessingStats:
    """Statistics for unified processing operation."""

    messages_processed: int = 0
    messages_written: int = 0
    attachments_processed: int = 0
    attachments_written: int = 0
    metadata_processed: int = 0
    metadata_written: int = 0
    chunks_processed: int = 0
    chunks_copied: int = 0  # Fast copied chunks
    chunks_decoded: int = 0  # Decoded chunks
    schemas_written: int = 0
    channels_written: int = 0
    errors_encountered: int = 0
    validation_errors: int = 0
    filter_rejections: int = 0


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
        except re.error as e:  # noqa: PERF203
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    return compiled


def topic_matches_patterns(topic: str, patterns: list[Pattern[str]]) -> bool:
    """Check if topic matches any of the given patterns."""
    if not patterns:
        return False
    return any(pattern.search(topic) for pattern in patterns)


class McapProcessor:
    """Unified MCAP processor combining recovery and filtering capabilities."""

    def __init__(self, options: ProcessingOptions) -> None:
        self.options = options
        self.stats = ProcessingStats()

        # Track schemas and channels
        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}
        self.written_schemas: set[int] = set()
        self.written_channels: set[int] = set()

        # Channel topic cache for filtering
        self.channel_topics: dict[int, str] = {}

    def should_include_topic(self, topic: str) -> bool:
        """Check if topic should be included based on filter criteria."""
        if self.options.include_topics:
            return topic_matches_patterns(topic, self.options.include_topics)
        if self.options.exclude_topics:
            return not topic_matches_patterns(topic, self.options.exclude_topics)
        # No topic filtering
        return True

    def should_include_time(self, log_time: int) -> bool:
        """Check if time should be included based on filter criteria."""
        return self.options.start_time <= log_time < self.options.end_time

    def process_schema(self, schema: Schema) -> None:
        """Process a schema record. Returns True if written."""
        self.schemas[schema.id] = schema

    def process_channel(self, channel: Channel) -> None:
        """Process a channel record. Returns True if written."""
        if channel.id in self.channels:
            return
        self.channels[channel.id] = channel
        self.channel_topics[channel.id] = channel.topic

    def ensure_schema_written(self, schema_id: int, writer: McapWriter) -> bool:
        """Ensure schema is written to output. Returns True if written."""
        if schema_id == 0 or schema_id in self.written_schemas:
            return False

        schema = self.schemas.get(schema_id)
        if not schema:
            return False

        writer.add_schema(
            name=schema.name,
            encoding=schema.encoding,
            data=schema.data,
            schema_id=schema.id,
        )
        self.written_schemas.add(schema_id)
        self.stats.schemas_written += 1
        return True

    def ensure_channel_written(self, channel_id: int, writer: McapWriter) -> bool:
        """Ensure channel is written to output. Returns True if written."""
        if channel_id in self.written_channels:
            return False

        channel = self.channels.get(channel_id)
        if not channel:
            return False

        # Ensure schema is written first
        self.ensure_schema_written(channel.schema_id, writer)

        writer.add_channel(
            topic=channel.topic,
            message_encoding=channel.message_encoding,
            schema_id=channel.schema_id,
            metadata=channel.metadata or {},
            channel_id=channel.id,
        )
        self.written_channels.add(channel_id)
        self.stats.channels_written += 1
        return True

    def process_message(self, message: Message, writer: McapWriter) -> None:
        """Process a message record. Returns True if written."""
        self.stats.messages_processed += 1

        if not self.should_include_time(message.log_time):
            return

        # 4. Topic filtering (for messages)
        topic = self.channel_topics.get(message.channel_id)
        if topic and not self.should_include_topic(topic):
            self.stats.filter_rejections += 1
            return

        # Ensure channel is written
        self.ensure_channel_written(message.channel_id, writer)

        writer.add_message(
            channel_id=message.channel_id,
            log_time=message.log_time,
            data=message.data,
            publish_time=message.publish_time,
        )
        self.stats.messages_written += 1

    def process_attachment(self, attachment: Attachment, writer: McapWriter) -> None:
        """Process an attachment record."""
        self.stats.attachments_processed += 1

        if not self.options.include_attachments:
            return

        if not self.should_include_time(attachment.log_time):
            return

        writer.add_attachment(
            log_time=attachment.log_time,
            create_time=attachment.create_time,
            name=attachment.name,
            media_type=attachment.media_type,
            data=attachment.data,
        )
        self.stats.attachments_written += 1

    def process_metadata(self, metadata: Metadata, writer: McapWriter) -> bool:
        """Process a metadata record. Returns True if written."""
        self.stats.metadata_processed += 1

        if not self.options.include_metadata:
            return False

        writer.add_metadata(name=metadata.name, data=metadata.metadata)
        self.stats.metadata_written += 1
        return True

    def process(
        self,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        file_size: int,
    ) -> ProcessingStats:
        """Main processing function combining recovery and filtering."""
        # Initialize writer
        writer = McapWriter(
            output_stream,
            chunk_size=self.options.chunk_size,
            compression=self.options.compression_type,
        )
        writer_started = False

        try:
            with file_progress("[bold blue]Processing MCAP...", console) as progress:
                task = progress.add_task("Processing", total=file_size)
                # Choose processing mode based on options
                emit_chunks = not self.options.always_decode_chunk
                stream_reader = read_message(
                    input_stream,
                    should_include=lambda c: self.should_include_topic(c.topic),
                    start_time=self.options.start_time,
                    end_time=self.options.end_time,
                    emit_chunks=emit_chunks,
                )

                last_chunk: Chunk | None = None
                last_chunk_message_indexes: list[MessageIndex] = []

                for record in stream_reader:
                    try:
                        # Update progress
                        progress.update(task, completed=input_stream.tell())

                        # Ensure writer is started before processing any records
                        if not writer_started:
                            if isinstance(record, Header):
                                writer.start(profile=record.profile, library=record.library)
                            else:
                                # Start with default values if no header found
                                writer.start()
                            writer_started = True

                        if not isinstance(record, MessageIndex) and last_chunk:
                            self._process_chunk_smart(
                                last_chunk, last_chunk_message_indexes, writer
                            )
                            last_chunk = None
                            last_chunk_message_indexes = []

                        if isinstance(record, Header):
                            # Header already processed above for writer start
                            pass

                        elif isinstance(record, Chunk):
                            self.stats.chunks_processed += 1
                            last_chunk = record
                            last_chunk_message_indexes = []
                        elif isinstance(record, MessageIndex):
                            last_chunk_message_indexes.append(record)
                        elif isinstance(record, Schema):
                            self.process_schema(record)
                        elif isinstance(record, Channel):
                            self.process_channel(record)
                        elif isinstance(record, Message):
                            self.process_message(record, writer)
                        elif isinstance(record, Attachment):
                            self.process_attachment(record, writer)
                        elif isinstance(record, Metadata):
                            self.process_metadata(record, writer)
                        elif isinstance(record, (DataEnd, Footer)):
                            break

                    except McapError as e:
                        if self.options.recovery_mode:
                            console.print(f"[yellow]Warning: Skipping invalid record: {e}[/yellow]")
                            self.stats.errors_encountered += 1
                            continue
                        raise

                if last_chunk:
                    # always decompress the last chunk
                    self._process_chunk_fallback(last_chunk, writer)

                # Complete progress
                if task and file_size:
                    progress.update(task, completed=file_size)

        finally:
            writer.finish()

        return self.stats

    def _can_copy_chunk_fast(self, chunk: Chunk, indexes: list[MessageIndex]) -> bool:
        """Check if entire chunk can be copied without decoding."""
        # Fast copy is possible if:
        # 1. We have chunk copying enabled
        # 2. We're in recovery mode (handle any errors)
        # 3. No topic filtering (would need to decode to check topics)
        # 4. Time filtering allows the entire chunk time range

        if self.options.include_topics or self.options.exclude_topics:
            for idx in indexes:
                channel = self.channels.get(idx.channel_id)
                if channel is None:
                    # contains unknown channel need decoding
                    return False
                if self.should_include_topic(channel.topic):
                    return False

        # Check if chunk time range intersects with our filter range
        chunk_start = chunk.message_start_time
        chunk_end = chunk.message_end_time

        # If chunk is entirely within our time range, we can copy it
        return chunk_start >= self.options.start_time and chunk_end < self.options.end_time

    def _process_chunk_smart(
        self, chunk: Chunk, indexes: list[MessageIndex], writer: McapWriter
    ) -> None:
        """Smart chunk processing with fast copying when possible."""
        decode = False

        # chunk time outside of limits
        if (
            chunk.message_end_time < self.options.start_time
            or chunk.message_start_time >= self.options.end_time
        ):
            return

        if self.options.include_topics:
            for idx in indexes:
                channel = self.channels.get(idx.channel_id)
                if channel is None:
                    # contains unknown channel need decoding
                    decode = True
                    break
                if not self.should_include_topic(channel.topic):
                    # Topic should be included
                    decode = True

        if decode or self.options.always_decode_chunk:
            self._process_chunk_fallback(chunk, writer)
        else:
            for idx in indexes:
                self.ensure_channel_written(idx.channel_id, writer)
            writer.add_chunk_with_indexes(chunk, indexes)
            self.stats.messages_written += sum(len(idx.records) for idx in indexes)
            self.stats.chunks_copied += 1
            return

    def _process_chunk_fallback(self, chunk: Chunk, writer: McapWriter) -> None:
        """Fallback to decode chunk into individual records."""
        try:
            chunk_records = breakup_chunk(chunk, validate_crc=True)
            self.stats.chunks_decoded += 1

            for chunk_record in chunk_records:
                if isinstance(chunk_record, Schema):
                    self.process_schema(chunk_record)
                elif isinstance(chunk_record, Channel):
                    self.process_channel(chunk_record)
                elif isinstance(chunk_record, Message):
                    self.process_message(chunk_record, writer)

        except McapError as e:
            if self.options.recovery_mode:
                console.print(f"[yellow]Warning: Failed to decode chunk: {e}[/yellow]")
                self.stats.errors_encountered += 1
            else:
                raise
