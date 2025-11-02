"""Rechunk command - reorganize MCAP messages into chunks based on topic patterns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from re import Pattern

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
    breakup_chunk,
    stream_reader,
)
from small_mcap.writer import _ChunkBuilder

from pymcap_cli.mcap_processor import compile_topic_patterns, str_to_compression_type
from pymcap_cli.utils import file_progress

console = Console()


class MessageGroup:
    """Manages a group of messages that will be chunked together independently."""

    def __init__(
        self, writer: McapWriter, chunk_size: int, compression_type: CompressionType
    ) -> None:
        self.writer = writer
        self.chunk_size = chunk_size
        self.message_count = 0
        # Each group has its own chunk builder for independent chunking
        self.chunk_builder = _ChunkBuilder(
            chunk_size=chunk_size,
            compression=compression_type,
            enable_crcs=writer.enable_crcs,
        )

    def add_message(self, message: Message) -> None:
        """Add message to this group's chunk builder."""
        # Add message to this group's chunk builder
        result = self.chunk_builder.add(message)

        # If chunk builder returns a completed chunk, write it immediately
        if result is not None:
            chunk, message_indexes = result
            self.writer.add_chunk_with_indexes(chunk, list(message_indexes.values()))

        self.message_count += 1

    def flush(self) -> None:
        """Flush any remaining messages in this group's chunk builder."""
        result = self.chunk_builder.finalize()
        if result is not None:
            chunk, message_indexes = result
            self.writer.add_chunk_with_indexes(chunk, list(message_indexes.values()))


class RechunkProcessor:
    """Process MCAP file and rechunk messages based on topic patterns."""

    def __init__(
        self,
        patterns: list[Pattern[str]],
        chunk_size: int,
        compression: str,
    ) -> None:
        self.patterns = patterns
        self.chunk_size = chunk_size
        self.compression = compression

        # Track schemas and channels
        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}
        self.written_schemas: set[int] = set()
        self.written_channels: set[int] = set()

        # Message groups (one per pattern + one for unmatched)
        self.groups: list[MessageGroup] = []

        # Stats
        self.messages_processed = 0
        self.messages_written = 0

    def ensure_schema_written(self, schema_id: int, writer: McapWriter) -> None:
        """Ensure schema is written to output."""
        if schema_id == 0 or schema_id in self.written_schemas:
            return

        schema = self.schemas.get(schema_id)
        if not schema:
            return

        writer.add_schema(
            name=schema.name,
            encoding=schema.encoding,
            data=schema.data,
            schema_id=schema.id,
        )
        self.written_schemas.add(schema_id)

    def ensure_channel_written(self, channel_id: int, writer: McapWriter) -> None:
        """Ensure channel is written to output."""
        if channel_id in self.written_channels:
            return

        channel = self.channels.get(channel_id)
        if not channel:
            return

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

    def find_matching_pattern_index(self, topic: str) -> int | None:
        """Find first pattern that matches topic. Returns pattern index or None."""
        for i, pattern in enumerate(self.patterns):
            if pattern.search(topic):
                return i
        return None

    def process(
        self,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        file_size: int,
    ) -> None:
        """Main processing function - streaming approach."""
        # Initialize writer
        compression_type = str_to_compression_type(self.compression)
        writer = McapWriter(
            output_stream,
            chunk_size=self.chunk_size,
            compression=compression_type,
        )

        try:
            writer_started = False

            with file_progress("[bold blue]Rechunking MCAP...", console) as progress:
                task = progress.add_task("Processing", total=file_size)

                records = stream_reader(input_stream, emit_chunks=True)

                try:
                    for record in records:
                        # Update progress
                        progress.update(task, completed=input_stream.tell())

                        if isinstance(record, Header):
                            # Start writer with header info
                            writer.start(profile=record.profile, library=record.library)
                            writer_started = True
                            # Initialize message groups (N patterns + 1 unmatched)
                            num_groups = len(self.patterns) + 1
                            self.groups = [
                                MessageGroup(writer, self.chunk_size, compression_type)
                                for _ in range(num_groups)
                            ]

                        elif isinstance(record, Chunk):
                            # Ensure writer is started
                            if not writer_started:
                                writer.start()
                                writer_started = True
                                num_groups = len(self.patterns) + 1
                                self.groups = [
                                    MessageGroup(writer, self.chunk_size, compression_type)
                                    for _ in range(num_groups)
                                ]

                            # Decode chunk and process records
                            try:
                                chunk_records = breakup_chunk(record, validate_crc=True)
                                for chunk_record in chunk_records:
                                    if isinstance(chunk_record, Schema):
                                        self.schemas[chunk_record.id] = chunk_record
                                    elif isinstance(chunk_record, Channel):
                                        self.channels[chunk_record.id] = chunk_record
                                    elif isinstance(chunk_record, Message):
                                        self._process_message(chunk_record, writer)
                            except McapError as e:
                                console.print(
                                    f"[yellow]Warning: Failed to decode chunk: {e}[/yellow]"
                                )

                        elif isinstance(record, MessageIndex):
                            pass  # Skip message indexes

                        elif isinstance(record, Schema):
                            self.schemas[record.id] = record

                        elif isinstance(record, Channel):
                            self.channels[record.id] = record

                        elif isinstance(record, Message):
                            # Ensure writer is started
                            if not writer_started:
                                writer.start()
                                writer_started = True
                                num_groups = len(self.patterns) + 1
                                self.groups = [
                                    MessageGroup(writer, self.chunk_size, compression_type)
                                    for _ in range(num_groups)
                                ]
                            self._process_message(record, writer)

                        elif isinstance(record, Attachment):
                            if not writer_started:
                                writer.start()
                                writer_started = True
                            writer.add_attachment(
                                log_time=record.log_time,
                                create_time=record.create_time,
                                name=record.name,
                                media_type=record.media_type,
                                data=record.data,
                            )

                        elif isinstance(record, Metadata):
                            if not writer_started:
                                writer.start()
                                writer_started = True
                            writer.add_metadata(name=record.name, metadata=record.metadata)

                        elif isinstance(record, (DataEnd, Footer)):
                            break

                    # Flush all groups at the end
                    for group in self.groups:
                        group.flush()

                    # Complete progress
                    progress.update(task, completed=file_size)

                except McapError as e:
                    console.print(f"[yellow]Warning: Error reading file: {e}[/yellow]")

        finally:
            writer.finish()

    def _process_message(self, message: Message, writer: McapWriter) -> None:
        """Process a single message - route to appropriate group."""
        self.messages_processed += 1

        # Get topic from channel
        channel = self.channels.get(message.channel_id)
        if not channel:
            # Channel not yet seen - skip message
            return

        # Find which pattern matches (first match wins)
        pattern_idx = self.find_matching_pattern_index(channel.topic)

        # Determine group index
        # pattern_idx None → last group (unmatched)
        # pattern_idx 0..N-1 → corresponding group
        group_index = pattern_idx if pattern_idx is not None else len(self.patterns)

        # Ensure channel is written
        self.ensure_channel_written(message.channel_id, writer)

        # Add message to group (group will auto-write chunks when full)
        self.groups[group_index].add_message(message)
        self.messages_written += 1


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the rechunk command parser to the subparsers."""
    parser = subparsers.add_parser(
        "rechunk",
        help="Reorganize MCAP messages into chunks based on topic patterns",
        description=(
            "Rechunk MCAP files by organizing messages into separate chunk groups "
            "based on topic regex patterns. Each pattern creates a separate chunk group, "
            "with unmatched topics going into their own group. Messages are written "
            "in a streaming fashion as they are read."
            "\n\nusage:\n  pymcap-cli rechunk in.mcap -o out.mcap -p '/camera.*' -p '/lidar.*'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file",
        help="Path to the MCAP file to rechunk",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (required)",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-p",
        "--pattern",
        action="append",
        default=[],
        help="Regex pattern for topic grouping (can be used multiple times). "
        "Topics matching the first pattern go into chunk group 1, "
        "second pattern → group 2, etc. Unmatched topics → separate group.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4 * 1024 * 1024,  # 4MB
        help="Chunk size in bytes (default: 4MB)",
    )

    parser.add_argument(
        "--compression",
        choices=["zstd", "lz4", "none"],
        default="zstd",
        help="Compression algorithm for output file (default: zstd)",
    )

    return parser


def handle_command(args: argparse.Namespace) -> None:
    """Handle the rechunk command execution."""
    input_file = Path(args.file)
    if not input_file.exists():
        console.print(f"[red]Error: Input file '{input_file}' does not exist[/red]")
        sys.exit(1)

    output_file = Path(args.output)
    file_size = input_file.stat().st_size

    # Compile patterns
    try:
        patterns = compile_topic_patterns(args.pattern)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if not patterns:
        console.print(
            "[yellow]Warning: No patterns specified. All messages will be in one group.[/yellow]"
        )

    # Create processor and run
    processor = RechunkProcessor(
        patterns=patterns,
        chunk_size=args.chunk_size,
        compression=args.compression,
    )

    with input_file.open("rb") as input_stream, output_file.open("wb") as output_stream:
        processor.process(input_stream, output_stream, file_size)

        # Report results
        console.print("[green]✓ Rechunking completed successfully![/green]")
        console.print(
            f"Processed {processor.messages_processed:,} messages, "
            f"wrote {processor.messages_written:,} messages"
        )

        if patterns:
            console.print(f"Used {len(patterns)} topic pattern(s) for grouping")
