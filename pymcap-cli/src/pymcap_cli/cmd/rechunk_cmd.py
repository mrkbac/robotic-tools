"""Rechunk command - reorganize MCAP messages into chunks based on topic patterns."""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from re import Pattern

import typer
from rich.console import Console
from small_mcap import (
    Attachment,
    Channel,
    CompressionType,
    DataEnd,
    Footer,
    Header,
    McapWriter,
    Message,
    Metadata,
    Schema,
    stream_reader,
)
from small_mcap.rebuild import rebuild_summary
from small_mcap.writer import _ChunkBuilder

from pymcap_cli.mcap_processor import compile_topic_patterns, str_to_compression_type
from pymcap_cli.utils import file_progress

console = Console()
app = typer.Typer()


class RechunkMode(Enum):
    """Mode for rechunking operation."""

    PATTERN = auto()  # Group by regex patterns
    ALL = auto()  # Each topic in its own chunk group
    AUTO = auto()  # Auto-group based on size (>15% threshold)


class CompressionChoice(str, Enum):
    """Compression algorithm choices."""

    ZSTD = "zstd"
    LZ4 = "lz4"
    NONE = "none"


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


class RechunkProcessor:
    """Process MCAP file and rechunk messages based on topic patterns."""

    def __init__(
        self,
        mode: RechunkMode,
        patterns: list[Pattern[str]],
        chunk_size: int,
        compression: str,
    ) -> None:
        self.mode = mode
        self.patterns = patterns
        self.chunk_size = chunk_size
        self.compression = compression

        # Track schemas and channels (shared with writer and chunk builders)
        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}

        # Map channel_id to its MessageGroup (populated dynamically)
        self.channel_to_group: dict[int, MessageGroup] = {}

        # For AUTO mode: track which channels are "large" (>15% of total)
        self.large_channels: set[int] = set()

        # Stats
        self.messages_processed = 0
        self.messages_written = 0

    def find_matching_pattern_index(self, topic: str) -> int | None:
        """Find first pattern that matches topic. Returns pattern index or None."""
        for i, pattern in enumerate(self.patterns):
            if pattern.search(topic):
                return i
        return None

    def analyze_for_auto_grouping(self, input_stream: BinaryIO) -> None:
        """Pre-analyze file to identify large channels (>15% of total uncompressed size)."""
        console.print("[dim]Analyzing file for auto-grouping...[/dim]")

        # Use rebuild_summary to estimate channel sizes
        rebuild_info = rebuild_summary(
            input_stream,
            validate_crc=False,
            calculate_channel_sizes=True,
            exact_sizes=False,  # Use fast estimation
        )

        if not rebuild_info.channel_sizes:
            console.print("[yellow]Warning: Could not determine channel sizes[/yellow]")
            return

        # Calculate 15% threshold
        total_size = sum(rebuild_info.channel_sizes.values())
        threshold = total_size * 0.15

        # Identify large channels
        self.large_channels = {
            ch_id for ch_id, size in rebuild_info.channel_sizes.items() if size > threshold
        }

        if self.large_channels:
            console.print(
                f"[dim]Found {len(self.large_channels)} large channel(s) "
                f"(>{threshold / 1024 / 1024:.1f}MB each)[/dim]"
            )

        # Reset stream position for main processing
        input_stream.seek(0)

    def process(
        self,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        file_size: int,
    ) -> None:
        """Main processing function - streaming approach."""
        # For AUTO mode, pre-analyze to identify large channels
        if self.mode == RechunkMode.AUTO:
            self.analyze_for_auto_grouping(input_stream)

        # Initialize writer and share schema/channel dicts for auto-ensure
        compression_type = str_to_compression_type(self.compression)
        writer = McapWriter(
            output_stream,
            chunk_size=self.chunk_size,
            compression=compression_type,
        )
        writer.schemas = self.schemas
        writer.channels = self.channels

        try:
            with file_progress("[bold blue]Rechunking MCAP...", console) as progress:
                task = progress.add_task("Processing", total=file_size)

                records = stream_reader(input_stream, emit_chunks=False)

                for record in records:
                    # Update progress
                    progress.update(task, completed=input_stream.tell())

                    if isinstance(record, Header):
                        # Start writer with header info
                        writer.start(profile=record.profile, library=record.library)

                    elif isinstance(record, Schema):
                        self.schemas[record.id] = record

                    elif isinstance(record, Channel):
                        self.channels[record.id] = record

                    elif isinstance(record, Message):
                        self._process_message(record, writer)

                    elif isinstance(record, Attachment):
                        writer.add_attachment(
                            log_time=record.log_time,
                            create_time=record.create_time,
                            name=record.name,
                            media_type=record.media_type,
                            data=record.data,
                        )

                    elif isinstance(record, Metadata):
                        writer.add_metadata(name=record.name, metadata=record.metadata)

                    elif isinstance(record, (DataEnd, Footer)):
                        break

                # Flush all unique groups at the end
                # Use set() to avoid flushing the same group multiple times
                for group in set(self.channel_to_group.values()):
                    group.flush()

                # Complete progress
                progress.update(task, completed=file_size)

        finally:
            writer.finish()

    def _process_message(self, message: Message, writer: McapWriter) -> None:
        """Process a single message - route to appropriate group based on mode."""
        self.messages_processed += 1

        # Get channel for this message
        channel = self.channels.get(message.channel_id)
        if not channel:
            # Channel not yet seen - skip message
            return

        # Get or create the MessageGroup for this channel
        if message.channel_id not in self.channel_to_group:
            group = self._create_group_for_channel(message.channel_id, channel, writer)
            self.channel_to_group[message.channel_id] = group

        # Ensure channel is written to main file (not in chunks)
        writer.ensure_channel_written(message.channel_id)

        # Add message to its group (chunk builder auto-ensures within chunks)
        self.channel_to_group[message.channel_id].add_message(message)
        self.messages_written += 1

    def _create_group_for_channel(
        self, channel_id: int, channel: Channel, writer: McapWriter
    ) -> MessageGroup:
        """Create appropriate MessageGroup for a channel based on mode."""
        compression_type = str_to_compression_type(self.compression)

        if self.mode == RechunkMode.ALL:
            # Each channel gets its own unique group
            return MessageGroup(
                writer, self.chunk_size, compression_type, self.schemas, self.channels
            )

        if self.mode == RechunkMode.AUTO:
            # Large channels get their own group, small channels share one
            if channel_id in self.large_channels:
                # Create unique group for large channel
                return MessageGroup(
                    writer, self.chunk_size, compression_type, self.schemas, self.channels
                )
            # Reuse shared group for small channels (if exists)
            # Look for any existing small-channel group
            for ch_id, group in self.channel_to_group.items():
                if ch_id not in self.large_channels:
                    return group
            # No shared group yet, create one
            return MessageGroup(
                writer, self.chunk_size, compression_type, self.schemas, self.channels
            )

        # RechunkMode.PATTERN
        # Find which pattern matches this channel's topic
        pattern_idx = self.find_matching_pattern_index(channel.topic)
        group_key = pattern_idx if pattern_idx is not None else -1  # -1 for unmatched

        # Check if we already have a group for this pattern
        for ch_id, group in self.channel_to_group.items():
            ch = self.channels.get(ch_id)
            if ch:
                other_pattern_idx = self.find_matching_pattern_index(ch.topic)
                other_key = other_pattern_idx if other_pattern_idx is not None else -1
                if other_key == group_key:
                    return group

        # No existing group for this pattern, create new one
        return MessageGroup(writer, self.chunk_size, compression_type, self.schemas, self.channels)


@app.command()
def rechunk(
    file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to the MCAP file to rechunk",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output filename (required)",
    ),
    pattern: list[str] = typer.Option(
        [],
        "-p",
        "--pattern",
        help="Regex pattern for topic grouping (can be used multiple times). "
        "Topics matching the first pattern go into chunk group 1, "
        "second pattern → group 2, etc. Unmatched topics → separate group.",
    ),
    all_topics: bool = typer.Option(
        False,
        "--all",
        help="If set, all messages are placed into their own chunk",
    ),
    auto: bool = typer.Option(
        False,
        "--auto",
        help="If set, automatically create chunk groups. "
        "Topics taking up more than 15% of the uncompressed total size get their own chunk.",
    ),
    chunk_size: int = typer.Option(
        4 * 1024 * 1024,
        "--chunk-size",
        help="Chunk size in bytes (default: 4MB)",
    ),
    compression: CompressionChoice = typer.Option(
        CompressionChoice.ZSTD,
        "--compression",
        help="Compression algorithm for output file (default: zstd)",
    ),
) -> None:
    """Reorganize MCAP messages into chunks based on topic patterns.

    Rechunk MCAP files by organizing messages into separate chunk groups
    based on topic regex patterns. Each pattern creates a separate chunk group,
    with unmatched topics going into their own group. Messages are written
    in a streaming fashion as they are read.

    Usage:
      pymcap-cli rechunk in.mcap -o out.mcap -p '/camera.*' -p '/lidar.*'
    """
    # Validate mutually exclusive options
    mode_count = sum([bool(pattern), all_topics, auto])
    if mode_count == 0:
        console.print("[red]Error: One of --pattern, --all, or --auto is required[/red]")
        raise typer.Exit(1)
    if mode_count > 1:
        console.print("[red]Error: --pattern, --all, and --auto are mutually exclusive[/red]")
        raise typer.Exit(1)

    input_file = Path(file)
    if not input_file.exists():
        console.print(f"[red]Error: Input file '{input_file}' does not exist[/red]")
        raise typer.Exit(1)

    output_file = Path(output)
    file_size = input_file.stat().st_size

    # Determine mode and compile patterns if needed
    mode: RechunkMode
    patterns: list[Pattern[str]] = []

    if all_topics:
        mode = RechunkMode.ALL
        console.print("[dim]Mode: Each topic in its own chunk group[/dim]")
    elif auto:
        mode = RechunkMode.AUTO
        console.print("[dim]Mode: Auto-grouping based on size (>15% threshold)[/dim]")
    else:
        mode = RechunkMode.PATTERN
        try:
            patterns = compile_topic_patterns(pattern)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        if not patterns:
            console.print(
                "[yellow]Warning: No patterns specified. "
                "All messages will be in one group.[/yellow]"
            )

    # Create processor and run
    processor = RechunkProcessor(
        mode=mode,
        patterns=patterns,
        chunk_size=chunk_size,
        compression=compression.value,
    )

    with input_file.open("rb") as input_stream, output_file.open("wb") as output_stream:
        processor.process(input_stream, output_stream, file_size)

        # Report results
        console.print("[green]✓ Rechunking completed successfully![/green]")
        console.print(
            f"Processed {processor.messages_processed:,} messages, "
            f"wrote {processor.messages_written:,} messages"
        )

        # Mode-specific stats
        num_unique_groups = len(set(processor.channel_to_group.values()))
        if mode == RechunkMode.ALL:
            console.print(f"Created {num_unique_groups} chunk group(s) (one per topic)")
        elif mode == RechunkMode.AUTO:
            console.print(
                f"Created {num_unique_groups} chunk group(s) "
                f"({len(processor.large_channels)} large, rest shared)"
            )
        elif mode == RechunkMode.PATTERN and patterns:
            console.print(f"Used {len(patterns)} topic pattern(s) for grouping")
