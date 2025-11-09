"""Filter command for pymcap-cli."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from pymcap_cli.mcap_processor import (
    McapProcessor,
    ProcessingOptions,
    compile_topic_patterns,
    parse_time_arg,
)

console = Console()
app = typer.Typer()


class CompressionType(str, Enum):
    """Compression algorithm types."""

    ZSTD = "zstd"
    LZ4 = "lz4"
    NONE = "none"


def parse_timestamp_args(date_or_nanos: str, nanoseconds: int, seconds: int) -> int:
    """Parse timestamp with precedence: date_or_nanos > nanoseconds > seconds."""
    if date_or_nanos:
        return parse_time_arg(date_or_nanos)
    if nanoseconds != 0:
        return nanoseconds
    return seconds * 1_000_000_000


def build_processing_options(
    include_topic_regex: list[str],
    exclude_topic_regex: list[str],
    start: str,
    start_nsecs: int,
    start_secs: int,
    end: str,
    end_nsecs: int,
    end_secs: int,
    include_metadata: bool,
    include_attachments: bool,
    chunk_size: int,
    compression: CompressionType,
) -> ProcessingOptions:
    """Build ProcessingOptions from filter command line arguments."""
    # Validate mutually exclusive topic filters
    if include_topic_regex and exclude_topic_regex:
        raise ValueError("Cannot use both --include-topic-regex and --exclude-topic-regex")

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
        # Enable recovery mode for robust filtering
        recovery_mode=True,
        always_decode_chunk=False,
        # Filter options
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        start_time=start_time,
        end_time=end_time,
        include_metadata=include_metadata,
        include_attachments=include_attachments,
        # Output options
        compression=compression.value,
        chunk_size=chunk_size,
    )


@app.command(name="filter")
def filter_cmd(
    file: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to the MCAP file to filter"
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Output filename (required)"),
    include_topic_regex: list[str] = typer.Option(
        [],
        "-y",
        "--include-topic-regex",
        help="Include messages with topic names matching this regex (can be used multiple times)",
    ),
    exclude_topic_regex: list[str] = typer.Option(
        [],
        "-n",
        "--exclude-topic-regex",
        help="Exclude messages with topic names matching this regex (can be used multiple times)",
    ),
    start: str = typer.Option(
        "",
        "-S",
        "--start",
        help="Include messages at or after this time (nanoseconds or RFC3339 date)",
    ),
    start_secs: int = typer.Option(
        0,
        "-s",
        "--start-secs",
        help="Include messages at or after this time in seconds (ignored if --start used)",
    ),
    start_nsecs: int = typer.Option(
        0,
        "--start-nsecs",
        help="(Deprecated, use --start) Include messages at or after this time in nanoseconds",
    ),
    end: str = typer.Option(
        "",
        "-E",
        "--end",
        help="Include messages before this time (nanoseconds or RFC3339 date)",
    ),
    end_secs: int = typer.Option(
        0,
        "-e",
        "--end-secs",
        help="Include messages before this time in seconds (ignored if --end used)",
    ),
    end_nsecs: int = typer.Option(
        0,
        "--end-nsecs",
        help="(Deprecated, use --end) Include messages before this time in nanoseconds",
    ),
    include_metadata: bool = typer.Option(
        False,
        "--include-metadata",
        help="Include metadata records in output",
    ),
    include_attachments: bool = typer.Option(
        False,
        "--include-attachments",
        help="Include attachment records in output",
    ),
    chunk_size: int = typer.Option(
        4 * 1024 * 1024,
        "--chunk-size",
        help="Chunk size of output file (default: 4MB)",
    ),
    compression: CompressionType = typer.Option(
        CompressionType.ZSTD,
        "--compression",
        help="Compression algorithm for output file (default: zstd)",
    ),
) -> None:
    """Copy filtered MCAP data to a new file.

    Filter an MCAP file by topic and time range to create a new file.
    When multiple regexes are used, topics that match any regex are
    included (or excluded).

    Usage:
      mcap filter in.mcap -o out.mcap -y /diagnostics -y /tf -y /camera_.*
    """
    try:
        processing_options = build_processing_options(
            include_topic_regex=include_topic_regex,
            exclude_topic_regex=exclude_topic_regex,
            start=start,
            start_nsecs=start_nsecs,
            start_secs=start_secs,
            end=end,
            end_nsecs=end_nsecs,
            end_secs=end_secs,
            include_metadata=include_metadata,
            include_attachments=include_attachments,
            chunk_size=chunk_size,
            compression=compression,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not file.exists():
        console.print(f"[red]Error: Input file '{file}' does not exist[/red]")
        raise typer.Exit(1)

    file_size = file.stat().st_size

    # Create processor and run
    processor = McapProcessor(processing_options)

    with file.open("rb") as input_stream, output.open("wb") as output_stream:
        try:
            stats = processor.process([input_stream], output_stream, [file_size])

            # Report results in filter-style format
            console.print("[green]✓ Filter completed successfully![/green]")
            console.print(
                f"Processed {stats.messages_processed:,} messages, "
                f"wrote {stats.writer_statistics.message_count:,} messages"
            )
            if stats.writer_statistics.attachment_count > 0:
                console.print(f"Wrote {stats.writer_statistics.attachment_count} attachments")
            if stats.writer_statistics.metadata_count > 0:
                console.print(f"Wrote {stats.writer_statistics.metadata_count} metadata records")
            console.print(
                f"Wrote {stats.writer_statistics.schema_count} schemas and "
                f"{stats.writer_statistics.channel_count} channels"
            )
            if stats.chunks_processed > 0:
                console.print(
                    f"Processed {stats.chunks_processed} chunks "
                    f"({stats.chunks_copied} fast copied, {stats.chunks_decoded} decoded)"
                )

        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during filtering: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def compress(
    file: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to the MCAP file to compress"
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Output filename (required)"),
    chunk_size: int = typer.Option(
        4 * 1024 * 1024,
        "--chunk-size",
        help="Chunk size of output file (default: 4MB)",
    ),
    compression: CompressionType = typer.Option(
        CompressionType.ZSTD,
        "--compression",
        help="Compression algorithm for output file (default: zstd)",
    ),
) -> None:
    """Create a compressed copy of an MCAP file.

    Copy data in an MCAP file to a new file, compressing the output.

    Usage:
      mcap compress in.mcap -o out.mcap
    """
    # Convert compress args to unified processing options (include everything)
    processing_options = ProcessingOptions(
        # Recovery mode with all content included
        recovery_mode=True,
        always_decode_chunk=False,
        # No filtering - include everything
        include_topics=[],
        exclude_topics=[],
        start_time=0,
        end_time=2**63 - 1,
        include_metadata=True,
        include_attachments=True,
        # Output options with specified compression
        compression=compression.value,
        chunk_size=chunk_size,
    )

    if not file.exists():
        console.print(f"[red]Error: Input file '{file}' does not exist[/red]")
        raise typer.Exit(1)

    file_size = file.stat().st_size

    console.print(f"[blue]Compressing '{file}' to '{output}'[/blue]")

    processor = McapProcessor(processing_options)

    with file.open("rb") as input_stream, output.open("wb") as output_stream:
        try:
            stats = processor.process([input_stream], output_stream, [file_size])

            console.print("[green]✓ Compression completed successfully![/green]")
            console.print(
                f"Processed {stats.messages_processed:,} messages, "
                f"wrote {stats.writer_statistics.message_count:,} messages"
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during compression: {e}[/red]")
            raise typer.Exit(1)
