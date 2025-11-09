"""Filter command for pymcap-cli."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pymcap_cli.autocompletion import complete_all_topics
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


@app.command(name="filter")
def filter_cmd(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Path to the MCAP file to filter",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            ...,
            "-o",
            "--output",
            help="Output filename",
            rich_help_panel="Output Options",
        ),
    ],
    include_topic_regex: Annotated[
        list[str] | None,
        typer.Option(
            "-y",
            "--include-topic-regex",
            help=(
                "Include messages with topic names matching this regex (can be used multiple times)"
            ),
            autocompletion=complete_all_topics,
            rich_help_panel="Topic Filtering",
            show_default="none",
        ),
    ] = None,
    exclude_topic_regex: Annotated[
        list[str] | None,
        typer.Option(
            "-n",
            "--exclude-topic-regex",
            help=(
                "Exclude messages with topic names matching this regex (can be used multiple times)"
            ),
            autocompletion=complete_all_topics,
            rich_help_panel="Topic Filtering",
            show_default="none",
        ),
    ] = None,
    start: Annotated[
        str,
        typer.Option(
            "-S",
            "--start",
            help="Include messages at or after this time (nanoseconds or RFC3339 date)",
            rich_help_panel="Time Filtering",
            show_default="beginning of recording",
        ),
    ] = "",
    start_secs: Annotated[
        int,
        typer.Option(
            "-s",
            "--start-secs",
            help="Include messages at or after this time in seconds (ignored if --start used)",
            rich_help_panel="Time Filtering",
            show_default=True,
        ),
    ] = 0,
    start_nsecs: Annotated[
        int,
        typer.Option(
            "--start-nsecs",
            help="(Deprecated, use --start) Include messages at or after this time in nanoseconds",
            hidden=True,
        ),
    ] = 0,
    end: Annotated[
        str,
        typer.Option(
            "-E",
            "--end",
            help="Include messages before this time (nanoseconds or RFC3339 date)",
            rich_help_panel="Time Filtering",
            show_default="end of recording",
        ),
    ] = "",
    end_secs: Annotated[
        int,
        typer.Option(
            "-e",
            "--end-secs",
            help="Include messages before this time in seconds (ignored if --end used)",
            rich_help_panel="Time Filtering",
            show_default=True,
        ),
    ] = 0,
    end_nsecs: Annotated[
        int,
        typer.Option(
            "--end-nsecs",
            help="(Deprecated, use --end) Include messages before this time in nanoseconds",
            hidden=True,
        ),
    ] = 0,
    include_metadata: Annotated[
        bool,
        typer.Option(
            "--include-metadata",
            help="Include metadata records in output",
            rich_help_panel="Content Filtering",
            show_default=True,
        ),
    ] = False,
    include_attachments: Annotated[
        bool,
        typer.Option(
            "--include-attachments",
            help="Include attachment records in output",
            rich_help_panel="Content Filtering",
            show_default=True,
        ),
    ] = False,
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            help="Chunk size of output file",
            rich_help_panel="Output Options",
            envvar="PYMCAP_CHUNK_SIZE",
            show_default="4MB",
        ),
    ] = 4 * 1024 * 1024,
    compression: Annotated[
        CompressionType,
        typer.Option(
            "--compression",
            help="Compression algorithm for output file",
            rich_help_panel="Output Options",
            envvar="PYMCAP_COMPRESSION",
            show_default=True,
        ),
    ] = CompressionType.ZSTD,
) -> None:
    """Copy filtered MCAP data to a new file.

    Filter an MCAP file by topic and time range to create a new file.
    When multiple regexes are used, topics that match any regex are
    included (or excluded).

    Usage:
      mcap filter in.mcap -o out.mcap -y /diagnostics -y /tf -y /camera_.*
    """
    include_topic_regex = include_topic_regex or []
    exclude_topic_regex = exclude_topic_regex or []

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

    processing_options = ProcessingOptions(
        # Enable recovery mode for robust filtering
        recovery_mode=True,
        always_decode_chunk=False,
        # Filter options
        include_topics=compile_topic_patterns(include_topic_regex),
        exclude_topics=compile_topic_patterns(exclude_topic_regex),
        start_time=start_time,
        end_time=end_time,
        include_metadata=include_metadata,
        include_attachments=include_attachments,
        # Output options
        compression=compression.value,
        chunk_size=chunk_size,
    )

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
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Path to the MCAP file to compress",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            ...,
            "-o",
            "--output",
            help="Output filename",
        ),
    ],
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            help="Chunk size of output file",
            envvar="PYMCAP_CHUNK_SIZE",
            show_default="4MB",
        ),
    ] = 4 * 1024 * 1024,
    compression: Annotated[
        CompressionType,
        typer.Option(
            "--compression",
            help="Compression algorithm for output file",
            envvar="PYMCAP_COMPRESSION",
            show_default=True,
        ),
    ] = CompressionType.ZSTD,
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
