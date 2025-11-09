"""Unified process command combining recovery and filtering capabilities."""

from __future__ import annotations

import contextlib
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pymcap_cli.mcap_processor import (
    McapProcessor,
    ProcessingOptions,
    compile_topic_patterns,
    parse_time_arg,
)

app = typer.Typer()
console = Console()


class CompressionType(str, Enum):
    """Compression algorithm choices."""

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
    include_topic_regex: list[str] | None,
    exclude_topic_regex: list[str] | None,
    start: str,
    start_nsecs: int,
    start_secs: int,
    end: str,
    end_nsecs: int,
    end_secs: int,
    include_metadata: bool,
    exclude_metadata: bool,
    include_attachments: bool,
    exclude_attachments: bool,
    compression: str,
    chunk_size: int,
    recovery_mode: bool,
    no_recovery: bool,
    always_decode_chunk: bool,
) -> ProcessingOptions:
    """Build ProcessingOptions from command line arguments."""
    # Handle None defaults for list parameters
    include_topic_regex = include_topic_regex or []
    exclude_topic_regex = exclude_topic_regex or []

    # Validate mutually exclusive options
    if include_topic_regex and exclude_topic_regex:
        raise ValueError("Cannot use both --include-topic-regex and --exclude-topic-regex")

    recovery = False if no_recovery and recovery_mode else not no_recovery

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

    # Handle content filtering
    include_meta = include_metadata and not exclude_metadata
    include_attach = include_attachments and not exclude_attachments

    return ProcessingOptions(
        # Recovery options
        recovery_mode=recovery,
        always_decode_chunk=always_decode_chunk,
        # Filter options
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        start_time=start_time,
        end_time=end_time,
        include_metadata=include_meta,
        include_attachments=include_attach,
        # Output options
        compression=compression,
        chunk_size=chunk_size,
    )


@app.command()
def process(
    file: Annotated[
        list[Path],
        typer.Argument(
            ...,
            exists=True,
            dir_okay=False,
            help="Path(s) to MCAP file(s) to process (or merge if multiple)",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Output filename (required)",
        ),
    ],
    # Recovery options
    recovery_mode: Annotated[
        bool,
        typer.Option(
            "--recovery-mode",
            help="Enable recovery mode (handle errors gracefully, default: enabled)",
        ),
    ] = True,
    no_recovery: Annotated[
        bool,
        typer.Option(
            "--no-recovery",
            help="Disable recovery mode (fail on any errors)",
        ),
    ] = False,
    always_decode_chunk: Annotated[
        bool,
        typer.Option(
            "-a",
            "--always-decode-chunk",
            help="Always decode chunks, never use fast copying",
        ),
    ] = False,
    # Topic filtering
    include_topic_regex: Annotated[
        list[str] | None,
        typer.Option(
            "-y",
            "--include-topic-regex",
            help=(
                "Include messages with topic names matching this regex (can be used multiple times)"
            ),
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
        ),
    ] = None,
    # Time filtering
    start: Annotated[
        str,
        typer.Option(
            "-S",
            "--start",
            help="Include messages at or after this time (nanoseconds or RFC3339 date)",
        ),
    ] = "",
    start_secs: Annotated[
        int,
        typer.Option(
            "-s",
            "--start-secs",
            help="Include messages at or after this time in seconds (ignored if --start used)",
        ),
    ] = 0,
    start_nsecs: Annotated[
        int,
        typer.Option(
            "--start-nsecs",
            help="(Deprecated, use --start) Include messages at or after this time in nanoseconds",
        ),
    ] = 0,
    end: Annotated[
        str,
        typer.Option(
            "-E",
            "--end",
            help="Include messages before this time (nanoseconds or RFC3339 date)",
        ),
    ] = "",
    end_secs: Annotated[
        int,
        typer.Option(
            "-e",
            "--end-secs",
            help="Include messages before this time in seconds (ignored if --end used)",
        ),
    ] = 0,
    end_nsecs: Annotated[
        int,
        typer.Option(
            "--end-nsecs",
            help="(Deprecated, use --end) Include messages before this time in nanoseconds",
        ),
    ] = 0,
    # Content filtering
    include_metadata: Annotated[
        bool,
        typer.Option(
            "--include-metadata",
            help="Include metadata records in output (default: enabled)",
        ),
    ] = True,
    exclude_metadata: Annotated[
        bool,
        typer.Option(
            "--exclude-metadata",
            help="Exclude metadata records from output",
        ),
    ] = False,
    include_attachments: Annotated[
        bool,
        typer.Option(
            "--include-attachments",
            help="Include attachment records in output (default: enabled)",
        ),
    ] = True,
    exclude_attachments: Annotated[
        bool,
        typer.Option(
            "--exclude-attachments",
            help="Exclude attachment records from output",
        ),
    ] = False,
    # Output options
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            help="Chunk size of output file (default: 4MB)",
        ),
    ] = 4 * 1024 * 1024,  # 4MB
    compression: Annotated[
        CompressionType,
        typer.Option(
            "--compression",
            help="Compression algorithm for output file (default: zstd)",
        ),
    ] = CompressionType.ZSTD,
) -> None:
    """Process MCAP files with unified recovery and filtering.

    Unified command for processing MCAP files. Combines recovery, filtering,
    and transformation capabilities in a single operation. Can handle corrupt files
    while applying topic/time filters and changing compression.

    Usage:
      mcap process in.mcap -o out.mcap -y /camera.* --recovery-mode
    """
    try:
        options = build_processing_options(
            include_topic_regex=include_topic_regex,
            exclude_topic_regex=exclude_topic_regex,
            start=start,
            start_nsecs=start_nsecs,
            start_secs=start_secs,
            end=end,
            end_nsecs=end_nsecs,
            end_secs=end_secs,
            include_metadata=include_metadata,
            exclude_metadata=exclude_metadata,
            include_attachments=include_attachments,
            exclude_attachments=exclude_attachments,
            compression=compression.value,
            chunk_size=chunk_size,
            recovery_mode=recovery_mode,
            no_recovery=no_recovery,
            always_decode_chunk=always_decode_chunk,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    # Handle single or multiple input files
    input_files = file

    # Validate all input files exist (Typer should handle this, but double-check)
    for input_file in input_files:
        if not input_file.exists():
            console.print(f"[red]Error: Input file '{input_file}' does not exist[/red]")
            sys.exit(1)

    output_file = output
    file_sizes = [f.stat().st_size for f in input_files]

    # Create processor and run
    processor = McapProcessor(options)

    # Use ExitStack to manage multiple file handles
    with contextlib.ExitStack() as stack:
        input_streams = [stack.enter_context(f.open("rb")) for f in input_files]
        output_stream = stack.enter_context(output_file.open("wb"))

        stats = processor.process(input_streams, output_stream, file_sizes)

        # Report results
        if len(input_files) > 1:
            console.print(f"[green]✓ Merged {len(input_files)} files successfully![/green]")
        else:
            console.print("[green]✓ Processing completed successfully![/green]")

        # Basic stats
        console.print(
            f"Processed {stats.messages_processed:,} messages, "
            f"wrote {stats.writer_statistics.message_count:,} messages"
        )

        # Content stats
        if stats.attachments_processed > 0 and stats.writer_statistics:
            console.print(
                f"Processed {stats.attachments_processed} attachments, "
                f"wrote {stats.writer_statistics.attachment_count}"
            )
        if stats.metadata_processed > 0 and stats.writer_statistics:
            console.print(
                f"Processed {stats.metadata_processed} metadata records, "
                f"wrote {stats.writer_statistics.metadata_count}"
            )

        # Schema/channel stats
        console.print(
            f"Wrote {stats.writer_statistics.schema_count} schemas and "
            f"{stats.writer_statistics.channel_count} channels"
        )

        # Performance stats
        if stats.chunks_processed > 0:
            console.print(
                f"Processed {stats.chunks_processed} chunks "
                f"({stats.chunks_copied} fast copied, {stats.chunks_decoded} decoded)"
            )

        # Error stats
        if stats.errors_encountered > 0:
            console.print(f"[yellow]Encountered {stats.errors_encountered} errors[/yellow]")
        if stats.validation_errors > 0:
            console.print(f"[yellow]Found {stats.validation_errors} validation errors[/yellow]")
        if stats.filter_rejections > 0:
            console.print(f"Filtered out {stats.filter_rejections} records")
