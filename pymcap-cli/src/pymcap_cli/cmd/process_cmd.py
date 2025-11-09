"""Unified process command combining recovery and filtering capabilities."""

from __future__ import annotations

import contextlib
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pymcap_cli.autocompletion import complete_all_topics
from pymcap_cli.mcap_processor import (
    AttachmentsMode,
    McapProcessor,
    MetadataMode,
    build_processing_options,
    confirm_output_overwrite,
    report_processing_stats,
)

app = typer.Typer()
console = Console()


class CompressionType(str, Enum):
    """Compression algorithm choices."""

    ZSTD = "zstd"
    LZ4 = "lz4"
    NONE = "none"


class RecoveryMode(str, Enum):
    """Recovery mode choices."""

    ENABLED = "enabled"
    DISABLED = "disabled"


@app.command(
    epilog="""
Examples:
  # Recover corrupted file with compression change
  pymcap-cli process corrupt.mcap -o fixed.mcap --compression lz4

  # Filter by topic and time in one pass
  pymcap-cli process in.mcap -o out.mcap -y '/camera.*' --start-secs 10

  # Merge multiple files
  pymcap-cli process file1.mcap file2.mcap file3.mcap -o merged.mcap

  # Process with metadata excluded
  pymcap-cli process in.mcap -o out.mcap --metadata exclude
"""
)
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
            rich_help_panel="Output Options",
        ),
    ],
    # Recovery options
    recovery_mode: Annotated[
        RecoveryMode,
        typer.Option(
            "--recovery",
            help="Recovery mode: enabled (handle errors gracefully) or disabled (fail on errors)",
            rich_help_panel="Recovery Options",
            show_default=True,
        ),
    ] = RecoveryMode.ENABLED,
    always_decode_chunk: Annotated[
        bool,
        typer.Option(
            "-a",
            "--always-decode-chunk",
            help="Always decode chunks, never use fast copying",
            rich_help_panel="Recovery Options",
            show_default=True,
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
            autocompletion=complete_all_topics,
            rich_help_panel="Topic Filtering",
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
        ),
    ] = None,
    # Time filtering
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
        ),
    ] = 0,
    start_nsecs: Annotated[
        int,
        typer.Option(
            "--start-nsecs",
            help=(
                "[DEPRECATED - use --start instead] "
                "Include messages at or after this time in nanoseconds"
            ),
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
        ),
    ] = 0,
    end_nsecs: Annotated[
        int,
        typer.Option(
            "--end-nsecs",
            help=(
                "[DEPRECATED - use --end instead] Include messages before this time in nanoseconds"
            ),
            hidden=True,
        ),
    ] = 0,
    # Content filtering
    metadata_mode: Annotated[
        MetadataMode,
        typer.Option(
            "--metadata",
            help="Metadata handling: include or exclude metadata records",
            rich_help_panel="Content Filtering",
            show_default=True,
        ),
    ] = MetadataMode.INCLUDE,
    attachments_mode: Annotated[
        AttachmentsMode,
        typer.Option(
            "--attachments",
            help="Attachments handling: include or exclude attachment records",
            rich_help_panel="Content Filtering",
            show_default=True,
        ),
    ] = AttachmentsMode.INCLUDE,
    # Output options
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            min=1,
            help="Chunk size of output file in bytes",
            rich_help_panel="Output Options",
            show_default="4MB",
        ),
    ] = 4 * 1024 * 1024,  # 4MB
    compression: Annotated[
        CompressionType,
        typer.Option(
            "--compression",
            help="Compression algorithm for output file",
            rich_help_panel="Output Options",
            show_default=True,
        ),
    ] = CompressionType.ZSTD,
    force: Annotated[
        bool,
        typer.Option(
            "-f",
            "--force",
            help="Force overwrite of output file without confirmation",
            rich_help_panel="Output Options",
            show_default=True,
        ),
    ] = False,
) -> None:
    """Process MCAP files with unified recovery and filtering.

    Unified command for processing MCAP files. Combines recovery, filtering,
    and transformation capabilities in a single operation. Can handle corrupt files
    while applying topic/time filters and changing compression.

    Usage:
      mcap process in.mcap -o out.mcap -y /camera.* --recovery-mode
    """
    # Confirm overwrite if needed (file existence validated by Typer)
    confirm_output_overwrite(output, force)

    # Build processing options
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
            metadata_mode=metadata_mode,
            attachments_mode=attachments_mode,
            compression=compression.value,
            chunk_size=chunk_size,
            recovery_mode=recovery_mode == RecoveryMode.ENABLED,
            always_decode_chunk=always_decode_chunk,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    input_files = file
    file_sizes = [f.stat().st_size for f in input_files]

    # Create processor and run
    processor = McapProcessor(options)

    # Use ExitStack to manage multiple file handles
    with contextlib.ExitStack() as stack:
        input_streams = [stack.enter_context(f.open("rb")) for f in input_files]
        output_stream = stack.enter_context(output.open("wb"))

        try:
            stats = processor.process(input_streams, output_stream, file_sizes)

            # Report results
            report_processing_stats(stats, console, len(input_files), "process")

        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during processing: {e}[/red]")
            raise typer.Exit(1) from None
