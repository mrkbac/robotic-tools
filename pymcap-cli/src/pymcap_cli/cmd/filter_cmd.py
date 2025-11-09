"""Filter command for pymcap-cli."""

from __future__ import annotations

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
from pymcap_cli.types import CompressionType

console = Console()
app = typer.Typer()


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
    metadata_mode: Annotated[
        MetadataMode,
        typer.Option(
            "--metadata",
            help="Metadata handling: include or exclude metadata records",
            rich_help_panel="Content Filtering",
            show_default=True,
        ),
    ] = MetadataMode.EXCLUDE,
    attachments_mode: Annotated[
        AttachmentsMode,
        typer.Option(
            "--attachments",
            help="Attachments handling: include or exclude attachment records",
            rich_help_panel="Content Filtering",
            show_default=True,
        ),
    ] = AttachmentsMode.EXCLUDE,
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            min=1,
            help="Chunk size of output file in bytes",
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
    """Copy filtered MCAP data to a new file.

    Filter an MCAP file by topic and time range to create a new file.
    When multiple regexes are used, topics that match any regex are
    included (or excluded).

    Usage:
      mcap filter in.mcap -o out.mcap -y /diagnostics -y /tf -y /camera_.*
    """
    # Confirm overwrite if needed (file existence validated by Typer)
    confirm_output_overwrite(output, force)

    # Build processing options
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
            metadata_mode=metadata_mode,
            attachments_mode=attachments_mode,
            compression=compression.value,
            chunk_size=chunk_size,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    file_size = file.stat().st_size

    # Create processor and run
    processor = McapProcessor(processing_options)

    with file.open("rb") as input_stream, output.open("wb") as output_stream:
        try:
            stats = processor.process([input_stream], output_stream, [file_size])

            # Report results
            report_processing_stats(stats, console, 1, "filter")

        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during filtering: {e}[/red]")
            raise typer.Exit(1) from None
