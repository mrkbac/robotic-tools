"""Merge command for pymcap-cli."""

from __future__ import annotations

import contextlib
from typing import Annotated

import typer
from rich.console import Console

from pymcap_cli.input_handler import open_input
from pymcap_cli.mcap_processor import (
    AttachmentsMode,
    McapProcessor,
    MetadataMode,
    build_processing_options,
    confirm_output_overwrite,
    report_processing_stats,
)
from pymcap_cli.types import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    OutputPathOption,
)

console = Console()
app = typer.Typer()


@app.command(name="merge")
def merge(
    files: Annotated[
        list[str],
        typer.Argument(
            help=(
                "Paths to MCAP files to merge (local files or HTTP/HTTPS URLs, 2 or more required)"
            ),
        ),
    ],
    output: OutputPathOption,
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
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
) -> None:
    """Merge multiple MCAP files into one.

    Merge multiple MCAP files chronologically by timestamp into a single output file.
    Messages are interleaved in time order across all input files.

    Usage:
      pymcap merge recording1.mcap recording2.mcap -o combined.mcap
      pymcap merge *.mcap -o all_recordings.mcap --compression lz4
    """
    # Validate inputs (merge requires at least 2 files, existence checked by Typer)
    if len(files) < 2:
        console.print("[red]Error: At least 2 input files are required for merging[/red]")
        raise typer.Exit(1)

    # Confirm overwrite if needed
    confirm_output_overwrite(output, force)

    # Build processing options (no topic/time filtering for merge)
    processing_options = build_processing_options(
        metadata_mode=metadata_mode,
        attachments_mode=attachments_mode,
        compression=compression.value,
        chunk_size=chunk_size,
    )
    # Create processor and run
    processor = McapProcessor(processing_options)

    # Use ExitStack to manage multiple file handles
    with contextlib.ExitStack() as stack:
        input_streams = []
        file_sizes = []

        for f in files:
            stream, size = stack.enter_context(open_input(f))
            input_streams.append(stream)
            file_sizes.append(size)

        output_stream = stack.enter_context(output.open("wb"))

        try:
            stats = processor.process(input_streams, output_stream, file_sizes)

            # Report results
            report_processing_stats(stats, console, len(files), "merge")

        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during merge: {e}[/red]")
            raise typer.Exit(1) from None
