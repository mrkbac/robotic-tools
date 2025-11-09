"""Compress command for pymcap-cli."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pymcap_cli.mcap_processor import (
    McapProcessor,
    ProcessingOptions,
    confirm_output_overwrite,
)
from pymcap_cli.types import CompressionType

console = Console()
app = typer.Typer()


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
    force: Annotated[
        bool,
        typer.Option(
            "-f",
            "--force",
            help="Force overwrite of output file without confirmation",
            show_default=True,
        ),
    ] = False,
) -> None:
    """Create a compressed copy of an MCAP file.

    Copy data in an MCAP file to a new file, compressing the output.

    Usage:
      mcap compress in.mcap -o out.mcap
    """
    # Confirm overwrite if needed (file existence validated by Typer)
    confirm_output_overwrite(output, force)

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
