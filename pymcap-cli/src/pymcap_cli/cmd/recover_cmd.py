from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pymcap_cli.mcap_processor import McapProcessor, ProcessingOptions

console = Console()


class CompressionType(str, Enum):
    """Compression algorithm choices."""

    zstd = "zstd"
    lz4 = "lz4"
    none = "none"


app = typer.Typer()


@app.command()
def recover(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Path to the MCAP file to recover",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            ...,
            "-o",
            "--output",
            help="Output filename (writes to stdout if not provided)",
        ),
    ],
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            help="Chunk size of output file (default: 4MB)",
        ),
    ] = 4 * 1024 * 1024,
    compression: Annotated[
        CompressionType,
        typer.Option(
            "--compression",
            help="Compression algorithm to use on output file (default: zstd)",
        ),
    ] = CompressionType.zstd,
    always_decode_chunk: Annotated[
        bool,
        typer.Option(
            "--always-decode-chunk",
            "-a",
            help="Always decode chunks, even if the file is not chunked",
        ),
    ] = False,
) -> None:
    """Recover data from a potentially corrupt MCAP file.

    This subcommand reads a potentially corrupt MCAP file and copies data to a new file.

    usage:
      mcap recover in.mcap -o out.mcap
    """
    input_file = file
    output_file = output
    file_size = input_file.stat().st_size

    # Convert recover options to unified processing options
    processing_options = ProcessingOptions(
        # Recovery mode enabled with all content included
        recovery_mode=True,
        always_decode_chunk=always_decode_chunk,
        # No filtering - include everything
        include_topics=[],
        exclude_topics=[],
        start_time=0,
        end_time=2**63 - 1,
        include_metadata=True,
        include_attachments=True,
        # Output options
        compression=compression.value,
        chunk_size=chunk_size,
    )

    # Create processor and run
    processor = McapProcessor(processing_options)

    with input_file.open("rb") as input_stream, output_file.open("wb") as output_stream:
        try:
            stats = processor.process([input_stream], output_stream, [file_size])

            # Report results in recovery-style format
            console.print("[green]✓ Recovery completed successfully![/green]")
            console.print(
                f"Recovered {stats.writer_statistics.message_count:,} messages, "
                f"{stats.writer_statistics.attachment_count} attachments, "
                f"and {stats.writer_statistics.metadata_count} metadata records."
            )
            if stats.chunks_processed > 0:
                console.print(
                    f"Processed {stats.chunks_processed} chunks "
                    f"({stats.chunks_copied} fast copied, {stats.chunks_decoded} decoded)."
                )
            if stats.errors_encountered > 0:
                console.print(f"Encountered {stats.errors_encountered} errors.")

        except RuntimeError as e:
            if "Writer not started" in str(e):
                # Empty file case - this is expected for empty/corrupt files
                console.print(
                    "[yellow]Warning: File appears to be empty or severely corrupted[/yellow]"
                )
                console.print("No valid MCAP data found to recover")
            else:
                console.print(f"[red]Error during recovery: {e}[/red]")
                raise typer.Exit(1)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during recovery: {e}[/red]")
            raise typer.Exit(1)
