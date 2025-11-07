from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from pymcap_cli.mcap_processor import McapProcessor, ProcessingOptions

console = Console()


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the recover command parser to the subparsers."""
    parser = subparsers.add_parser(
        "recover",
        help="Recover data from a potentially corrupt MCAP file",
        description=(
            "This subcommand reads a potentially corrupt MCAP file and copies data to a new file."
            "\n\nusage:\n  mcap recover in.mcap -o out.mcap"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file",
        help="Path to the MCAP file to recover",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (writes to stdout if not provided)",
        type=str,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4 * 1024 * 1024,  # 4MB
        help="Chunk size of output file (default: 4MB)",
    )

    parser.add_argument(
        "--compression",
        choices=["zstd", "lz4", "none"],
        default="zstd",
        help="Compression algorithm to use on output file (default: zstd)",
    )

    parser.add_argument(
        "-a",
        "--always-decode-chunk",
        action="store_true",
        help="Always decode chunks, even if the file is not chunked",
    )

    return parser


def handle_command(args: argparse.Namespace) -> None:
    """Handle the recover command execution using unified processor."""
    input_file = Path(args.file)
    if not input_file.exists():
        console.print(f"[red]Error: Input file '{input_file}' does not exist[/red]")
        sys.exit(1)

    output_file = Path(args.output)
    file_size = input_file.stat().st_size

    # Convert recover options to unified processing options
    processing_options = ProcessingOptions(
        # Recovery mode enabled with all content included
        recovery_mode=True,
        always_decode_chunk=args.always_decode_chunk,
        # No filtering - include everything
        include_topics=[],
        exclude_topics=[],
        start_time=0,
        end_time=2**63 - 1,
        include_metadata=True,
        include_attachments=True,
        # Output options
        compression=args.compression,
        chunk_size=args.chunk_size,
    )

    # Create processor and run
    processor = McapProcessor(processing_options)

    with input_file.open("rb") as input_stream, output_file.open("wb") as output_stream:
        try:
            stats = processor.process(input_stream, output_stream, file_size)

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
                sys.exit(1)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during recovery: {e}[/red]")
            sys.exit(1)
