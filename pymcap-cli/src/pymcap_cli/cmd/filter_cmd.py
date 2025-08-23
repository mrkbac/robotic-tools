"""Filter command for pymcap-cli."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from pymcap_cli.mcap_processor import (
    McapProcessor,
    ProcessingOptions,
    compile_topic_patterns,
    parse_time_arg,
)

console = Console()


def parse_timestamp_args(date_or_nanos: str, nanoseconds: int, seconds: int) -> int:
    """Parse timestamp with precedence: date_or_nanos > nanoseconds > seconds."""
    if date_or_nanos:
        return parse_time_arg(date_or_nanos)
    if nanoseconds != 0:
        return nanoseconds
    return seconds * 1_000_000_000


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the filter command parser to the subparsers."""
    parser = subparsers.add_parser(
        "filter",
        help="Copy filtered MCAP data to a new file",
        description=(
            "Filter an MCAP file by topic and time range to create a new file. "
            "When multiple regexes are used, topics that match any regex are "
            "included (or excluded)."
            "\\n\\nusage:\\n  mcap filter in.mcap -o out.mcap -y /diagnostics -y /tf -y /camera_.*"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file",
        help="Path to the MCAP file to filter",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (required)",
        type=str,
        required=True,
    )

    # Topic filtering
    parser.add_argument(
        "-y",
        "--include-topic-regex",
        action="append",
        default=[],
        help="Include messages with topic names matching this regex (can be used multiple times)",
    )

    parser.add_argument(
        "-n",
        "--exclude-topic-regex",
        action="append",
        default=[],
        help="Exclude messages with topic names matching this regex (can be used multiple times)",
    )

    # Time filtering - multiple formats for compatibility with Go CLI
    parser.add_argument(
        "-S",
        "--start",
        help="Include messages at or after this time (nanoseconds or RFC3339 date)",
        type=str,
        default="",
    )

    parser.add_argument(
        "-s",
        "--start-secs",
        help="Include messages at or after this time in seconds (ignored if --start used)",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--start-nsecs",
        help="(Deprecated, use --start) Include messages at or after this time in nanoseconds",
        type=int,
        default=0,
    )

    parser.add_argument(
        "-E",
        "--end",
        help="Include messages before this time (nanoseconds or RFC3339 date)",
        type=str,
        default="",
    )

    parser.add_argument(
        "-e",
        "--end-secs",
        help="Include messages before this time in seconds (ignored if --end used)",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-nsecs",
        help="(Deprecated, use --end) Include messages before this time in nanoseconds",
        type=int,
        default=0,
    )

    # Content filtering
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata records in output",
    )

    parser.add_argument(
        "--include-attachments",
        action="store_true",
        help="Include attachment records in output",
    )

    # Output options
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
        help="Compression algorithm for output file (default: zstd)",
    )

    return parser


def add_compress_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the compress command parser."""
    parser = subparsers.add_parser(
        "compress",
        help="Create a compressed copy of an MCAP file",
        description=(
            "Copy data in an MCAP file to a new file, compressing the output."
            "\\n\\nusage:\\n  mcap compress in.mcap -o out.mcap"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file",
        help="Path to the MCAP file to compress",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (required)",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4 * 1024 * 1024,
        help="Chunk size of output file (default: 4MB)",
    )

    parser.add_argument(
        "--compression",
        choices=["zstd", "lz4", "none"],
        default="zstd",
        help="Compression algorithm for output file (default: zstd)",
    )

    return parser


def build_processing_options(args: argparse.Namespace) -> ProcessingOptions:
    """Build ProcessingOptions from filter command line arguments."""
    # Validate mutually exclusive topic filters
    if args.include_topic_regex and args.exclude_topic_regex:
        raise ValueError("Cannot use both --include-topic-regex and --exclude-topic-regex")

    # Parse time arguments
    try:
        start_time = parse_timestamp_args(
            getattr(args, "start", ""),
            getattr(args, "start_nsecs", 0),
            getattr(args, "start_secs", 0),
        )
        end_time = parse_timestamp_args(
            getattr(args, "end", ""), getattr(args, "end_nsecs", 0), getattr(args, "end_secs", 0)
        )
    except ValueError as e:
        raise ValueError(f"Time parsing error: {e}") from e

    # Default end time to max if not specified
    if end_time == 0:
        end_time = 2**63 - 1

    # Validate time range
    if end_time < start_time:
        raise ValueError("End time cannot be before start time")

    # Compile topic patterns
    include_topics = compile_topic_patterns(getattr(args, "include_topic_regex", []))
    exclude_topics = compile_topic_patterns(getattr(args, "exclude_topic_regex", []))

    return ProcessingOptions(
        # Enable recovery mode for robust filtering
        recovery_mode=True,
        always_decode_chunk=False,
        # Filter options
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        start_time=start_time,
        end_time=end_time,
        include_metadata=getattr(args, "include_metadata", False),
        include_attachments=getattr(args, "include_attachments", False),
        # Output options
        compression=getattr(args, "compression", "zstd"),
        chunk_size=getattr(args, "chunk_size", 4 * 1024 * 1024),
    )


def handle_filter_command(args: argparse.Namespace) -> None:
    """Handle the filter command execution using unified processor."""
    try:
        processing_options = build_processing_options(args)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    input_file = Path(args.file)
    if not input_file.exists():
        console.print(f"[red]Error: Input file '{input_file}' does not exist[/red]")
        sys.exit(1)

    output_file = Path(args.output)
    file_size = input_file.stat().st_size

    # Create processor and run
    processor = McapProcessor(processing_options)

    with input_file.open("rb") as input_stream, output_file.open("wb") as output_stream:
        try:
            stats = processor.process(input_stream, output_stream, file_size)

            # Report results in filter-style format
            console.print("[green]✓ Filter completed successfully![/green]")
            console.print(
                f"Processed {stats.messages_processed:,} messages, "
                f"wrote {stats.messages_written:,} messages"
            )
            if stats.attachments_written > 0:
                console.print(f"Wrote {stats.attachments_written} attachments")
            if stats.metadata_written > 0:
                console.print(f"Wrote {stats.metadata_written} metadata records")
            console.print(
                f"Wrote {stats.schemas_written} schemas and {stats.channels_written} channels"
            )
            if stats.chunks_processed > 0:
                console.print(
                    f"Processed {stats.chunks_processed} chunks "
                    f"({stats.chunks_copied} fast copied, {stats.chunks_decoded} decoded)"
                )

        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during filtering: {e}[/red]")
            sys.exit(1)


def handle_compress_command(args: argparse.Namespace) -> None:
    """Handle the compress command execution."""
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
        compression=args.compression,
        chunk_size=args.chunk_size,
    )

    input_file = Path(args.file)
    if not input_file.exists():
        console.print(f"[red]Error: Input file '{input_file}' does not exist[/red]")
        sys.exit(1)

    output_file = Path(args.output)
    file_size = input_file.stat().st_size

    console.print(f"[blue]Compressing '{args.file}' to '{args.output}'[/blue]")

    processor = McapProcessor(processing_options)

    with input_file.open("rb") as input_stream, output_file.open("wb") as output_stream:
        try:
            stats = processor.process(input_stream, output_stream, file_size)

            console.print("[green]✓ Compression completed successfully![/green]")
            console.print(
                f"Processed {stats.messages_processed:,} messages, "
                f"wrote {stats.messages_written:,} messages"
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during compression: {e}[/red]")
            sys.exit(1)
