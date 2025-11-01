"""Unified process command combining recovery and filtering capabilities."""

from __future__ import annotations

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


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the process command parser to the subparsers."""
    parser = subparsers.add_parser(
        "process",
        help="Process MCAP files with unified recovery and filtering",
        description=(
            "Unified command for processing MCAP files. Combines recovery, filtering, "
            "and transformation capabilities in a single operation. Can handle corrupt files "
            "while applying topic/time filters and changing compression."
            "\\n\\nusage:\\n  mcap process in.mcap -o out.mcap -y /camera.* --recover"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file",
        help="Path to the MCAP file to process",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (required)",
        type=str,
        required=True,
    )

    # Recovery options
    recovery_group = parser.add_argument_group("Recovery options")
    recovery_group.add_argument(
        "--recovery-mode",
        action="store_true",
        default=True,
        help="Enable recovery mode (handle errors gracefully, default: enabled)",
    )
    recovery_group.add_argument(
        "--no-recovery",
        action="store_true",
        help="Disable recovery mode (fail on any errors)",
    )
    recovery_group.add_argument(
        "-a",
        "--always-decode-chunk",
        action="store_true",
        help="Always decode chunks, never use fast copying",
    )

    # Topic filtering
    topic_group = parser.add_argument_group("Topic filtering")
    topic_group.add_argument(
        "-y",
        "--include-topic-regex",
        action="append",
        default=[],
        help="Include messages with topic names matching this regex (can be used multiple times)",
    )
    topic_group.add_argument(
        "-n",
        "--exclude-topic-regex",
        action="append",
        default=[],
        help="Exclude messages with topic names matching this regex (can be used multiple times)",
    )

    # Time filtering
    time_group = parser.add_argument_group("Time filtering")
    time_group.add_argument(
        "-S",
        "--start",
        help="Include messages at or after this time (nanoseconds or RFC3339 date)",
        type=str,
        default="",
    )
    time_group.add_argument(
        "-s",
        "--start-secs",
        help="Include messages at or after this time in seconds (ignored if --start used)",
        type=int,
        default=0,
    )
    time_group.add_argument(
        "--start-nsecs",
        help="(Deprecated, use --start) Include messages at or after this time in nanoseconds",
        type=int,
        default=0,
    )
    time_group.add_argument(
        "-E",
        "--end",
        help="Include messages before this time (nanoseconds or RFC3339 date)",
        type=str,
        default="",
    )
    time_group.add_argument(
        "-e",
        "--end-secs",
        help="Include messages before this time in seconds (ignored if --end used)",
        type=int,
        default=0,
    )
    time_group.add_argument(
        "--end-nsecs",
        help="(Deprecated, use --end) Include messages before this time in nanoseconds",
        type=int,
        default=0,
    )

    # Content filtering
    content_group = parser.add_argument_group("Content filtering")
    content_group.add_argument(
        "--include-metadata",
        action="store_true",
        default=True,
        help="Include metadata records in output (default: enabled)",
    )
    content_group.add_argument(
        "--exclude-metadata",
        action="store_true",
        help="Exclude metadata records from output",
    )
    content_group.add_argument(
        "--include-attachments",
        action="store_true",
        default=True,
        help="Include attachment records in output (default: enabled)",
    )
    content_group.add_argument(
        "--exclude-attachments",
        action="store_true",
        help="Exclude attachment records from output",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--chunk-size",
        type=int,
        default=4 * 1024 * 1024,  # 4MB
        help="Chunk size of output file (default: 4MB)",
    )
    output_group.add_argument(
        "--compression",
        choices=["zstd", "lz4", "none"],
        default="zstd",
        help="Compression algorithm for output file (default: zstd)",
    )

    return parser


def build_processing_options(args: argparse.Namespace) -> ProcessingOptions:
    """Build ProcessingOptions from command line arguments."""
    # Validate mutually exclusive options
    if args.include_topic_regex and args.exclude_topic_regex:
        raise ValueError("Cannot use both --include-topic-regex and --exclude-topic-regex")

    recovery_mode = False if args.no_recovery and args.recovery_mode else not args.no_recovery

    # Parse time arguments
    try:
        start_time = parse_timestamp_args(args.start, args.start_nsecs, args.start_secs)
        end_time = parse_timestamp_args(args.end, args.end_nsecs, args.end_secs)
    except ValueError as e:
        raise ValueError(f"Time parsing error: {e}") from e

    # Default end time to max if not specified
    if end_time == 0:
        end_time = 2**63 - 1

    # Validate time range
    if end_time < start_time:
        raise ValueError("End time cannot be before start time")

    # Compile topic patterns
    include_topics = compile_topic_patterns(args.include_topic_regex)
    exclude_topics = compile_topic_patterns(args.exclude_topic_regex)

    # Handle content filtering
    include_metadata = args.include_metadata and not args.exclude_metadata
    include_attachments = args.include_attachments and not args.exclude_attachments

    return ProcessingOptions(
        # Recovery options
        recovery_mode=recovery_mode,
        always_decode_chunk=args.always_decode_chunk,
        # Filter options
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        start_time=start_time,
        end_time=end_time,
        include_metadata=include_metadata,
        include_attachments=include_attachments,
        # Output options
        compression=args.compression,
        chunk_size=args.chunk_size,
    )


def handle_command(args: argparse.Namespace) -> None:
    """Handle the process command execution."""
    try:
        options = build_processing_options(args)
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
    processor = McapProcessor(options)

    with input_file.open("rb") as input_stream, output_file.open("wb") as output_stream:
        stats = processor.process(input_stream, output_stream, file_size)

        # Report results
        console.print("[green]✓ Processing completed successfully![/green]")

        # Basic stats
        console.print(
            f"Processed {stats.messages_processed:,} messages, "
            f"wrote {stats.messages_written:,} messages"
        )

        # Content stats
        if stats.attachments_processed > 0:
            console.print(
                f"Processed {stats.attachments_processed} attachments, "
                f"wrote {stats.attachments_written}"
            )
        if stats.metadata_processed > 0:
            console.print(
                f"Processed {stats.metadata_processed} metadata records, "
                f"wrote {stats.metadata_written}"
            )

        # Schema/channel stats
        console.print(
            f"Wrote {stats.schemas_written} schemas and {stats.channels_written} channels"
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
