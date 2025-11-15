"""Unified process command combining recovery and filtering capabilities."""

import contextlib
import sys
from enum import Enum
from typing import Annotated

from cyclopts import Group, Parameter
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
from pymcap_cli.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    OutputPathOption,
)

console = Console()

# Parameter groups
RECOVERY_GROUP = Group("Recovery Options")
FILTERING_GROUP = Group("Topic Filtering")
TIME_FILTERING_GROUP = Group("Time Filtering")
CONTENT_FILTERING_GROUP = Group("Content Filtering")


class RecoveryMode(str, Enum):
    """Recovery mode choices."""

    ENABLED = "enabled"
    DISABLED = "disabled"


def process(
    file: list[str],
    output: OutputPathOption,
    *,
    recovery_mode: Annotated[
        RecoveryMode,
        Parameter(
            name=["--recovery"],
            group=RECOVERY_GROUP,
        ),
    ] = RecoveryMode.ENABLED,
    always_decode_chunk: Annotated[
        bool,
        Parameter(
            name=["-a", "--always-decode-chunk"],
            group=RECOVERY_GROUP,
        ),
    ] = False,
    include_topic_regex: Annotated[
        list[str] | None,
        Parameter(
            name=["-y", "--include-topic-regex"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    exclude_topic_regex: Annotated[
        list[str] | None,
        Parameter(
            name=["-n", "--exclude-topic-regex"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    start: Annotated[
        str,
        Parameter(
            name=["-S", "--start"],
            group=TIME_FILTERING_GROUP,
        ),
    ] = "",
    start_secs: Annotated[
        int,
        Parameter(
            name=["-s", "--start-secs"],
            group=TIME_FILTERING_GROUP,
        ),
    ] = 0,
    start_nsecs: Annotated[
        int,
        Parameter(
            name=["--start-nsecs"],
            group=TIME_FILTERING_GROUP,
        ),
    ] = 0,
    end: Annotated[
        str,
        Parameter(
            name=["-E", "--end"],
            group=TIME_FILTERING_GROUP,
        ),
    ] = "",
    end_secs: Annotated[
        int,
        Parameter(
            name=["-e", "--end-secs"],
            group=TIME_FILTERING_GROUP,
        ),
    ] = 0,
    end_nsecs: Annotated[
        int,
        Parameter(
            name=["--end-nsecs"],
            group=TIME_FILTERING_GROUP,
        ),
    ] = 0,
    metadata_mode: Annotated[
        MetadataMode,
        Parameter(
            name=["--metadata"],
            group=CONTENT_FILTERING_GROUP,
        ),
    ] = MetadataMode.INCLUDE,
    attachments_mode: Annotated[
        AttachmentsMode,
        Parameter(
            name=["--attachments"],
            group=CONTENT_FILTERING_GROUP,
        ),
    ] = AttachmentsMode.INCLUDE,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
) -> None:
    """Process MCAP files with unified recovery and filtering.

    Unified command for processing MCAP files. Combines recovery, filtering,
    and transformation capabilities in a single operation. Can handle corrupt files
    while applying topic/time filters and changing compression.

    Parameters
    ----------
    file
        Path(s) to MCAP file(s) to process (local files or HTTP/HTTPS URLs, or merge if multiple).
    output
        Output filename.
    recovery_mode
        Recovery mode: enabled (handle errors gracefully) or disabled (fail on errors).
    always_decode_chunk
        Always decode chunks, never use fast copying.
    include_topic_regex
        Include messages with topic names matching this regex (can be used multiple times).
    exclude_topic_regex
        Exclude messages with topic names matching this regex (can be used multiple times).
    start
        Include messages at or after this time (nanoseconds or RFC3339 date).
    start_secs
        Include messages at or after this time in seconds (ignored if --start used).
    start_nsecs
        [DEPRECATED - use --start instead] Include messages at or after this time in nanoseconds.
    end
        Include messages before this time (nanoseconds or RFC3339 date).
    end_secs
        Include messages before this time in seconds (ignored if --end used).
    end_nsecs
        [DEPRECATED - use --end instead] Include messages before this time in nanoseconds.
    metadata_mode
        Metadata handling: include or exclude metadata records.
    attachments_mode
        Attachments handling: include or exclude attachment records.
    chunk_size
        Chunk size of output file in bytes.
    compression
        Compression algorithm for output file.
    force
        Force overwrite of output file without confirmation.

    Examples
    --------
    ```
    # Recover corrupted file with compression change
    pymcap-cli process corrupt.mcap -o fixed.mcap --compression lz4

    # Filter by topic and time in one pass
    pymcap-cli process in.mcap -o out.mcap -y '/camera.*' --start-secs 10

    # Merge multiple files
    pymcap-cli process file1.mcap file2.mcap file3.mcap -o merged.mcap

    # Process with metadata excluded
    pymcap-cli process in.mcap -o out.mcap --metadata exclude
    ```
    """
    # Confirm overwrite if needed
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
        sys.exit(1)

    input_files = file

    # Create processor and run
    processor = McapProcessor(options)

    # Use ExitStack to manage multiple file handles
    with contextlib.ExitStack() as stack:
        input_streams = []
        file_sizes = []

        for f in input_files:
            stream, size = stack.enter_context(open_input(f))
            input_streams.append(stream)
            file_sizes.append(size)

        output_stream = stack.enter_context(output.open("wb"))

        try:
            stats = processor.process(input_streams, output_stream, file_sizes)

            # Report results
            report_processing_stats(stats, console, len(input_files), "process")

        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during processing: {e}[/red]")
            sys.exit(1)
