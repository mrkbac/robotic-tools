"""Filter command for pymcap-cli."""

from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console

from pymcap_cli.input_handler import open_input
from pymcap_cli.mcap_processor import (
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)
from pymcap_cli.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    OutputPathOption,
)
from pymcap_cli.utils import (
    AttachmentsMode,
    MetadataMode,
    confirm_output_overwrite,
)

console = Console()

# Parameter groups
TOPIC_FILTERING_GROUP = Group("Topic Filtering")
TIME_FILTERING_GROUP = Group("Time Filtering")
CONTENT_FILTERING_GROUP = Group("Content Filtering")


def filter_cmd(
    file: str,
    output: OutputPathOption,
    *,
    include_topic_regex: Annotated[
        list[str] | None,
        Parameter(
            name=["-y", "--include-topic-regex"],
            group=TOPIC_FILTERING_GROUP,
        ),
    ] = None,
    exclude_topic_regex: Annotated[
        list[str] | None,
        Parameter(
            name=["-n", "--exclude-topic-regex"],
            group=TOPIC_FILTERING_GROUP,
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
    ] = MetadataMode.EXCLUDE,
    attachments_mode: Annotated[
        AttachmentsMode,
        Parameter(
            name=["--attachments"],
            group=CONTENT_FILTERING_GROUP,
        ),
    ] = AttachmentsMode.EXCLUDE,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
) -> int:
    """Copy filtered MCAP data to a new file.

    Filter an MCAP file by topic and time range to create a new file.
    When multiple regexes are used, topics that match any regex are
    included (or excluded).

    Parameters
    ----------
    file
        Path to the MCAP file to filter (local file or HTTP/HTTPS URL).
    output
        Output filename.
    include_topic_regex
        Include messages with topic names matching this regex (can be used multiple times).
    exclude_topic_regex
        Exclude messages with topic names matching this regex (can be used multiple times).
    start
        Include messages at or after this time (nanoseconds or RFC3339 date).
    start_secs
        Include messages at or after this time in seconds (ignored if --start used).
    start_nsecs
        (Deprecated, use --start) Include messages at or after this time in nanoseconds.
    end
        Include messages before this time (nanoseconds or RFC3339 date).
    end_secs
        Include messages before this time in seconds (ignored if --end used).
    end_nsecs
        (Deprecated, use --end) Include messages before this time in nanoseconds.
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
    mcap filter in.mcap -o out.mcap -y /diagnostics -y /tf -y /camera_.*
    ```
    """
    # Confirm overwrite if needed
    confirm_output_overwrite(output, force)

    with open_input(file) as (f, file_size), output.open("wb") as output_stream:
        # Build input options directly with raw CLI args
        try:
            input_opts = InputOptions(
                stream=f,
                file_size=file_size,
                include_topic_regex=include_topic_regex,
                exclude_topic_regex=exclude_topic_regex,
                start=start,
                start_nsecs=start_nsecs,
                start_secs=start_secs,
                end=end,
                end_nsecs=end_nsecs,
                end_secs=end_secs,
                include_metadata=metadata_mode == MetadataMode.INCLUDE,
                include_attachments=attachments_mode == AttachmentsMode.INCLUDE,
            )
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1

        # Build processing options
        processing_options = ProcessingOptions(
            inputs=[input_opts],
            output=OutputOptions(
                compression=compression.value,
                chunk_size=chunk_size,
            ),
        )

        # Create processor and run
        processor = McapProcessor(processing_options)

        try:
            stats = processor.process(output_stream)

            console.print("[green]✓ Filter completed successfully![/green]")
            console.print(stats)

        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during filtering: {e}[/red]")
            return 1

    return 0
