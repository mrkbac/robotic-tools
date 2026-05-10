"""Unified process command combining recovery and filtering capabilities."""

import logging
from enum import Enum
from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console

from pymcap_cli.cmd._run_processor import (
    finalize_delete_source,
    resolve_overwrite_policy,
    run_processor,
)
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
)
from pymcap_cli.types.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    DeleteSourceOption,
    ForceOverwriteOption,
    NoClobberOption,
    OutputPathOption,
)
from pymcap_cli.utils import (
    AttachmentsMode,
    MetadataMode,
)

logger = logging.getLogger(__name__)
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
            name=["-t", "--topics", "--include-topic-regex", "-y"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    exclude_topic_regex: Annotated[
        list[str] | None,
        Parameter(
            name=["-x", "--exclude-topics", "--exclude-topic-regex", "-n"],
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
            name=["--start-secs"],
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
            name=["--end-secs"],
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
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
) -> int:
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
    no_clobber
        Fail instead of prompting if the output file already exists.
    delete_source
        Delete source file(s) after the output is validated (header + summary).
        URL inputs and any source whose path equals the output are skipped.

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
    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        logger.error("--force and --no-clobber cannot be used together.")
        return 1

    try:
        input_options = InputOptions.from_args(
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
            always_decode_chunk=always_decode_chunk,
        )
    except ValueError as e:
        logger.error(str(e))  # noqa: TRY400
        return 1

    try:
        result = run_processor(
            files=file,
            output=output,
            input_options=input_options,
            output_options=OutputOptions(
                compression=compression,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        if len(file) > 1:
            logger.info(f"[green]✓ Successfully processed {len(file)} files![/green]")
        else:
            logger.info("[green]✓ Processing completed successfully![/green]")
        console.print(result.stats)
    except Exception:
        logger.exception("Error during processing")
        return 1

    if delete_source:
        return finalize_delete_source(sources=list(file), outputs=[output])

    return 0
