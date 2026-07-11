"""Filter command for pymcap-cli."""

import logging
from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console

from pymcap_cli.cmd._run_processor import (
    finalize_delete_source,
    resolve_overwrite_policy,
    run_processor,
)
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
)
from pymcap_cli.types.types_manual import (
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
TOPIC_FILTERING_GROUP = Group("Topic Filtering")
TIME_FILTERING_GROUP = Group("Time Filtering")
CONTENT_FILTERING_GROUP = Group("Content Filtering")
LATCHING_GROUP = Group("Latching")


def filter_cmd(
    file: str,
    output: OutputPathOption,
    *,
    include_topic_regex: Annotated[
        list[str] | None,
        Parameter(
            name=["-t", "--topics", "--include-topic-regex", "-y"],
            group=TOPIC_FILTERING_GROUP,
        ),
    ] = None,
    exclude_topic_regex: Annotated[
        list[str] | None,
        Parameter(
            name=["-x", "--exclude-topics", "--exclude-topic-regex", "-n"],
            group=TOPIC_FILTERING_GROUP,
        ),
    ] = None,
    include_topic_glob: Annotated[
        list[str] | None,
        Parameter(
            name=["--topic-glob"],
            group=TOPIC_FILTERING_GROUP,
            help=(
                "Include topics matching this shell-style glob "
                "(repeatable, combines with --topics)."
            ),
        ),
    ] = None,
    exclude_topic_glob: Annotated[
        list[str] | None,
        Parameter(
            name=["--exclude-topic-glob"],
            group=TOPIC_FILTERING_GROUP,
            help="Exclude topics matching this shell-style glob (repeatable).",
        ),
    ] = None,
    invert_topics: Annotated[
        bool,
        Parameter(
            name=["--invert-topics"],
            group=TOPIC_FILTERING_GROUP,
            help="Invert the include/exclude topic decision.",
        ),
    ] = False,
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
    invert_time: Annotated[
        bool,
        Parameter(
            name=["--invert-time"],
            group=TIME_FILTERING_GROUP,
            help="Invert the time-window decision (drop messages INSIDE [start, end]).",
        ),
    ] = False,
    is_early_bail_enabled: Annotated[
        bool,
        Parameter(
            name=["--early-bail"],
            group=TIME_FILTERING_GROUP,
            help=(
                "Assume input messages are ordered by log_time and stop scanning "
                "after --end is reached. Unindexed out-of-order messages after "
                "that point will be ignored."
            ),
        ),
    ] = False,
    latch: Annotated[
        list[str] | None,
        Parameter(
            name=["--latch"],
            group=LATCHING_GROUP,
            help=(
                "Topic regex (repeatable) whose latest message should be replayed "
                "into the output even if --topics or --start would otherwise drop "
                "it. Useful for /tf_static and other transient-local topics."
            ),
        ),
    ] = None,
    latch_from_metadata: Annotated[
        bool,
        Parameter(
            name=["--latch-from-metadata"],
            group=LATCHING_GROUP,
            help=(
                "Also auto-detect latched channels by reading the MCAP "
                "'offered_qos_profiles' metadata for durability=transient_local."
            ),
        ),
    ] = False,
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
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
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
    is_early_bail_enabled
        Stop scanning at the first message at or after --end. Requires monotonic
        input log_time ordering for correctness.
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
    mcap filter in.mcap -o out.mcap -y /diagnostics -y /tf -y /camera_.*
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
            include_topic_glob=include_topic_glob,
            exclude_topic_glob=exclude_topic_glob,
            start=start,
            start_nsecs=start_nsecs,
            start_secs=start_secs,
            end=end,
            end_nsecs=end_nsecs,
            end_secs=end_secs,
            include_metadata=metadata_mode == MetadataMode.INCLUDE,
            include_attachments=attachments_mode == AttachmentsMode.INCLUDE,
            latch_topics=latch,
            latch_from_metadata=latch_from_metadata,
            invert_topics=invert_topics,
            invert_time=invert_time,
            is_early_bail_enabled=is_early_bail_enabled,
        )
    except ValueError as e:
        logger.error(str(e))  # noqa: TRY400
        return 1

    try:
        result = run_processor(
            files=[file],
            output=output,
            input_options=input_options,
            output_options=OutputOptions(
                compression=compression,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        logger.info("[green]✓ Filter completed successfully![/green]")
        console.print(result.stats)
    except Exception:
        logger.exception("Error during filtering")
        return 1

    if delete_source:
        return finalize_delete_source(sources=[file], outputs=[output])

    return 0
