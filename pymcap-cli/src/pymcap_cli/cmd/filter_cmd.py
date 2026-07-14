"""Filter command for pymcap-cli."""

import logging
from typing import Annotated

from cyclopts import Parameter
from rich.console import Console

from pymcap_cli.cmd._cli_options import (
    TIME_FILTERING_GROUP,
    TOPIC_FILTERING_GROUP,
    AttachmentsModeOption,
    ChunkSizeOption,
    CompressionOption,
    DeleteSourceOption,
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    ForceOverwriteOption,
    LatchFromMetadataOption,
    LatchOption,
    MetadataModeOption,
    NoClobberOption,
    OutputPathOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
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
from pymcap_cli.utils import (
    AttachmentsMode,
    MetadataMode,
)

logger = logging.getLogger(__name__)
console = Console()


def filter_cmd(
    file: str,
    output: OutputPathOption,
    *,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    invert_topics: Annotated[
        bool,
        Parameter(
            name=["--invert-topics"],
            group=TOPIC_FILTERING_GROUP,
            help="Invert the include/exclude topic decision.",
        ),
    ] = False,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    invert_time: Annotated[
        bool,
        Parameter(
            name=["--invert-time"],
            group=TIME_FILTERING_GROUP,
            help="Invert the time-window decision (drop messages INSIDE [start, end]).",
        ),
    ] = False,
    early_bail: EarlyBailOption = False,
    latch: LatchOption = None,
    latch_from_metadata: LatchFromMetadataOption = False,
    metadata_mode: MetadataModeOption = MetadataMode.EXCLUDE,
    attachments_mode: AttachmentsModeOption = AttachmentsMode.EXCLUDE,
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
    topic
        Include a topic regex using full-match semantics (repeatable).
    exclude_topic
        Exclude a topic regex using full-match semantics (repeatable).
    start
        Include messages at or after this time (nanoseconds or RFC3339 date).
    end
        Include messages before this time (nanoseconds or RFC3339 date).
    early_bail
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
    pymcap-cli filter in.mcap -o out.mcap -t /diagnostics -t /tf -t '/camera_.*'
    ```
    """
    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        logger.error("--force and --no-clobber cannot be used together.")
        return 1
    if early_bail and invert_time:
        logger.error("--early-bail cannot be combined with --invert-time")
        return 1

    try:
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
        input_options = InputOptions.from_message_filter(
            message_filter,
            include_metadata=metadata_mode == MetadataMode.INCLUDE,
            include_attachments=attachments_mode == AttachmentsMode.INCLUDE,
            latch_topics=latch,
            latch_from_metadata=latch_from_metadata,
            invert_topics=invert_topics,
            invert_time=invert_time,
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
