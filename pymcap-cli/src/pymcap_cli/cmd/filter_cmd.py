"""Filter command for pymcap-cli."""

import logging
from typing import Annotated

from cyclopts import Parameter
from rich.console import Console

from pymcap_cli.cmd._arg_constraints import MutuallyExclusive, constraint_group
from pymcap_cli.cmd._cli_options import (
    TIME_FILTERING_GROUP,
    TOPIC_FILTERING_GROUP,
    AttachmentsModeOption,
    ChunkSizeOption,
    CompressionOption,
    DeleteSourceOption,
    EarlyBailOption,
    EndTimeOption,
    ExcludeAttachmentsOption,
    ExcludeMetadataOption,
    ExcludeTopicOption,
    ForceOverwriteOption,
    LatchFromMetadataOption,
    LatchOption,
    MetadataModeOption,
    NoChunksOption,
    NoClobberOption,
    NoCrcOption,
    OrderOption,
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
from pymcap_cli.core.ordered_rewrite import reorder_output
from pymcap_cli.utils import (
    AttachmentsMode,
    MetadataMode,
)

logger = logging.getLogger(__name__)
console = Console()

# --early-bail assumes a forward scan to --end; inverting the window contradicts it.
_TIME_MODE_CONSTRAINT = constraint_group(MutuallyExclusive())


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
            group=[TIME_FILTERING_GROUP, _TIME_MODE_CONSTRAINT],
            help="Invert the time-window decision (drop messages INSIDE [start, end]).",
        ),
    ] = False,
    early_bail: Annotated[
        EarlyBailOption, Parameter(group=[TIME_FILTERING_GROUP, _TIME_MODE_CONSTRAINT])
    ] = False,
    latch: LatchOption = None,
    latch_from_metadata: LatchFromMetadataOption = False,
    metadata_mode: MetadataModeOption = MetadataMode.INCLUDE,
    attachments_mode: AttachmentsModeOption = AttachmentsMode.INCLUDE,
    exclude_metadata: ExcludeMetadataOption = False,
    exclude_attachments: ExcludeAttachmentsOption = False,
    order: OrderOption = "preserve",
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    no_crc: NoCrcOption = False,
    no_chunks: NoChunksOption = False,
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
        Metadata handling: include or exclude metadata records. Defaults to include
        (filtering is lossless by default).
    attachments_mode
        Attachments handling: include or exclude attachment records. Defaults to
        include (filtering is lossless by default).
    exclude_metadata
        Drop metadata records. Shorthand for ``--metadata exclude``.
    exclude_attachments
        Drop attachment records. Shorthand for ``--attachments exclude``.
    order
        Message ordering in the output: preserve (default), log_time, or topic.
    chunk_size
        Chunk size of output file in bytes.
    compression
        Compression algorithm for output file.
    no_crc
        Do not write CRC checksums in the output.
    no_chunks
        Write messages unchunked (no Chunk records) in the output.
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

    include_metadata = metadata_mode == MetadataMode.INCLUDE and not exclude_metadata
    include_attachments = attachments_mode == AttachmentsMode.INCLUDE and not exclude_attachments

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
            # The fast chunk-copy path re-emits stored chunk bytes verbatim, which
            # would bypass --no-chunks / --no-crc. Force per-message decode so the
            # writer's chunking and CRC settings actually take effect.
            always_decode_chunk=no_chunks or no_crc,
            include_metadata=include_metadata,
            include_attachments=include_attachments,
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
                enable_crcs=not no_crc,
                use_chunking=not no_chunks,
                overwrite_policy=overwrite_policy,
            ),
        )
        logger.info("[green]✓ Filter completed successfully![/green]")
        console.print(result.stats)
    except Exception:
        logger.exception("Error during filtering")
        return 1

    if order != "preserve":
        try:
            reorder_output(
                output,
                order=order,
                compression=compression,
                chunk_size=chunk_size,
                enable_crcs=not no_crc,
                use_chunking=not no_chunks,
            )
        except Exception:
            logger.exception("Error while reordering output")
            return 1

    if delete_source:
        return finalize_delete_source(sources=[file], outputs=[output])

    return 0
