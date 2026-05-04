"""Merge command for pymcap-cli."""

import logging
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
CONTENT_FILTERING_GROUP = Group("Content Filtering")


def merge(
    files: list[str],
    output: OutputPathOption,
    *,
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
    """Merge multiple MCAP files into one.

    Merge multiple MCAP files chronologically by timestamp into a single output file.
    Messages are interleaved in time order across all input files.

    Parameters
    ----------
    files
        Paths to MCAP files to merge (local files or HTTP/HTTPS URLs, 2 or more required).
    output
        Output filename.
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
    pymcap-cli merge recording1.mcap recording2.mcap -o combined.mcap
    pymcap-cli merge *.mcap -o all_recordings.mcap --compression lz4
    ```
    """
    if len(files) < 2:
        logger.error("At least 2 input files are required for merging")
        return 1

    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        logger.error("--force and --no-clobber cannot be used together.")
        return 1

    try:
        result = run_processor(
            files=files,
            output=output,
            input_options=InputOptions.from_args(
                include_metadata=metadata_mode == MetadataMode.INCLUDE,
                include_attachments=attachments_mode == AttachmentsMode.INCLUDE,
            ),
            output_options=OutputOptions(
                compression=compression.value,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        logger.info(f"[green]✓ Successfully merged {len(files)} files![/green]")
        console.print(result.stats)
    except Exception:
        logger.exception("Error during merge")
        return 1

    if delete_source:
        return finalize_delete_source(sources=list(files), outputs=[output])

    return 0
