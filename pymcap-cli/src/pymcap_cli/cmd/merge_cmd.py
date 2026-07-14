"""Merge command for pymcap-cli."""

import logging

from rich.console import Console

from pymcap_cli.cmd._cli_options import (
    AttachmentsModeOption,
    ChunkSizeOption,
    CompressionOption,
    DedupIdenticalOption,
    DeleteSourceOption,
    ForceOverwriteOption,
    MetadataModeOption,
    NoClobberOption,
    OutputPathOption,
)
from pymcap_cli.cmd._run_processor import (
    finalize_delete_source,
    processing_had_errors,
    resolve_overwrite_policy,
    run_processor,
)
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
)
from pymcap_cli.core.processors.base import InputProcessor  # noqa: TC001 - runtime annotation below
from pymcap_cli.core.processors.dedup import DedupIdenticalProcessor
from pymcap_cli.core.rosbag2_layout import expand_bag_paths
from pymcap_cli.utils import (
    AttachmentsMode,
    MetadataMode,
    output_overwrites_input,
)

logger = logging.getLogger(__name__)
console = Console()


def merge(
    files: list[str],
    output: OutputPathOption,
    *,
    metadata_mode: MetadataModeOption = MetadataMode.INCLUDE,
    attachments_mode: AttachmentsModeOption = AttachmentsMode.INCLUDE,
    dedup_identical: DedupIdenticalOption = False,
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
    files = expand_bag_paths(files)
    if len(files) < 2:
        logger.error("At least 2 input files are required for merging")
        return 1

    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        logger.error("--force and --no-clobber cannot be used together.")
        return 1

    if any(output_overwrites_input(f, output) for f in files):
        logger.error("Output path is the same file as an input; choose a different output file.")
        return 1

    dedup_processor = DedupIdenticalProcessor() if dedup_identical else None
    extra_processors: list[InputProcessor] | None = [dedup_processor] if dedup_processor else None

    try:
        result = run_processor(
            files=files,
            output=output,
            input_options=InputOptions.from_args(
                include_metadata=metadata_mode == MetadataMode.INCLUDE,
                include_attachments=attachments_mode == AttachmentsMode.INCLUDE,
                extra_processors=extra_processors,
            ),
            output_options=OutputOptions(
                compression=compression,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        logger.info(f"[green]✓ Successfully merged {len(files)} files![/green]")
        console.print(result.stats)
        if dedup_processor and dedup_processor.dropped_count:
            console.print(
                f"[dim]Dedup dropped {dedup_processor.dropped_count:,} duplicate message(s).[/dim]"
            )
    except Exception:
        logger.exception("Error during merge")
        return 1

    if delete_source:
        if processing_had_errors(result.stats):
            logger.error("Processing reported errors — source file(s) preserved.")
            return 1
        return finalize_delete_source(sources=list(files), outputs=[output])

    return 0
