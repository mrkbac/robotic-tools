"""Merge command for pymcap-cli."""

from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console

from pymcap_cli.cmd._run_processor import run_processor
from pymcap_cli.core.mcap_processor import InputOptions, OutputOptions
from pymcap_cli.types.types_manual import (
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

    Examples
    --------
    ```
    pymcap-cli merge recording1.mcap recording2.mcap -o combined.mcap
    pymcap-cli merge *.mcap -o all_recordings.mcap --compression lz4
    ```
    """
    if len(files) < 2:
        console.print("[red]Error: At least 2 input files are required for merging[/red]")
        return 1

    confirm_output_overwrite(output, force)

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
            ),
        )
        console.print(f"[green]✓ Successfully merged {len(files)} files![/green]")
        console.print(result.stats)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error during merge: {e}[/red]")
        return 1

    return 0
