"""Merge command for pymcap-cli."""

import contextlib
import sys
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
) -> None:
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
    # Validate inputs (merge requires at least 2 files)
    if len(files) < 2:
        console.print("[red]Error: At least 2 input files are required for merging[/red]")
        sys.exit(1)

    # Confirm overwrite if needed
    confirm_output_overwrite(output, force)

    # Use ExitStack to manage multiple file handles
    with contextlib.ExitStack() as stack:
        input_options: list[InputOptions] = []

        for f in files:
            stream, size = stack.enter_context(open_input(f))
            input_options.append(
                InputOptions(
                    stream=stream,
                    file_size=size,
                    include_metadata=metadata_mode == MetadataMode.INCLUDE,
                    include_attachments=attachments_mode == AttachmentsMode.INCLUDE,
                )
            )

        output_stream = stack.enter_context(output.open("wb"))

        # Build processing options
        processing_options = ProcessingOptions(
            inputs=input_options,
            output=OutputOptions(
                compression=compression.value,
                chunk_size=chunk_size,
            ),
        )

        # Create processor and run
        processor = McapProcessor(processing_options)

        try:
            stats = processor.process(output_stream)

            console.print(f"[green]✓ Successfully merged {len(files)} files![/green]")
            console.print(stats)

        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during merge: {e}[/red]")
            sys.exit(1)
