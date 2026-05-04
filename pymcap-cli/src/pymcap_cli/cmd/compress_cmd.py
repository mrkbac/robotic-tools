"""Compress command for pymcap-cli."""

import logging

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

logger = logging.getLogger(__name__)
console = Console()


def compress(
    file: str,
    output: OutputPathOption,
    *,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
) -> int:
    """Create a compressed copy of an MCAP file.

    Copy data in an MCAP file to a new file, compressing the output.

    Parameters
    ----------
    file
        Path to the MCAP file to compress (local file or HTTP/HTTPS URL).
    output
        Output filename.
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
    pymcap-cli compress in.mcap -o out.mcap
    ```
    """
    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        logger.error("--force and --no-clobber cannot be used together.")
        return 1

    logger.info(f"Compressing '{file}' to '{output}'")

    try:
        result = run_processor(
            files=[file],
            output=output,
            # Do not force always_decode_chunk — the processor now has a
            # chunk-level RECOMPRESS path that avoids per-message parsing.
            input_options=InputOptions.from_args(),
            output_options=OutputOptions(
                compression=compression.value,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        logger.info("[green]✓ Compression completed successfully![/green]")
        console.print(result.stats)
    except Exception:
        logger.exception("Error during compression")
        return 1

    if delete_source:
        return finalize_delete_source(sources=[file], outputs=[output])

    return 0
