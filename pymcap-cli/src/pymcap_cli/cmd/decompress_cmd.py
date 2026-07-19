"""Decompress command for pymcap-cli — write an uncompressed copy of an MCAP file."""

import logging
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console

from pymcap_cli.cmd._cli_options import (
    ChunkSizeOption,
    DeleteSourceOption,
    ForceOverwriteOption,
    InPlaceOption,
    NoClobberOption,
    OutputPathOption,
)
from pymcap_cli.cmd._run_processor import (
    finalize_delete_source,
    finalize_replace_source,
    in_place_temp_path,
    processing_had_errors,
    resolve_overwrite_policy,
    run_processor,
)
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from pymcap_cli.utils import output_overwrites_input

logger = logging.getLogger(__name__)
console = Console()


def decompress(
    file: str,
    output: OutputPathOption | None = None,
    *,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
    in_place: InPlaceOption = False,
) -> int:
    """Create an uncompressed copy of an MCAP file.

    Copy data in an MCAP file to a new file, leaving chunks uncompressed. This is
    the inverse of ``compress`` — messages, attachments, and metadata are
    preserved unchanged.

    Parameters
    ----------
    file
        Path to the MCAP file to decompress (local file or HTTP/HTTPS URL).
    output
        Output filename. Required unless --in-place is given.
    chunk_size
        Chunk size of output file in bytes.
    force
        Force overwrite of output file without confirmation.
    no_clobber
        Fail instead of prompting if the output file already exists.
    delete_source
        Delete source file(s) after the output is validated (header + summary).
        URL inputs and any source whose path equals the output are skipped.
    in_place
        Decompress to a temp file next to the source and, after the output is
        validated (header + summary), atomically replace the source with it.
        Local files only; mutually exclusive with --output and --delete-source.

    Examples
    --------
    ```
    pymcap-cli decompress in.mcap -o out.mcap
    pymcap-cli decompress in.mcap --in-place
    ```
    """
    if in_place:
        if output is not None:
            logger.error("--in-place and --output cannot be used together.")
            return 1
        if delete_source:
            logger.error("--in-place and --delete-source cannot be used together.")
            return 1
        if urlparse(file).scheme in ("http", "https"):
            logger.error("--in-place requires a local file, not a URL.")
            return 1
        output = in_place_temp_path(Path(file))
        overwrite_policy = OverwriteCollisionPolicy.OVERWRITE
    else:
        if output is None:
            logger.error("Either --output or --in-place is required.")
            return 1
        if output_overwrites_input(file, output):
            logger.error(
                "Output path is the same file as the input. "
                "Use --in-place to decompress in place safely."
            )
            return 1
        overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)

    logger.info(f"Decompressing '{file}' to '{output}'")

    try:
        result = run_processor(
            files=[file],
            output=output,
            input_options=InputOptions.from_args(),
            output_options=OutputOptions(
                compression="none",
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        logger.info("[green]✓ Decompression completed successfully![/green]")
        console.print(result.stats)
    except Exception:
        logger.exception("Error during decompression")
        output.unlink(missing_ok=True)
        return 1

    if in_place:
        if processing_had_errors(result.stats):
            logger.error("Processing reported errors — source preserved, not replaced in place.")
            output.unlink(missing_ok=True)
            return 1
        return finalize_replace_source(source=Path(file), tmp_output=output)

    if delete_source:
        if processing_had_errors(result.stats):
            logger.error("Processing reported errors — source file preserved.")
            return 1
        return finalize_delete_source(sources=[file], outputs=[output], require_lossless=True)

    return 0
