"""Compress command for pymcap-cli."""

import logging
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console

from pymcap_cli.cmd._cli_options import (
    ChunkSizeOption,
    CompressionLevelOption,
    CompressionOption,
    DeleteSourceOption,
    FastCompressionOption,
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
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from pymcap_cli.utils import output_overwrites_input

logger = logging.getLogger(__name__)
console = Console()

# zstd "fast" mode: a negative level trades ~5% larger output for roughly 2x
# compression throughput. Used by --fast.
_FAST_ZSTD_LEVEL = -1


def compress(
    file: str,
    output: OutputPathOption | None = None,
    *,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
    in_place: InPlaceOption = False,
    compression_level: CompressionLevelOption = None,
    fast: FastCompressionOption = False,
) -> int:
    """Create a compressed copy of an MCAP file.

    Copy data in an MCAP file to a new file, compressing the output.

    Parameters
    ----------
    file
        Path to the MCAP file to compress (local file or HTTP/HTTPS URL).
    output
        Output filename. Required unless --in-place is given.
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
    in_place
        Compress to a temp file next to the source and, after the output is
        validated (header + summary), atomically replace the source with it.
        Local files only; mutually exclusive with --output and --delete-source.
    compression_level
        zstd level. Omit for the library default (3). Negative levels select the
        fast modes — much higher throughput for slightly larger output. Ignored
        for non-zstd compression. Mutually exclusive with --fast.
    fast
        Shortcut for a fast zstd level (roughly 2x throughput, ~5% larger
        output). Equivalent to ``--compression-level -1``.

    Examples
    --------
    ```
    pymcap-cli compress in.mcap -o out.mcap
    pymcap-cli compress in.mcap --in-place
    pymcap-cli compress in.mcap -o out.mcap --fast
    ```
    """
    if fast and compression_level is not None:
        logger.error("--fast and --compression-level cannot be used together.")
        return 1
    zstd_level = _FAST_ZSTD_LEVEL if fast else compression_level
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
                "Use --in-place to compress in place safely."
            )
            return 1
        policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
        if policy is None:
            logger.error("--force and --no-clobber cannot be used together.")
            return 1
        overwrite_policy = policy

    logger.info(f"Compressing '{file}' to '{output}'")

    try:
        result = run_processor(
            files=[file],
            output=output,
            # Do not force always_decode_chunk — the processor now has a
            # chunk-level RECOMPRESS path that avoids per-message parsing.
            input_options=InputOptions.from_args(),
            output_options=OutputOptions(
                compression=compression,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
                zstd_level=zstd_level,
            ),
        )
        logger.info("[green]✓ Compression completed successfully![/green]")
        console.print(result.stats)
    except Exception:
        logger.exception("Error during compression")
        # The output was opened "wb" (truncated) before processing, so a failed run
        # leaves an empty/partial file. Remove it rather than leaving a 0-byte result.
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
