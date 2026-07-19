"""Sort command for pymcap-cli — rewrite an MCAP file with messages reordered."""

import logging
from pathlib import Path
from urllib.parse import urlparse

from pymcap_cli.cmd._cli_options import (
    ChunkSizeOption,
    CompressionOption,
    DeleteSourceOption,
    ForceOverwriteOption,
    InPlaceOption,
    NoChunksOption,
    NoClobberOption,
    NoCrcOption,
    OrderOption,
    OutputPathOption,
)
from pymcap_cli.cmd._run_processor import (
    finalize_delete_source,
    finalize_replace_source,
    in_place_temp_path,
    resolve_overwrite_policy,
)
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.mcap_processor import OverwriteCollisionPolicy
from pymcap_cli.core.ordered_rewrite import rewrite_ordered
from pymcap_cli.utils import confirm_output_overwrite, output_overwrites_input

logger = logging.getLogger(__name__)


def sort(
    file: str,
    output: OutputPathOption | None = None,
    *,
    order: OrderOption = "log_time",
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    no_crc: NoCrcOption = False,
    no_chunks: NoChunksOption = False,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
    in_place: InPlaceOption = False,
) -> int:
    """Rewrite an MCAP file with its messages reordered.

    All messages are buffered in memory and reordered with a stable sort, so
    equal keys keep their stored order. Schemas, channels, attachments, and
    metadata are preserved.

    Parameters
    ----------
    file
        Path to the MCAP file to sort (local file or HTTP/HTTPS URL).
    output
        Output filename. Required unless --in-place is given.
    order
        Message ordering: log_time (default) or topic. 'preserve' keeps stored
        order (a plain rewrite).
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
    in_place
        Sort to a temp file next to the source and, after validation, atomically
        replace the source. Local files only; mutually exclusive with --output.

    Examples
    --------
    ```
    pymcap-cli sort in.mcap -o out.mcap
    pymcap-cli sort in.mcap --order topic -o out.mcap
    pymcap-cli sort in.mcap --in-place
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
    else:
        if output is None:
            logger.error("Either --output or --in-place is required.")
            return 1
        if output_overwrites_input(file, output):
            logger.error(
                "Output path is the same file as the input. Use --in-place to sort in place safely."
            )
            return 1
        policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
        if output.exists() and policy != OverwriteCollisionPolicy.OVERWRITE:
            if policy == OverwriteCollisionPolicy.ERROR:
                logger.error(f"Output file '{output}' already exists.")
                return 1
            confirm_output_overwrite(output, force=False)

    logger.info(f"Sorting '{file}' -> '{output}' by {order}")

    try:
        count = rewrite_ordered(
            input_path=file,
            output_path=output,
            order=order,
            compression=compression,
            chunk_size=chunk_size,
            enable_crcs=not no_crc,
            use_chunking=not no_chunks,
        )
    except Exception:
        logger.exception("Error during sort")
        output.unlink(missing_ok=True)
        return 1

    logger.info(f"[green]✓ Sorted {count} messages[/green]")

    if in_place:
        return finalize_replace_source(source=Path(file), tmp_output=output)

    if delete_source:
        return finalize_delete_source(sources=[file], outputs=[output], require_lossless=True)

    return 0
