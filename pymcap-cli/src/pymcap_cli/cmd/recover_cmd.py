import logging

from rich.console import Console
from small_mcap import WriterNotStartedError

from pymcap_cli.cmd._cli_options import (
    AlwaysDecodeChunkOption,
    ChunkSizeOption,
    CompressionOption,
    DeleteSourceOption,
    ForceOverwriteOption,
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

logger = logging.getLogger(__name__)
console = Console()


def recover(
    file: str,
    output: OutputPathOption,
    *,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    always_decode_chunk: AlwaysDecodeChunkOption = False,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
) -> int:
    """Recover data from a potentially corrupt MCAP file.

    This command reads a potentially corrupt MCAP file and copies data to a new file.

    The exit code signals how much was recovered: ``0`` when the whole input was read
    cleanly, ``3`` when output was produced but the input was truncated or corrupt so
    some data was lost, and ``1`` when nothing could be recovered.

    Parameters
    ----------
    file
        Path to the MCAP file to recover (local file or HTTP/HTTPS URL).
    output
        Output filename.
    chunk_size
        Chunk size of output file in bytes.
    compression
        Compression algorithm for output file.
    always_decode_chunk
        Always decode chunks, even if the file is not chunked.
    force
        Force overwrite of output file without confirmation.
    no_clobber
        Fail instead of prompting if the output file already exists.
    delete_source
        Delete source file(s) after the output is validated (header + summary).
        URL inputs and any source whose path equals the output are skipped.
        Skipped if recovery yielded no valid MCAP data.

    Examples
    --------
    ```
    pymcap-cli recover in.mcap -o out.mcap
    ```
    """
    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)

    recovered = False
    lossy = False
    try:
        result = run_processor(
            files=[file],
            output=output,
            input_options=InputOptions.from_args(always_decode_chunk=always_decode_chunk),
            output_options=OutputOptions(
                compression=compression,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        recovered = result.stats.writer_statistics.message_count > 0
        lossy = processing_had_errors(result.stats)
        if not recovered:
            logger.warning("No valid MCAP data found to recover")
        elif lossy:
            logger.warning("Recovery completed with data loss — input was truncated or corrupt")
        else:
            logger.info("[green]✓ Recovery completed successfully![/green]")
        console.print(result.stats)
    except WriterNotStartedError:
        logger.warning("File appears to be empty or severely corrupted")
        logger.warning("No valid MCAP data found to recover")
    except (FileNotFoundError, FileExistsError) as e:
        logger.error(str(e))  # noqa: TRY400
        return 1
    except RuntimeError:
        logger.exception("Error during recovery")
        return 1
    except Exception:
        logger.exception("Error during recovery")
        return 1

    if not recovered:
        return 1

    if delete_source:
        delete_code = finalize_delete_source(sources=[file], outputs=[output])
        if delete_code != 0:
            return delete_code

    return 3 if lossy else 0
