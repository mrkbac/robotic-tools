from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console
from small_mcap import WriterNotStartedError

from pymcap_cli.cmd._run_processor import resolve_overwrite_policy, run_processor
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
)
from pymcap_cli.types.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    NoClobberOption,
    OutputPathOption,
)

console = Console()

# Parameter groups
RECOVERY_OPTIONS_GROUP = Group("Recovery Options")


def recover(
    file: str,
    output: OutputPathOption,
    *,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    always_decode_chunk: Annotated[
        bool,
        Parameter(
            name=["-a", "--always-decode-chunk"],
            group=RECOVERY_OPTIONS_GROUP,
        ),
    ] = False,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
) -> int:
    """Recover data from a potentially corrupt MCAP file.

    This command reads a potentially corrupt MCAP file and copies data to a new file.

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

    Examples
    --------
    ```
    pymcap-cli recover in.mcap -o out.mcap
    ```
    """
    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        console.print("[red]Error: --force and --no-clobber cannot be used together.[/red]")
        return 1

    try:
        result = run_processor(
            files=[file],
            output=output,
            input_options=InputOptions.from_args(always_decode_chunk=always_decode_chunk),
            output_options=OutputOptions(
                compression=compression.value,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        console.print("[green]✓ Recovery completed successfully![/green]")
        console.print(result.stats)
    except WriterNotStartedError:
        console.print("[yellow]Warning: File appears to be empty or severely corrupted[/yellow]")
        console.print("No valid MCAP data found to recover")
    except RuntimeError as e:
        console.print(f"[red]Error during recovery: {e}[/red]")
        return 1
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error during recovery: {e}[/red]")
        return 1

    return 0
