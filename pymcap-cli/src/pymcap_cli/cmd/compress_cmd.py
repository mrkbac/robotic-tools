"""Compress command for pymcap-cli."""

from rich.console import Console

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


def compress(
    file: str,
    output: OutputPathOption,
    *,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
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

    Examples
    --------
    ```
    pymcap-cli compress in.mcap -o out.mcap
    ```
    """
    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        console.print("[red]Error: --force and --no-clobber cannot be used together.[/red]")
        return 1

    console.print(f"[blue]Compressing '{file}' to '{output}'[/blue]")

    try:
        result = run_processor(
            files=[file],
            output=output,
            input_options=InputOptions.from_args(always_decode_chunk=True),
            output_options=OutputOptions(
                compression=compression.value,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        console.print("[green]✓ Compression completed successfully![/green]")
        console.print(result.stats)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error during compression: {e}[/red]")
        return 1

    return 0
