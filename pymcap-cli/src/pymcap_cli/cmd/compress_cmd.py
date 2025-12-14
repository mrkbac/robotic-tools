"""Compress command for pymcap-cli."""

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
from pymcap_cli.utils import confirm_output_overwrite

console = Console()


def compress(
    file: str,
    output: OutputPathOption,
    *,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
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

    Examples
    --------
    ```
    pymcap-cli compress in.mcap -o out.mcap
    ```
    """
    # Confirm overwrite if needed
    confirm_output_overwrite(output, force)

    console.print(f"[blue]Compressing '{file}' to '{output}'[/blue]")

    with open_input(file) as (f, file_size), output.open("wb") as output_stream:
        # Build input options - include everything, force decode for re-compression
        input_opts = InputOptions(
            stream=f,
            file_size=file_size,
            always_decode_chunk=True,  # Force decode for compression change
        )

        # Build processing options
        processing_options = ProcessingOptions(
            inputs=[input_opts],
            output=OutputOptions(
                compression=compression.value,
                chunk_size=chunk_size,
            ),
        )

        processor = McapProcessor(processing_options)

        try:
            stats = processor.process(output_stream)

            console.print("[green]✓ Compression completed successfully![/green]")
            console.print(stats)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during compression: {e}[/red]")
            return 1

    return 0
