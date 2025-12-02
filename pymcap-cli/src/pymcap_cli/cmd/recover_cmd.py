from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console

from pymcap_cli.input_handler import open_input
from pymcap_cli.mcap_processor import (
    MAX_INT64,
    McapProcessor,
    ProcessingOptions,
    confirm_output_overwrite,
)
from pymcap_cli.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
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

    Examples
    --------
    ```
    pymcap-cli recover in.mcap -o out.mcap
    ```
    """
    # Confirm overwrite if needed
    confirm_output_overwrite(output, force)

    # Convert recover options to unified processing options
    processing_options = ProcessingOptions(
        # Recovery mode enabled with all content included
        recovery_mode=True,
        always_decode_chunk=always_decode_chunk,
        # No filtering - include everything
        include_topics=[],
        exclude_topics=[],
        start_time=0,
        end_time=MAX_INT64,
        include_metadata=True,
        include_attachments=True,
        # Output options
        compression=compression.value,
        chunk_size=chunk_size,
    )

    # Create processor and run
    processor = McapProcessor(processing_options)

    with open_input(file) as (f, file_size), output.open("wb") as output_stream:
        try:
            stats = processor.process([f], output_stream, [file_size])

            console.print("[green]✓ Recovery completed successfully![/green]")
            console.print(stats)

        except RuntimeError as e:
            if "Writer not started" in str(e):
                # Empty file case - this is expected for empty/corrupt files
                console.print(
                    "[yellow]Warning: File appears to be empty or severely corrupted[/yellow]"
                )
                console.print("No valid MCAP data found to recover")
            else:
                console.print(f"[red]Error during recovery: {e}[/red]")
                return 1
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during recovery: {e}[/red]")
            return 1

    return 0
