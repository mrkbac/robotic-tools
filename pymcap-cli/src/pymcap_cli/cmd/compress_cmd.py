"""Compress command for pymcap-cli."""

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

    # Convert compress args to unified processing options (include everything)
    processing_options = ProcessingOptions(
        # Recovery mode with all content included
        recovery_mode=True,
        always_decode_chunk=False,
        # No filtering - include everything
        include_topics=[],
        exclude_topics=[],
        start_time=0,
        end_time=MAX_INT64,
        include_metadata=True,
        include_attachments=True,
        # Output options with specified compression
        compression=compression.value,
        chunk_size=chunk_size,
    )
    console.print(f"[blue]Compressing '{file}' to '{output}'[/blue]")

    processor = McapProcessor(processing_options)

    with open_input(file) as (f, file_size), output.open("wb") as output_stream:
        try:
            stats = processor.process([f], output_stream, [file_size])

            console.print("[green]✓ Compression completed successfully![/green]")
            console.print(
                f"Processed {stats.messages_processed:,} messages, "
                f"wrote {stats.writer_statistics.message_count:,} messages"
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during compression: {e}[/red]")
            return 1

    return 0
