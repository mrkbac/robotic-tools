"""Rechunk command - reorganize MCAP messages into chunks based on topic patterns."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Group, Parameter
from rich.console import Console

from pymcap_cli.input_handler import open_input
from pymcap_cli.mcap_processor import (
    McapProcessor,
    ProcessingOptions,
    RechunkStrategy,
    compile_topic_patterns,
    confirm_output_overwrite,
)
from pymcap_cli.types import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    OutputPathOption,
)

if TYPE_CHECKING:
    from re import Pattern

console = Console()

# Parameter groups
STRATEGY_GROUP = Group("Rechunking Strategy")


def rechunk(
    file: str,
    output: OutputPathOption,
    *,
    strategy: Annotated[
        RechunkStrategy,
        Parameter(
            name=["-s", "--strategy"],
            group=STRATEGY_GROUP,
        ),
    ] = RechunkStrategy.AUTO,
    pattern: Annotated[
        list[str] | None,
        Parameter(
            name=["-p", "--pattern"],
            group=STRATEGY_GROUP,
        ),
    ] = None,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
) -> None:
    """Reorganize MCAP messages into chunks based on topic patterns.

    Rechunk MCAP files by organizing messages into separate chunk groups
    based on topic regex patterns. Each pattern creates a separate chunk group,
    with unmatched topics going into their own group. Messages are written
    in a streaming fashion as they are read.

    Parameters
    ----------
    file
        Path to the MCAP file to rechunk (local file or HTTP/HTTPS URL).
    output
        Output filename.
    strategy
        Rechunking strategy: pattern (group by regex), all (each topic separate),
        or auto (size-based grouping).
    pattern
        Regex pattern for topic grouping (can be used multiple times, only with
        --strategy=pattern). Topics matching the first pattern go into chunk group 1,
        second pattern → group 2, etc. Unmatched topics → separate group.
    chunk_size
        Chunk size of output file in bytes.
    compression
        Compression algorithm for output file.
    force
        Force overwrite of output file without confirmation.

    Examples
    --------
    ```
    # Pattern-based grouping
    pymcap-cli rechunk in.mcap --strategy pattern -p '/camera.*' -p '/lidar.*' -o out.mcap

    # Auto-grouping by size (topics >15% get own chunk)
    pymcap-cli rechunk in.mcap --strategy auto -o out.mcap

    # Each topic in separate chunk
    pymcap-cli rechunk in.mcap --strategy all -o out.mcap

    # Custom chunk size with pattern strategy
    pymcap-cli rechunk in.mcap --strategy pattern -p '/high_freq.*' --chunk-size 8388608 -o out.mcap
    ```
    """
    # Validate pattern is provided when using PATTERN strategy
    if strategy == RechunkStrategy.PATTERN and not pattern:
        console.print(
            "[red]Error: --strategy=pattern requires at least one regex pattern. "
            "Use -p PATTERN[/red]"
        )
        sys.exit(1)

    output_file = Path(output)

    # Confirm overwrite if needed
    confirm_output_overwrite(output_file, force)

    # Compile patterns if using PATTERN strategy
    patterns: list[Pattern[str]] = []

    if strategy == RechunkStrategy.ALL:
        console.print("[dim]Strategy: Each topic in its own chunk group[/dim]")
    elif strategy == RechunkStrategy.AUTO:
        console.print("[dim]Strategy: Auto-grouping based on size (>15% threshold)[/dim]")
    else:  # PATTERN
        console.print("[dim]Strategy: Pattern-based grouping[/dim]")
        try:
            patterns = compile_topic_patterns(pattern or [])
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

        if not patterns:
            console.print(
                "[yellow]Warning: No patterns specified. "
                "All messages will be in one group.[/yellow]"
            )

    # Build processing options with rechunking enabled
    options = ProcessingOptions(
        rechunk_strategy=strategy,
        rechunk_patterns=patterns,
        compression=compression.value,
        chunk_size=chunk_size,
        recovery_mode=True,  # Always enable recovery mode
    )

    # Create processor and run
    processor = McapProcessor(options)

    with open_input(file) as (input_stream, file_size), output_file.open("wb") as output_stream:
        stats = processor.process([input_stream], output_stream, [file_size])

        # Report results
        console.print("[green]✓ Rechunking completed successfully![/green]")
        console.print(
            f"Processed {stats.messages_processed:,} messages, "
            f"wrote {stats.writer_statistics.message_count:,} messages"
        )

        # Strategy-specific stats
        num_unique_groups = len(processor.rechunk_groups)
        if strategy == RechunkStrategy.ALL:
            console.print(f"Created {num_unique_groups} chunk group(s) (one per topic)")
        elif strategy == RechunkStrategy.AUTO:
            console.print(
                f"Created {num_unique_groups} chunk group(s) "
                f"({len(processor.large_channels)} large, rest shared)"
            )
        elif strategy == RechunkStrategy.PATTERN and patterns:
            console.print(f"Used {len(patterns)} topic pattern(s) for grouping")
