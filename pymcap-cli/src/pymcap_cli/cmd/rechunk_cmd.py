"""Rechunk command - reorganize MCAP messages into chunks based on topic patterns."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Group, Parameter
from rich.console import Console

from pymcap_cli.cmd._run_processor import (
    finalize_delete_source,
    resolve_overwrite_policy,
    run_processor,
)
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    RechunkStrategy,
)
from pymcap_cli.types.types_manual import (
    ChunkSizeOption,
    CompressionOption,
    DeleteSourceOption,
    ForceOverwriteOption,
    NoClobberOption,
    OutputPathOption,
)
from pymcap_cli.utils import compile_topic_patterns

if TYPE_CHECKING:
    from re import Pattern

logger = logging.getLogger(__name__)
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
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
) -> int:
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
    no_clobber
        Fail instead of prompting if the output file already exists.
    delete_source
        Delete source file(s) after the output is validated (header + summary).
        URL inputs and any source whose path equals the output are skipped.

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
        logger.error("--strategy=pattern requires at least one regex pattern. Use -p PATTERN")
        return 1

    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        logger.error("--force and --no-clobber cannot be used together.")
        return 1

    output_file = Path(output)

    # Compile patterns if using PATTERN strategy
    patterns: list[Pattern[str]] = []

    if strategy == RechunkStrategy.ALL:
        logger.info("Strategy: Each topic in its own chunk group")
    elif strategy == RechunkStrategy.AUTO:
        logger.info("Strategy: Auto-grouping based on size (>15% threshold)")
    else:  # PATTERN
        logger.info("Strategy: Pattern-based grouping")
        try:
            patterns = compile_topic_patterns(pattern or [])
        except ValueError as e:
            logger.error(str(e))  # noqa: TRY400
            return 1

        if not patterns:
            logger.warning("No patterns specified. All messages will be in one group.")

    try:
        result = run_processor(
            files=[file],
            output=output_file,
            input_options=InputOptions.from_args(),
            output_options=OutputOptions(
                rechunk_strategy=strategy,
                rechunk_patterns=patterns,
                compression=compression,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
    except Exception:
        logger.exception("Error during rechunking")
        return 1

    # Report results
    logger.info("[green]✓ Rechunking completed successfully![/green]")
    console.print(
        f"Processed {result.stats.messages_processed:,} messages, "
        f"wrote {result.stats.writer_statistics.message_count:,} messages"
    )

    # Strategy-specific stats
    assert result.processor.output_manager is not None
    num_unique_groups = sum(
        len(segment.rechunk_groups) for segment in result.processor.output_manager.segments.values()
    )
    if strategy == RechunkStrategy.ALL:
        console.print(f"Created {num_unique_groups} chunk group(s) (one per topic)")
    elif strategy == RechunkStrategy.AUTO:
        console.print(
            f"Created {num_unique_groups} chunk group(s) "
            f"({len(result.processor.large_channels)} large, rest shared)"
        )
    elif strategy == RechunkStrategy.PATTERN and patterns:
        console.print(f"Used {len(patterns)} topic pattern(s) for grouping")

    if delete_source:
        return finalize_delete_source(sources=[file], outputs=[output_file])

    return 0
