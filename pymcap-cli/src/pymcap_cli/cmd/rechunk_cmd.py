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
from pymcap_cli.types.size import parse_size_bytes
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
    ] = RechunkStrategy.ALL,
    pattern: Annotated[
        list[str] | None,
        Parameter(
            name=["-p", "--pattern", "--rechunk-pattern"],
            group=STRATEGY_GROUP,
        ),
    ] = None,
    schema_pattern: Annotated[
        list[str] | None,
        Parameter(
            name=["--schema-pattern", "--rechunk-schema-pattern"],
            group=STRATEGY_GROUP,
            help=(
                "Regex pattern matched against Schema.name (e.g. "
                "'sensor_msgs/.*Image.*'). Repeatable; participates in the "
                "same first-match-wins chain as --pattern, evaluated after "
                "topic patterns. Streaming-safe — schema is known at channel "
                "registration time."
            ),
        ),
    ] = None,
    max_groups: Annotated[
        int | None,
        Parameter(
            name=["--max-groups", "--rechunk-max-groups"],
            group=STRATEGY_GROUP,
            help=(
                "Cap on concurrent chunk groups per output segment. When the "
                "cap is hit, further channels share the most-recently-created "
                "group as an overflow pool."
            ),
        ),
    ] = None,
    max_memory: Annotated[
        str | None,
        Parameter(
            name=["--max-memory", "--rechunk-max-memory"],
            group=STRATEGY_GROUP,
            help=(
                "Cap on total uncompressed bytes buffered across all chunk "
                "groups in a segment (e.g. '256MB', '1GiB'). When exceeded, "
                "the largest in-flight chunk is flushed prematurely. "
                "Streaming-safe and orthogonal to --max-groups."
            ),
        ),
    ] = None,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
) -> int:
    """Reorganize MCAP messages into chunks based on topic or schema patterns.

    Rechunk MCAP files by organizing messages into separate chunk groups
    based on topic / schema regex patterns. Each pattern creates a separate
    chunk group, with unmatched topics going into their own group. Messages
    are written in a streaming fashion as they are read.

    Parameters
    ----------
    file
        Path to the MCAP file to rechunk (local file or HTTP/HTTPS URL).
    output
        Output filename.
    strategy
        Rechunking strategy: ``none`` (fast-copy when possible),
        ``pattern`` (group by topic/schema regex), ``all`` (each topic separate).
        Defaults to ``all``.
    pattern
        Regex pattern matched against ``Channel.topic`` for grouping
        (repeatable; only with ``--strategy=pattern``). Topics matching the
        first pattern go into chunk group 1, second → group 2, etc.
    schema_pattern
        Regex pattern matched against ``Schema.name`` (repeatable). Evaluated
        after ``--pattern``; first match across both lists wins.
    max_groups
        Hard cap on concurrent chunk groups per output segment. Overflow
        channels share the last-created group.
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
    # Pattern-based grouping by topic
    pymcap-cli rechunk in.mcap --strategy pattern -p '/camera.*' -p '/lidar.*' -o out.mcap

    # Group every image channel into its own chunk regardless of topic name
    pymcap-cli rechunk in.mcap --strategy pattern \\
        --schema-pattern 'sensor_msgs/.*Image.*' -o out.mcap

    # Each topic in its own chunk, but cap memory at 8 concurrent groups
    pymcap-cli rechunk in.mcap --strategy all --max-groups 8 -o out.mcap
    ```
    """
    # Validate pattern is provided when using PATTERN strategy
    if strategy == RechunkStrategy.PATTERN and not pattern and not schema_pattern:
        logger.error("--strategy=pattern requires at least one --pattern or --schema-pattern.")
        return 1
    if max_groups is not None and max_groups < 1:
        logger.error("--max-groups must be >= 1.")
        return 1

    max_memory_bytes: int | None = None
    if max_memory is not None:
        try:
            max_memory_bytes = parse_size_bytes(max_memory)
        except ValueError:
            logger.exception(f"Error parsing --max-memory {max_memory!r}")
            return 1
        if max_memory_bytes < 1:
            logger.error("--max-memory must be >= 1 byte.")
            return 1

    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        logger.error("--force and --no-clobber cannot be used together.")
        return 1

    output_file = Path(output)

    # Compile patterns if using PATTERN strategy
    patterns: list[Pattern[str]] = []
    schema_patterns: list[Pattern[str]] = []

    if strategy == RechunkStrategy.NONE:
        logger.info("Strategy: None — preserving input chunking via fast-copy where possible")
    elif strategy == RechunkStrategy.ALL:
        logger.info("Strategy: Each topic in its own chunk group")
    else:  # PATTERN
        logger.info("Strategy: Pattern-based grouping (topic + schema)")
        try:
            patterns = compile_topic_patterns(pattern or [])
            schema_patterns = compile_topic_patterns(schema_pattern or [])
        except ValueError as e:
            logger.error(str(e))  # noqa: TRY400
            return 1

    try:
        result = run_processor(
            files=[file],
            output=output_file,
            input_options=InputOptions.from_args(),
            output_options=OutputOptions(
                rechunk_strategy=strategy,
                rechunk_patterns=patterns,
                rechunk_schema_patterns=schema_patterns,
                rechunk_max_groups=max_groups,
                rechunk_max_memory=max_memory_bytes,
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
    elif strategy == RechunkStrategy.PATTERN and (patterns or schema_patterns):
        console.print(
            f"Used {len(patterns)} topic pattern(s) and "
            f"{len(schema_patterns)} schema pattern(s) for grouping"
        )

    if delete_source:
        return finalize_delete_source(sources=[file], outputs=[output_file])

    return 0
