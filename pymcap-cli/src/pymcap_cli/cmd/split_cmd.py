"""Split command - divide MCAP files into multiple output segments."""

import logging
from pathlib import Path
from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console
from ros_parser.message_path import MessagePathError

from pymcap_cli.cmd._run_processor import finalize_delete_source, resolve_overwrite_policy
from pymcap_cli.cmd._run_processor_multi import run_processor_multi
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.mcap_processor import InputOptions, OutputOptions
from pymcap_cli.core.processors.duration_split import DurationSplitProcessor
from pymcap_cli.core.processors.expression_split import ExpressionSplitProcessor
from pymcap_cli.core.processors.timestamp_split import TimestampSplitProcessor
from pymcap_cli.types.duration import parse_duration_ns
from pymcap_cli.types.types_manual import (
    ChunkSizeOption,
    CompressionOption,
    DeleteSourceOption,
    ForceOverwriteOption,
    NoClobberOption,
)
from pymcap_cli.utils import parse_time_arg

logger = logging.getLogger(__name__)
console = Console()

# Parameter groups
SPLIT_GROUP = Group("Split Mode")
OUTPUT_GROUP = Group("Output Options")
LATCHING_GROUP = Group("Latching")


def split(
    file: str,
    *,
    duration: Annotated[
        str | None,
        Parameter(
            name=["--duration"],
            group=SPLIT_GROUP,
            help="Split every N time units (e.g. 60s, 1.5m, 1h); bare numbers are seconds",
        ),
    ] = None,
    split_at: Annotated[
        list[str] | None,
        Parameter(
            name=["--split-at"],
            group=SPLIT_GROUP,
            help="Split at specific timestamps (ns or RFC3339, repeatable)",
        ),
    ] = None,
    expression: Annotated[
        str | None,
        Parameter(
            name=["-E", "--expression"],
            group=SPLIT_GROUP,
            help=(
                "Split whenever a ros-parser message path changes value, e.g. "
                "'/gps/fix.status.status' (value-change trigger) or "
                "'/detections.objects[:]{confidence>0.8}' (predicate trigger: "
                "match/no-match transitions). Segments are numbered — use "
                "'{index:03d}' in --output-template. Messages on other topics "
                "follow the current segment (sticky). Chunks with no "
                "target-topic messages fast-copy without decoding."
            ),
        ),
    ] = None,
    output_template: Annotated[
        str,
        Parameter(
            name=["-t", "--output-template"],
            group=OUTPUT_GROUP,
            help="Output file naming template (e.g. 'output_{index:03d}.mcap')",
        ),
    ] = "output_{index:03d}.mcap",
    latch: Annotated[
        list[str] | None,
        Parameter(
            name=["--latch"],
            group=LATCHING_GROUP,
            help=(
                "Topic regex (repeatable) whose latest message is replayed into "
                "every output segment. Use for /tf_static and other transient-local "
                "topics that consumers need at the start of each segment."
            ),
        ),
    ] = None,
    latch_from_metadata: Annotated[
        bool,
        Parameter(
            name=["--latch-from-metadata"],
            group=LATCHING_GROUP,
            help=(
                "Also auto-detect latched channels by reading the MCAP "
                "'offered_qos_profiles' metadata for durability=transient_local."
            ),
        ),
    ] = False,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
) -> int:
    """Split an MCAP file into multiple output segments.

    Supports duration-based splitting (every N seconds/minutes/hours),
    timestamp-based splitting (at specific points), or both combined.

    Parameters
    ----------
    file
        Path to the MCAP file to split (local file or HTTP/HTTPS URL).
    duration
        Split interval, e.g. "60s", "1.5m", "1h"; bare numbers are seconds.
    split_at
        Timestamps at which to split (ns integer or RFC3339 format).
    expression
        ros-parser message path; each distinct value becomes a segment.
    output_template
        Python format string for output filenames. Available variables:
        {index}, {index1}, {key}, {start_time}, {start_time_iso}, {end_time}.
    chunk_size
        Chunk size of output file in bytes.
    compression
        Compression algorithm for output file.
    force
        Force overwrite of output files without confirmation.
    no_clobber
        Fail instead of prompting if any split output path already exists.
    delete_source
        Delete the source file after every split output is validated
        (header + summary). URL inputs and any source whose path equals one
        of the outputs are skipped.

    Examples
    --------
    ```
    # Split every 60 seconds
    pymcap-cli split input.mcap --duration 60s

    # Split every 5 minutes with custom naming
    pymcap-cli split input.mcap --duration 5m --output-template "seg_{index:03d}.mcap"

    # Split at specific timestamps
    pymcap-cli split input.mcap --split-at 1000000000 --split-at 2000000000

    # Split at RFC3339 timestamps
    pymcap-cli split input.mcap --split-at "2024-01-15T10:00:00Z"

    # Split by a message-path expression (new segment on each value change)
    pymcap-cli split input.mcap --expression '/gps/fix.status.status' \\
        -t 'gps_{index:03d}.mcap'

    # Filter-triggered split: new segment when confidence>0.8 turns on/off
    pymcap-cli split input.mcap \\
        --expression '/detections.objects[:]{confidence>0.8}' \\
        -t 'hits_{index:03d}.mcap'
    ```
    """
    if not duration and not split_at and not expression:
        logger.error("Specify --duration, --split-at and/or --expression to define split points.")
        return 1

    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        logger.error("--force and --no-clobber cannot be used together.")
        return 1

    # Parse split-at timestamps. Relative anchors are not supported here.
    split_points: list[int] = []
    if split_at:
        for ts in split_at:
            try:
                split_points.append(parse_time_arg(ts))
            except ValueError:
                logger.exception(f"Error parsing timestamp '{ts}'")
                return 1

    # Build split processors
    processors = []
    if duration:
        try:
            duration_ns = parse_duration_ns(duration)
        except ValueError:
            logger.exception(f"Error parsing duration '{duration}'")
            return 1
        if duration_ns <= 0:
            logger.error("Duration must be positive.")
            return 1
        processors.append(DurationSplitProcessor(duration_ns))
        logger.info(f"Duration split: every {duration} ({duration_ns:,} ns)")

    if split_points:
        processors.append(TimestampSplitProcessor(split_points))
        logger.info(f"Timestamp split: {len(split_points)} point(s)")

    if expression:
        try:
            processors.append(ExpressionSplitProcessor(expression))
        except MessagePathError:
            logger.exception(f"Error parsing expression '{expression}'")
            return 1
        logger.info(f"Expression split: {expression}")

    # Display split mode
    modes = []
    if duration:
        modes.append("Duration")
    if split_points:
        modes.append("Timestamp")
    if expression:
        modes.append("Expression")
    logger.info(f"Mode: {' + '.join(modes)} split")

    input_options = InputOptions.from_args(
        latch_topics=latch,
        latch_from_metadata=latch_from_metadata,
    )

    try:
        result = run_processor_multi(
            files=[file],
            input_options=input_options,
            output_options=OutputOptions(
                processors=processors,
                output_template=output_template,
                compression=compression,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
    except Exception:
        logger.exception("Error during splitting")
        return 1

    # Report results
    logger.info("[green]✓ Splitting completed successfully![/green]")
    console.print(
        f"Processed {result.stats.messages_processed:,} messages, "
        f"wrote {result.stats.writer_statistics.message_count:,} messages"
    )

    # Per-segment stats
    if result.stats.writer_statistics:
        stats = result.stats.writer_statistics
        console.print(f"Time range: {stats.message_start_time:,} - {stats.message_end_time:,} ns")

    assert result.processor.output_manager is not None
    console.print(f"Created {len(result.processor.output_manager.segments)} output file(s)")
    for _, segment in sorted(
        result.processor.output_manager.segments.items(), key=lambda x: x[1].index
    ):
        seg_stats = segment.writer.statistics
        console.print(
            f"  [{segment.index}] {segment.path}: "
            f"{seg_stats.message_count:,} messages, "
            f"{bytes_to_human(seg_stats.chunk_count * chunk_size)} "
            f"({seg_stats.chunk_count} chunks)"
        )

    if delete_source:
        outputs = [
            Path(segment.path) for segment in result.processor.output_manager.segments.values()
        ]
        return finalize_delete_source(sources=[file], outputs=outputs)

    return 0


def bytes_to_human(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"
