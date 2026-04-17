"""Split command - divide MCAP files into multiple output segments."""

from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console

from pymcap_cli.cmd._run_processor import resolve_overwrite_policy
from pymcap_cli.cmd._run_processor_multi import run_processor_multi
from pymcap_cli.core.mcap_processor import OutputOptions
from pymcap_cli.core.processors import (
    DurationSplitProcessor,
    TimestampSplitProcessor,
)
from pymcap_cli.types.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    NoClobberOption,
)
from pymcap_cli.utils import parse_time_arg

console = Console()

# Parameter groups
SPLIT_GROUP = Group("Split Mode")
OUTPUT_GROUP = Group("Output Options")


def _parse_duration(value: str) -> int:
    """Parse duration string to nanoseconds. Supports s, m, h suffixes or raw ns."""
    value = value.strip().lower()
    if value.endswith("ns"):
        return int(value[:-2])
    if value.endswith("us"):
        return int(value[:-2]) * 1_000
    if value.endswith("ms"):
        return int(value[:-2]) * 1_000_000
    if value.endswith("s"):
        return int(value[:-1]) * 1_000_000_000
    if value.endswith("m"):
        return int(value[:-1]) * 60 * 1_000_000_000
    if value.endswith("h"):
        return int(value[:-1]) * 3600 * 1_000_000_000
    # Plain integer = nanoseconds
    return int(value)


def split(
    file: str,
    *,
    duration: Annotated[
        str | None,
        Parameter(
            name=["--duration"],
            group=SPLIT_GROUP,
            help="Split every N time units (e.g. 60s, 5m, 1h) or nanoseconds",
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
    output_template: Annotated[
        str,
        Parameter(
            name=["-t", "--output-template"],
            group=OUTPUT_GROUP,
            help="Output file naming template (e.g. 'output_{index:03d}.mcap')",
        ),
    ] = "output_{index:03d}.mcap",
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
) -> int:
    """Split an MCAP file into multiple output segments.

    Supports duration-based splitting (every N seconds/minutes/hours),
    timestamp-based splitting (at specific points), or both combined.

    Parameters
    ----------
    file
        Path to the MCAP file to split (local file or HTTP/HTTPS URL).
    duration
        Split interval, e.g. "60s", "5m", "1h", or raw nanoseconds.
    split_at
        Timestamps at which to split (ns integer or RFC3339 format).
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
    ```
    """
    if not duration and not split_at:
        console.print(
            "[red]Error: Specify --duration and/or --split-at to define split points.[/red]"
        )
        return 1

    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        console.print("[red]Error: --force and --no-clobber cannot be used together.[/red]")
        return 1

    # Parse split-at timestamps
    split_points: list[int] = []
    if split_at:
        for ts in split_at:
            try:
                split_points.append(parse_time_arg(ts))
            except ValueError as e:
                console.print(f"[red]Error parsing timestamp '{ts}': {e}[/red]")
                return 1

    # Build split processors
    processors = []
    if duration:
        try:
            duration_ns = _parse_duration(duration)
        except ValueError as e:
            console.print(f"[red]Error parsing duration '{duration}': {e}[/red]")
            return 1
        if duration_ns <= 0:
            console.print("[red]Error: Duration must be positive.[/red]")
            return 1
        processors.append(DurationSplitProcessor(duration_ns))
        console.print(f"[dim]Duration split: every {duration} ({duration_ns:,} ns)[/dim]")

    if split_points:
        processors.append(TimestampSplitProcessor(split_points))
        console.print(f"[dim]Timestamp split: {len(split_points)} point(s)[/dim]")

    # Display split mode
    if len(processors) > 1:
        console.print("[dim]Mode: Duration + Timestamp split[/dim]")
    elif duration:
        console.print("[dim]Mode: Duration split[/dim]")
    else:
        console.print("[dim]Mode: Timestamp split[/dim]")

    result = run_processor_multi(
        files=[file],
        output_options=OutputOptions(
            processors=processors,
            output_template=output_template,
            compression=compression.value,
            chunk_size=chunk_size,
            overwrite_policy=overwrite_policy,
        ),
    )

    # Report results
    console.print("[green]✓ Splitting completed successfully![/green]")
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

    return 0


def bytes_to_human(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"
