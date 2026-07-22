"""Shared utilities for MCAP transform commands (roscompress, rosdecompress)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pymcap_cli.display.osc_utils import OSCProgressColumn
from pymcap_cli.log_setup import ERR, OUT

if TYPE_CHECKING:
    from collections.abc import Callable

    from small_mcap import Channel, Schema, Summary


def create_progress(*, title: str) -> Progress:
    """Create a rich progress bar with the standard column layout, on stderr."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        OSCProgressColumn(title=title),
        console=ERR,
    )


def count_included_messages(
    summary: Summary | None,
    should_include: Callable[[Channel, Schema | None], bool] | None = None,
) -> int | None:
    """Total message count from an already-read summary, or None.

    Returns None when the summary is missing or carries no per-channel counts.
    When ``should_include`` is provided, sum only the counts for channels whose
    ``(channel, schema)`` pair passes the predicate — matching the filter used
    by ``small_mcap.read_message_decoded`` so progress totals reflect what will
    actually be iterated.
    """
    if not (summary and summary.statistics and summary.statistics.channel_message_counts):
        return None
    counts = summary.statistics.channel_message_counts
    if should_include is None:
        return sum(counts.values())
    total = 0
    for channel_id, count in counts.items():
        channel = summary.channels.get(channel_id)
        if channel is None:
            continue
        schema = summary.schemas.get(channel.schema_id) if channel.schema_id else None
        if should_include(channel, schema):
            total += count
    return total


def print_size_comparison(input_size: int, output_size: int) -> None:
    """Print input/output file size comparison to stdout."""
    if input_size > 0:
        ratio = output_size / input_size
        OUT.print(f"\n[cyan]Input size:[/cyan] {input_size / 1024 / 1024:.1f} MB")
        OUT.print(f"[cyan]Output size:[/cyan] {output_size / 1024 / 1024:.1f} MB")
        reduction_pct = (1 - ratio) * 100
        if reduction_pct > 0:
            OUT.print(f"[green]Reduction:[/green] {reduction_pct:.1f}%")
        else:
            OUT.print(f"[yellow]Size change:[/yellow] {-reduction_pct:.1f}% increase")
    else:
        OUT.print(f"\n[cyan]Output size:[/cyan] {output_size / 1024 / 1024:.1f} MB")
