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
from small_mcap import get_summary

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.display.osc_utils import OSCProgressColumn
from pymcap_cli.log_setup import ERR, OUT

if TYPE_CHECKING:
    from collections.abc import Callable

    from small_mcap import Channel, DecodedMessage, McapWriter, Schema, Summary


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


def get_total_message_count(
    file: str,
    should_include: Callable[[Channel, Schema | None], bool] | None = None,
) -> int | None:
    """Read the MCAP summary and return the total message count, or None.

    Thin wrapper over :func:`count_included_messages` for callers that only
    need the count and have no summary in hand.
    """
    with open_input(file) as (f, _file_size):
        summary = get_summary(f)
    return count_included_messages(summary, should_include)


def ensure_schema(
    writer: McapWriter,
    schema_name: str,
    encoding: str,
    data: bytes,
    schema_ids: dict[str, int],
) -> int:
    """Register a schema if not already registered and return its ID."""
    if schema_name not in schema_ids:
        sid = max(schema_ids.values(), default=0) + 1
        writer.add_schema(sid, schema_name, encoding, data)
        schema_ids[schema_name] = sid
    return schema_ids[schema_name]


def ensure_channel(
    writer: McapWriter,
    topic: str,
    message_encoding: str,
    schema_id: int,
    channel_ids: dict[str, int],
    metadata: dict[str, str] | None = None,
) -> int:
    """Register a channel if not already registered and return its ID."""
    if topic not in channel_ids:
        cid = max(channel_ids.values(), default=0) + 1
        writer.add_channel(
            channel_id=cid,
            topic=topic,
            message_encoding=message_encoding,
            schema_id=schema_id,
            metadata=metadata,
        )
        channel_ids[topic] = cid
    return channel_ids[topic]


def copy_message(
    msg: DecodedMessage,
    writer: McapWriter,
    schema_ids: dict[str, int],
    channel_ids: dict[str, int],
) -> None:
    """Copy a message unchanged to the output writer, registering schema/channel as needed."""
    topic = msg.channel.topic
    if topic not in channel_ids:
        if msg.schema:
            schema_id = ensure_schema(
                writer, msg.schema.name, msg.schema.encoding, msg.schema.data, schema_ids
            )
        else:
            schema_id = 0
        ensure_channel(
            writer,
            topic,
            msg.channel.message_encoding,
            schema_id,
            channel_ids,
            msg.channel.metadata,
        )

    writer.add_message(
        channel_id=channel_ids[topic],
        log_time=msg.message.log_time,
        data=msg.message.data,
        publish_time=msg.message.publish_time,
    )


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
