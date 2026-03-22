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

if TYPE_CHECKING:
    from rich.console import Console
    from small_mcap import McapWriter
    from small_mcap.reader import DecodedMessage


def create_progress(console: Console, *, title: str) -> Progress:
    """Create a rich progress bar with the standard column layout."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        OSCProgressColumn(title=title),
        console=console,
    )


def get_total_message_count(file: str) -> int | None:
    """Read MCAP summary and return total message count, or None."""
    from small_mcap import get_summary  # noqa: PLC0415

    from pymcap_cli.core.input_handler import open_input  # noqa: PLC0415

    with open_input(file) as (f, _file_size):
        if (
            (summary := get_summary(f))
            and summary.statistics
            and summary.statistics.channel_message_counts
        ):
            return sum(summary.statistics.channel_message_counts.values())
    return None


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


def print_size_comparison(console: Console, input_size: int, output_size: int) -> None:
    """Print input/output file size comparison."""
    if input_size > 0:
        ratio = output_size / input_size
        console.print(f"\n[cyan]Input size:[/cyan] {input_size / 1024 / 1024:.1f} MB")
        console.print(f"[cyan]Output size:[/cyan] {output_size / 1024 / 1024:.1f} MB")
        reduction_pct = (1 - ratio) * 100
        if reduction_pct > 0:
            console.print(f"[green]Reduction:[/green] {reduction_pct:.1f}%")
        else:
            console.print(f"[yellow]Size change:[/yellow] {-reduction_pct:.1f}% increase")
    else:
        console.print(f"\n[cyan]Output size:[/cyan] {output_size / 1024 / 1024:.1f} MB")
