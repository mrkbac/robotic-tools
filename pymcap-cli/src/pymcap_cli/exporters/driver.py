"""Generic export driver.

Iterates decoded messages from an MCAP file and dispatches them to per-topic
:class:`~pymcap_cli.exporters.base.TopicWriter` instances created by the
selected :class:`~pymcap_cli.exporters.base.Exporter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from small_mcap import read_message_decoded

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.mcap_transform import create_progress, get_total_message_count
from pymcap_cli.exporters._common import (
    make_should_include,
    unique_topic_filename,
)
from pymcap_cli.exporters.base import TopicContext
from pymcap_cli.utils import MAX_INT64

if TYPE_CHECKING:
    from pathlib import Path

    from pymcap_cli.exporters.base import Exporter, TopicWriter


def run_export(
    *,
    file: str,
    output: str | Path | None,
    exporter: Exporter,
    topics: list[str] | None = None,
    force: bool = False,
    num_workers: int = 8,
    start_time_ns: int = 0,
    end_time_ns: int | None = None,
    console: Console | None = None,
) -> int:
    """Drive an export run end-to-end.

    Returns the process exit code (0 on success).

    ``output`` may be ``None`` for exporters whose
    :meth:`Exporter.validate_output` accepts that — they manage the output
    location internally (e.g. plot's optional ``--output``: when omitted,
    the figure is shown interactively).

    ``start_time_ns`` / ``end_time_ns`` filter messages at the reader level.
    ``end_time_ns=None`` means "no upper bound".
    """
    console = console or Console()

    resolved_output = exporter.validate_output(output, force=force, console=console)
    if resolved_output is None:
        return 1

    should_include = make_should_include(topics=topics, accepts_schema=exporter.accepts)

    try:
        total = get_total_message_count(file)
    except (FileNotFoundError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    writers: dict[int, TopicWriter] = {}
    used_filenames: set[str] = set()
    counts: dict[int, int] = {}
    error_count = 0

    console.print(f"[cyan]Input:[/cyan] {file}")
    console.print(f"[cyan]Output:[/cyan] {resolved_output}")
    console.print(f"[cyan]Format:[/cyan] {exporter.name}")

    exporter.setup(console, resolved_output)

    read_buffer_bytes = 4 * 1024 * 1024
    try:
        with (
            open_input(file, buffering=read_buffer_bytes) as (stream, _size),
            create_progress(console, title=f"Exporting to {exporter.name}") as progress,
        ):
            task_id = progress.add_task("Processing messages", total=total)

            for msg in read_message_decoded(
                stream,
                should_include=should_include,
                decoder_factories=exporter.decoder_factories(),
                num_workers=num_workers,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns if end_time_ns is not None else MAX_INT64,
            ):
                progress.advance(task_id)
                topic = msg.channel.topic
                writer_key = int(msg.channel.id)

                writer = writers.get(writer_key)
                if writer is None:
                    safe = unique_topic_filename(topic, used_filenames)
                    used_filenames.add(safe)
                    ctx = TopicContext(
                        topic=topic,
                        schema=msg.schema,
                        channel=msg.channel,
                        writer_key=writer_key,
                        output_path=resolved_output,
                        safe_filename=safe,
                        force=force,
                    )
                    writer = exporter.open_topic(ctx)
                    writers[writer_key] = writer

                try:
                    writer.write(msg)
                except Exception as exc:  # noqa: BLE001
                    error_count += 1
                    console.print(
                        f"[yellow]Warning:[/yellow] failed to write message on {topic}: {exc}"
                    )
                else:
                    counts[writer_key] = counts.get(writer_key, 0) + 1
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                error_count += 1
                console.print(f"[yellow]Warning:[/yellow] writer close failed: {exc}")
        try:
            exporter.finish(console, resolved_output, counts)
        except Exception as exc:  # noqa: BLE001
            error_count += 1
            console.print(f"[yellow]Warning:[/yellow] exporter finish failed: {exc}")

    if not writers:
        console.print("[yellow]No messages exported (no matching topics).[/yellow]")
        return 1

    written = sum(counts.values())
    if error_count:
        console.print(
            f"[yellow]Exported {written} messages across {len(writers)} topic(s) "
            f"to {resolved_output} — {error_count} message(s) failed.[/yellow]"
        )
        return 1
    console.print(
        f"[green]Exported {written} messages[/green] across {len(writers)} "
        f"topic(s) to {resolved_output}"
    )
    return 0
