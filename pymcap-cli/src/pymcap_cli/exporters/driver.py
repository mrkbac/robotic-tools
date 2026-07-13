"""Generic export driver.

Iterates decoded messages from an MCAP file and dispatches them to per-topic
:class:`~pymcap_cli.exporters.base.TopicWriter` instances created by the
selected :class:`~pymcap_cli.exporters.base.Exporter`.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from small_mcap import get_summary, read_message_decoded

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.mcap_transform import count_included_messages, create_progress
from pymcap_cli.core.message_filter import MessageFilterOptions
from pymcap_cli.exporters._common import (
    make_should_include,
    unique_topic_filename,
)
from pymcap_cli.exporters._summary_hints import warn_topic_coverage
from pymcap_cli.exporters.base import TopicContext

if TYPE_CHECKING:
    from pathlib import Path

    from pymcap_cli.exporters.base import Exporter, TopicWriter

logger = logging.getLogger(__name__)


def run_export(
    *,
    file: str,
    output: str | Path | None,
    exporter: Exporter,
    message_filter: MessageFilterOptions | None = None,
    required_topics: list[str] | None = None,
    force: bool = False,
    num_workers: int = 8,
) -> int:
    """Drive an export run end-to-end.

    Returns the process exit code (0 on success).

    ``output`` may be ``None`` for exporters whose
    :meth:`Exporter.validate_output` accepts that — they manage the output
    location internally (e.g. plot's optional ``--output``: when omitted,
    the figure is shown interactively).

    ``message_filter`` is the canonical topic/time selection shared by every
    file-reading command. ``required_topics`` is an additional command-owned
    restriction, used by plot paths and other semantic readers.
    """
    filters = message_filter or MessageFilterOptions()
    should_include = make_should_include(
        message_filter=filters,
        accepts_schema=exporter.accepts,
        required_topics=required_topics,
    )

    # Read the summary exactly once, then derive both the progress total and
    # the pre-scan coverage warnings from it.
    try:
        with open_input(file) as (stream, _size):
            summary = get_summary(stream)
    except (FileNotFoundError, OSError):
        logger.exception(f"Error reading {file}")
        return 1

    try:
        resolved_filter = filters.resolve(summary)
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    resolved_output = exporter.validate_output(output, force=force)
    if resolved_output is None:
        return 1

    total = count_included_messages(summary, should_include)
    warn_topic_coverage(summary, file, None)

    writers: dict[int, TopicWriter] = {}
    used_filenames: set[str] = set()
    counts: dict[int, int] = {}
    error_count = 0

    logger.info(f"Input: {file}")
    logger.info(f"Output: {resolved_output}")
    logger.info(f"Format: {exporter.name}")

    exporter.setup(resolved_output)

    read_buffer_bytes = 4 * 1024 * 1024
    try:
        with (
            open_input(file, buffering=read_buffer_bytes) as (stream, _size),
            create_progress(title=f"Exporting to {exporter.name}") as progress,
        ):
            task_id = progress.add_task("Processing messages", total=total)

            reader_end = sys.maxsize if resolved_filter.early_bail else resolved_filter.end_time_ns
            for msg in read_message_decoded(
                stream,
                should_include=should_include,
                decoder_factories=exporter.decoder_factories(),
                num_workers=num_workers,
                start_time_ns=resolved_filter.start_time_ns,
                end_time_ns=reader_end,
            ):
                log_time = msg.message.log_time
                if resolved_filter.early_bail and log_time >= resolved_filter.end_time_ns:
                    break
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
                    logger.warning(f"failed to write message on {topic}: {exc}")
                else:
                    counts[writer_key] = counts.get(writer_key, 0) + 1
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                error_count += 1
                logger.warning(f"writer close failed: {exc}")
        try:
            exporter.finish(resolved_output, counts)
        except Exception as exc:  # noqa: BLE001
            error_count += 1
            logger.warning(f"exporter finish failed: {exc}")

    if not writers:
        logger.warning("No messages exported (no matching topics).")
        return 1

    written = sum(counts.values())
    if error_count:
        logger.warning(
            f"Exported {written} messages across {len(writers)} topic(s) "
            f"to {resolved_output} — {error_count} message(s) failed."
        )
        return 1
    logger.info(
        f"[green]Exported {written} messages[/green] across {len(writers)} "
        f"topic(s) to {resolved_output}"
    )
    return 0
