"""TF tree command - display transform tree from MCAP file."""

import logging
import sys
from typing import Annotated

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Console
from rich.live import Live
from small_mcap import get_summary, include_topics, read_message_decoded

from pymcap_cli.cmd._cli_options import (
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    StartTimeOption,
    StaticOnlyOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.tf_findings import collect_tf_findings, has_error_findings
from pymcap_cli.core.tf_tree import TF_STATIC_TOPIC, TF_TOPIC, TfGraph, add_tf_message
from pymcap_cli.display.tf_render import TF_COMPACT_WIDTH, build_findings_table, build_tf_table

logger = logging.getLogger(__name__)
console = Console()

DISPLAY_GROUP = Group("Display")


def tftree(
    file: str,
    *,
    static_only: StaticOnlyOption = False,
    change_only: Annotated[
        bool,
        Parameter(
            name=["--change-only"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
) -> int:
    """Display TF transform tree from MCAP file.

    Parameters
    ----------
    file
        Path to MCAP file (local file or HTTP/HTTPS URL).
    static_only
        Show only static transforms (/tf_static).
    change_only
        Update display only when tree structure changes (new frames added).
    """
    graph = TfGraph()

    topics = [TF_STATIC_TOPIC]
    if not static_only:
        topics.append(TF_TOPIC)

    try:
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    try:
        with open_input(file) as (f, _file_size), Live(console=console) as live:
            resolved_filter = message_filter.resolve(get_summary(f))
            for msg in read_message_decoded(
                f,
                should_include=message_filter.create_channel_predicate(include_topics(topics)),
                decoder_factories=[DecoderFactory()],
                start_time_ns=resolved_filter.start_time_ns,
                end_time_ns=(
                    sys.maxsize if resolved_filter.early_bail else resolved_filter.end_time_ns
                ),
            ):
                if (
                    resolved_filter.early_bail
                    and msg.message.log_time >= resolved_filter.end_time_ns
                ):
                    break
                tree_changed = add_tf_message(graph, msg.channel.topic, msg.decoded_message)

                compact = console.width < TF_COMPACT_WIDTH
                if not change_only or tree_changed:
                    table = build_tf_table(graph.transforms, graph.counts, compact=compact)
                    if table:
                        live.update(table)

            table = build_tf_table(
                graph.transforms, graph.counts, compact=console.width < TF_COMPACT_WIDTH
            )
            if table:
                live.update(table)

        findings = collect_tf_findings(graph)
        if findings:
            console.print()
            console.print(build_findings_table(findings))

    except (OSError, ValueError, RuntimeError):
        logger.exception("Error reading MCAP file")
        return 1

    return 1 if has_error_findings(findings) else 0
