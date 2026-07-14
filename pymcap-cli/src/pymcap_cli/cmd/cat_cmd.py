"""Cat command for pymcap-cli - stream MCAP messages to stdout."""

import json
import logging
import re
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import IO, TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from small_mcap import DecodedMessage

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import Schema
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from ros_parser.message_path import (
    MessagePath,
    MessagePathError,
    parse_message_path,
)
from small_mcap import Channel, JSONDecoderFactory, get_summary, read_message_decoded

from pymcap_cli.cmd._message_filter_options import (
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    StartTimeOption,
    TopicOption,
    create_message_filter,
)
from pymcap_cli.cmd._message_path_options import (
    MessagePathVariablesOption,
    create_message_path_variables,
)
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.display.cat_helpers import SchemaCache, plan_for_query, query_result_is_empty
from pymcap_cli.display.message_render import (
    SMART_BYTES_INLINE_LIMIT,
    TTY_BYTES_TRUNCATE,
    BytesMode,
    changed_leaf_paths,
    message_matches_grep,
    message_to_dict,
    render_message_flat,
    render_message_tree,
)
from pymcap_cli.utils import ProgressTrackingIO, file_progress

logger = logging.getLogger(__name__)
console_out = Console()

FILTERING_GROUP = Group("Filtering")
OUTPUT_GROUP = Group("Output")


def cat(
    file: str,
    *,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    query: Annotated[
        list[str] | None,
        Parameter(
            name=["-q", "--query"],
            group=FILTERING_GROUP,
            help=(
                "MessagePath expression scoping output to one topic and/or subfield. "
                "Repeat for additional topics."
            ),
        ),
    ] = None,
    var: MessagePathVariablesOption = None,
    grep: Annotated[
        str | None,
        Parameter(
            name=["-g", "--grep"],
            group=FILTERING_GROUP,
            help=(
                "Regex applied to every scalar value in the decoded message. "
                "Messages with no match are skipped. Bytes-like fields are not "
                "searched. Composes with --query: the regex runs on the post-"
                "query result so '--query <path> --grep <re>' scopes the search."
            ),
        ),
    ] = None,
    grep_ignore_case: Annotated[
        bool,
        Parameter(
            name=["-i", "--grep-ignore-case"],
            group=FILTERING_GROUP,
        ),
    ] = False,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    limit: Annotated[
        int | None,
        Parameter(
            name=["-l", "--limit"],
            group=OUTPUT_GROUP,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        Parameter(
            name=["-o", "--output"],
            group=OUTPUT_GROUP,
        ),
    ] = None,
    bytes_mode: Annotated[
        BytesMode,
        Parameter(
            name=["--bytes"],
            group=OUTPUT_GROUP,
            help=(
                "How to render `bytes` fields in JSON output. `smart` (default) "
                f"inlines payloads ≤{SMART_BYTES_INLINE_LIMIT} bytes as int lists "
                "and collapses larger ones to `<N bytes>` so `cat` stays readable "
                "on messages with Image/PointCloud2 payloads. Use `ints` for the "
                "full int list, `base64` for a compact serialisable string, or "
                "`skip` to always drop the payload."
            ),
        ),
    ] = BytesMode.SMART,
    flat: Annotated[
        bool,
        Parameter(
            name=["--flat"],
            group=OUTPUT_GROUP,
            help=(
                "In a terminal, print one `dotted.path: value` line per leaf "
                "instead of a tree. Greppable, and handy with --query."
            ),
        ),
    ] = False,
    changed: Annotated[
        bool,
        Parameter(
            name=["--changed"],
            group=OUTPUT_GROUP,
            help=(
                "In a terminal, highlight values that changed since the previous "
                "message on the same topic. The full message is still shown."
            ),
        ),
    ] = False,
) -> int:
    """Stream MCAP messages to stdout.

    Decodes ROS2 messages and outputs as JSON. When output is to a terminal (TTY),
    displays messages in a Rich table. When piped, outputs JSONL (one JSON per line).

    Examples:
      # Display messages in a table (interactive)
      pymcap-cli cat recording.mcap

      # Pipe to file as JSONL
      pymcap-cli cat recording.mcap > messages.jsonl

      # Write to file with progress bar
      pymcap-cli cat recording.mcap -o messages.jsonl

      # Filter specific topics
      pymcap-cli cat recording.mcap --topic /camera/image

      # Filter by time range
      pymcap-cli cat recording.mcap --start @10s --end @20s

      # Limit output
      pymcap-cli cat recording.mcap --limit 100

      # Query specific field using message path
      pymcap-cli cat recording.mcap --query '/odom.pose.position.x'

      # Query fields from multiple topics
      pymcap-cli cat recording.mcap -q '/odom.pose' -q '/imu.angular_velocity'

      # Filter array elements
      pymcap-cli cat recording.mcap --query '/detections.objects[:]{confidence>0.8}'

      # Skip binary data (images, pointclouds)
      pymcap-cli cat recording.mcap --bytes skip

      # Base64-encode binary data
      pymcap-cli cat recording.mcap --bytes base64
    """

    try:
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
        variables = create_message_path_variables(var) if query or var else {}
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    parsed_queries: dict[str, MessagePath] = {}
    query_reprs: dict[str, str] = {}
    for query_repr in query or []:
        try:
            parsed_query = parse_message_path(query_repr)
        except Exception:
            logger.exception("Invalid query syntax")
            return 1
        if parsed_query.topic in parsed_queries:
            logger.error(
                f"Only one --query per topic is supported; "
                f"topic '{parsed_query.topic}' was specified more than once"
            )
            return 1
        parsed_queries[parsed_query.topic] = parsed_query
        query_reprs[parsed_query.topic] = query_repr

    grep_pattern: re.Pattern[str] | None = None
    if grep:
        try:
            grep_pattern = re.compile(grep, re.IGNORECASE if grep_ignore_case else 0)
        except re.error:
            logger.exception("Invalid --grep regex")
            return 1

    writing_to_file = output is not None
    is_tty = not writing_to_file and sys.stdout.isatty()

    message_count = 0
    validated_topics: set[str] = set()
    summary_query_topics: set[str] | None = None
    schema_cache = SchemaCache()
    previous_by_topic: dict[str, Any] = {}

    def base_predicate(
        channel: Channel,
        _schema: Schema | None,
    ) -> bool:
        return not parsed_queries or channel.topic in parsed_queries

    should_include_message = message_filter.create_channel_predicate(base_predicate)

    def _to_jsonl(msg: "DecodedMessage", data: Any) -> str:
        entry: dict[str, Any] = {
            "topic": msg.channel.topic,
            "sequence": msg.message.sequence,
            "log_time": msg.message.log_time,
            "publish_time": msg.message.publish_time,
        }
        if msg.schema:
            entry["schema"] = msg.schema.name
        entry["message"] = message_to_dict(data, bytes_mode=bytes_mode)
        return json.dumps(entry, separators=(",", ":"))

    try:
        with open_input(file) as (input_stream, file_size), ExitStack() as stack:
            try:
                summary = get_summary(input_stream)
                resolved_filter = message_filter.resolve(summary)
            except ValueError as exc:
                logger.error(str(exc))  # noqa: TRY400
                return 1
            if summary is not None:
                available_topics = {channel.topic for channel in summary.channels.values()}
                summary_query_topics = parsed_queries.keys() & available_topics
            stream: IO[bytes] = input_stream
            if writing_to_file and file_size:
                progress = file_progress("[bold blue]Reading MCAP...")
                progress.start()
                stack.callback(progress.stop)
                task = progress.add_task("Processing", total=file_size)
                stream = ProgressTrackingIO(input_stream, task, progress, input_stream.tell())

            out_file = stack.enter_context(output.open("w")) if output else None

            for msg in read_message_decoded(
                stream,
                decoder_factories=[JSONDecoderFactory(), DecoderFactory()],
                start_time_ns=resolved_filter.start_time_ns,
                end_time_ns=(
                    sys.maxsize if resolved_filter.early_bail else resolved_filter.end_time_ns
                ),
                should_include=should_include_message,
            ):
                if (
                    resolved_filter.early_bail
                    and msg.message.log_time >= resolved_filter.end_time_ns
                ):
                    break
                if limit is not None and message_count >= limit:
                    break

                parsed_query = parsed_queries.get(msg.channel.topic)

                # Validate query against schema on first message of each topic
                if parsed_query is not None and msg.channel.topic not in validated_topics:
                    validated_topics.add(msg.channel.topic)

                    if msg.schema is None:
                        logger.warning(
                            f"Cannot validate query for topic '{msg.channel.topic}' "
                            "(no schema available)"
                        )
                    elif not schema_cache.validate_query(
                        parsed_query,
                        msg.schema,
                        msg.channel.topic,
                        query_repr=query_reprs[msg.channel.topic],
                    ):
                        return 1

                # Apply query filter
                if parsed_query is not None:
                    try:
                        data = parsed_query.apply(msg.decoded_message, variables)
                        if query_result_is_empty(data):
                            continue
                    except MessagePathError as e:
                        logger.warning(f"Filter error on {msg.channel.topic}: {e}")
                        continue
                else:
                    data = msg.decoded_message

                if grep_pattern is not None and not message_matches_grep(data, grep_pattern):
                    continue

                message_count += 1

                if is_tty:
                    schema = msg.schema
                    header = Text()
                    header.append(msg.channel.topic, style="bold cyan")
                    header.append(" @ ", style="dim")
                    header.append(str(msg.message.log_time), style="green")
                    header.append(" [", style="dim")
                    header.append(schema.name if schema else "unknown", style="yellow")
                    header.append("]", style="dim")

                    root_plan = schema_cache.enum_plan(schema) if schema is not None else None
                    plan = plan_for_query(root_plan, parsed_query)

                    changed_paths = None
                    if changed:
                        current_topic = msg.channel.topic
                        previous = previous_by_topic.get(current_topic)
                        changed_paths = (
                            changed_leaf_paths(previous, data) if previous is not None else None
                        )
                        previous_by_topic[current_topic] = data

                    if flat:
                        console_out.print(header)
                        for flat_line in render_message_flat(
                            data,
                            plan,
                            bytes_mode=bytes_mode,
                            truncate_bytes=TTY_BYTES_TRUNCATE,
                            changed_paths=changed_paths,
                        ):
                            console_out.print(flat_line)
                    else:
                        tree = render_message_tree(
                            data,
                            plan,
                            title=header,
                            bytes_mode=bytes_mode,
                            truncate_bytes=TTY_BYTES_TRUNCATE,
                            changed_paths=changed_paths,
                        )
                        console_out.print(Panel(tree, border_style="blue", expand=False))
                else:
                    line = _to_jsonl(msg, data)
                    if out_file is not None:
                        out_file.write(line + "\n")
                    else:
                        print(line, file=sys.stdout)  # noqa: T201

        if writing_to_file:
            logger.info(f"Wrote {message_count:,} messages to {output}")

        found_query_topics = (
            summary_query_topics if summary_query_topics is not None else validated_topics
        )
        missing_query_topics = parsed_queries.keys() - found_query_topics
        if missing_query_topics:
            if len(missing_query_topics) == 1:
                missing_topic = next(iter(missing_query_topics))
                logger.error(f"Topic '{missing_topic}' not found in MCAP file")
            else:
                topics = ", ".join(f"'{topic}'" for topic in sorted(missing_query_topics))
                logger.error(f"Topics {topics} not found in MCAP file")
            return 1

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0

    except Exception:
        logger.exception("Error reading MCAP")
        return 1

    return 0
