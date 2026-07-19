"""Cat command for pymcap-cli - stream MCAP messages to stdout."""

import json
import logging
import re
import sys
from contextlib import ExitStack
from typing import IO, TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from small_mcap import DecodedMessage

from cyclopts import Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import Schema
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from ros_parser.message_path import NO_OUTPUT, MessagePathError, MessagePathEvaluator
from small_mcap import Channel, JSONDecoderFactory, get_summary, read_message_decoded

from pymcap_cli.cmd._cli_options import (
    FILTERING_GROUP,
    MESSAGE_PATH_GROUP,
    VAR_REQUIRES_QUERY,
    BytesModeOption,
    ChangedOption,
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    FlatOption,
    GrepIgnoreCaseOption,
    GrepOption,
    MessageLimitOption,
    MessagePathVariablesOption,
    OptionalOutputPathOption,
    QueryOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.cmd._message_path_options import create_message_path_variables
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.display.cat_helpers import (
    SchemaCache,
    parse_cat_queries,
    plan_for_query,
    query_result_is_empty,
)
from pymcap_cli.display.message_render import (
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


def cat(
    file: str,
    *,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    query: Annotated[QueryOption, Parameter(group=[FILTERING_GROUP, VAR_REQUIRES_QUERY])] = None,
    var: Annotated[
        MessagePathVariablesOption, Parameter(group=[MESSAGE_PATH_GROUP, VAR_REQUIRES_QUERY])
    ] = None,
    grep: GrepOption = None,
    grep_ignore_case: GrepIgnoreCaseOption = False,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    limit: MessageLimitOption = None,
    output: OptionalOutputPathOption = None,
    bytes_mode: BytesModeOption = BytesMode.SMART,
    flat: FlatOption = False,
    changed: ChangedOption = False,
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

      # Cross-message stream transforms (one value per message)
      pymcap-cli cat recording.mcap -q '/odom.pose.position.x.@@delta'

      # Cross-message reducers (one value at end of stream)
      pymcap-cli cat recording.mcap -q '/imu.linear_acceleration.@norm.@@max'

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

    try:
        parsed_queries = parse_cat_queries(query)
        stream_evaluators = {
            topic: MessagePathEvaluator(parsed.path)
            for topic, parsed in parsed_queries.items()
            if parsed.path.has_stream
        }
    except Exception:
        logger.exception("Invalid query syntax")
        return 1

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

                query_for_topic = parsed_queries.get(msg.channel.topic)
                parsed_query = query_for_topic.path if query_for_topic is not None else None

                # Validate query against schema on first message of each topic
                if query_for_topic is not None and msg.channel.topic not in validated_topics:
                    validated_topics.add(msg.channel.topic)

                    if msg.schema is None:
                        logger.warning(
                            f"Cannot validate query for topic '{msg.channel.topic}' "
                            "(no schema available)"
                        )
                    elif not schema_cache.validate_query(
                        query_for_topic.path,
                        msg.schema,
                        msg.channel.topic,
                        query_repr=query_for_topic.source,
                    ):
                        return 1

                # Apply query filter
                if parsed_query is not None:
                    evaluator = stream_evaluators.get(msg.channel.topic)
                    try:
                        if evaluator is not None:
                            data = evaluator.observe(
                                msg.decoded_message, msg.message.log_time, variables
                            )
                            if data is NO_OUTPUT:
                                continue
                        else:
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

            # Stream reducers (@@count, @@max, ...) emit one value at end of stream
            for stream_topic, evaluator in stream_evaluators.items():
                try:
                    final = evaluator.finalize(variables)
                except MessagePathError as e:
                    logger.warning(f"Filter error on {stream_topic}: {e}")
                    continue
                if final is NO_OUTPUT:
                    continue
                source = parsed_queries[stream_topic].source
                value = message_to_dict(final, bytes_mode=bytes_mode)
                if is_tty:
                    reduced_line = Text()
                    reduced_line.append(source, style="bold cyan")
                    reduced_line.append(" = ", style="dim")
                    reduced_line.append(json.dumps(value), style="green")
                    console_out.print(reduced_line)
                else:
                    reduced_entry = json.dumps(
                        {"topic": stream_topic, "query": source, "value": value},
                        separators=(",", ":"),
                    )
                    if out_file is not None:
                        out_file.write(reduced_entry + "\n")
                    else:
                        print(reduced_entry, file=sys.stdout)  # noqa: T201

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
