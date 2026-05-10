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
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    MessagePath,
    MessagePathError,
    ValidationError,
    parse_message_path,
)
from ros_parser.models import MessageDefinition
from small_mcap import Channel, JSONDecoderFactory, read_message_decoded

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.display.message_render import (
    SMART_BYTES_INLINE_LIMIT,
    TTY_BYTES_TRUNCATE,
    BytesMode,
    EnumPlan,
    build_enum_plan,
    message_matches_grep,
    message_to_dict,
    render_message_tree,
    resolve_msgdef_by_name,
)
from pymcap_cli.utils import MAX_INT64, ProgressTrackingIO, file_progress, parse_timestamp_args

logger = logging.getLogger(__name__)
console_out = Console()

FILTERING_GROUP = Group("Filtering")
OUTPUT_GROUP = Group("Output")


def cat(
    file: str,
    *,
    topics: Annotated[
        list[str] | None,
        Parameter(
            name=["-t", "--topics"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    exclude_topics: Annotated[
        list[str] | None,
        Parameter(
            name=["-x", "--exclude-topics", "-n"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    query: Annotated[
        str | None,
        Parameter(
            name=["-q", "--query"],
            group=FILTERING_GROUP,
        ),
    ] = None,
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
    start: Annotated[
        str,
        Parameter(
            name=["-S", "--start"],
            group=FILTERING_GROUP,
        ),
    ] = "",
    start_secs: Annotated[
        int,
        Parameter(
            name=["-s", "--start-secs"],
            group=FILTERING_GROUP,
        ),
    ] = 0,
    end: Annotated[
        str,
        Parameter(
            name=["-E", "--end"],
            group=FILTERING_GROUP,
        ),
    ] = "",
    end_secs: Annotated[
        int,
        Parameter(
            name=["-e", "--end-secs"],
            group=FILTERING_GROUP,
        ),
    ] = 0,
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
      pymcap-cli cat recording.mcap --topics /camera/image

      # Filter by time range
      pymcap-cli cat recording.mcap --start-secs 10 --end-secs 20

      # Limit output
      pymcap-cli cat recording.mcap --limit 100

      # Query specific field using message path
      pymcap-cli cat recording.mcap --query '/odom.pose.position.x'

      # Filter array elements
      pymcap-cli cat recording.mcap --query '/detections.objects[:]{confidence>0.8}'

      # Skip binary data (images, pointclouds)
      pymcap-cli cat recording.mcap --bytes skip

      # Base64-encode binary data
      pymcap-cli cat recording.mcap --bytes base64
    """

    start_time_ns = parse_timestamp_args(start, 0, start_secs) or 0
    end_time_ns = parse_timestamp_args(end, 0, end_secs)
    end_time_ns = MAX_INT64 if end_time_ns is None else end_time_ns

    parsed_query = None
    if query:
        try:
            parsed_query = parse_message_path(query)
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

    try:
        topic_patterns = [re.compile(pattern) for pattern in topics] if topics else []
        exclude_topic_patterns = (
            [re.compile(pattern) for pattern in exclude_topics] if exclude_topics else []
        )
    except re.error:
        logger.exception("Invalid topic regex")
        return 1

    writing_to_file = output is not None
    is_tty = not writing_to_file and sys.stdout.isatty()

    message_count = 0
    validated_topics: set[str] = set()
    parsed_schemas: dict[int, dict[str, MessageDefinition] | None] = {}
    enum_plans: dict[int, EnumPlan | None] = {}

    def should_include_message(
        channel: Channel,
        _schema: Schema | None,
    ) -> bool:
        topic = channel.topic

        if parsed_query:
            return topic == parsed_query.topic and not any(
                p.search(topic) for p in exclude_topic_patterns
            )
        if topic_patterns and not any(p.search(topic) for p in topic_patterns):
            return False

        return not any(p.search(topic) for p in exclude_topic_patterns)

    def _get_parsed_schema(schema: Schema) -> dict[str, MessageDefinition] | None:
        if schema.id in parsed_schemas:
            return parsed_schemas[schema.id]
        try:
            parsed = parse_schema_to_definitions(schema.name, schema.data)
        except Exception:
            logger.exception(f"Failed to parse schema '{schema.name}'")
            parsed = None
        parsed_schemas[schema.id] = parsed
        return parsed

    def _get_enum_plan(schema: Schema) -> EnumPlan | None:
        if schema.id in enum_plans:
            return enum_plans[schema.id]
        parsed = _get_parsed_schema(schema)
        plan = build_enum_plan(schema.name, parsed) if parsed else None
        enum_plans[schema.id] = plan
        return plan

    def _validate_query(query_path: MessagePath, schema: Schema, topic: str) -> int | None:
        """Validate query against schema. Returns 1 on error, None on success."""
        all_definitions = _get_parsed_schema(schema)
        if all_definitions is None:
            return None
        try:
            root_msgdef = resolve_msgdef_by_name(schema.name, all_definitions)
            if root_msgdef is None:
                logger.warning(f"Could not find message definition for schema '{schema.name}'")
            else:
                query_path.validate(root_msgdef, all_definitions)
        except ValidationError:
            logger.exception(f"Query validation error for topic '{topic}'")
            logger.exception(f"Query: {query}  Schema: {schema.name}")
            return 1
        return None

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
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                should_include=should_include_message,
            ):
                if limit is not None and message_count >= limit:
                    break

                # Validate query against schema on first message of each topic
                if parsed_query and msg.channel.topic not in validated_topics:
                    validated_topics.add(msg.channel.topic)

                    if msg.schema is None:
                        logger.warning(
                            f"Cannot validate query for topic '{msg.channel.topic}' "
                            "(no schema available)"
                        )
                    elif _validate_query(parsed_query, msg.schema, msg.channel.topic):
                        return 1

                # Apply query filter
                if parsed_query:
                    try:
                        data = parsed_query.apply(msg.decoded_message)
                        if data is None:
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

                    plan = None if parsed_query or schema is None else _get_enum_plan(schema)

                    tree = render_message_tree(
                        data,
                        plan,
                        title=header,
                        bytes_mode=bytes_mode,
                        truncate_bytes=TTY_BYTES_TRUNCATE,
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

        if parsed_query and not validated_topics:
            logger.error(f"Topic '{parsed_query.topic}' not found in MCAP file")
            return 1

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0

    except Exception:
        logger.exception("Error reading MCAP")
        return 1

    return 0
