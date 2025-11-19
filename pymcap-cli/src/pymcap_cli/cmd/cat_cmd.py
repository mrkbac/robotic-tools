"""Cat command for pymcap-cli - stream MCAP messages to stdout."""

import json
import re
import sys
from typing import Annotated, Any

from cyclopts import Group, Parameter
from lark.exceptions import LarkError
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import Schema
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.text import Text
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import MessagePathError, ValidationError, parse_message_path
from small_mcap import JSONDecoderFactory
from small_mcap.reader import read_message_decoded
from small_mcap.records import Channel

from pymcap_cli.input_handler import open_input
from pymcap_cli.mcap_processor import parse_timestamp_args

console_err = Console(stderr=True)  # Use stderr for errors
console_out = Console()  # Use stdout for data output

# Parameter groups
FILTERING_GROUP = Group("Filtering")
OUTPUT_GROUP = Group("Output")


def message_to_dict(obj: Any) -> Any:
    """Recursively convert a message object to a JSON-serializable dict.

    Handles:
    - Dataclass objects with __slots__ → dict
    - Lists/tuples → lists
    - bytes/bytearray/memoryview → list of ints
    - Other types → as-is
    """
    # Handle dataclass-like objects with __slots__
    if hasattr(obj, "__slots__"):
        result = {}
        for slot in obj.__slots__:
            # Skip private/internal fields
            if slot.startswith("_"):
                continue
            value = getattr(obj, slot, None)
            result[slot] = message_to_dict(value)
        return result

    # Handle sequences
    if isinstance(obj, (list, tuple)):
        return [message_to_dict(item) for item in obj]

    # Handle bytes-like objects as list of integers
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return list(obj)

    # Handle other standard types (int, float, str, bool, None)
    return obj


def truncate_for_display(obj: Any, max_bytes: int = 100, max_array: int = 5) -> Any:
    """Truncate large data structures for display in table mode.

    Args:
        obj: Object to truncate
        max_bytes: Maximum number of bytes to show for byte arrays
        max_array: Maximum number of array elements to show

    Returns:
        Truncated representation suitable for display
    """
    # Handle dataclass-like objects with __slots__
    if hasattr(obj, "__slots__"):
        result = {}
        for slot in obj.__slots__:
            if slot.startswith("_"):
                continue
            value = getattr(obj, slot, None)
            result[slot] = truncate_for_display(value, max_bytes, max_array)
        return result

    # Handle bytes-like objects - show size and preview
    if isinstance(obj, (bytes, bytearray, memoryview)):
        size = len(obj)
        if size > max_bytes:
            preview = list(obj[:max_bytes])
            return f"<{size} bytes: {preview}...>"
        return list(obj)

    # Handle sequences - truncate long arrays
    if isinstance(obj, (list, tuple)):
        if len(obj) > max_array:
            truncated = [
                truncate_for_display(item, max_bytes, max_array) for item in obj[:max_array]
            ]
            return [*truncated, f"... ({len(obj) - max_array} more items)"]
        return [truncate_for_display(item, max_bytes, max_array) for item in obj]

    return obj


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
            name=["-n", "--exclude-topics"],
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
) -> None:
    """Stream MCAP messages to stdout.

    Decodes ROS2 messages and outputs as JSON. When output is to a terminal (TTY),
    displays messages in a Rich table. When piped, outputs JSONL (one JSON per line).

    Examples:
      # Display messages in a table (interactive)
      pymcap-cli cat recording.mcap

      # Pipe to file as JSONL
      pymcap-cli cat recording.mcap > messages.jsonl

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
    """

    start_time_ns = parse_timestamp_args(start, 0, start_secs)
    end_time_ns = parse_timestamp_args(end, 0, end_secs)
    # Default end time to max if not specified
    if end_time_ns == 0:
        end_time_ns = 2**63 - 1

    # Parse message path query if provided
    parsed_query = None
    if query:
        try:
            parsed_query = parse_message_path(query)
        except LarkError as e:
            console_err.print(f"[red]Invalid query syntax: {e}[/red]")
            sys.exit(1)

    # Always decode messages
    decoder_factories = [JSONDecoderFactory(), DecoderFactory()]

    # Detect if output is to a TTY (terminal) or piped
    is_tty = sys.stdout.isatty()

    message_count = 0
    validated_topics: set[str] = set()  # Track which topics have been validated

    def should_include_message(
        channel: Channel,
        _schema: Schema | None,
    ) -> bool:
        """Check if a message should be included based on filters."""
        topic = channel.topic

        # If query is specified, filter by query topic (smart filtering)
        if parsed_query:
            if topic != parsed_query.topic:
                return False
        elif topics and not any(re.search(pattern, topic) for pattern in topics):
            # Otherwise use explicit topic filters
            return False

        return not (exclude_topics and any(re.search(pattern, topic) for pattern in exclude_topics))

    try:
        with open_input(file) as (input_stream, _):
            for msg in read_message_decoded(
                input_stream,
                decoder_factories=decoder_factories,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                should_include=should_include_message,
            ):
                # Check limit
                if limit is not None and message_count >= limit:
                    break
                message_count += 1

                # Validate query against schema on first message of each topic
                if parsed_query and msg.channel.topic not in validated_topics:
                    validated_topics.add(msg.channel.topic)

                    if msg.schema is None:
                        console_err.print(
                            f"[yellow]Warning: Cannot validate query for topic "
                            f"'{msg.channel.topic}' (no schema available)[/yellow]"
                        )
                    else:
                        try:
                            # Parse schema into message definitions
                            all_definitions = parse_schema_to_definitions(
                                msg.schema.name, msg.schema.data
                            )

                            # Get the root message definition
                            root_msgdef = all_definitions.get(msg.schema.name)
                            if root_msgdef is None:
                                # Try short name
                                short_name = "/".join(
                                    [msg.schema.name.split("/")[0], msg.schema.name.split("/")[-1]]
                                )
                                root_msgdef = all_definitions.get(short_name)

                            if root_msgdef is None:
                                console_err.print(
                                    f"[yellow]Warning: Could not find message definition "
                                    f"for schema '{msg.schema.name}'[/yellow]"
                                )
                            else:
                                # Validate the query against the schema
                                parsed_query.validate(root_msgdef, all_definitions)

                        except ValidationError as e:
                            console_err.print(
                                f"[red]Query validation error for topic "
                                f"'{msg.channel.topic}':[/red]"
                            )
                            console_err.print(f"[red]{e}[/red]")
                            console_err.print(
                                f"\n[yellow]Query:[/yellow] {query}\n"
                                f"[yellow]Schema:[/yellow] {msg.schema.name}"
                            )
                            sys.exit(1)

                # Filter data if query is specified
                if parsed_query:
                    try:
                        data = parsed_query.apply(msg.decoded_message)
                        # If filter returned None, skip this message
                        if data is None:
                            continue
                    except MessagePathError as e:
                        console_err.print(
                            f"[yellow]Filter error on {msg.channel.topic}: {e}[/yellow]",
                        )
                        continue
                else:
                    data = msg.decoded_message

                # Output data (pretty or raw)
                if is_tty:
                    # Pretty output with Rich
                    truncated = truncate_for_display(data)
                    json_str = json.dumps(message_to_dict(truncated), indent=2)

                    header = Text()
                    header.append(msg.channel.topic, style="bold cyan")
                    header.append(" @ ", style="dim")
                    header.append(str(msg.message.log_time), style="green")
                    header.append(" [", style="dim")
                    schema_name = msg.schema.name if msg.schema else "unknown"
                    header.append(schema_name, style="yellow")
                    header.append("]", style="dim")

                    panel = Panel(
                        JSON(json_str),
                        title=header,
                        border_style="blue",
                        expand=False,
                    )
                    console_out.print(panel)
                else:
                    # Raw JSONL output
                    output = {
                        "topic": msg.channel.topic,
                        "sequence": msg.message.sequence,
                        "log_time": msg.message.log_time,
                        "publish_time": msg.message.publish_time,
                    }
                    if msg.schema:
                        output["schema"] = msg.schema.name
                    output["message"] = message_to_dict(data)

                    print(json.dumps(output, separators=(",", ":")), file=sys.stdout)  # noqa: T201

    except KeyboardInterrupt:
        # Allow graceful exit with Ctrl+C
        console_err.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)

    except Exception as e:  # noqa: BLE001
        console_err.print(f"[red]Error reading MCAP: {e}[/red]")
        sys.exit(1)
