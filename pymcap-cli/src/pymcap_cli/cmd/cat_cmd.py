"""Cat command for pymcap-cli - stream MCAP messages to stdout."""

import base64
import json
import re
import sys
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import IO, TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from small_mcap.reader import DecodedMessage

from cyclopts import Group, Parameter
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

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.utils import MAX_INT64, ProgressTrackingIO, file_progress, parse_timestamp_args

console_err = Console(stderr=True)
console_out = Console()

FILTERING_GROUP = Group("Filtering")
OUTPUT_GROUP = Group("Output")

_TTY_BYTES_TRUNCATE = 32


class BytesMode(str, Enum):
    """How to serialize bytes fields in JSON output."""

    INTS = "ints"
    BASE64 = "base64"
    SKIP = "skip"


def message_to_dict(
    obj: Any,
    *,
    bytes_mode: BytesMode = BytesMode.INTS,
    truncate_bytes: int = 0,
) -> Any:
    """Recursively convert a message object to a JSON-serializable dict.

    Handles __slots__-based objects, sequences, and bytes-like objects.
    The ``truncate_bytes`` parameter is used for TTY display to keep output manageable.
    """
    recurse = lambda v: message_to_dict(v, bytes_mode=bytes_mode, truncate_bytes=truncate_bytes)  # noqa: E731

    if hasattr(obj, "__slots__"):
        return {
            slot: recurse(getattr(obj, slot, None))
            for slot in obj.__slots__
            if not slot.startswith("_")
        }

    if isinstance(obj, (list, tuple)):
        return [recurse(item) for item in obj]

    if isinstance(obj, (bytes, bytearray, memoryview)):
        total = len(obj)
        if bytes_mode == BytesMode.SKIP:
            return f"<{total} bytes>"
        if bytes_mode == BytesMode.BASE64:
            return base64.b64encode(bytes(obj)).decode("ascii")
        if truncate_bytes and total > truncate_bytes:
            return [*list(obj[:truncate_bytes]), f"... ({total} bytes total)"]
        return list(obj)

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
        ),
    ] = BytesMode.INTS,
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
    # Default end time to max if not specified
    if end_time_ns is None:
        end_time_ns = MAX_INT64

    # Parse message path query if provided
    parsed_query = None
    if query:
        try:
            parsed_query = parse_message_path(query)
        except Exception as e:  # noqa: BLE001
            console_err.print(f"[red]Invalid query syntax: {e}[/red]")
            return 1

    # Determine output mode
    writing_to_file = output is not None
    is_tty = not writing_to_file and sys.stdout.isatty()

    message_count = 0
    validated_topics: set[str] = set()

    def should_include_message(
        channel: Channel,
        _schema: Schema | None,
    ) -> bool:
        """Check if a message should be included based on filters."""
        topic = channel.topic

        if parsed_query:
            if topic != parsed_query.topic:
                return False
        elif topics and not any(re.search(pattern, topic) for pattern in topics):
            return False

        return not (exclude_topics and any(re.search(pattern, topic) for pattern in exclude_topics))

    def _validate_query(msg_schema_name: str, msg_schema_data: bytes, topic: str) -> int | None:
        """Validate query against schema. Returns 1 on error, None on success."""
        try:
            all_definitions = parse_schema_to_definitions(msg_schema_name, msg_schema_data)
            root_msgdef = all_definitions.get(msg_schema_name)
            if root_msgdef is None:
                parts = msg_schema_name.split("/")
                root_msgdef = all_definitions.get(f"{parts[0]}/{parts[-1]}")

            if root_msgdef is None:
                console_err.print(
                    f"[yellow]Warning: Could not find message definition "
                    f"for schema '{msg_schema_name}'[/yellow]"
                )
            else:
                parsed_query.validate(root_msgdef, all_definitions)  # type: ignore[union-attr]
        except ValidationError as e:
            console_err.print(f"[red]Query validation error for topic '{topic}':[/red]")
            console_err.print(f"[red]{e}[/red]")
            console_err.print(
                f"\n[yellow]Query:[/yellow] {query}\n[yellow]Schema:[/yellow] {msg_schema_name}"
            )
            return 1
        return None

    def _to_jsonl(msg: "DecodedMessage", data: Any) -> str:
        """Serialize a decoded message to a compact JSON line."""
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
                progress = file_progress("[bold blue]Reading MCAP...", console_err)
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
                        err = _validate_query(msg.schema.name, msg.schema.data, msg.channel.topic)
                        if err:
                            return err

                # Apply query filter
                if parsed_query:
                    try:
                        data = parsed_query.apply(msg.decoded_message)
                        if data is None:
                            continue
                    except MessagePathError as e:
                        console_err.print(
                            f"[yellow]Filter error on {msg.channel.topic}: {e}[/yellow]",
                        )
                        continue
                else:
                    data = msg.decoded_message

                if is_tty:
                    json_str = json.dumps(
                        message_to_dict(
                            data,
                            bytes_mode=bytes_mode,
                            truncate_bytes=_TTY_BYTES_TRUNCATE,
                        ),
                        indent=2,
                    )

                    header = Text()
                    header.append(msg.channel.topic, style="bold cyan")
                    header.append(" @ ", style="dim")
                    header.append(str(msg.message.log_time), style="green")
                    header.append(" [", style="dim")
                    header.append(msg.schema.name if msg.schema else "unknown", style="yellow")
                    header.append("]", style="dim")

                    console_out.print(
                        Panel(
                            JSON(json_str),
                            title=header,
                            border_style="blue",
                            expand=False,
                        )
                    )
                else:
                    line = _to_jsonl(msg, data)
                    if out_file is not None:
                        out_file.write(line + "\n")
                    else:
                        print(line, file=sys.stdout)  # noqa: T201

        if writing_to_file:
            console_err.print(
                f"Wrote [bold]{message_count:,}[/bold] messages to [cyan]{output}[/cyan]"
            )

        if parsed_query and not validated_topics:
            console_err.print(
                f"[red]Error: Topic '{parsed_query.topic}' not found in MCAP file[/red]"
            )
            return 1

    except KeyboardInterrupt:
        console_err.print("\n[yellow]Interrupted by user[/yellow]")
        return 0

    except Exception as e:  # noqa: BLE001
        console_err.print(f"[red]Error reading MCAP: {e}[/red]")
        return 1

    return 0
