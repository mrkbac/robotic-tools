"""Cat command for pymcap-cli - stream MCAP messages to stdout."""

import json
import re
import sys
from typing import Annotated, Any

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Console
from small_mcap.reader import DecodedMessage, read_message_decoded

from pymcap_cli.input_handler import open_input
from pymcap_cli.mcap_processor import AttachmentsMode, MetadataMode, build_processing_options

console = Console(stderr=True)  # Use stderr for errors, stdout for data

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


def format_message_json(msg: DecodedMessage) -> str:
    """Format a decoded message as a single-line JSON object."""
    output = {
        "topic": msg.channel.topic,
        "sequence": msg.message.sequence,
        "log_time": msg.message.log_time,
        "publish_time": msg.message.publish_time,
    }

    # Add schema info if available
    if msg.schema:
        output["schema"] = msg.schema.name

    # Add decoded message if available
    if msg.decoded_message is not None:
        output["message"] = message_to_dict(msg.decoded_message)

    return json.dumps(output, separators=(",", ":"))


def format_message_text(msg: DecodedMessage) -> str:
    """Format a message as human-readable text."""
    topic = msg.channel.topic
    log_time = msg.message.log_time
    schema = msg.schema.name if msg.schema else "unknown"

    return f"{topic} [{log_time}] ({schema})"


def should_include_message(
    msg: DecodedMessage,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    start_ns: int,
    end_ns: int,
) -> bool:
    """Check if a message should be included based on filters."""
    # Check time range
    if start_ns > 0 and msg.message.log_time < start_ns:
        return False
    if end_ns > 0 and msg.message.log_time >= end_ns:
        return False

    topic = msg.channel.topic

    # Check topic filters
    if include_patterns and not any(re.search(pattern, topic) for pattern in include_patterns):
        return False

    return not (exclude_patterns and any(re.search(pattern, topic) for pattern in exclude_patterns))


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
    json_output: Annotated[
        bool,
        Parameter(
            name=["--json"],
            group=OUTPUT_GROUP,
        ),
    ] = False,
    limit: Annotated[
        int | None,
        Parameter(
            name=["-l", "--limit"],
            group=OUTPUT_GROUP,
        ),
    ] = None,
) -> None:
    """Stream MCAP messages to stdout.

    By default, prints topic, timestamp, and schema for each message.
    With --json, decodes ROS2 messages and outputs full message content as JSON.

    Examples:
      # Print all messages
      pymcap-cli cat recording.mcap

      # Output specific topics as JSON
      pymcap-cli cat recording.mcap --topics /camera/image --json

      # Filter by time range
      pymcap-cli cat recording.mcap --start-secs 10 --end-secs 20

      # Limit output
      pymcap-cli cat recording.mcap --limit 100
    """
    # Parse time filters
    try:
        options = build_processing_options(
            include_topic_regex=None,  # We'll handle filtering manually
            exclude_topic_regex=None,
            start=start,
            start_nsecs=0,
            start_secs=start_secs,
            end=end,
            end_nsecs=0,
            end_secs=end_secs,
            metadata_mode=MetadataMode.EXCLUDE,
            attachments_mode=AttachmentsMode.EXCLUDE,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    start_ns = options.start_time
    end_ns = options.end_time

    # Setup decoder factory if JSON output requested
    decoder_factories = [DecoderFactory()] if json_output else []

    message_count = 0

    try:
        with open_input(file) as (input_stream, _):
            for msg in read_message_decoded(input_stream, decoder_factories=decoder_factories):
                # Apply filters
                if not should_include_message(msg, topics, exclude_topics, start_ns, end_ns):
                    continue

                # Format and output message
                if json_output:
                    # Check if message was decoded (only CDR messages with
                    # DecoderFactory will be decoded)
                    if msg.decoded_message is None:
                        # Skip non-CDR messages when JSON output requested
                        console.print(
                            f"[yellow]Warning: Skipping message on {msg.channel.topic} "
                            f"(encoding '{msg.channel.message_encoding}' not supported "
                            f"for JSON output)[/yellow]"
                        )
                        continue
                    output_line = format_message_json(msg)
                else:
                    output_line = format_message_text(msg)

                # Write to stdout
                print(output_line, file=sys.stdout)  # noqa: T201

                message_count += 1

                # Check limit
                if limit is not None and message_count >= limit:
                    break

    except KeyboardInterrupt:
        # Allow graceful exit with Ctrl+C
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)

    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error reading MCAP: {e}[/red]")
        sys.exit(1)
