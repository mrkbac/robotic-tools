"""Helper for parsing MCAP schema data into ROS1 message definitions."""

import re
from collections.abc import Callable

from ros_parser.models import Field, MessageDefinition, Type
from ros_parser.ros1_msg.parser import parse_message_string

# Built-in ROS1 types that may not be included in schema data
# ROS1 uses time/duration as aggregate types (unlike ROS2's builtin_interfaces)
BUILTIN_TYPES = {
    "std_msgs/Header": MessageDefinition(
        name="std_msgs/Header",
        fields_all=[
            Field(Type(type_name="uint32"), "seq"),
            Field(Type(type_name="time"), "stamp"),
            Field(Type(type_name="string"), "frame_id"),
        ],
    ),
    "time": MessageDefinition(
        name="time",
        fields_all=[
            Field(Type(type_name="uint32"), "secs"),
            Field(Type(type_name="uint32"), "nsecs"),
        ],
    ),
    "duration": MessageDefinition(
        name="duration",
        fields_all=[
            Field(Type(type_name="int32"), "secs"),
            Field(Type(type_name="int32"), "nsecs"),
        ],
    ),
}


def parse_schema_to_definitions(
    schema_name: str,
    schema_data: bytes,
) -> dict[str, MessageDefinition]:
    """Parse MCAP schema data into a dictionary of ROS1 message definitions.

    MCAP schemas for ROS1 messages contain the full message definition text,
    including all nested message types. This function parses that text and
    returns a dictionary mapping message names to their definitions.

    Args:
        schema_name: The schema name (e.g., "geometry_msgs/Pose")
        schema_data: The raw schema data bytes (will be decoded as UTF-8)

    Returns:
        Dictionary mapping message names to MessageDefinition objects.
        Keys include both full paths (e.g., "geometry_msgs/Pose")
        and short names (e.g., "Pose").
        Also includes built-in types like "std_msgs/Header", "time", "duration".

    Example:
        >>> schema_name = "geometry_msgs/Pose"
        >>> schema_data = b"Point position\\nQuaternion orientation\\n===\\n..."
        >>> definitions = parse_schema_to_definitions(schema_name, schema_data)
        >>> "geometry_msgs/Pose" in definitions
        True
        >>> "Pose" in definitions
        True
    """
    schema_text = schema_data.decode("utf-8")
    definitions: dict[str, MessageDefinition] = {}

    # Add built-in types
    definitions.update(BUILTIN_TYPES)

    # Parse all message definitions in the schema
    for_each_msgdef(
        schema_name,
        schema_text,
        lambda full_name, short_name, msgdef: _add_msgdef(
            definitions, full_name, short_name, msgdef
        ),
    )

    return definitions


def for_each_msgdef(
    schema_name: str,
    schema_text: str,
    fn: Callable[[str, str, MessageDefinition], None],
) -> None:
    """Parse schema text and call fn for each message definition found.

    MCAP schema data can contain multiple message definitions separated by "===".
    Each section may start with "MSG: package/Name" to indicate the message name.

    Args:
        schema_name: The main schema name
        schema_text: The schema text containing one or more message definitions
        fn: Callback function(full_name, short_name, msgdef) to call for each definition
    """
    cur_schema_name = schema_name

    # Remove empty lines
    schema_text = "\n".join([s for s in schema_text.splitlines() if s.strip()])

    # Split schema_text by separator lines containing at least 3 = characters
    for cur_schema_text in re.split(r"^={3,}$", schema_text, flags=re.MULTILINE):
        cur_schema_text = cur_schema_text.strip()  # noqa: PLW2901

        # Check for a "MSG: pkg_name/msg_name" line
        match = re.match(r"^MSG:\s+(\S+)$", cur_schema_text, flags=re.MULTILINE)
        if match:
            cur_schema_name = match.group(1)
            # Remove this line from the message definition
            cur_schema_text = re.sub(  # noqa: PLW2901
                r"^MSG:\s+(\S+)$", "", cur_schema_text, flags=re.MULTILINE
            )

        # Parse the package and message names from the schema name
        # ROS1 uses "package/MessageName" format (no /msg/ in middle)
        # e.g., "geometry_msgs/Point" -> package="geometry_msgs", msg="Point"
        parts = cur_schema_name.split("/")
        pkg_name = parts[0] if parts else ""
        msg_name = parts[-1] if parts else cur_schema_name

        # ROS1 uses short name: "package/MessageName" (already in this format)
        short_name = f"{pkg_name}/{msg_name}" if pkg_name else msg_name

        # Parse the message with the package context
        msgdef = parse_message_string(cur_schema_text, context_package_name=pkg_name)

        # Set the short name on the message definition
        msgdef = MessageDefinition(
            name=short_name,
            fields_all=msgdef.fields_all,
        )

        fn(cur_schema_name, short_name, msgdef)


def _add_msgdef(
    definitions: dict[str, MessageDefinition],
    full_name: str,
    short_name: str,
    msgdef: MessageDefinition,
) -> None:
    """Add a message definition to the dictionary with multiple key formats."""
    # Add with both full name and short name for easier lookup
    definitions[full_name] = msgdef
    if short_name != full_name:
        definitions[short_name] = msgdef

    # Also add with just the message name (e.g., "Point") for simple lookups
    msg_name_only = full_name.split("/")[-1]
    if msg_name_only not in definitions:
        definitions[msg_name_only] = msgdef
