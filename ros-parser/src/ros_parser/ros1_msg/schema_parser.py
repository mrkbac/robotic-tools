"""Helper for parsing MCAP schema data into ROS1 message definitions."""

from collections.abc import Callable

from ros_parser._utils import add_msgdef_to_dict, for_each_msgdef_in_schema
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
    for_each_msgdef_in_schema(
        schema_name,
        schema_text,
        parse_message_string,
        lambda full_name, short_name, msgdef: add_msgdef_to_dict(
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
    for_each_msgdef_in_schema(schema_name, schema_text, parse_message_string, fn)
