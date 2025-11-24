"""Helper for parsing MCAP schema data into ROS message definitions."""

from collections.abc import Callable

from ros_parser._utils import add_msgdef_to_dict, for_each_msgdef_in_schema
from ros_parser.models import Field, MessageDefinition, Type
from ros_parser.ros2_msg.parser import parse_message_string

# Built-in ROS2 types that may not be included in schema data
BUILTIN_TYPES = {
    "builtin_interfaces/Time": MessageDefinition(
        name="builtin_interfaces/Time",
        fields_all=[
            Field(Type(type_name="int32"), "sec"),
            Field(Type(type_name="uint32"), "nanosec"),
        ],
    ),
    "builtin_interfaces/Duration": MessageDefinition(
        name="builtin_interfaces/Duration",
        fields_all=[
            Field(Type(type_name="int32"), "sec"),
            Field(Type(type_name="uint32"), "nanosec"),
        ],
    ),
}


def parse_schema_to_definitions(
    schema_name: str,
    schema_data: bytes,
) -> dict[str, MessageDefinition]:
    """Parse MCAP schema data into a dictionary of message definitions.

    MCAP schemas for ROS2 messages contain the full message definition text,
    including all nested message types. This function parses that text and
    returns a dictionary mapping message names to their definitions.

    Args:
        schema_name: The schema name (e.g., "geometry_msgs/msg/Pose")
        schema_data: The raw schema data bytes (will be decoded as UTF-8)

    Returns:
        Dictionary mapping message names to MessageDefinition objects.
        Keys include both full paths (e.g., "geometry_msgs/msg/Point")
        and short names (e.g., "geometry_msgs/Point").
        Also includes built-in types like "builtin_interfaces/Time".

    Example:
        >>> schema_name = "geometry_msgs/msg/Pose"
        >>> schema_data = b"Point position\\nQuaternion orientation\\n===\\n..."
        >>> definitions = parse_schema_to_definitions(schema_name, schema_data)
        >>> "geometry_msgs/msg/Pose" in definitions
        True
        >>> "geometry_msgs/Point" in definitions
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
    Each section may start with "MSG: package/msg/Name" to indicate the message name.

    Args:
        schema_name: The main schema name
        schema_text: The schema text containing one or more message definitions
        fn: Callback function(full_name, short_name, msgdef) to call for each definition
    """
    for_each_msgdef_in_schema(schema_name, schema_text, parse_message_string, fn)
