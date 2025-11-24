"""Shared utilities for ROS message parsing."""

import re
from collections.abc import Callable

from ros_parser.models import MessageDefinition

# Standard escape sequence mapping for ROS string literals
_ESCAPE_MAP = {
    "\\'": "'",
    '\\"': '"',
    "\\a": "\a",
    "\\b": "\b",
    "\\f": "\f",
    "\\n": "\n",
    "\\r": "\r",
    "\\t": "\t",
    "\\v": "\v",
    "\\\\": "\\",
}

# Regex patterns for escape sequence processing
_OCTAL_ESCAPE_PATTERN = re.compile(r"\\([0-7]{1,3})")
_HEX_ESCAPE_PATTERN = re.compile(r"\\x([0-9a-fA-F]{2})")
_UNICODE_4_ESCAPE_PATTERN = re.compile(r"\\u([0-9a-fA-F]{4})")
_UNICODE_8_ESCAPE_PATTERN = re.compile(r"\\U([0-9a-fA-F]{8})")

# Schema separator pattern (3 or more = characters on their own line)
_SCHEMA_SEPARATOR_PATTERN = re.compile(r"^={3,}$", flags=re.MULTILINE)
# MSG header pattern (e.g., "MSG: package/msg/Name")
_MSG_HEADER_PATTERN = re.compile(r"^MSG:\s+(\S+)$", flags=re.MULTILINE)


def unescape_string(s: str) -> str:
    """Process escape sequences in ROS string literals.

    Handles:
    - Standard C-style escapes (\\n, \\t, \\r, etc.)
    - Octal escapes (\\012)
    - Hex escapes (\\x10)
    - Unicode escapes (\\u1010, \\U0002F804)

    Args:
        s: The string with escape sequences to process

    Returns:
        The string with escape sequences converted to actual characters
    """
    result = s
    for escaped, unescaped in _ESCAPE_MAP.items():
        result = result.replace(escaped, unescaped)

    # Handle octal escapes (\012)
    result = _OCTAL_ESCAPE_PATTERN.sub(lambda m: chr(int(m.group(1), 8)), result)

    # Handle hex escapes (\x10)
    result = _HEX_ESCAPE_PATTERN.sub(lambda m: chr(int(m.group(1), 16)), result)

    # Handle unicode escapes (\u1010, \U0002F804)
    result = _UNICODE_4_ESCAPE_PATTERN.sub(lambda m: chr(int(m.group(1), 16)), result)
    return _UNICODE_8_ESCAPE_PATTERN.sub(lambda m: chr(int(m.group(1), 16)), result)


def for_each_msgdef_in_schema(
    schema_name: str,
    schema_text: str,
    parse_fn: Callable[[str, str | None], MessageDefinition],
    callback: Callable[[str, str, MessageDefinition], None],
) -> None:
    """Parse schema text and call callback for each message definition found.

    MCAP schema data can contain multiple message definitions separated by "===".
    Each section may start with "MSG: package/msg/Name" to indicate the message name.

    This is shared logic between ROS1 and ROS2 schema parsing - the only difference
    is the parser function used to parse individual message definitions.

    Args:
        schema_name: The main schema name (e.g., "geometry_msgs/msg/Pose" for ROS2
                    or "geometry_msgs/Pose" for ROS1)
        schema_text: The schema text containing one or more message definitions
        parse_fn: Parser function that takes (message_text, package_name) and returns
                 a MessageDefinition
        callback: Function called for each definition with (full_name, short_name, msgdef)
    """
    cur_schema_name = schema_name

    # Remove empty lines
    schema_text = "\n".join([s for s in schema_text.splitlines() if s.strip()])

    # Split schema_text by separator lines containing at least 3 = characters
    for cur_section in _SCHEMA_SEPARATOR_PATTERN.split(schema_text):
        section_text = cur_section.strip()

        # Check for a "MSG: pkg_name/msg_name" line
        match = _MSG_HEADER_PATTERN.match(section_text)
        if match:
            cur_schema_name = match.group(1)
            # Remove this line from the message definition
            section_text = _MSG_HEADER_PATTERN.sub("", section_text)

        # Parse the package and message names from the schema name
        # e.g., "geometry_msgs/msg/Point" -> package="geometry_msgs", msg="Point"
        # or   "geometry_msgs/Point" -> package="geometry_msgs", msg="Point"
        parts = cur_schema_name.split("/")
        pkg_name = parts[0] if parts else ""
        msg_name = parts[-1] if parts else cur_schema_name

        # Create short name: "package/MessageName" (without "/msg/" in middle)
        short_name = f"{pkg_name}/{msg_name}" if pkg_name else msg_name

        # Parse the message with the package context
        msgdef = parse_fn(section_text, pkg_name if pkg_name else None)

        # Set the short name on the message definition
        msgdef = MessageDefinition(
            name=short_name,
            fields_all=msgdef.fields_all,
        )

        callback(cur_schema_name, short_name, msgdef)


def add_msgdef_to_dict(
    definitions: dict[str, MessageDefinition],
    full_name: str,
    short_name: str,
    msgdef: MessageDefinition,
) -> None:
    """Add a message definition to the dictionary with multiple key formats.

    This is the standard callback for for_each_msgdef_in_schema that adds
    the definition with multiple keys for flexible lookup.

    Args:
        definitions: Dictionary to add definitions to
        full_name: Full schema name (e.g., "geometry_msgs/msg/Point")
        short_name: Short name (e.g., "geometry_msgs/Point")
        msgdef: The parsed message definition
    """
    # Add with both full name and short name for easier lookup
    definitions[full_name] = msgdef
    if short_name != full_name:
        definitions[short_name] = msgdef

    # Also add with just the message name (e.g., "Point") for simple lookups
    msg_name_only = full_name.split("/")[-1]
    if msg_name_only not in definitions:
        definitions[msg_name_only] = msgdef
