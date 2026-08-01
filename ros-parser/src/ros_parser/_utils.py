"""Shared utilities for ROS message parsing."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from ros_parser.models import MessageDefinition

if TYPE_CHECKING:
    from typing_extensions import Self

# Standard escape sequence mapping for ROS string literals
_ESCAPE_MAP = {
    "'": "'",
    '"': '"',
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
}

# Schema separator pattern (3 or more = characters on their own line)
_SCHEMA_SEPARATOR_PATTERN = re.compile(r"^={3,}$", flags=re.MULTILINE)
# MSG header pattern (e.g., "MSG: package/msg/Name")
_MSG_HEADER_PATTERN = re.compile(r"^MSG:\s+(\S+)$", flags=re.MULTILINE)

_INTEGER_RANGES: dict[str, tuple[int, int]] = {
    "char": (0, 2**8 - 1),
    "uint8": (0, 2**8 - 1),
    "int8": (-(2**7), 2**7 - 1),
    "uint16": (0, 2**16 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "uint32": (0, 2**32 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "uint64": (0, 2**64 - 1),
    "int64": (-(2**63), 2**63 - 1),
}


def integer_type_range(type_name: str, *, is_ros1: bool) -> tuple[int, int] | None:
    """Return the ROS-version-specific range for an integer primitive."""
    if type_name == "byte":
        return (-(2**7), 2**7 - 1) if is_ros1 else (0, 2**8 - 1)
    return _INTEGER_RANGES.get(type_name)


class IntegerLiteral(int):
    """Integer value that retains its source spelling until type validation."""

    source: str

    def __new__(cls, value: int, source: str) -> Self:
        instance = super().__new__(cls, value)
        instance.source = source
        return instance


class FloatLiteral(float):
    """Float value that retains its source spelling until type validation."""

    source: str

    def __new__(cls, value: float, source: str) -> Self:
        instance = super().__new__(cls, value)
        instance.source = source
        return instance


class UnquotedLiteral(str):
    """Unquoted word that can be interpreted according to its field type."""


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
    result: list[str] = []
    index = 0
    hex_digits = frozenset("0123456789abcdefABCDEF")
    octal_digits = frozenset("01234567")

    while index < len(s):
        if s[index] != "\\":
            result.append(s[index])
            index += 1
            continue

        if index + 1 >= len(s):
            result.append("\\")
            index += 1
            continue

        token = s[index + 1]
        unescaped = _ESCAPE_MAP.get(token)
        if unescaped is not None:
            result.append(unescaped)
            index += 2
            continue

        if s[index + 1] == "x" and index + 4 <= len(s):
            digits = s[index + 2 : index + 4]
            if all(digit in hex_digits for digit in digits):
                result.append(chr(int(digits, 16)))
                index += 4
                continue

        if s[index + 1] == "u" and index + 6 <= len(s):
            digits = s[index + 2 : index + 6]
            if all(digit in hex_digits for digit in digits):
                result.append(chr(int(digits, 16)))
                index += 6
                continue

        if s[index + 1] == "U" and index + 10 <= len(s):
            digits = s[index + 2 : index + 10]
            if all(digit in hex_digits for digit in digits):
                result.append(chr(int(digits, 16)))
                index += 10
                continue

        if s[index + 1] in octal_digits:
            end = index + 2
            while end < len(s) and end < index + 4 and s[end] in octal_digits:
                end += 1
            result.append(chr(int(s[index + 1 : end], 8)))
            index = end
            continue

        # Preserve unknown or incomplete escapes literally, and consume both
        # characters so a following letter cannot be interpreted a second time.
        result.extend(("\\", s[index + 1]))
        index += 2

    return "".join(result)


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
            section_text = _MSG_HEADER_PATTERN.sub("", section_text).strip()

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
