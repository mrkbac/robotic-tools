"""
Reference ROS1 message parser implementation.

This is a simplified port of genmsg's msg_loader.py for testing purposes.
Used to verify our Lark-based parser produces the same results as the reference implementation.
"""

from dataclasses import dataclass

# Constants from genmsg/base.py
SEP = "/"
CONSTCHAR = "="
COMMENTCHAR = "#"
HEADER = "Header"
HEADER_FULL_NAME = "std_msgs/Header"
TIME = "time"
DURATION = "duration"

# Primitive types from genmsg/msgs.py
PRIMITIVE_TYPES = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
    "string",
    "bool",
    # deprecated:
    "char",
    "byte",
]
BUILTIN_TYPES = [*PRIMITIVE_TYPES, TIME, DURATION]


@dataclass
class ReferenceField:
    """Represents a field in the reference parser."""

    field_type: str
    name: str


@dataclass
class ReferenceConstant:
    """Represents a constant in the reference parser."""

    field_type: str
    name: str
    value: bool | int | float | str


@dataclass
class ReferenceMessageDef:
    """Simplified message definition for reference comparison."""

    fields: list[ReferenceField]
    constants: list[ReferenceConstant]


def bare_msg_type(msg_type: str) -> str:
    """Get the bare type without array brackets."""
    if "[" in msg_type:
        return msg_type[: msg_type.find("[")]
    return msg_type


def is_builtin(msg_type_name: str) -> bool:
    """Check if type is a built-in type."""
    return bare_msg_type(msg_type_name) in BUILTIN_TYPES


def _strip_comments(line: str) -> str:
    """Strip comments from a line."""
    return line.split(COMMENTCHAR)[0].strip()


def _convert_constant_value(field_type: str, val: str) -> bool | int | float | str:
    """Convert constant value string to proper type."""
    val = val.strip()

    if field_type == "string":
        return val
    if field_type == "bool":
        # ROS1 accepts 0/1 or true/false
        if val.lower() in ("true", "1"):
            return True
        if val.lower() in ("false", "0"):
            return False
        raise ValueError(f"Invalid bool value: {val}")
    if field_type in ("float32", "float64"):
        return float(val)
    if field_type in PRIMITIVE_TYPES:
        # Integer types
        # Handle hex, octal, binary
        if val.lower().startswith("0x"):
            return int(val, 16)
        if val.lower().startswith("0b"):
            return int(val, 2)
        if val.lower().startswith("0o"):
            return int(val, 8)
        return int(val)
    raise ValueError(f"Unknown constant type: {field_type}")


def _load_constant_line(orig_line: str) -> ReferenceConstant:
    """Parse a constant line."""
    clean_line = _strip_comments(orig_line)
    line_splits = [s for s in [x.strip() for x in clean_line.split(" ")] if s]
    field_type = line_splits[0]

    if field_type == "string":
        # strings contain anything to the right of the equals sign
        idx = orig_line.find(CONSTCHAR)
        name = orig_line[orig_line.find(" ") + 1 : idx].strip()
        val = orig_line[idx + 1 :]
    else:
        line_splits = [x.strip() for x in " ".join(line_splits[1:]).split(CONSTCHAR)]
        if len(line_splits) != 2:
            raise ValueError(f"Invalid constant declaration: {orig_line}")
        name = line_splits[0]
        val = line_splits[1]

    val_converted = _convert_constant_value(field_type, val)
    return ReferenceConstant(field_type, name, val_converted)


def _load_field_line(orig_line: str, package_context: str | None) -> tuple[str, str]:
    """Parse a field line and return (field_type, name)."""
    clean_line = _strip_comments(orig_line)
    line_splits = [s for s in [x.strip() for x in clean_line.split(" ")] if s]
    if len(line_splits) != 2:
        raise ValueError(f"Invalid declaration: {orig_line}")
    field_type, name = line_splits

    # Resolve type based on context
    if package_context and SEP not in field_type:
        if field_type == HEADER:
            field_type = HEADER_FULL_NAME
        elif not is_builtin(bare_msg_type(field_type)):
            field_type = f"{package_context}/{field_type}"
    elif field_type == HEADER:
        field_type = HEADER_FULL_NAME

    return field_type, name


def parse_message_string(text: str, package_context: str | None = None) -> ReferenceMessageDef:
    """
    Parse a ROS1 message definition string using the reference implementation.

    Args:
        text: The message definition text
        package_context: Package name for resolving local types

    Returns:
        ReferenceMessageDef with fields and constants
    """
    fields: list[ReferenceField] = []
    constants: list[ReferenceConstant] = []

    for orig_line in text.split("\n"):
        clean_line = _strip_comments(orig_line)
        if not clean_line:
            continue  # ignore empty lines
        if CONSTCHAR in clean_line:
            constants.append(_load_constant_line(orig_line))
        else:
            field_type, name = _load_field_line(orig_line, package_context)
            fields.append(ReferenceField(field_type, name))

    return ReferenceMessageDef(fields, constants)
