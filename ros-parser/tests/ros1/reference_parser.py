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
class ReferenceType:
    """Normalized type shape produced independently of ros_parser."""

    base_type: str
    package_name: str | None
    is_array: bool
    array_size: int | None
    is_upper_bound: bool = False


@dataclass
class ReferenceField:
    """Represents a field in the reference parser."""

    type: ReferenceType
    name: str


@dataclass
class ReferenceConstant:
    """Represents a constant in the reference parser."""

    type: ReferenceType
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
    """Strip comments while preserving a hash inside a quoted string."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote is not None:
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == COMMENTCHAR and quote is None:
            return line[:index].strip()
    return line.strip()


def _parse_type(type_text: str, package_context: str | None = None) -> ReferenceType:
    """Parse a ROS1 type token into an independent normalized shape."""
    base_type = type_text
    is_array = False
    array_size = None
    if "[" in type_text:
        if not type_text.endswith("]"):
            raise ValueError(f"Invalid array type: {type_text}")
        base_type, array_suffix = type_text[:-1].split("[", maxsplit=1)
        is_array = True
        if array_suffix:
            array_size = int(array_suffix)

    if base_type == HEADER:
        package_name = "std_msgs"
        base_type = HEADER
    elif base_type in BUILTIN_TYPES:
        package_name = None
    elif SEP in base_type:
        package_name, base_type = base_type.split(SEP, maxsplit=1)
    else:
        package_name = package_context

    return ReferenceType(
        base_type=base_type,
        package_name=package_name,
        is_array=is_array,
        array_size=array_size,
    )


def _convert_constant_value(type_: ReferenceType, val: str) -> bool | int | float | str:
    """Convert constant value string to proper type."""
    val = val.strip()
    field_type = type_.base_type

    if field_type == "string":
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            return val[1:-1]
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
        result = int(val)
        ranges = {
            "byte": (-128, 127),
            "char": (0, 255),
            "uint8": (0, 255),
            "int8": (-128, 127),
            "uint16": (0, 65_535),
            "int16": (-32_768, 32_767),
            "uint32": (0, 4_294_967_295),
            "int32": (-2_147_483_648, 2_147_483_647),
            "uint64": (0, 18_446_744_073_709_551_615),
            "int64": (-9_223_372_036_854_775_808, 9_223_372_036_854_775_807),
        }
        lower, upper = ranges[field_type]
        if not lower <= result <= upper:
            raise ValueError(f"Integer constant out of range for {field_type}: {result}")
        return result
    raise ValueError(f"Unknown constant type: {field_type}")


def _load_constant_line(orig_line: str) -> ReferenceConstant:
    """Parse a constant line."""
    clean_line = _strip_comments(orig_line)
    line_splits = [s for s in [x.strip() for x in clean_line.split(" ")] if s]
    field_type_text = line_splits[0]
    type_ = _parse_type(field_type_text)

    if field_type_text == "string":
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

    if type_.is_array or type_.package_name is not None:
        raise ValueError("ROS1 constants must be primitive, non-array types")
    val_converted = _convert_constant_value(type_, val)
    return ReferenceConstant(type_, name, val_converted)


def _load_field_line(orig_line: str, package_context: str | None) -> ReferenceField:
    """Parse a field line into its normalized type and name."""
    clean_line = _strip_comments(orig_line)
    line_splits = [s for s in [x.strip() for x in clean_line.split(" ")] if s]
    if len(line_splits) != 2:
        raise ValueError(f"Invalid declaration: {orig_line}")
    field_type, name = line_splits
    return ReferenceField(_parse_type(field_type, package_context), name)


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
            fields.append(_load_field_line(orig_line, package_context))

    return ReferenceMessageDef(fields, constants)
