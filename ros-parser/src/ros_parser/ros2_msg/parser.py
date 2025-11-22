"""ROS2 message parser using Lark."""

import re
from pathlib import Path
from typing import Any, ClassVar, cast

# Import from standalone parser (pre-compiled grammar)
from .._lark_standalone_runtime import Token, Transformer
from ..models import (
    ActionDefinition,
    Constant,
    Field,
    MessageDefinition,
    ServiceDefinition,
    Type,
)
from ._standalone_parser import Lark_StandAlone


class MessageTransformer(Transformer[Any, MessageDefinition]):
    """Transforms Lark parse tree into ROS2 message data structures."""

    # Type normalization is NOT done at parse time
    # The reference parser keeps the original type names (byte, char) as-is
    # Normalization happens at semantic analysis time, not parsing time
    _TYPE_ALIASES: ClassVar[dict[str, str]] = {}

    def __init__(self, context_package_name: str | None = None) -> None:
        """
        Initialize transformer with optional context package name.

        Args:
            context_package_name: Package name to use for resolving local types
        """
        super().__init__()
        self.context_package_name = context_package_name

    def start(self, items: list[Any]) -> MessageDefinition:
        """Process the complete message definition."""
        fields_all: list[Field | Constant] = []

        for item in items:
            if item is None:
                continue
            if isinstance(item, (Field, Constant)):
                fields_all.append(item)

        return MessageDefinition(name=None, fields_all=fields_all)

    def content(self, items: list[Any]) -> Field | Constant | None:
        """Extract content from line."""
        if len(items) == 0:
            return None
        item = items[0]
        assert isinstance(item, (Field, Constant)) or item is None
        return item

    def field_or_constant(self, items: list[Any]) -> Field | Constant:
        """Create a Field or Constant based on the tail."""
        type_spec = items[0]
        name = str(items[1])
        tail = items[2] if len(items) > 2 else None

        # tail is a tuple: (is_constant: bool, value: bool | int | float | str | list | None)
        if tail and tail[0]:  # is_constant
            # Validate that constants only use primitive types (per ROS 2 spec)
            if isinstance(type_spec, Type):
                if type_spec.package_name is not None:
                    raise ValueError(
                        f"Constant '{name}' uses complex type '{type_spec}'. "
                        "Constants must use primitive types only."
                    )
                if type_spec.is_array:
                    raise ValueError(
                        f"Constant '{name}' uses array type '{type_spec}'. "
                        "Constants must use primitive types only (no arrays)."
                    )
            return Constant(type=type_spec, name=name, value=tail[1])
        # is_field
        default_value = tail[1] if tail else None
        return Field(type=type_spec, name=name, default_value=default_value)

    def field_or_const_tail(
        self, items: list[Any]
    ) -> tuple[bool, bool | int | float | str | list[Any] | None]:
        """
        Return (is_constant, value) tuple.

        The subrule (constant_tail/default_tail) determines the type.
        """
        if not items or items[0] is None:
            return (False, None)
        item = items[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        return item

    def constant_tail(self, items: list[Any]) -> tuple[bool, bool | int | float | str | None]:
        """Handle constant definition - returns (True, value)."""
        # Filter out None
        non_none_items = [
            item for item in items if item is not None and not isinstance(item, Token)
        ]
        if non_none_items:
            return (True, non_none_items[0])
        return (True, None)

    def default_tail(
        self, items: list[Any]
    ) -> tuple[bool, bool | int | float | str | list[Any] | None]:
        """Handle field default value - returns (False, value)."""
        # Filter out None
        non_none_items = [
            item for item in items if item is not None and not isinstance(item, Token)
        ]
        if non_none_items:
            return (False, non_none_items[0])
        return (False, None)

    def identifier(self, items: list[Token]) -> str:
        """Return identifier after validation."""
        name = str(items[0])
        # Validate no consecutive underscores
        if "__" in name:
            raise ValueError(f"Identifier '{name}' contains consecutive underscores")
        return name

    def type_spec(self, items: list[Any]) -> Type:
        """Build a Type from primitive/complex type and optional bounds/arrays."""
        base_type = items[0]

        # Extract optional string_bound and array_spec
        string_bound = None
        array_spec = None

        for item in items[1:]:
            if isinstance(item, int):
                string_bound = item
            elif isinstance(item, tuple):
                array_spec = item

        # Build the Type
        if isinstance(base_type, Type):
            # Complex type
            type_obj = Type(
                type_name=base_type.type_name,
                package_name=base_type.package_name,
                string_upper_bound=string_bound,
            )
        else:
            # Primitive type
            type_name = self._normalize_type(str(base_type))
            type_obj = Type(type_name=type_name, string_upper_bound=string_bound)

        # Apply array specification if present
        if array_spec:
            is_array, array_size, is_upper_bound = array_spec
            type_obj = Type(
                type_name=type_obj.type_name,
                package_name=type_obj.package_name,
                string_upper_bound=type_obj.string_upper_bound,
                is_array=is_array,
                array_size=array_size,
                is_upper_bound=is_upper_bound,
            )

        return type_obj

    def string_type(self, items: list[Token]) -> str:
        """Return the string type name."""
        return str(items[0])

    def numeric_type(self, items: list[Token]) -> str:
        """Return the numeric type name."""
        return str(items[0])

    def complex_type(self, items: list[Any]) -> Type:
        """Build a complex type with package name."""
        # items[0] is PACKAGE_NAME, items[1] is TYPE_IDENTIFIER
        package_name = str(items[0])
        type_name = str(items[1])

        return Type(type_name=type_name, package_name=package_name)

    def local_type(self, items: list[Token]) -> Type:
        """
        Build a local type (complex type without package prefix).

        Local types inherit the package name from the context (the current message's package).
        """
        type_name = str(items[0])
        # Use context package name if available, otherwise None
        return Type(type_name=type_name, package_name=self.context_package_name)

    def string_bound(self, items: list[Token]) -> int:
        """Extract string bound value."""
        return int(items[0])

    def array_spec(self, items: list[Any]) -> tuple[bool, int | None, bool]:
        """Extract array specification from child rule."""
        # The child rule (unbounded_array, fixed_array, or bounded_array) returns a tuple
        item = items[0]
        assert isinstance(item, tuple)
        assert len(item) == 3
        return item

    def unbounded_array(self, items: list[Any]) -> tuple[bool, int | None, bool]:  # noqa: ARG002
        """Return array specification for unbounded array."""
        return (True, None, False)

    def fixed_array(self, items: list[Token]) -> tuple[bool, int, bool]:
        """Return array specification for fixed-size array."""
        return (True, int(items[0]), False)

    def bounded_array(self, items: list[Token]) -> tuple[bool, int, bool]:
        """Return array specification for bounded array."""
        return (True, int(items[0]), True)

    def default_value(self, items: list[Any]) -> bool | int | float | str | list[Any]:
        """Return default value."""
        item = items[0]
        assert isinstance(item, (bool, int, float, str, list))
        return item

    def constant_value(self, items: list[Any]) -> bool | int | float | str:
        """Return constant value."""
        item = items[0]
        assert isinstance(item, (bool, int, float, str))
        return item

    def array_literal(self, items: list[Any]) -> list[Any]:
        """Build array from primitive values."""
        return list(items)

    def primitive_literal(self, items: list[Any]) -> bool | int | float | str:
        """Return primitive literal value."""
        item = items[0]
        assert isinstance(item, (bool, int, float, str))
        return item

    def quoted_string(self, items: list[Token]) -> str:
        """
        Parse quoted string, handling escape sequences.

        Lark's ESCAPED_STRING includes quotes, so we remove them and unescape.
        """
        string_token = str(items[0])
        # Remove surrounding quotes
        if (string_token.startswith("'") and string_token.endswith("'")) or (
            string_token.startswith('"') and string_token.endswith('"')
        ):
            string_content = string_token[1:-1]
        else:
            string_content = string_token

        # Use custom unescaping for ROS-specific escape sequences
        return self._unescape_string(string_content)

    def boolean_literal(self, items: list[Token]) -> bool:
        """Parse boolean value - returns True for 'true'/'True'/'1', False otherwise."""
        value = str(items[0])
        return value.lower() in ("true", "1")

    def numeric_literal(self, items: list[Token]) -> int | float:
        """
        Parse numeric value with support for different bases and floats.

        Handles: hex (0x...), binary (0b...), octal (0o...), decimal int, and floats.
        """
        value_str = str(items[0])

        # Handle different number bases using Python's int() base parameter
        if "x" in value_str.lower():
            return int(value_str, 16)
        if "b" in value_str.lower():
            return int(value_str, 2)
        if "o" in value_str.lower():
            return int(value_str, 8)

        # Handle decimal integers and floats
        # If contains '.', 'e', or 'E', it's a float
        if "." in value_str or "e" in value_str.lower():
            return float(value_str)
        return int(value_str)

    def unquoted_string(self, items: list[Token]) -> str:
        """Parse unquoted string."""
        string_content = str(items[0]).strip()
        # Handle escape sequences even in unquoted strings
        return self._unescape_string(string_content)

    # Helper methods

    @staticmethod
    def _normalize_type(type_name: str) -> str:
        """Normalize type aliases to standard types."""
        return MessageTransformer._TYPE_ALIASES.get(type_name, type_name)

    @staticmethod
    def _unescape_string(s: str) -> str:
        """Process escape sequences in strings."""
        # Handle standard escape sequences
        escape_map = {
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

        result = s
        for escaped, unescaped in escape_map.items():
            result = result.replace(escaped, unescaped)

        # Handle octal escapes (\012)
        result = re.sub(r"\\([0-7]{1,3})", lambda m: chr(int(m.group(1), 8)), result)

        # Handle hex escapes (\x10)
        result = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), result)

        # Handle unicode escapes (\u1010, \U0002F804)
        result = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), result)
        return re.sub(r"\\U([0-9a-fA-F]{8})", lambda m: chr(int(m.group(1), 16)), result)

    # Line filtering - return None for lines we want to filter out
    def line(self, items: list[Any]) -> Field | Constant | str | None:
        """Process a line and filter out None values."""
        if len(items) == 0 or items[0] is None:
            return None
        item = items[0]
        assert isinstance(item, (Field, Constant, str)) or item is None
        return item


def parse_string(message_string: str, context_package_name: str | None = None) -> MessageDefinition:
    """
    Parse a message definition from a string.

    Args:
        message_string: The message definition as a string
        context_package_name: Package name for resolving local types (optional)

    Returns:
        Parsed MessageDefinition object
    """
    # Strip trailing whitespace from each line to avoid grammar ambiguities
    lines = message_string.splitlines()  # Don't keep newlines
    cleaned_lines = [line.rstrip(" \t") for line in lines]
    cleaned = "\n".join(cleaned_lines)
    # Also strip trailing newlines/whitespace at the end of the entire string
    cleaned = cleaned.rstrip()

    # Create parser with context package name
    transformer: Any = MessageTransformer(context_package_name=context_package_name)
    parser: Any = Lark_StandAlone(transformer=transformer)
    return cast("MessageDefinition", parser.parse(cleaned))


def parse_file(file_path: str | Path, package_name: str | None = None) -> MessageDefinition:
    """
    Parse a message definition from a file.

    Args:
        file_path: Path to the .msg file
        package_name: Package name for resolving local types.
                     If None, will attempt to infer from file path.

    Returns:
        Parsed MessageDefinition object
    """
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")

    # Infer package name from path if not provided
    if package_name is None:
        package_name = _infer_package_name(path)

    return parse_string(content, context_package_name=package_name)


def _infer_package_name(file_path: Path) -> str | None:
    """
    Infer package name from file path.

    Expects pattern: .../package_name/{msg,srv,action}/FileName.{msg,srv,action}
    Falls back to parent directory name if standard pattern not found.
    """
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part in ("msg", "srv", "action") and i > 0:
            return parts[i - 1]
    # Fallback: use parent directory name (handles non-standard layouts like msg_ros1)
    return file_path.parent.name


# Backwards compatibility aliases
parse_message_string = parse_string
parse_message_file = parse_file


def parse_service_string(
    service_name: str, service_string: str, package_name: str | None = None
) -> ServiceDefinition:
    """
    Parse a ROS2 service definition from a string.

    Args:
        service_name: Name of the service (e.g., "GetMap")
        service_string: The service definition as a string
        package_name: Optional package name (e.g., "nav_msgs")

    Returns:
        Parsed ServiceDefinition object
    """
    lines = service_string.splitlines()
    separator_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]

    if len(separator_indices) != 1:
        raise ValueError(
            "Service definition must have exactly one '---' separator between request and response"
        )

    sep_idx = separator_indices[0]
    request_string = "\n".join(lines[:sep_idx])
    response_string = "\n".join(lines[sep_idx + 1 :])

    request = parse_string(request_string, context_package_name=package_name)
    response = parse_string(response_string, context_package_name=package_name)

    # Set the message names if not already set
    full_name = f"{package_name}/{service_name}" if package_name else service_name
    if request.name is None:
        request.name = f"{full_name}_Request"
    if response.name is None:
        response.name = f"{full_name}_Response"

    return ServiceDefinition(name=full_name, request=request, response=response)


def parse_service_file(file_path: str | Path, package_name: str | None = None) -> ServiceDefinition:
    """
    Parse a ROS2 service definition from a file.

    Args:
        file_path: Path to the .srv file
        package_name: Optional package name (will be inferred from path if not provided)

    Returns:
        Parsed ServiceDefinition object
    """
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")

    # Extract service name from filename
    service_name = path.stem

    # Try to infer package name from path if not provided
    if package_name is None:
        # Look for pattern: .../package_name/srv/ServiceName.srv
        parts = path.parts
        if "srv" in parts:
            srv_idx = parts.index("srv")
            if srv_idx > 0:
                package_name = parts[srv_idx - 1]

    return parse_service_string(service_name, content, package_name)


def parse_action_string(
    action_name: str, action_string: str, package_name: str | None = None
) -> ActionDefinition:
    """
    Parse a ROS2 action definition from a string.

    Args:
        action_name: Name of the action
        action_string: The action definition as a string
        package_name: Optional package name

    Returns:
        Parsed ActionDefinition object
    """
    lines = action_string.splitlines()
    separator_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]

    if len(separator_indices) != 2:
        raise ValueError(
            "Action definition must have exactly two '---' separators "
            "for goal, result, and feedback"
        )

    sep1, sep2 = separator_indices
    goal_string = "\n".join(lines[:sep1])
    result_string = "\n".join(lines[sep1 + 1 : sep2])
    feedback_string = "\n".join(lines[sep2 + 1 :])

    goal = parse_string(goal_string, context_package_name=package_name)
    result = parse_string(result_string, context_package_name=package_name)
    feedback = parse_string(feedback_string, context_package_name=package_name)

    # Set the message names if not already set
    full_name = f"{package_name}/{action_name}" if package_name else action_name
    if goal.name is None:
        goal.name = f"{full_name}_Goal"
    if result.name is None:
        result.name = f"{full_name}_Result"
    if feedback.name is None:
        feedback.name = f"{full_name}_Feedback"

    return ActionDefinition(name=full_name, goal=goal, result=result, feedback=feedback)


def parse_action_file(file_path: str | Path, package_name: str | None = None) -> ActionDefinition:
    """
    Parse a ROS2 action definition from a file.

    Args:
        file_path: Path to the .action file
        package_name: Optional package name (will be inferred from path if not provided)

    Returns:
        Parsed ActionDefinition object
    """
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")

    # Extract action name from filename
    action_name = path.stem

    # Try to infer package name from path if not provided
    if package_name is None:
        # Look for pattern: .../package_name/action/ActionName.action
        parts = path.parts
        if "action" in parts:
            action_idx = parts.index("action")
            if action_idx > 0:
                package_name = parts[action_idx - 1]

    return parse_action_string(action_name, content, package_name)
