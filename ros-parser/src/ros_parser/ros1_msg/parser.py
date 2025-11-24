"""ROS1 message parser using Lark."""

from pathlib import Path
from typing import Any, ClassVar, cast

# Import from standalone parser (pre-compiled grammar)
from .._lark_standalone_runtime import Token, Transformer
from .._utils import unescape_string

# Import models (shared between ROS1 and ROS2)
from ..models import Constant, Field, MessageDefinition, ServiceDefinition, Type
from ._standalone_parser import Lark_StandAlone


class Ros1MessageTransformer(Transformer[Any, MessageDefinition]):
    """Transforms Lark parse tree into ROS1 message data structures."""

    # ROS1 type normalization is NOT done at parse time (same as ROS2)
    # The reference parser keeps the original type names (byte, char) as-is
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
        if not items:
            return None
        item = items[0]
        if not (isinstance(item, (Field, Constant)) or item is None):
            raise TypeError(f"Expected Field, Constant, or None, got {type(item).__name__}")
        return item

    def field_or_constant(self, items: list[Any]) -> Field | Constant:
        """Create a Field or Constant based on presence of constant_tail."""
        type_spec = items[0]
        name = str(items[1])
        constant_value = items[2] if len(items) > 2 else None

        # ROS1 only has constants (with =), no default values
        if constant_value is not None:
            # Validate that constants only use primitive types
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
            return Constant(type=type_spec, name=name, value=constant_value)
        # is_field (no default value in ROS1)
        return Field(type=type_spec, name=name, default_value=None)

    def constant_tail(self, items: list[Any]) -> bool | int | float | str:
        """Handle constant definition - returns the value."""
        # Filter out None and tokens
        non_none_items: list[bool | int | float | str] = [
            item for item in items if item is not None and not isinstance(item, Token)
        ]
        if non_none_items:
            val = non_none_items[0]
            if not isinstance(val, (bool, int, float, str)):
                raise TypeError(f"Expected bool, int, float, or str, got {type(val).__name__}")
            return val
        return ""

    def identifier(self, items: list[Token]) -> str:
        """Return identifier after validation."""
        name = str(items[0])
        # Validate no consecutive underscores
        if "__" in name:
            raise ValueError(f"Identifier '{name}' contains consecutive underscores")
        return name

    def type_spec(self, items: list[Any]) -> Type:
        """Build a Type from primitive/complex type and optional array."""
        base_type = items[0]

        # Extract optional array_spec
        array_spec = None
        for item in items[1:]:
            if isinstance(item, tuple):
                array_spec = item

        # Build the Type
        if isinstance(base_type, Type):
            # Complex type
            type_obj = Type(
                type_name=base_type.type_name,
                package_name=base_type.package_name,
            )
        else:
            # Primitive type
            type_name = self._normalize_type(str(base_type))
            type_obj = Type(type_name=type_name)

        # Apply array specification if present
        if array_spec:
            is_array, array_size = array_spec
            type_obj = Type(
                type_name=type_obj.type_name,
                package_name=type_obj.package_name,
                is_array=is_array,
                array_size=array_size,
                is_upper_bound=False,  # ROS1 doesn't have bounded arrays
            )

        return type_obj

    def primitive_type(self, items: list[Token]) -> str:
        """Return the primitive type name."""
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
        Special case: "Header" resolves to "std_msgs/Header" in ROS1.
        """
        type_name = str(items[0])

        # ROS1 special case: "Header" auto-resolves to std_msgs/Header
        if type_name == "Header":
            return Type(type_name="Header", package_name="std_msgs")

        # Use context package name if available, otherwise None
        return Type(type_name=type_name, package_name=self.context_package_name)

    def array_spec(self, items: list[Any]) -> tuple[bool, int | None]:
        """Extract array specification from child rule."""
        # The child rule (unbounded_array or fixed_array) returns a tuple
        item = items[0]
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"Expected tuple of length 2, got {type(item).__name__}")
        return item

    def unbounded_array(self, items: list[Any]) -> tuple[bool, int | None]:  # noqa: ARG002
        """Return array specification for unbounded array."""
        return (True, None)

    def fixed_array(self, items: list[Token]) -> tuple[bool, int]:
        """Return array specification for fixed-size array."""
        return (True, int(items[0]))

    def constant_value(self, items: list[Any]) -> bool | int | float | str:
        """Return constant value."""
        item = items[0]
        if not isinstance(item, (bool, int, float, str)):
            raise TypeError(f"Expected bool, int, float, or str, got {type(item).__name__}")
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
        return unescape_string(string_content)

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
        return unescape_string(string_content)

    # Helper methods

    @staticmethod
    def _normalize_type(type_name: str) -> str:
        """Normalize type aliases to standard types."""
        return Ros1MessageTransformer._TYPE_ALIASES.get(type_name, type_name)

    # Line filtering - return None for lines we want to filter out
    def line(self, items: list[Any]) -> Field | Constant | str | None:
        """Process a line and filter out None values."""
        if not items or items[0] is None:
            return None
        item = items[0]
        if not isinstance(item, (Field, Constant, str)):
            raise TypeError(f"Expected Field, Constant, or str, got {type(item).__name__}")
        return item


def parse_string(message_string: str, context_package_name: str | None = None) -> MessageDefinition:
    """
    Parse a ROS1 message definition from a string.

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
    transformer: Any = Ros1MessageTransformer(context_package_name=context_package_name)
    parser: Any = Lark_StandAlone(transformer=transformer)
    return cast("MessageDefinition", parser.parse(cleaned))


def parse_file(file_path: str | Path, package_name: str | None = None) -> MessageDefinition:
    """
    Parse a ROS1 message definition from a file.

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

    Expects pattern: .../package_name/{msg,srv}/FileName.{msg,srv}
    Falls back to parent directory name if standard pattern not found.
    """
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part in ("msg", "srv") and i > 0:
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
    Parse a ROS1 service definition from a string.

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
    Parse a ROS1 service definition from a file.

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
