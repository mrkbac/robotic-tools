"""Data models for ROS2 message definitions."""

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property


class PrimitiveType(str, Enum):
    """ROS primitive type names as enumeration.

    This enum provides type-safe access to primitive type names
    with autocomplete support in IDEs.
    """

    BOOL = "bool"
    BYTE = "byte"
    CHAR = "char"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    INT64 = "int64"
    UINT64 = "uint64"
    STRING = "string"
    WSTRING = "wstring"
    TIME = "time"
    DURATION = "duration"


# Set of all primitive type names (for backwards compatibility)
PRIMITIVE_TYPE_NAMES = {member.value for member in PrimitiveType}

# Type aliases for primitive values
PrimitiveValue = bool | int | float | str
ArrayValue = list[PrimitiveValue]


@dataclass(frozen=True)
class Type:
    """Represents a ROS2 type (primitive or complex, with optional array and bounds)."""

    # Base type (e.g., "string", "int32", "geometry_msgs/Point")
    type_name: str

    # Package name for complex types (None for primitives and local types)
    package_name: str | None = None

    # Array specification
    is_array: bool = False
    array_size: int | None = None  # Fixed size (e.g., [5])
    is_upper_bound: bool = False  # Whether array_size is an upper bound (e.g., [<=5])

    # String upper bound (e.g., string<=10)
    string_upper_bound: int | None = None

    @property
    def is_primitive(self) -> bool:
        """Check if this is a primitive type based on type name."""
        return self.type_name in PRIMITIVE_TYPE_NAMES

    @property
    def is_dynamic_array(self) -> bool:
        """Check if this is a dynamic/unbounded array."""
        return self.is_array and (not self.array_size or self.is_upper_bound)

    @property
    def is_fixed_array(self) -> bool:
        """Check if this is a fixed-size array."""
        return self.is_array and self.array_size is not None and not self.is_upper_bound

    def __str__(self) -> str:
        """Return the string representation of the type."""
        if self.package_name:
            result = f"{self.package_name}/{self.type_name}"
        else:
            result = self.type_name
            if self.string_upper_bound:
                result += f"<={self.string_upper_bound}"

        if self.is_array:
            result += "["
            if self.is_upper_bound:
                result += "<="
            if self.array_size is not None:
                result += str(self.array_size)
            result += "]"

        return result


@dataclass
class Field:
    """Represents a field in a ROS2 message."""

    type: Type
    name: str
    default_value: PrimitiveValue | ArrayValue | None = None

    def __str__(self) -> str:
        """Return the string representation of the field."""
        result = f"{self.type} {self.name}"
        if self.default_value is not None:
            if isinstance(self.default_value, str):
                result += f" '{self.default_value}'"
            elif isinstance(self.default_value, list):
                result += f" {self.default_value}"
            else:
                result += f" {self.default_value}"
        return result


@dataclass
class Constant:
    """Represents a constant definition in a ROS2 message."""

    type: Type
    name: str
    value: PrimitiveValue

    def __post_init__(self) -> None:
        """Validate constant after initialization."""
        if not self.type.is_primitive or self.type.is_array:
            raise TypeError("Constants must be primitive, non-array types")

    def __str__(self) -> str:
        """Return the string representation of the constant."""
        value_str = f"'{self.value}'" if isinstance(self.value, str) else str(self.value)
        return f"{self.type} {self.name}={value_str}"


@dataclass
class MessageDefinition:
    """Represents a complete ROS2 message definition."""

    # Message name (e.g., "geometry_msgs/Point" or None for anonymous messages)
    name: str | None
    fields_all: list[Field | Constant] = field(default_factory=list)

    @cached_property
    def fields(self) -> list[Field]:
        """Return only Field objects from fields_all."""
        return [item for item in self.fields_all if isinstance(item, Field)]

    @cached_property
    def constants(self) -> list[Constant]:
        """Return only Constant objects from fields_all."""
        return [item for item in self.fields_all if isinstance(item, Constant)]

    def __post_init__(self) -> None:
        """Validate message definition after initialization."""
        # Collect all names from fields_all
        all_names = [item.name for item in self.fields_all]

        # Check for duplicate names (across both fields and constants)
        if len(all_names) != len(set(all_names)):
            duplicates = {name for name in all_names if all_names.count(name) > 1}
            raise ValueError(f"Duplicate field/constant names: {', '.join(sorted(duplicates))}")

    def __str__(self) -> str:
        """Return the string representation of the message."""
        lines = []
        if self.name:
            lines.append(f"# {self.name}")

        # Output in original order from fields_all
        lines.extend(str(item) for item in self.fields_all)

        return "\n".join(lines)


@dataclass
class ServiceDefinition:
    """Represents a ROS2 service definition (request and response)."""

    name: str  # Service name (e.g., "nav_msgs/GetMap")
    request: MessageDefinition
    response: MessageDefinition

    def __str__(self) -> str:
        """Return the string representation of the service."""
        lines = [f"# {self.name}"]
        lines.append(str(self.request))
        lines.append("---")
        lines.append(str(self.response))
        return "\n".join(lines)


@dataclass
class ActionDefinition:
    """Represents a ROS2 action definition (goal, result, feedback)."""

    name: str  # Action name
    goal: MessageDefinition
    result: MessageDefinition
    feedback: MessageDefinition

    def __str__(self) -> str:
        """Return the string representation of the action."""
        lines = [f"# {self.name}"]
        lines.append(str(self.goal))
        lines.append("---")
        lines.append(str(self.result))
        lines.append("---")
        lines.append(str(self.feedback))
        return "\n".join(lines)
