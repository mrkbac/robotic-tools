"""Pure plan data structures for ROS2 message parsing.

This module contains only data structures and enums for representing
message parsing plans. It has no dependencies on specific parser
implementations, allowing the same plans to be executed by different
backends (interpreted, compiled, code-generated, etc.).
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from types import SimpleNamespace
from typing import Any, ClassVar, Literal


class McapROS2DecodeError(Exception):
    """Raised if a MCAP message record cannot be decoded as a ROS2 message."""


# Type aliases for plan data structures
DecodedMessage = SimpleNamespace
DecoderFunction = Callable[[bytes | memoryview], DecodedMessage]
EncoderFunction = Callable[[Any], bytes | memoryview]
PrimitiveValue = bool | int | float | str
DefaultValue = PrimitiveValue | list[PrimitiveValue]


class TypeId(IntEnum):
    """Type identifiers for primitive types."""

    BOOL = 1
    BYTE = 2
    CHAR = 3
    FLOAT32 = 4
    FLOAT64 = 5
    INT8 = 6
    UINT8 = 7
    INT16 = 8
    UINT16 = 9
    INT32 = 10
    UINT32 = 11
    INT64 = 12
    UINT64 = 13
    STRING = 14
    WSTRING = 15  # not supported right now
    PADDING = 16  # Alignment padding bytes


class ActionType(IntEnum):
    """Action types for plan-based parsing."""

    PRIMITIVE = 1
    """Parse primitive type, data contains TypeId."""
    PRIMITIVE_ARRAY = 2
    """Parse array of primitive types, data contains TypeId and optional length."""
    COMPLEX = 3
    """Parse complex type, data contains PlanList."""
    COMPLEX_ARRAY = 4
    """Parse array of complex types, data contains PlanList and optional length."""

    # Optimization actions
    PRIMITIVE_GROUP = 10
    """Group multiple primitive fields into a single action, data contains list of field names."""


# Base types for actions


@dataclass(slots=True, frozen=True)
class PrimitiveAction:
    type: ClassVar[Literal[ActionType.PRIMITIVE]] = ActionType.PRIMITIVE
    target: str
    data: TypeId
    default_value: PrimitiveValue | None = None


@dataclass(slots=True, frozen=True)
class PrimitiveArrayAction:
    type: ClassVar[Literal[ActionType.PRIMITIVE_ARRAY]] = ActionType.PRIMITIVE_ARRAY
    target: str
    data: TypeId
    size: int | None
    is_upper_bound: bool = False  # Whether size is an upper bound (<=N)
    default_value: list[PrimitiveValue] | None = None


@dataclass(slots=True, frozen=True)
class PrimitiveGroupAction:
    type: ClassVar[Literal[ActionType.PRIMITIVE_GROUP]] = ActionType.PRIMITIVE_GROUP
    targets: list[tuple[str, TypeId, PrimitiveValue | None]]  # (name, type_id, default_value)


@dataclass(slots=True, frozen=True)
class ComplexAction:
    type: ClassVar[Literal[ActionType.COMPLEX]] = ActionType.COMPLEX
    target: str
    plan: "PlanList"


@dataclass(slots=True, frozen=True)
class ComplexArrayAction:
    type: ClassVar[Literal[ActionType.COMPLEX_ARRAY]] = ActionType.COMPLEX_ARRAY
    target: str
    plan: "PlanList"
    size: int | None
    is_upper_bound: bool = False  # Whether size is an upper bound (<=N)


# Union type for all possible actions
PlanAction = (
    PrimitiveAction
    | PrimitiveArrayAction
    | PrimitiveGroupAction
    | ComplexAction
    | ComplexArrayAction
)

PlanActions = list[PlanAction]  # field_name, action
PlanList = tuple[type, PlanActions]  # target_type, actions


# String to TypeId mapping for plan generation
STRING_TO_TYPE_ID = {
    "bool": TypeId.BOOL,
    "byte": TypeId.BYTE,
    "char": TypeId.CHAR,
    "float32": TypeId.FLOAT32,
    "float64": TypeId.FLOAT64,
    "int8": TypeId.INT8,
    "uint8": TypeId.UINT8,
    "int16": TypeId.INT16,
    "uint16": TypeId.UINT16,
    "int32": TypeId.INT32,
    "uint32": TypeId.UINT32,
    "int64": TypeId.INT64,
    "uint64": TypeId.UINT64,
    "string": TypeId.STRING,
    "wstring": TypeId.WSTRING,
}

# Type metadata for code generation
TYPE_INFO: dict[TypeId, str] = {
    TypeId.BOOL: "?",
    TypeId.BYTE: "B",
    TypeId.CHAR: "b",  # signed byte, not unsigned
    TypeId.FLOAT32: "f",
    TypeId.FLOAT64: "d",
    TypeId.INT8: "b",
    TypeId.UINT8: "B",
    TypeId.INT16: "h",
    TypeId.UINT16: "H",
    TypeId.INT32: "i",
    TypeId.UINT32: "I",
    TypeId.INT64: "q",
    TypeId.UINT64: "Q",
    TypeId.PADDING: "x",  # Padding bytes, no size
}
UTF8_FUNC_NAME = "_decode_utf8"
