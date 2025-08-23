"""Message Path Grammar Parser using Lark.

This module provides a parser for Foxglove/ROS message path expressions
that enables navigation and filtering of nested data structures.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer
from lark.exceptions import LarkError


@dataclass
class Variable:
    """Represents a variable reference ($var)."""

    name: str
    repr: str


@dataclass
class SliceNotation:
    """Represents slice notation [start:end]."""

    start: int | Variable | None
    end: int | Variable | None
    repr: str


@dataclass
class PathComponent:
    """Base class for message path components."""

    repr: str


@dataclass
class FieldAccess(PathComponent):
    """Represents field access in a message path (.field)."""

    field: str


@dataclass
class ArrayAccess(PathComponent):
    """Represents array access in a message path ([index] or [slice])."""

    index: int | SliceNotation | Variable


@dataclass
class Filter(PathComponent):
    """Represents a filter in a message path ({condition})."""

    field: str
    operator: str
    value: str | int | float | bool | Variable


@dataclass
class ParsedMessagePath:
    """Result of parsing a message path."""

    topic_name: str
    message_path: list[PathComponent]
    modifier: str | None


class MessagePathTransformer(
    Transformer[
        Token,
        ParsedMessagePath
        | FieldAccess
        | ArrayAccess
        | Filter
        | SliceNotation
        | Variable
        | str
        | int
        | bool
        | dict[str, str | int | float | bool | Variable],
    ]
):
    """Transforms the parse tree into structured data."""

    def message_path(self, items: list[Any]) -> ParsedMessagePath:
        """Transform the main message path rule."""
        topic_name = items[0]
        path_components = [item for item in items[1:] if isinstance(item, PathComponent)]
        modifier = next(
            (
                item
                for item in items[1:]
                if isinstance(item, str) and not isinstance(item, PathComponent)
            ),
            None,
        )

        return ParsedMessagePath(
            topic_name=topic_name,
            message_path=path_components,
            modifier=modifier,
        )

    def topic_complex(self, items: list[str]) -> str:
        """Transform complex topic path."""
        return items[0]

    def topic_path(self, items: list[Token]) -> str:
        """Transform topic path with slashes."""
        parts = ["/"]  # Always starts with /
        for item in items:
            if hasattr(item, "value") and item.value != "/":
                if len(parts) > 1:  # Add slash between parts (not after initial /)
                    parts.append("/")
                parts.append(item.value)
        return "".join(parts)

    def topic_plain(self, items: list[Token]) -> str:
        """Transform plain topic name."""
        return items[0].value

    def topic_quoted(self, items: list[Token]) -> str:
        """Transform quoted topic name."""
        return items[0].value.strip("\"'")

    def path_component(self, items: list[PathComponent]) -> PathComponent:
        return items[0]

    def field_access(self, items: list[Token]) -> FieldAccess:
        """Transform field access (.field)."""
        field_name = items[0].value  # CNAME token
        return FieldAccess(field=field_name, repr=f".{field_name}")

    def array_access(self, items: list[int | SliceNotation | Variable]) -> ArrayAccess:
        """Transform array access ([index])."""
        index = items[0]
        repr_str = f"[{getattr(index, 'repr', index)}]"
        return ArrayAccess(index=index, repr=repr_str)

    def array_index(self, items: list[Any]) -> int | SliceNotation | Variable:
        """Transform array index."""
        value = items[0]
        return int(value.value) if hasattr(value, "value") else value

    def slice_start_end(self, items: list[Any]) -> SliceNotation:
        """Transform slice with both start and end ([start:end])."""
        start, end = items[0], items[1]
        return self._create_slice(start, end)

    def slice_start_only(self, items: list[Any]) -> SliceNotation:
        """Transform slice with only start ([start:])."""
        return self._create_slice(items[0], None)

    def slice_end_only(self, items: list[Any]) -> SliceNotation:
        """Transform slice with only end ([:end])."""
        return self._create_slice(None, items[0])

    def slice_full(self, _items: list[Token]) -> SliceNotation:
        """Transform full slice ([:])."""
        return self._create_slice(None, None)

    def _create_slice(
        self, start: int | Variable | None, end: int | Variable | None
    ) -> SliceNotation:
        """Helper to create SliceNotation with consistent repr generation."""
        start_repr = str(start) if start is not None else ""
        end_repr = str(end) if end is not None else ""
        repr_str = f"{start_repr}:{end_repr}"
        return SliceNotation(start=start, end=end, repr=repr_str)

    def slice_part(self, items: list[Any]) -> int | Variable:
        """Transform slice part."""
        value = items[0]
        return int(value.value) if hasattr(value, "value") else value

    def filter(self, items: list[Any]) -> Filter:
        """Transform filter ({condition})."""
        comparison = items[0]  # comparison
        field = comparison["field"]
        operator = comparison["operator"]
        value = comparison["value"]

        repr_str = f"{{{field}{operator}{value}}}"
        return Filter(field=field, operator=operator, value=value, repr=repr_str)

    def comparison(self, items: list[Any]) -> dict[str, str | int | float | bool | Variable]:
        """Transform comparison expression."""
        field = items[0]
        operator = items[1]
        value = items[2]
        return {"field": field, "operator": operator, "value": value}

    def field_name(self, items: list[Token]) -> str:
        """Transform field name (may be dotted)."""
        return ".".join(item.value for item in items if hasattr(item, "value"))

    # Operator transformations
    def eq(self, _items: list[Token]) -> str:
        return "=="

    def ne(self, _items: list[Token]) -> str:
        return "!="

    def ge(self, _items: list[Token]) -> str:
        return ">="

    def le(self, _items: list[Token]) -> str:
        return "<="

    def gt(self, _items: list[Token]) -> str:
        return ">"

    def lt(self, _items: list[Token]) -> str:
        return "<"

    def value(self, items: list[Any]) -> str | int | float | bool | Variable:
        """Transform value."""
        value = items[0]
        if hasattr(
            value, "value"
        ):  # Token (SIGNED_INT, SIGNED_FLOAT, ESCAPED_STRING, or SINGLE_QUOTED_STRING)
            if value.type == "SIGNED_INT":
                return int(value.value)
            if value.type == "SIGNED_FLOAT":
                return float(value.value)
            if value.type in ("ESCAPED_STRING", "SINGLE_QUOTED_STRING"):
                return value.value.strip("\"'")
            return value.value
        return value

    # Boolean transformations
    def true(self, _items: list[Token]) -> bool:
        return True

    def false(self, _items: list[Token]) -> bool:
        return False

    def variable(self, items: list[Token]) -> Variable:
        """Transform variable ($var)."""
        name = items[0].value if items and hasattr(items[0], "value") else ""
        return Variable(name=name, repr=f"${name}")

    def modifier(self, items: list[Token]) -> str:
        return items[0].value


class MessagePathParser:
    """Parser for Foxglove/ROS message paths."""

    def __init__(self) -> None:
        """Initialize the parser with the grammar."""
        grammar_path = Path(__file__).parent / "grammar.lark"
        with grammar_path.open() as f:
            grammar = f.read()

        self.parser = Lark(
            grammar,
            parser="lalr",
            transformer=MessagePathTransformer(),
        )

    def parse(self, message_path: str) -> ParsedMessagePath:
        """Parse a message path string.

        Args:
            message_path: The message path string to parse

        Returns:
            ParsedMessagePath: The parsed message path structure

        Raises:
            LarkError: If the message path is invalid
        """
        try:
            result = self.parser.parse(message_path)
            # When using a transformer, Lark returns the transformed result
            # Verify this is actually what we expect
            if not isinstance(result, ParsedMessagePath):
                msg = f"Parser returned unexpected type: {type(result)}"
                raise TypeError(msg)
            return result  # noqa: TRY300
        except LarkError as e:
            msg = f"Failed to parse message path '{message_path}': {e}"
            raise ValueError(msg) from e


def parse_message_path(message_path: str) -> ParsedMessagePath:
    """Parse a message path string.

    Convenience function for parsing message paths.

    Args:
        message_path: The message path string to parse

    Returns:
        ParsedMessagePath: The parsed message path structure

    Raises:
        ValueError: If the message path is invalid
    """
    parser = MessagePathParser()
    return parser.parse(message_path)
