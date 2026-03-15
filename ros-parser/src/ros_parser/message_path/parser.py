"""Message path parser for Foxglove message path syntax."""

from typing import Any

from .._lark_standalone_runtime import Token, Transformer
from ._standalone_parser import Lark_StandAlone
from .models import (
    ArrayIndex,
    ArraySlice,
    Comparison,
    ComparisonOperator,
    CompoundFilter,
    FieldAccess,
    Filter,
    FilterExpression,
    FilterFieldRef,
    FilterValue,
    InExpression,
    MathModifier,
    MessagePath,
    Variable,
)


class MessagePathTransformer(Transformer[Token, MessagePath]):
    """Transforms Lark parse tree into message path data structures."""

    def message_path(self, items: list[Any]) -> MessagePath:
        """Build MessagePath from topic and segments."""
        topic = items[0]
        segments = items[1:]
        return MessagePath(topic=topic, segments=segments)

    def topic_ref(self, items: list[Token]) -> str:
        """Extract topic name - supports hierarchical topic paths."""
        # Join all identifiers with slashes to reconstruct full topic path
        return "/" + "/".join(str(item) for item in items)

    def field_access(self, items: list[Token]) -> FieldAccess:
        """Build FieldAccess from field name."""
        field_name = str(items[0])
        return FieldAccess(field_name=field_name)

    def array_index(self, items: list[Any]) -> ArrayIndex:
        """Build ArrayIndex from index value."""
        index = items[0]
        # Handle Token or Variable
        if isinstance(index, Variable):
            return ArrayIndex(index=index)
        # It's a Token with integer
        return ArrayIndex(index=int(index))

    def slice_both(self, items: list[Any]) -> ArraySlice:
        """Handle [start:end] slice."""
        start = items[0]
        end = items[1]
        return ArraySlice(start=start, end=end)

    def slice_start_only(self, items: list[Any]) -> ArraySlice:
        """Handle [start:] slice."""
        start = items[0]
        return ArraySlice(start=start, end=None)

    def slice_end_only(self, items: list[Any]) -> ArraySlice:
        """Handle [:end] slice."""
        end = items[0]
        return ArraySlice(start=None, end=end)

    def slice_all(self, items: list[Any]) -> ArraySlice:  # noqa: ARG002
        """Handle [:] slice."""
        return ArraySlice(start=None, end=None)

    def slice_value(self, items: list[Any]) -> int | Variable:
        """Return slice value (int or Variable)."""
        item = items[0]
        if isinstance(item, Variable):
            return item
        return int(item)

    def filter(self, items: list[Any]) -> Filter:
        """Build Filter from a filter expression."""
        return Filter(expression=items[0])

    def or_expr(self, items: list[FilterExpression]) -> CompoundFilter:
        """Build OR compound filter."""
        return CompoundFilter(op="or", children=items)

    def and_expr(self, items: list[FilterExpression]) -> CompoundFilter:
        """Build AND compound filter."""
        return CompoundFilter(op="and", children=items)

    def not_filter(self, items: list[FilterExpression]) -> CompoundFilter:
        """Build NOT compound filter."""
        return CompoundFilter(op="not", children=[items[0]])

    def comparison(self, items: list[Any]) -> Comparison:
        """Build Comparison from field path, operator, and value."""
        field_path: str = items[0]
        operator = ComparisonOperator(str(items[1]))
        value: FilterValue = items[2]
        return Comparison(field_path=field_path, operator=operator, value=value)

    def in_expr(self, items: list[Any]) -> InExpression:
        """Build InExpression from field path and list of values."""
        field_path: str = items[0]
        values: list[FilterValue] = items[1:]
        return InExpression(field_path=field_path, values=values)

    def filter_literal(self, items: list[Any]) -> FilterValue:
        """Pass through a literal filter value."""
        return items[0]

    def filter_field_ref(self, items: list[Any]) -> FilterFieldRef:
        """Build a field reference for cross-field comparison."""
        return FilterFieldRef(field_path=items[0])

    def field_path(self, items: list[Token]) -> str:
        """Build field path string from identifiers."""
        return ".".join(str(item) for item in items)

    def math_modifier_with_args(self, items: list[Any]) -> MathModifier:
        """Build MathModifier with arguments from parsed tokens."""
        operation = str(items[0])  # IDENTIFIER token
        # items[1] is the list of arguments from modifier_args
        arguments = items[1] if len(items) > 1 else []
        return MathModifier(operation=operation, arguments=arguments)

    def math_modifier_no_args(self, items: list[Any]) -> MathModifier:
        """Build MathModifier without arguments from parsed tokens."""
        operation = str(items[0])  # IDENTIFIER token
        return MathModifier(operation=operation, arguments=[])

    def modifier_args(self, items: list[Any]) -> list[int | float | Variable]:
        """Collect all modifier arguments into a flat list."""
        return items

    def variable(self, items: list[Token]) -> Variable:
        """Build Variable from name."""
        name = str(items[0])
        return Variable(name=name)

    def BOOLEAN(self, token: Token) -> bool:  # noqa: N802
        """Transform BOOLEAN terminal."""
        return str(token) == "true"

    def SIGNED_NUMBER(self, token: Token) -> int | float:  # noqa: N802
        """Transform SIGNED_NUMBER terminal."""
        s = str(token)
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)

    def STRING(self, token: Token) -> str:  # noqa: N802
        """Transform STRING terminal."""
        return self._parse_string(str(token))

    def IDENTIFIER(self, token: Token) -> str:  # noqa: N802
        """Transform IDENTIFIER terminal."""
        return str(token)

    def _parse_string(self, s: str) -> str:
        """Parse string literal, removing quotes and handling escapes."""
        # Remove surrounding quotes
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1]
        # Unescape standard escape sequences and return
        return (
            s.replace("\\'", "'")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace("\\\\", "\\")
        )


_parser: Any = Lark_StandAlone(transformer=MessagePathTransformer())  # type: ignore[arg-type]


def parse_message_path(path: str) -> MessagePath:
    """
    Parse a message path string into a MessagePath object.

    Args:
        path: The message path string (e.g., "/topic.field[0]{x>5}")

    Returns:
        Parsed MessagePath object

    Raises:
        LarkError: If the path syntax is invalid
    """
    return _parser.parse(path)
