"""Message path parser for Foxglove message path syntax."""

from typing import Any

from .._lark_standalone_runtime import Token, Transformer
from ._standalone_parser import Lark_StandAlone
from .models import (
    ArrayIndex,
    ArraySlice,
    ComparisonOperator,
    FieldAccess,
    Filter,
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
        """Build Filter from field path, operator, and value."""
        field_path = items[0]
        operator_str = str(items[1])  # COMPARISON_OP token
        value = items[2]  # Already transformed by terminal methods or is Variable

        # Convert operator string to enum
        operator = ComparisonOperator(operator_str)

        return Filter(field_path=field_path, operator=operator, value=value)

    def field_path(self, items: list[Token]) -> str:
        """Build field path string from identifiers."""
        # Join all identifiers with dots
        return ".".join(str(item) for item in items)

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
    return _parser.parse(path)  # type: ignore[no-any-return]
