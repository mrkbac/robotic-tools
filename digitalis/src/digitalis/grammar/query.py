"""Query execution engine for message paths.

This module provides functionality to apply parsed message paths
to Python objects with __slots__ or primitive types.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .parser import (
    ArrayAccess,
    FieldAccess,
    Filter,
    ParsedMessagePath,
    PathComponent,
    SliceNotation,
    Variable,
    parse_message_path,
)


class QueryError(Exception):
    """Exception raised when query execution fails."""


@dataclass
class QueryResult:
    """Result of a query execution."""

    value: Any
    path: str


class QueryExecutor:
    """Executes queries on Python objects."""

    def __init__(self, variables: dict[str, Any] | None = None) -> None:
        """Initialize query executor.

        Args:
            variables: Dictionary of variables for substitution
        """
        self.variables = variables or {}

    def execute(self, parsed_path: ParsedMessagePath, obj: Any) -> Any:
        """Execute a parsed message path on an object.

        Args:
            parsed_path: The parsed message path
            obj: The object to query

        Returns:
            The result of the query

        Raises:
            QueryError: If the query cannot be executed
        """
        # Start with the root object
        current = obj

        # Check if the topic name is actually a field name we need to access
        topic_name = parsed_path.topic_name
        if topic_name and not topic_name.startswith("/") and topic_name != "topic":
            # This is a field name, treat it as field access
            try:
                # For dicts, prioritize key access over attribute access
                if isinstance(current, dict) and topic_name in current:
                    current = current[topic_name]
                elif hasattr(current, topic_name):
                    current = getattr(current, topic_name)
                else:
                    raise QueryError(
                        f"Field '{topic_name}' not found on object of type {type(current).__name__}"
                    )
            except (AttributeError, KeyError) as e:
                raise QueryError(f"Cannot access field '{topic_name}': {e}") from e

        # Apply each path component in sequence
        for component in parsed_path.message_path:
            current = self._apply_component(component, current)

        return current

    def _apply_component(self, component: PathComponent, obj: Any) -> Any:
        """Apply a single path component to an object.

        Args:
            component: The path component to apply
            obj: The current object

        Returns:
            The result after applying the component
        """
        if isinstance(component, FieldAccess):
            return self._apply_field_access(component, obj)
        if isinstance(component, ArrayAccess):
            return self._apply_array_access(component, obj)
        if isinstance(component, Filter):
            return self._apply_filter(component, obj)
        raise QueryError(f"Unsupported path component: {type(component)}")

    def _apply_field_access(self, field_access: FieldAccess, obj: Any) -> Any:
        """Apply field access to an object.

        Args:
            field_access: The field access component
            obj: The object to access

        Returns:
            The field value
        """
        field_name = field_access.field

        # If obj is a list/sequence, apply field access to each element
        if isinstance(obj, (list, tuple)) and not isinstance(obj, str):
            try:
                return [self._apply_field_access(field_access, item) for item in obj]
            except QueryError:
                # If field access fails on elements, fall through to try on the container
                pass

        try:
            # For dicts, prioritize key access over attribute access
            if isinstance(obj, dict) and field_name in obj:
                return obj[field_name]

            # Try attribute access (for objects with __slots__ or regular attributes)
            if hasattr(obj, field_name):
                return getattr(obj, field_name)

            # Field not found
            raise QueryError(
                f"Field '{field_name}' not found on object of type {type(obj).__name__}"
            )

        except (AttributeError, KeyError, TypeError) as e:
            raise QueryError(f"Cannot access field '{field_name}': {e}") from e

    def _apply_array_access(self, array_access: ArrayAccess, obj: Any) -> Any:
        """Apply array access to an object.

        Args:
            array_access: The array access component
            obj: The object to index

        Returns:
            The indexed value or slice
        """
        # Allow indexing on sequences and dicts
        if not isinstance(obj, (Sequence, dict, str)):
            raise QueryError(f"Cannot index object of type {type(obj).__name__}")

        index = array_access.index

        try:
            if isinstance(index, int):
                return obj[index]
            if isinstance(index, SliceNotation):
                if isinstance(obj, dict):
                    raise QueryError("Cannot slice dictionary objects")
                return self._apply_slice(index, obj)
            if isinstance(index, Variable):
                resolved_index = self._resolve_variable(index)
                if isinstance(resolved_index, int):
                    return obj[resolved_index]
                raise QueryError(f"Variable {index.name} must resolve to an integer")
            raise QueryError(f"Unsupported index type: {type(index)}")

        except (IndexError, KeyError, TypeError) as e:
            raise QueryError(f"Array access failed: {e}") from e

    def _apply_slice(self, slice_notation: SliceNotation, obj: Sequence[Any]) -> list[Any]:
        """Apply slice notation to a sequence.

        Args:
            slice_notation: The slice notation
            obj: The sequence to slice

        Returns:
            The sliced sequence as a list
        """
        start = self._resolve_slice_value(slice_notation.start)
        end = self._resolve_slice_value(slice_notation.end)

        try:
            return list(obj[start:end])
        except (TypeError, ValueError) as e:
            raise QueryError(f"Slice operation failed: {e}") from e

    def _resolve_slice_value(self, value: int | Variable | None) -> int | None:
        """Resolve a slice value (handling variables).

        Args:
            value: The slice value to resolve

        Returns:
            The resolved integer value or None
        """
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, Variable):
            resolved = self._resolve_variable(value)
            if not isinstance(resolved, int):
                raise QueryError(f"Variable {value.name} must resolve to an integer")
            return resolved
        raise QueryError(f"Invalid slice value type: {type(value)}")

    def _apply_filter(self, filter_comp: Filter, obj: Any) -> Any:
        """Apply filter to an object or sequence.

        Args:
            filter_comp: The filter component
            obj: The object or sequence to filter

        Returns:
            Filtered results (single item or list)
        """
        # If obj is a sequence, filter each item
        if isinstance(obj, Sequence) and not isinstance(obj, str):
            return [item for item in obj if self._evaluate_filter(filter_comp, item)]
        # Single object - return it if it passes the filter, empty list otherwise
        if self._evaluate_filter(filter_comp, obj):
            return [obj]
        return []

    def _evaluate_filter(self, filter_comp: Filter, obj: Any) -> bool:
        """Evaluate a filter condition on an object.

        Args:
            filter_comp: The filter component
            obj: The object to test

        Returns:
            True if the object passes the filter
        """
        try:
            # Get the field value for comparison
            field_value = self._get_nested_field_value(filter_comp.field, obj)

            # Resolve the comparison value (might be a variable)
            comparison_value = self._resolve_filter_value(filter_comp.value)

            # Perform the comparison
            return self._compare_values(field_value, filter_comp.operator, comparison_value)

        except Exception as e:
            raise QueryError(f"Filter evaluation failed: {e}") from e

    def _get_nested_field_value(self, field_path: str, obj: Any) -> Any:
        """Get a potentially nested field value.

        Args:
            field_path: Dot-separated field path (e.g., "pose.position.x")
            obj: The object to query

        Returns:
            The field value
        """
        current = obj
        for field_name in field_path.split("."):
            if hasattr(current, field_name):
                current = getattr(current, field_name)
            elif isinstance(current, dict) and field_name in current:
                current = current[field_name]
            else:
                raise QueryError(f"Field '{field_name}' not found in field path '{field_path}'")

        return current

    def _resolve_filter_value(self, value: Any) -> Any:
        """Resolve a filter value (handling variables).

        Args:
            value: The value to resolve

        Returns:
            The resolved value
        """
        if isinstance(value, Variable):
            return self._resolve_variable(value)
        return value

    def _resolve_variable(self, variable: Variable) -> Any:
        """Resolve a variable to its value.

        Args:
            variable: The variable to resolve

        Returns:
            The variable value

        Raises:
            QueryError: If the variable is not found
        """
        if variable.name not in self.variables:
            raise QueryError(f"Variable '{variable.name}' not found in context")

        return self.variables[variable.name]

    def _compare_values(self, left: Any, operator: str, right: Any) -> bool:
        """Compare two values using the given operator.

        Args:
            left: Left operand
            operator: Comparison operator
            right: Right operand

        Returns:
            The comparison result
        """
        try:
            if operator == "==":
                return bool(left == right)
            if operator == "!=":
                return bool(left != right)
            if operator == "<":
                return bool(left < right)
            if operator == "<=":
                return bool(left <= right)
            if operator == ">":
                return bool(left > right)
            if operator == ">=":
                return bool(left >= right)
            raise QueryError(f"Unsupported comparison operator: {operator}")

        except TypeError as e:
            raise QueryError(
                f"Cannot compare {type(left).__name__} and {type(right).__name__}: {e}"
            ) from e


def parse_query(query: str, obj: Any, variables: dict[str, Any] | None = None) -> Any:
    """Parse a query string and apply it to an object.

    Convenience function that combines parsing and query execution.
    Since the input object already represents the correct topic,
    we add a dummy topic name for parsing if needed.

    Args:
        query: The query string to parse and execute
        obj: The object to query (already represents the correct topic)
        variables: Optional variables for substitution

    Returns:
        The query result

    Raises:
        ValueError: If the query cannot be parsed
        QueryError: If the query cannot be executed
    """
    # Add dummy topic name if query doesn't start with one
    if not query.startswith("/") and not query.startswith('"') and not query.startswith("'"):
        # For field access or filters, prefix with dummy topic
        if query.startswith((".", "{")):
            query = f"/topic{query}"
        elif query.startswith("["):
            # Direct array access
            query = f"/topic{query}"
        else:
            # Field name without dot
            query = f"/topic.{query}"

    parsed = parse_message_path(query)
    return apply_query(parsed, obj, variables)


def apply_query(
    parsed_path: ParsedMessagePath, obj: Any, variables: dict[str, Any] | None = None
) -> Any:
    """Apply a parsed message path to an object.

    Args:
        parsed_path: The parsed message path
        obj: The object to query
        variables: Optional variables for substitution

    Returns:
        The query result

    Raises:
        QueryError: If the query cannot be executed
    """
    executor = QueryExecutor(variables)
    return executor.execute(parsed_path, obj)
