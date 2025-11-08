from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MessagePathError(Exception):
    """Exception raised when message path operations fail."""


class ComparisonOperator(Enum):
    EQUAL = "=="
    NOT_EQUAL = "!="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="


@dataclass
class Variable:
    name: str


_VariableStore = dict[str, Any]


class Action(ABC):
    @abstractmethod
    def apply(self, obj: Any, variables: _VariableStore) -> Any:
        """Apply this action to an object."""


@dataclass
class FieldAccess(Action):
    field_name: str

    def apply(self, obj: Any, _variables: _VariableStore) -> Any:
        """Access a field from an object, supporting both attribute and dict-like access."""
        # Try dict/mapping access first
        if isinstance(obj, Mapping):
            if self.field_name in obj:
                return obj[self.field_name]
            raise MessagePathError(
                f"Field '{self.field_name}' not found in mapping with keys: {list(obj.keys())}"
            )

        # Try attribute access for objects
        try:
            return getattr(obj, self.field_name)
        except AttributeError:
            pass

        # If we get here, the field doesn't exist
        obj_type = type(obj).__name__
        raise MessagePathError(
            f"Field '{self.field_name}' not found on object of type '{obj_type}'"
        )


@dataclass
class ArrayIndex(Action):
    index: int | Variable

    def apply(self, obj: Any, variables: _VariableStore) -> Any:
        """Index into a sequence (list, tuple, str) by position."""
        idx = variables[self.index.name] if isinstance(self.index, Variable) else self.index

        if not isinstance(obj, Sequence):
            obj_type = type(obj).__name__
            raise MessagePathError(
                f"ArrayIndex can only be applied to sequences (list, tuple, str), got '{obj_type}'"
            )

        try:
            return obj[idx]
        except IndexError as e:
            raise MessagePathError(
                f"Index {idx} out of range for sequence of length {len(obj)}"
            ) from e


@dataclass
class ArraySlice(Action):
    start: int | Variable | None
    end: int | Variable | None

    def apply(self, obj: Any, variables: _VariableStore) -> Any:
        """
        Slice a sequence with INCLUSIVE end index (per Foxglove spec).

        Unlike Python slicing where [1:3] returns indices 1 and 2,
        the Foxglove spec requires [1:3] to return indices 1, 2, and 3.
        """
        start_idx = variables[self.start.name] if isinstance(self.start, Variable) else self.start
        end_idx = variables[self.end.name] if isinstance(self.end, Variable) else self.end

        if not isinstance(obj, Sequence):
            obj_type = type(obj).__name__
            raise MessagePathError(
                f"ArraySlice can only be applied to sequences (list, tuple, str), got '{obj_type}'"
            )

        # Convert to inclusive end index for Foxglove spec compliance
        # Python slicing is exclusive, but Foxglove spec is inclusive
        if end_idx is not None:
            # For negative indices like -1 (last element), convert to None to include it
            # For all other indices (positive or negative), add 1 to make it inclusive
            end_idx = None if end_idx == -1 else end_idx + 1

        try:
            return obj[start_idx:end_idx]
        except IndexError as e:
            raise MessagePathError(
                f"Slice [{start_idx}:{end_idx}] out of range for sequence of length {len(obj)}"
            ) from e


@dataclass
class Filter(Action):
    field_path: str
    operator: ComparisonOperator
    value: int | float | str | bool | Variable

    def apply(self, obj: Any, variables: _VariableStore) -> list[Any]:
        """
        Filter a sequence based on a field comparison.

        Returns a new list containing only items where the field comparison evaluates to true.
        """
        if not isinstance(obj, (list, tuple)):
            obj_type = type(obj).__name__
            raise MessagePathError(
                f"Filter can only be applied to lists or tuples, got '{obj_type}'"
            )

        filtered: list[Any] = []
        for item in obj:
            try:
                field_value = self._get_field_value(item)
            except MessagePathError:
                # If field doesn't exist on this item, skip it
                continue

            compare_value = (
                variables[self.value.name] if isinstance(self.value, Variable) else self.value
            )
            if self._compare(field_value, compare_value):
                filtered.append(item)
        return filtered

    def _get_field_value(self, obj: Any) -> Any:
        """Extract field value from nested field path (e.g., 'pose.x')."""
        field_value = obj
        for part in self.field_path.split("."):
            # Use FieldAccess logic for consistent field access
            if isinstance(field_value, Mapping):
                if part not in field_value:
                    raise MessagePathError(f"Field '{part}' not found in mapping")
                field_value = field_value[part]
            else:
                try:
                    field_value = getattr(field_value, part)
                except AttributeError:
                    obj_type = type(field_value).__name__
                    raise MessagePathError(
                        f"Field '{part}' not found on object of type '{obj_type}'"
                    ) from None
        return field_value

    def _compare(self, field_value: Any, compare_value: Any) -> bool:
        """Compare two values according to the operator."""
        try:
            if self.operator == ComparisonOperator.EQUAL:
                return bool(field_value == compare_value)
            if self.operator == ComparisonOperator.NOT_EQUAL:
                return bool(field_value != compare_value)

            # For ordering comparisons, ensure types are comparable
            # Allow int/float mixing but catch other type mismatches
            if self.operator == ComparisonOperator.LESS_THAN:
                return bool(field_value < compare_value)
            if self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
                return bool(field_value <= compare_value)
            if self.operator == ComparisonOperator.GREATER_THAN:
                return bool(field_value > compare_value)
            if self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
                return bool(field_value >= compare_value)

            raise MessagePathError(f"Unsupported comparison operator: {self.operator}")
        except TypeError as e:
            # Comparison between incompatible types
            raise MessagePathError(
                f"Cannot compare {type(field_value).__name__} with {type(compare_value).__name__} "
                f"using operator {self.operator.value}"
            ) from e


@dataclass
class MessagePath:
    topic: str
    segments: list[FieldAccess | ArrayIndex | ArraySlice | Filter]

    def apply(self, obj: Any, variables: _VariableStore | None = None) -> Any:
        """
        Apply all segments in the message path to an object.

        Args:
            obj: The object to apply the message path to
            variables: Optional dictionary of variable values for substitution

        Returns:
            The result after applying all segments

        Raises:
            MessagePathError: If any segment fails to apply
        """
        if variables is None:
            variables = {}

        result = obj
        for segment in self.segments:
            result = segment.apply(result, variables)
        return result
