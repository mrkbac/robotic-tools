import contextlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ros_parser.message_definition import MessageDefinition, Type


class MessagePathError(Exception):
    """Exception raised when message path operations fail."""


class ValidationError(Exception):
    """Error raised when a message path is invalid for a given message definition."""


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

    @abstractmethod
    def validate(
        self,
        current_type: "Type",
        current_msgdef: "MessageDefinition | None",
        all_definitions: dict[str, "MessageDefinition"],
    ) -> tuple["Type", "MessageDefinition | None"]:
        """Validate this action against a schema and return the resulting type.

        Args:
            current_type: The current type before applying this action
            current_msgdef: The current message definition (if complex type)
            all_definitions: Dict of all message definitions for resolving types

        Returns:
            Tuple of (resulting_type, resulting_msgdef) after applying this action

        Raises:
            ValidationError: If this action is invalid for the current type
        """


@dataclass
class FieldAccess(Action):
    field_name: str

    def apply(self, obj: Any, variables: _VariableStore) -> Any:  # noqa: ARG002
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

    def validate(
        self,
        current_type: "Type",
        current_msgdef: "MessageDefinition | None",
        all_definitions: dict[str, "MessageDefinition"],
    ) -> tuple["Type", "MessageDefinition | None"]:
        """Validate field access and return the field's type."""

        # Cannot access fields on primitives
        if current_type.is_primitive:
            raise ValidationError(
                f"Cannot access field '{self.field_name}' on primitive type '{current_type}'"
            )

        # Cannot access fields on arrays directly - need to index/slice first
        if current_type.is_array:
            raise ValidationError(
                f"Cannot access field '{self.field_name}' on array type '{current_type}'. "
                "Use array indexing or slicing first"
            )

        # Get the message definition if we don't already have it
        if current_msgdef is None:
            current_msgdef = _get_message_definition(current_type, all_definitions)

        # Find the field
        field = next((f for f in current_msgdef.fields if f.name == self.field_name), None)
        if not field:
            available = [f.name for f in current_msgdef.fields]
            raise ValidationError(
                f"Field '{self.field_name}' not found in message '{current_type}'. "
                f"Available fields: {', '.join(available) if available else 'none'}"
            )

        # Return the field's type and its message definition if it's complex
        field_msgdef = None
        if not field.type.is_primitive and not field.type.is_array:
            # It's a complex type, try to get its definition
            with contextlib.suppress(ValidationError):
                field_msgdef = _get_message_definition(field.type, all_definitions)

        return field.type, field_msgdef


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

    def validate(
        self,
        current_type: "Type",
        current_msgdef: "MessageDefinition | None",  # noqa: ARG002
        all_definitions: dict[str, "MessageDefinition"],
    ) -> tuple["Type", "MessageDefinition | None"]:
        """Validate array index and return the element type."""

        if not current_type.is_array:
            raise ValidationError(f"Cannot apply array index to non-array type '{current_type}'")

        # Return element type (unwrap array)
        element_type = Type(
            type_name=current_type.type_name,
            package_name=current_type.package_name,
            is_array=False,
            array_size=None,
            is_upper_bound=False,
            string_upper_bound=current_type.string_upper_bound,
        )

        # Get message definition if it's a complex type
        element_msgdef = None
        if not element_type.is_primitive:
            with contextlib.suppress(ValidationError):
                element_msgdef = _get_message_definition(element_type, all_definitions)

        return element_type, element_msgdef


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

    def validate(
        self,
        current_type: "Type",
        current_msgdef: "MessageDefinition | None",
        all_definitions: dict[str, "MessageDefinition"],  # noqa: ARG002
    ) -> tuple["Type", "MessageDefinition | None"]:
        """Validate array slice and return the array type (slices preserve array type)."""
        if not current_type.is_array:
            raise ValidationError(f"Cannot apply array slice to non-array type '{current_type}'")

        # For slices, the result is still an array (same type as input)
        # No need to look up message definition as it's the same as current
        return current_type, current_msgdef


@dataclass
class Filter(Action):
    field_path: str
    operator: ComparisonOperator
    value: int | float | str | bool | Variable

    def apply(self, obj: Any, variables: _VariableStore) -> Any:
        """
        Filter a sequence or single object based on a field comparison.

        For sequences (list/tuple): Returns a new list with only matching items.
        For single objects: Returns the object if it matches, or None if it doesn't.
        """
        # Handle single objects
        if not isinstance(obj, (list, tuple)):
            try:
                field_value = self._get_field_value(obj)
            except MessagePathError:
                # Field doesn't exist on this object
                return None

            compare_value = (
                variables[self.value.name] if isinstance(self.value, Variable) else self.value
            )
            # Return the object if it matches, None otherwise
            return obj if self._compare(field_value, compare_value) else None

        # Handle sequences (lists/tuples)
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

    def validate(
        self,
        current_type: "Type",
        current_msgdef: "MessageDefinition | None",
        all_definitions: dict[str, "MessageDefinition"],
    ) -> tuple["Type", "MessageDefinition | None"]:
        """Validate filter operation and return the same type."""

        # Determine the type to validate the filter field path against
        validate_type = current_type
        validate_msgdef = current_msgdef

        # If it's an array, validate against the element type
        if current_type.is_array:
            validate_type = Type(
                type_name=current_type.type_name,
                package_name=current_type.package_name,
                is_array=False,
                array_size=None,
                is_upper_bound=False,
                string_upper_bound=current_type.string_upper_bound,
            )
            # Get element's message definition if complex
            validate_msgdef = None
            if not validate_type.is_primitive:
                with contextlib.suppress(ValidationError):
                    validate_msgdef = _get_message_definition(validate_type, all_definitions)

        # Validate the filter's field path
        field_parts = self.field_path.split(".")
        working_type = validate_type
        working_msgdef = validate_msgdef

        for part in field_parts:
            if working_type.is_primitive:
                raise ValidationError(
                    f"Cannot access field '{part}' on primitive type '{working_type}' "
                    f"in filter field path '{self.field_path}'"
                )

            if working_type.is_array:
                raise ValidationError(
                    f"Cannot access field '{part}' on array type '{working_type}' "
                    f"in filter field path '{self.field_path}'. "
                    "Nested array filtering is not supported"
                )

            if working_msgdef is None:
                working_msgdef = _get_message_definition(working_type, all_definitions)

            field = next((f for f in working_msgdef.fields if f.name == part), None)
            if not field:
                available = [f.name for f in working_msgdef.fields]
                raise ValidationError(
                    f"Field '{part}' not found in message '{working_type}' "
                    f"in filter field path '{self.field_path}'. "
                    f"Available fields: {', '.join(available) if available else 'none'}"
                )

            working_type = field.type
            # Update working_msgdef if it's a complex type
            working_msgdef = None
            if not working_type.is_primitive and not working_type.is_array:
                with contextlib.suppress(ValidationError):
                    working_msgdef = _get_message_definition(working_type, all_definitions)

        # Filter returns the same type as input
        return current_type, current_msgdef


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

    def validate(
        self,
        message_def: "MessageDefinition",
        all_definitions: dict[str, "MessageDefinition"],
    ) -> None:
        """Validate that this message path is valid for a given message definition.

        Args:
            message_def: Root message definition for the topic
            all_definitions: Dict of all message definitions for resolving complex types

        Raises:
            ValidationError: If the path is invalid with detailed error message
        """

        # Start with a pseudo-type representing the root message
        current_type = Type(
            type_name=message_def.name.split("/")[-1] if message_def.name else "Message",
            package_name=_extract_package_name(message_def.name) if message_def.name else None,
            is_array=False,
            array_size=None,
            is_upper_bound=False,
            string_upper_bound=None,
        )

        # Track the current message definition (for the root)
        current_msgdef: MessageDefinition | None = message_def

        for segment in self.segments:
            current_type, current_msgdef = segment.validate(
                current_type, current_msgdef, all_definitions
            )


def _get_message_definition(
    type_: "Type",
    all_definitions: dict[str, "MessageDefinition"],
) -> "MessageDefinition":
    """Get the message definition for a type.

    Tries multiple key formats to find the definition:
    - package_name/msg/type_name (e.g., "geometry_msgs/msg/Point")
    - package_name/type_name (e.g., "geometry_msgs/Point")
    - type_name (e.g., "Point")
    """
    if type_.is_primitive:
        raise ValidationError(f"Cannot get message definition for primitive type '{type_}'")

    # Try with package name first
    if type_.package_name:
        # Try full path with /msg/
        type_path = f"{type_.package_name}/msg/{type_.type_name}"
        if type_path in all_definitions:
            return all_definitions[type_path]

        # Try without /msg/
        type_path = f"{type_.package_name}/{type_.type_name}"
        if type_path in all_definitions:
            return all_definitions[type_path]

    # Try just the type name
    if type_.type_name in all_definitions:
        return all_definitions[type_.type_name]

    raise ValidationError(f"Message definition not found for type '{type_}'")


def _extract_package_name(full_name: str | None) -> str | None:
    """Extract package name from full message name like 'geometry_msgs/msg/Point'."""
    if not full_name:
        return None
    parts = full_name.split("/")
    if len(parts) >= 2:
        return parts[0]
    return None
