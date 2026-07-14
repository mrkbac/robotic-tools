import array
import contextlib
import difflib
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeAlias, cast

from ros_parser.models import MessageDefinition, PrimitiveValue, Type


class MessagePathError(Exception):
    """Exception raised when message path operations fail."""


class FieldResolutionError(MessagePathError):
    """Exception raised when a field path cannot be resolved on an object."""


class ValidationError(Exception):
    """Error raised when a message path is invalid for a given message definition."""


# Sentinel for "attribute absent" so field access can avoid try/except getattr.
_MISSING = object()

# Types treated as numeric arrays for element-wise math modifiers. ``str`` is
# intentionally excluded; decoded numeric arrays are ``memoryview``/``bytes``.
_ARRAY_TYPES = (list, tuple, memoryview, bytearray, bytes, array.array)


def _lookup_field(obj: Any, name: str) -> Any:
    """Resolve a field by dict key or attribute, returning ``_MISSING`` if absent.

    The branch order is correctness-critical, not just a fast path:

    1. plain ``dict`` — subscript (collision-safe for keys named like dict methods);
    2. dataclass — attribute access, skipping the ``Mapping`` ABC ``isinstance`` cost
       (decoded ROS messages are slotted dataclasses, the hot case);
    3. any other ``Mapping`` — subscript *before* attribute access, so a key that
       shadows a method name (e.g. ``keys``/``items``) yields the value, not the
       bound method;
    4. any other object — attribute access.
    """
    if type(obj) is dict:
        return cast("dict[str, Any]", obj)[name] if name in obj else _MISSING
    if getattr(type(obj), "__dataclass_fields__", None) is not None:
        return getattr(obj, name, _MISSING)
    if isinstance(obj, Mapping):
        return cast("Mapping[str, Any]", obj)[name] if name in obj else _MISSING
    return getattr(obj, name, _MISSING)


def _available_fields(obj: object) -> list[str]:
    """Best-effort list of field/key names on an object, for error suggestions."""
    dataclass_fields = getattr(type(obj), "__dataclass_fields__", None)
    if dataclass_fields is not None:
        return list(dataclass_fields)
    if isinstance(obj, Mapping):
        return [str(key) for key in obj]
    return []


def _field_not_found_message(obj: object, name: str) -> str:
    """Build a 'field not found' message with a 'did you mean' hint when possible."""
    base = f"Field '{name}' not found on object of type '{type(obj).__name__}'"
    available = _available_fields(obj)
    if not available:
        return base
    close = difflib.get_close_matches(name, available, n=1)
    if close:
        return f"{base}. Did you mean '{close[0]}'?"
    return f"{base}. Available fields: {', '.join(available)}"


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


MessagePathVariable: TypeAlias = bool | int | float | str
MessagePathVariables: TypeAlias = Mapping[str, MessagePathVariable]
_VariableStore: TypeAlias = MessagePathVariables


def _resolve_variable(variable: Variable, variables: _VariableStore) -> MessagePathVariable:
    try:
        return variables[variable.name]
    except KeyError as exc:
        raise MessagePathError(f"Variable '${variable.name}' was not provided") from exc


def _resolve_integer_variable(variable: Variable, variables: _VariableStore) -> int:
    value = _resolve_variable(variable, variables)
    if type(value) is not int:
        raise MessagePathError(f"Variable '${variable.name}' must be an integer")
    return value


def _resolve_numeric_variable(variable: Variable, variables: _VariableStore) -> int | float:
    value = _resolve_variable(variable, variables)
    if type(value) not in (int, float):
        raise MessagePathError(f"Variable '${variable.name}' must be numeric")
    return cast("int | float", value)


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

    def apply(self, obj: Any, variables: _VariableStore) -> Any:
        """Access a field, mapping element-wise over a list/tuple.

        After a slice or filter yields a sequence, ``.field`` reads that field
        from each element (Foxglove semantics), so ``arr[:].x`` and
        ``arr{f}.x`` return a list of the values.
        """
        obj_type = type(obj)
        if obj_type is list or obj_type is tuple:
            return [self.apply(item, variables) for item in obj]
        name = self.field_name
        value = _lookup_field(obj, name)
        if value is not _MISSING:
            return value
        raise MessagePathError(_field_not_found_message(obj, name))

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
            # It's a complex type, try to get its definition.
            # ValidationError is suppressed because the type may reference an external
            # message that isn't in all_definitions - this is valid for partial validation
            # (e.g., when validating paths without all dependencies loaded).
            with contextlib.suppress(ValidationError):
                field_msgdef = _get_message_definition(field.type, all_definitions)

        return field.type, field_msgdef


@dataclass
class ArrayIndex(Action):
    index: int | Variable

    def apply(self, obj: Any, variables: _VariableStore) -> Any:
        """Index into a sequence (list, tuple, str) by position."""
        idx = (
            _resolve_integer_variable(self.index, variables)
            if isinstance(self.index, Variable)
            else self.index
        )

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

        # Get message definition if it's a complex type.
        # ValidationError is suppressed because the element type may reference an external
        # message not in all_definitions - this allows partial validation without all deps.
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
        start_idx = (
            _resolve_integer_variable(self.start, variables)
            if isinstance(self.start, Variable)
            else self.start
        )
        end_idx = (
            _resolve_integer_variable(self.end, variables)
            if isinstance(self.end, Variable)
            else self.end
        )

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
class FilterFieldRef:
    """A reference to a field path used as a value in filter comparisons (cross-field)."""

    field_path: str
    _parts: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _constant_value: PrimitiveValue | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _is_constant: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._parts = tuple(self.field_path.split("."))

    def resolve(self, obj: Any) -> Any:
        """Resolve the field path against an object."""
        if self._is_constant:
            return self._constant_value
        return _resolve_parts(obj, self._parts)

    def bind_constant(self, value: PrimitiveValue) -> None:
        """Resolve this ambiguous bare identifier as a schema enum constant."""
        self._constant_value = value
        self._is_constant = True

    def bind_field(self) -> None:
        """Resolve this identifier as a cross-field reference."""
        self._constant_value = None
        self._is_constant = False


FilterValue = int | float | str | bool | Variable | FilterFieldRef
CurrentFilterValue = int | float | str | bool | Variable


def _resolve_parts(obj: Any, parts: tuple[str, ...]) -> Any:
    """Extract a value by walking pre-split field-path parts (e.g. ('pose', 'x'))."""
    value = obj
    for part in parts:
        nxt = _lookup_field(value, part)
        if nxt is _MISSING:
            raise FieldResolutionError(_field_not_found_message(value, part))
        value = nxt
    return value


def _resolve_field_path(obj: Any, field_path: str) -> Any:
    """Extract field value from a dotted field path (e.g. 'pose.x')."""
    return _resolve_parts(obj, tuple(field_path.split(".")))


def _resolve_filter_value(val: FilterValue, obj: Any, variables: _VariableStore) -> Any:
    """Resolve a filter value to a concrete value for comparison."""
    if isinstance(val, Variable):
        return _resolve_variable(val, variables)
    if isinstance(val, FilterFieldRef):
        return val.resolve(obj)
    return val


def _compare(field_value: Any, operator: ComparisonOperator, compare_value: Any) -> bool:
    """Compare two values according to the operator."""
    try:
        if operator == ComparisonOperator.EQUAL:
            return bool(field_value == compare_value)
        if operator == ComparisonOperator.NOT_EQUAL:
            return bool(field_value != compare_value)
        if operator == ComparisonOperator.LESS_THAN:
            return bool(field_value < compare_value)
        if operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            return bool(field_value <= compare_value)
        if operator == ComparisonOperator.GREATER_THAN:
            return bool(field_value > compare_value)
        if operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return bool(field_value >= compare_value)
        raise MessagePathError(f"Unsupported comparison operator: {operator}")
    except TypeError as e:
        raise MessagePathError(
            f"Cannot compare {type(field_value).__name__} with {type(compare_value).__name__} "
            f"using operator {operator.value}"
        ) from e


class FilterExpression(ABC):
    """Base class for filter expressions (used inside {})."""

    @abstractmethod
    def evaluate(self, obj: Any, variables: _VariableStore) -> bool:
        """Evaluate this filter expression against an object."""


@dataclass
class Comparison(FilterExpression):
    """A single comparison: field_path op value."""

    field_path: str
    operator: ComparisonOperator
    value: FilterValue
    _parts: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._parts = tuple(self.field_path.split("."))

    def evaluate(self, obj: Any, variables: _VariableStore) -> bool:
        field_value = _resolve_parts(obj, self._parts)
        compare_value = _resolve_filter_value(self.value, obj, variables)
        return _compare(field_value, self.operator, compare_value)


@dataclass
class CurrentValueComparison(FilterExpression):
    """A comparison against the current scalar or primitive sequence element."""

    operator: ComparisonOperator
    value: CurrentFilterValue

    def evaluate(self, obj: Any, variables: _VariableStore) -> bool:
        compare_value = _resolve_filter_value(self.value, obj, variables)
        return _compare(obj, self.operator, compare_value)


@dataclass
class CurrentValueInExpression(FilterExpression):
    """A membership test against the current scalar or primitive sequence element."""

    values: list[CurrentFilterValue]

    def evaluate(self, obj: Any, variables: _VariableStore) -> bool:
        resolved = [_resolve_filter_value(value, obj, variables) for value in self.values]
        return obj in resolved


@dataclass
class InExpression(FilterExpression):
    """Membership test: field_path in [val1, val2, ...]."""

    field_path: str
    values: list[FilterValue]
    _parts: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._parts = tuple(self.field_path.split("."))

    def evaluate(self, obj: Any, variables: _VariableStore) -> bool:
        field_value = _resolve_parts(obj, self._parts)
        resolved = [_resolve_filter_value(v, obj, variables) for v in self.values]
        return field_value in resolved


@dataclass
class CompoundFilter(FilterExpression):
    """Boolean combination of filter expressions."""

    op: Literal["and", "or", "not"]
    children: list[FilterExpression]

    def evaluate(self, obj: Any, variables: _VariableStore) -> bool:
        if self.op == "and":
            return all(child.evaluate(obj, variables) for child in self.children)
        if self.op == "or":
            return any(child.evaluate(obj, variables) for child in self.children)
        # not
        return not self.children[0].evaluate(obj, variables)


@dataclass
class Filter(Action):
    expression: FilterExpression

    def apply(self, obj: Any, variables: _VariableStore) -> Any:
        """
        Filter a sequence or single object based on a filter expression.

        For sequences: Returns a new list with only matching items.
        For single objects: Returns the object if it matches, or None if it doesn't.
        """
        if not isinstance(obj, _ARRAY_TYPES):
            try:
                return obj if self.expression.evaluate(obj, variables) else None
            except FieldResolutionError:
                return None

        filtered: list[Any] = []
        for item in obj:
            try:
                if self.expression.evaluate(item, variables):
                    filtered.append(item)
            except FieldResolutionError:
                continue
        return filtered

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
            validate_msgdef = None
            if not validate_type.is_primitive:
                with contextlib.suppress(ValidationError):
                    validate_msgdef = _get_message_definition(validate_type, all_definitions)

        # Validate all field paths in the expression
        self._validate_expression(self.expression, validate_type, validate_msgdef, all_definitions)

        # Filter returns the same type as input
        return current_type, current_msgdef

    def _validate_expression(
        self,
        expr: FilterExpression,
        validate_type: "Type",
        validate_msgdef: "MessageDefinition | None",
        all_definitions: dict[str, "MessageDefinition"],
    ) -> None:
        """Recursively validate all field paths in a filter expression."""
        if isinstance(expr, Comparison):
            field_type, field_owner = _validate_field_path(
                expr.field_path, validate_type, validate_msgdef, all_definitions
            )
            if isinstance(expr.value, FilterFieldRef):
                self._validate_filter_field_ref(
                    expr.value,
                    field_type,
                    field_owner,
                    validate_type,
                    validate_msgdef,
                    all_definitions,
                )
        elif isinstance(expr, (CurrentValueComparison, CurrentValueInExpression)):
            if not validate_type.is_primitive:
                raise ValidationError(
                    "current-value filter can only be applied to a primitive scalar "
                    f"or array of primitives, got '{validate_type}'"
                )
        elif isinstance(expr, InExpression):
            field_type, field_owner = _validate_field_path(
                expr.field_path, validate_type, validate_msgdef, all_definitions
            )
            for val in expr.values:
                if isinstance(val, FilterFieldRef):
                    self._validate_filter_field_ref(
                        val,
                        field_type,
                        field_owner,
                        validate_type,
                        validate_msgdef,
                        all_definitions,
                    )
        elif isinstance(expr, CompoundFilter):
            for child in expr.children:
                self._validate_expression(child, validate_type, validate_msgdef, all_definitions)

    def _validate_filter_field_ref(
        self,
        ref: FilterFieldRef,
        compared_type: "Type",
        compared_owner: "MessageDefinition",
        validate_type: "Type",
        validate_msgdef: "MessageDefinition | None",
        all_definitions: dict[str, "MessageDefinition"],
    ) -> None:
        """Resolve a bare identifier as a field reference or schema enum constant."""
        try:
            _validate_field_path(ref.field_path, validate_type, validate_msgdef, all_definitions)
        except ValidationError:
            constant = next(
                (
                    item
                    for item in compared_owner.constants
                    if item.name == ref.field_path
                    and item.type.type_name == compared_type.type_name
                ),
                None,
            )
            if "." in ref.field_path or constant is None:
                raise
            ref.bind_constant(constant.value)
        else:
            ref.bind_field()


# --- Math modifier framework -------------------------------------------------
# Each modifier registers its metadata (kind, input requirements, return type)
# right next to its implementation via the @modifier decorator, so the rules live
# with the consumer instead of in separate parallel tables.

_FLOAT64_TYPE = Type(type_name="float64", package_name=None)
_NUMERIC_TYPE_NAMES = {
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
}


# How a modifier is dispatched/validated:
#   scalar     numeric in → numeric out, applied element-wise over arrays
#   object     operates on a whole message value (norm, rpy, to_sec, ...)
#   timeseries needs history; raises without a TransformContext
ModifierKind = Literal["scalar", "object", "aggregate", "timeseries"]


@dataclass(frozen=True)
class _Modifier:
    """How a math modifier is dispatched and validated."""

    func: Callable[..., Any]
    array_reducer: Callable[[Any], Any] | None = None
    kind: ModifierKind = "scalar"
    requires_fields: tuple[str, ...] = ()
    requires_array: bool = False
    accepts_array: bool = False
    preserves_element_type: bool = False
    return_type: "Type | None" = None
    return_def: "MessageDefinition | None" = None


_MODIFIERS: dict[str, _Modifier] = {}


def modifier(
    name: str,
    *,
    kind: ModifierKind = "scalar",
    array_reducer: Callable[[Any], Any] | None = None,
    requires_fields: tuple[str, ...] = (),
    requires_array: bool = False,
    accepts_array: bool = False,
    preserves_element_type: bool = False,
    return_type: "Type | None" = None,
    return_def: "MessageDefinition | None" = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a math modifier and its metadata, co-located with the function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _MODIFIERS[name] = _Modifier(
            func=func,
            array_reducer=array_reducer,
            kind=kind,
            requires_fields=requires_fields,
            requires_array=requires_array,
            accepts_array=accepts_array,
            preserves_element_type=preserves_element_type,
            return_type=return_type,
            return_def=return_def,
        )
        return func

    return decorator


@dataclass
class MathModifier(Action):
    """Apply mathematical operations to numeric values."""

    operation: str
    arguments: list[int | float | Variable]

    def _spec(self) -> _Modifier:
        spec = _MODIFIERS.get(self.operation)
        if spec is None:
            raise MessagePathError(f"Unknown math modifier '{self.operation}'")
        return spec

    def apply(self, obj: Any, variables: _VariableStore) -> Any:
        """Apply the math operation to the object, supporting element-wise array operations."""
        spec = self._spec()
        if spec.kind == "aggregate" and self.arguments:
            raise MessagePathError(
                f"Aggregate modifier '{self.operation}' does not accept arguments"
            )

        # Resolve arguments (convert Variables to actual values)
        resolved_args: list[int | float] = [
            _resolve_numeric_variable(arg, variables) if isinstance(arg, Variable) else arg
            for arg in self.arguments
        ]

        if spec.array_reducer is not None and not resolved_args and isinstance(obj, _ARRAY_TYPES):
            try:
                return spec.array_reducer(obj)
            except MessagePathError:
                raise
            except Exception as exc:
                raise MessagePathError(f"Error in '{self.operation}': {exc!s}") from exc

        if spec.kind == "aggregate":
            if not isinstance(obj, _ARRAY_TYPES):
                raise MessagePathError(
                    f"Aggregate modifier '{self.operation}' requires a numeric array, "
                    f"got {type(obj).__name__}"
                )
            return self._apply_operation(obj, resolved_args)

        # Object-level functions operate on the whole object (not element-wise)
        if spec.kind == "object":
            return self._apply_operation(obj, resolved_args)

        # Element-wise over numeric arrays. Decoded numeric arrays are memoryview
        # (e.g. float64[]) or bytes (uint8[]); also accept list/tuple/array/bytearray.
        if isinstance(obj, _ARRAY_TYPES):
            result = [self._apply_operation(item, resolved_args) for item in obj]
            return tuple(result) if isinstance(obj, tuple) else result

        # Apply to single value
        return self._apply_operation(obj, resolved_args)

    def _apply_operation(
        self, value: Any, args: list[int | float]
    ) -> int | float | tuple[float, ...]:
        """Apply the math operation to a single value."""
        spec = self._spec()

        # Object-level functions accept non-numeric inputs (whole message values)
        if spec.kind in ("object", "aggregate"):
            try:
                return spec.func(value, *args) if args else spec.func(value)
            except MessagePathError:
                raise
            except Exception as e:
                raise MessagePathError(f"Error in '{self.operation}': {e!s}") from e

        # Scalar / time-series functions require a numeric input
        if not isinstance(value, (int, float)):
            raise MessagePathError(
                f"Math modifier '{self.operation}' can only be applied to numeric types, "
                f"got {type(value).__name__}"
            )

        # Check for NaN
        if isinstance(value, float) and math.isnan(value):
            raise MessagePathError(f"Math modifier '{self.operation}' received NaN value")

        try:
            result: int | float = spec.func(value, *args)
        except ValueError as e:
            # Math domain errors (sqrt of negative, log of negative, etc.)
            raise MessagePathError(f"Math error in '{self.operation}': {e!s}") from e
        except ZeroDivisionError as e:
            raise MessagePathError(f"Division by zero in '{self.operation}'") from e
        return result

    def _validate_object_function_input(
        self, spec: _Modifier, current_type: "Type", current_msgdef: "MessageDefinition | None"
    ) -> None:
        """Validate the input type/shape for an object-level math function."""
        op = self.operation
        if spec.requires_array:
            if not current_type.is_array:
                raise ValidationError(
                    f"Math modifier '{op}' requires an array, got '{current_type}'"
                )
            return
        if current_type.is_array:
            if spec.accepts_array:
                if (
                    not current_type.is_primitive
                    or current_type.type_name not in _NUMERIC_TYPE_NAMES
                ):
                    raise ValidationError(
                        f"Math modifier '{op}' requires a numeric array, got '{current_type}'"
                    )
                return
            raise ValidationError(
                f"Math modifier '{op}' cannot be applied to array type '{current_type}'"
            )
        # rpy/quat/to_sec/to_nsec and object-form norm need a message with known fields
        required = spec.requires_fields
        if current_type.is_primitive:
            raise ValidationError(
                f"Math modifier '{op}' requires a message with fields {', '.join(required)}, "
                f"got primitive '{current_type.type_name}'"
            )
        # Field presence is only checkable when the definition is available
        # (external types are validated leniently elsewhere in this module).
        if current_msgdef is not None:
            available = {f.name for f in current_msgdef.fields}
            missing = [name for name in required if name not in available]
            if missing:
                raise ValidationError(
                    f"Math modifier '{op}' requires fields {', '.join(required)}; "
                    f"'{current_type}' is missing {', '.join(missing)}"
                )

    def validate(
        self,
        current_type: "Type",
        current_msgdef: "MessageDefinition | None",
        all_definitions: dict[str, "MessageDefinition"],  # noqa: ARG002
    ) -> tuple["Type", "MessageDefinition | None"]:
        """Validate that the math modifier can be applied to the current type."""
        spec = _MODIFIERS.get(self.operation)
        if spec is None:
            raise ValidationError(f"Unknown math modifier '{self.operation}'")

        if spec.array_reducer is not None and not self.arguments and current_type.is_array:
            if not current_type.is_primitive or current_type.type_name not in _NUMERIC_TYPE_NAMES:
                raise ValidationError(
                    f"Math modifier '{self.operation}' requires a numeric array, "
                    f"got '{current_type}'"
                )
            if spec.preserves_element_type:
                return (
                    Type(
                        type_name=current_type.type_name,
                        package_name=current_type.package_name,
                        is_array=False,
                        array_size=None,
                        is_upper_bound=False,
                        string_upper_bound=current_type.string_upper_bound,
                    ),
                    None,
                )
            return spec.return_type or _FLOAT64_TYPE, None

        if spec.kind == "aggregate":
            if self.arguments:
                raise ValidationError(
                    f"Aggregate modifier '{self.operation}' does not accept arguments"
                )
            if (
                not current_type.is_array
                or not current_type.is_primitive
                or current_type.type_name not in _NUMERIC_TYPE_NAMES
            ):
                raise ValidationError(
                    f"Aggregate modifier '{self.operation}' requires a numeric array, "
                    f"got '{current_type}'"
                )
            if spec.preserves_element_type:
                return (
                    Type(
                        type_name=current_type.type_name,
                        package_name=current_type.package_name,
                        is_array=False,
                        array_size=None,
                        is_upper_bound=False,
                        string_upper_bound=current_type.string_upper_bound,
                    ),
                    None,
                )
            return spec.return_type or _FLOAT64_TYPE, None

        # Math operations work on both single numeric values and arrays of numeric values
        working_type = current_type

        # If it's an array, check the element type
        if current_type.is_array:
            working_type = Type(
                type_name=current_type.type_name,
                package_name=current_type.package_name,
                is_array=False,
                array_size=None,
                is_upper_bound=False,
                string_upper_bound=current_type.string_upper_bound,
            )

        # Object-level functions require specific input shapes (not any scalar).
        if spec.kind == "object":
            self._validate_object_function_input(spec, current_type, current_msgdef)
            if spec.return_def is not None:
                result_type = Type(type_name=spec.return_def.name or "unknown", package_name=None)
                return result_type, spec.return_def
            return spec.return_type or _FLOAT64_TYPE, None

        # Time-series functions work on numeric types, preserve type
        if spec.kind == "timeseries":
            return current_type, current_msgdef

        # Check if the base type is numeric
        if not working_type.is_primitive:
            raise ValidationError(
                f"Math modifier '{self.operation}' can only be applied to numeric types, "
                f"got complex type '{working_type}'"
            )

        # Check if it's a numeric primitive (int, float, double, etc.)
        if working_type.type_name not in _NUMERIC_TYPE_NAMES:
            raise ValidationError(
                f"Math modifier '{self.operation}' can only be applied to numeric types, "
                f"got '{working_type.type_name}'"
            )

        # Math modifiers preserve the type (single value -> single value, array -> array)
        return current_type, current_msgdef


@dataclass
class MessagePath:
    topic: str
    segments: list[FieldAccess | ArrayIndex | ArraySlice | Filter | MathModifier]

    def apply(self, obj: Any, variables: MessagePathVariables | None = None) -> Any:
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
            if result is None:
                return None
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
        self.resolve_type(message_def, all_definitions)

    def resolve_type(
        self,
        message_def: "MessageDefinition",
        all_definitions: dict[str, "MessageDefinition"],
    ) -> tuple["Type", "MessageDefinition | None"]:
        """Validate this path and return its resulting type and message definition."""

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

        # A slice or filter opens an "array context": the following field
        # accesses map element-wise over the array (Foxglove semantics). A raw
        # array field (no slice/index) does not, so ``arr.field`` still errors.
        in_array_context = False

        for segment in self.segments:
            if isinstance(segment, FieldAccess) and in_array_context and current_type.is_array:
                element_type = Type(
                    type_name=current_type.type_name,
                    package_name=current_type.package_name,
                    is_array=False,
                    array_size=None,
                    is_upper_bound=False,
                    string_upper_bound=current_type.string_upper_bound,
                )
                field_type, current_msgdef = segment.validate(element_type, None, all_definitions)
                # The mapped result is an array of the field's type.
                current_type = Type(
                    type_name=field_type.type_name,
                    package_name=field_type.package_name,
                    is_array=True,
                    array_size=None,
                    is_upper_bound=False,
                    string_upper_bound=field_type.string_upper_bound,
                )
                continue

            current_type, current_msgdef = segment.validate(
                current_type, current_msgdef, all_definitions
            )
            if isinstance(segment, (ArraySlice, Filter)):
                in_array_context = True
            elif isinstance(segment, ArrayIndex):
                in_array_context = False

        return current_type, current_msgdef


def _validate_field_path(
    field_path: str,
    validate_type: "Type",
    validate_msgdef: "MessageDefinition | None",
    all_definitions: dict[str, "MessageDefinition"],
) -> tuple["Type", "MessageDefinition"]:
    """Validate a field path against a type schema."""
    field_parts = field_path.split(".")
    working_type = validate_type
    working_msgdef = validate_msgdef
    owner_msgdef: MessageDefinition | None = None

    for part in field_parts:
        if working_type.is_primitive:
            raise ValidationError(
                f"Cannot access field '{part}' on primitive type '{working_type}' "
                f"in filter field path '{field_path}'"
            )

        if working_type.is_array:
            raise ValidationError(
                f"Cannot access field '{part}' on array type '{working_type}' "
                f"in filter field path '{field_path}'. "
                "Nested array filtering is not supported"
            )

        if working_msgdef is None:
            working_msgdef = _get_message_definition(working_type, all_definitions)

        owner_msgdef = working_msgdef

        field = next((f for f in working_msgdef.fields if f.name == part), None)
        if not field:
            available = [f.name for f in working_msgdef.fields]
            raise ValidationError(
                f"Field '{part}' not found in message '{working_type}' "
                f"in filter field path '{field_path}'. "
                f"Available fields: {', '.join(available) if available else 'none'}"
            )

        working_type = field.type
        working_msgdef = None
        if not working_type.is_primitive and not working_type.is_array:
            with contextlib.suppress(ValidationError):
                working_msgdef = _get_message_definition(working_type, all_definitions)

    if owner_msgdef is None:
        raise ValidationError(f"Filter field path '{field_path}' does not resolve to a field")
    return working_type, owner_msgdef


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


# Importing modifiers registers every ``.@op`` implementation into ``_MODIFIERS``.
# Done at the bottom to break the models <-> modifiers import cycle.
from ros_parser.message_path import modifiers as _modifiers  # noqa: E402, F401
