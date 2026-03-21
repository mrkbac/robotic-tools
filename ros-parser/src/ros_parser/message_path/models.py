import contextlib
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from operator import neg
from typing import Any, Literal

from ros_parser.models import Field, MessageDefinition, Type


class MessagePathError(Exception):
    """Exception raised when message path operations fail."""


class FieldResolutionError(MessagePathError):
    """Exception raised when a field path cannot be resolved on an object."""


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
class FilterFieldRef:
    """A reference to a field path used as a value in filter comparisons (cross-field)."""

    field_path: str

    def resolve(self, obj: Any) -> Any:
        """Resolve the field path against an object."""
        return _resolve_field_path(obj, self.field_path)


FilterValue = int | float | str | bool | Variable | FilterFieldRef


def _resolve_field_path(obj: Any, field_path: str) -> Any:
    """Extract field value from nested field path (e.g., 'pose.x')."""
    value = obj
    for part in field_path.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                raise FieldResolutionError(f"Field '{part}' not found in mapping")
            value = value[part]
        else:
            try:
                value = getattr(value, part)
            except AttributeError:
                obj_type = type(value).__name__
                raise FieldResolutionError(
                    f"Field '{part}' not found on object of type '{obj_type}'"
                ) from None
    return value


def _resolve_filter_value(val: FilterValue, obj: Any, variables: _VariableStore) -> Any:
    """Resolve a filter value to a concrete value for comparison."""
    if isinstance(val, Variable):
        return variables[val.name]
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

    def evaluate(self, obj: Any, variables: _VariableStore) -> bool:
        field_value = _resolve_field_path(obj, self.field_path)
        compare_value = _resolve_filter_value(self.value, obj, variables)
        return _compare(field_value, self.operator, compare_value)


@dataclass
class InExpression(FilterExpression):
    """Membership test: field_path in [val1, val2, ...]."""

    field_path: str
    values: list[FilterValue]

    def evaluate(self, obj: Any, variables: _VariableStore) -> bool:
        field_value = _resolve_field_path(obj, self.field_path)
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

        For sequences (list/tuple): Returns a new list with only matching items.
        For single objects: Returns the object if it matches, or None if it doesn't.
        """
        if not isinstance(obj, (list, tuple)):
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
            _validate_field_path(expr.field_path, validate_type, validate_msgdef, all_definitions)
            if isinstance(expr.value, FilterFieldRef):
                _validate_field_path(
                    expr.value.field_path, validate_type, validate_msgdef, all_definitions
                )
        elif isinstance(expr, InExpression):
            _validate_field_path(expr.field_path, validate_type, validate_msgdef, all_definitions)
            for val in expr.values:
                if isinstance(val, FilterFieldRef):
                    _validate_field_path(
                        val.field_path, validate_type, validate_msgdef, all_definitions
                    )
        elif isinstance(expr, CompoundFilter):
            for child in expr.children:
                self._validate_expression(child, validate_type, validate_msgdef, all_definitions)


def _add(value: float, *args: float) -> float:
    """Add multiple values."""
    return value + sum(args)


def _sub(value: float, *args: float) -> float:
    """Subtract multiple values from the initial value."""
    return value - sum(args)


def _mul(value: float, *args: float) -> float:
    """Multiply by multiple values."""
    result = value
    for arg in args:
        result *= arg
    return result


def _div(value: float, divisor: float) -> float:
    """Divide by multiple values with zero check."""
    if divisor == 0:
        raise ZeroDivisionError("Division by zero")
    return value / divisor


def _round_with_arg(value: float, precision: float | None = None) -> int | float:
    """Round with optional precision argument."""
    if precision is None:
        return round(value)
    return round(value, int(precision))


def _min(*args: float) -> float:
    """Return minimum of values."""
    return min(args)


def _max(*args: float) -> float:
    """Return maximum of values."""
    return max(args)


def _wrap_angle(value: float) -> float:
    """Wrap angle to [-pi, pi] range."""
    return (value + math.pi) % (2 * math.pi) - math.pi


def _sign(value: float) -> int:
    """Return the sign of a numeric value: 1, -1, or 0."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


_FUNCTIONS_NO_ARGS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "ceil": math.ceil,
    "cos": math.cos,
    "floor": math.floor,
    "log": math.log,
    "log1p": math.log1p,
    "log2": math.log2,
    "log10": math.log10,
    "negative": neg,
    "sign": _sign,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "trunc": math.trunc,
    "add": _add,
    "sub": _sub,
    "mul": _mul,
    "div": _div,
    "round": _round_with_arg,
    "min": _min,
    "max": _max,
    "degrees": math.degrees,
    "radians": math.radians,
    "wrap_angle": _wrap_angle,
}

TIMESERIES_OPS: set[str] = {"delta", "derivative", "timedelta"}


def _timeseries_sentinel(value: float) -> float:  # noqa: ARG001
    """Sentinel for time-series functions. Raises when called without TransformContext."""
    raise MessagePathError(
        "Time-series function requires TransformContext. "
        "Use digitalis transforms.apply_with_history() instead."
    )


for _op in TIMESERIES_OPS:
    _FUNCTIONS_NO_ARGS[_op] = _timeseries_sentinel


def _get_field(obj: Any, name: str) -> Any:
    """Get a field from an object, trying mapping access first, then attribute access."""
    if isinstance(obj, Mapping):
        try:
            return obj[name]
        except KeyError:
            raise MessagePathError(f"Field '{name}' not found in mapping") from None
    try:
        return getattr(obj, name)
    except AttributeError:
        raise MessagePathError(
            f"Field '{name}' not found on object of type '{type(obj).__name__}'"
        ) from None


def _norm(obj: Any) -> float:
    """Euclidean norm of object with x/y/z fields."""
    try:
        x = _get_field(obj, "x")
        y = _get_field(obj, "y")
        z = _get_field(obj, "z")
    except MessagePathError:
        raise MessagePathError("norm requires an object with x, y, z fields") from None
    return math.sqrt(x * x + y * y + z * z)


@dataclass
class EulerAngles:
    """Roll, pitch, yaw Euler angles (radians). Supports attribute access like Foxglove."""

    roll: float
    pitch: float
    yaw: float


@dataclass
class Quaternion:
    """Quaternion (x, y, z, w). Supports attribute access like Foxglove."""

    x: float
    y: float
    z: float
    w: float


def _quaternion_to_euler(obj: Any) -> EulerAngles:
    """Convert quaternion (x,y,z,w) to EulerAngles(roll, pitch, yaw)."""
    try:
        x = _get_field(obj, "x")
        y = _get_field(obj, "y")
        z = _get_field(obj, "z")
        w = _get_field(obj, "w")
    except MessagePathError:
        raise MessagePathError("rpy requires an object with x, y, z, w fields") from None

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return EulerAngles(roll=roll, pitch=pitch, yaw=yaw)


def _euler_to_quaternion(obj: Any) -> Quaternion:
    """Convert (roll, pitch, yaw) stored as (x, y, z) to Quaternion(x, y, z, w)."""
    try:
        roll = _get_field(obj, "x")
        pitch = _get_field(obj, "y")
        yaw = _get_field(obj, "z")
    except MessagePathError:
        raise MessagePathError(
            "quat requires an object with x, y, z fields (roll, pitch, yaw)"
        ) from None

    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return Quaternion(x=qx, y=qy, z=qz, w=qw)


def _magnitude(obj: Any) -> float:
    """L2 norm of a list/array/sequence of numbers."""
    if isinstance(obj, (list, tuple)):
        return math.sqrt(sum(v * v for v in obj))
    # Try to iterate (numpy arrays, etc.)
    try:
        values = list(obj)
        return math.sqrt(sum(v * v for v in values))
    except TypeError as e:
        raise MessagePathError("magnitude requires a list or array of numbers") from e


_OBJECT_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "norm": _norm,
    "rpy": _quaternion_to_euler,
    "quat": _euler_to_quaternion,
    "magnitude": _magnitude,
}

_FLOAT64_TYPE = Type(type_name="float64", package_name=None)

_OBJECT_FUNCTION_RETURN_TYPES: dict[str, MessageDefinition] = {
    "rpy": MessageDefinition(
        name="EulerAngles",
        fields_all=[
            Field(type=_FLOAT64_TYPE, name="roll"),
            Field(type=_FLOAT64_TYPE, name="pitch"),
            Field(type=_FLOAT64_TYPE, name="yaw"),
        ],
    ),
    "quat": MessageDefinition(
        name="Quaternion",
        fields_all=[
            Field(type=_FLOAT64_TYPE, name="x"),
            Field(type=_FLOAT64_TYPE, name="y"),
            Field(type=_FLOAT64_TYPE, name="z"),
            Field(type=_FLOAT64_TYPE, name="w"),
        ],
    ),
}


@dataclass
class MathModifier(Action):
    """Apply mathematical operations to numeric values."""

    operation: str
    arguments: list[int | float | Variable]

    def apply(self, obj: Any, variables: _VariableStore) -> Any:
        """Apply the math operation to the object, supporting element-wise array operations."""
        # Resolve arguments (convert Variables to actual values)
        resolved_args = [
            variables[arg.name] if isinstance(arg, Variable) else arg for arg in self.arguments
        ]

        # Object-level functions operate on the whole object (not element-wise)
        if self.operation in _OBJECT_FUNCTIONS:
            return self._apply_operation(obj, resolved_args)

        # Check if obj is a sequence for element-wise operation
        if isinstance(obj, (list, tuple)):
            # Apply element-wise
            result = [self._apply_operation(item, resolved_args) for item in obj]
            # Return same type as input (list or tuple)
            return type(obj)(result) if isinstance(obj, tuple) else result

        # Apply to single value
        return self._apply_operation(obj, resolved_args)

    def _apply_operation(
        self, value: Any, args: list[int | float]
    ) -> int | float | tuple[float, ...]:
        """Apply the math operation to a single value."""
        # Check object-level functions first (these accept non-numeric inputs)
        if func := _OBJECT_FUNCTIONS.get(self.operation):
            try:
                return func(value, *args) if args else func(value)
            except MessagePathError:
                raise
            except Exception as e:
                raise MessagePathError(f"Error in '{self.operation}': {e!s}") from e

        # Validate that value is numeric for scalar functions
        if not isinstance(value, (int, float)):
            raise MessagePathError(
                f"Math modifier '{self.operation}' can only be applied to numeric types, "
                f"got {type(value).__name__}"
            )

        # Check for NaN
        if isinstance(value, float) and math.isnan(value):
            raise MessagePathError(f"Math modifier '{self.operation}' received NaN value")

        try:
            if func := _FUNCTIONS_NO_ARGS.get(self.operation):
                result: int | float = func(value, *args)
                return result

            # Unknown operation
            raise MessagePathError(f"Unknown math modifier '{self.operation}'")

        except ValueError as e:
            # Math domain errors (sqrt of negative, log of negative, etc.)
            raise MessagePathError(f"Math error in '{self.operation}': {e!s}") from e
        except ZeroDivisionError as e:
            raise MessagePathError(f"Division by zero in '{self.operation}'") from e

    def validate(
        self,
        current_type: "Type",
        current_msgdef: "MessageDefinition | None",
        all_definitions: dict[str, "MessageDefinition"],  # noqa: ARG002
    ) -> tuple["Type", "MessageDefinition | None"]:
        """Validate that the math modifier can be applied to the current type."""
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

        # Object-level functions work on complex types
        if self.operation in _OBJECT_FUNCTIONS:
            result_def = _OBJECT_FUNCTION_RETURN_TYPES.get(self.operation)
            if result_def is not None:
                result_type = Type(type_name=result_def.name or "unknown", package_name=None)
                return result_type, result_def

            # norm, magnitude → scalar float
            return _FLOAT64_TYPE, None

        # Time-series functions work on numeric types, preserve type
        if self.operation in TIMESERIES_OPS:
            return current_type, current_msgdef

        # Check if the base type is numeric
        if not working_type.is_primitive:
            raise ValidationError(
                f"Math modifier '{self.operation}' can only be applied to numeric types, "
                f"got complex type '{working_type}'"
            )

        # Check if it's a numeric primitive (int, float, double, etc.)
        numeric_types = {
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
        if working_type.type_name not in numeric_types:
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


def _validate_field_path(
    field_path: str,
    validate_type: "Type",
    validate_msgdef: "MessageDefinition | None",
    all_definitions: dict[str, "MessageDefinition"],
) -> None:
    """Validate a field path against a type schema."""
    field_parts = field_path.split(".")
    working_type = validate_type
    working_msgdef = validate_msgdef

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
