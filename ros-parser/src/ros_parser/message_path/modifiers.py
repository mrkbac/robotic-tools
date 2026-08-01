"""Math modifier implementations for message-path ``.@op`` segments.

Each function registers itself — together with its dispatch ``kind`` and input
requirements — with the modifier registry in
:mod:`ros_parser.message_path.models` via the ``@modifier`` decorator. Importing
this module populates that registry as a side effect; ``models`` imports it so
the registry is always complete.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from operator import neg
from typing import Any

from ros_parser.message_path.models import (
    _FLOAT64_TYPE,
    _MISSING,
    MessagePathError,
    _field_not_found_message,
    _lookup_field,
    modifier,
)
from ros_parser.models import Field, MessageDefinition, Type

_INT64_TYPE = Type(type_name="int64", package_name=None)

_EULER_RETURN_DEF = MessageDefinition(
    name="EulerAngles",
    fields_all=[Field(type=_FLOAT64_TYPE, name=n) for n in ("roll", "pitch", "yaw")],
)
_QUAT_RETURN_DEF = MessageDefinition(
    name="Quaternion",
    fields_all=[Field(type=_FLOAT64_TYPE, name=n) for n in ("x", "y", "z", "w")],
)


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


def _get_field(obj: Any, name: str) -> Any:
    """Get a field from an object, supporting both dict and attribute access."""
    value = _lookup_field(obj, name)
    if value is not _MISSING:
        return value
    raise MessagePathError(_field_not_found_message(obj, name))


@modifier("add", min_args=1)
def _add(value: float, *args: float) -> float:
    """Add multiple values."""
    return value + sum(args)


@modifier("sub", min_args=1)
def _sub(value: float, *args: float) -> float:
    """Subtract multiple values from the initial value."""
    return value - sum(args)


@modifier("mul", min_args=1)
def _mul(value: float, *args: float) -> float:
    """Multiply by multiple values."""
    result = value
    for arg in args:
        result *= arg
    return result


@modifier("div", min_args=1, max_args=1)
def _div(value: float, divisor: float) -> float:
    """Divide by multiple values with zero check."""
    if divisor == 0:
        raise ZeroDivisionError("Division by zero")
    return value / divisor


@modifier("round", min_args=0, max_args=1)
def _round_with_arg(value: float, precision: float | None = None) -> int | float:
    """Round with optional precision argument."""
    if precision is None:
        return round(value)
    return round(value, int(precision))


def _numeric_array(obj: Any, operation: str) -> list[int | float]:
    try:
        iterator = iter(obj)
    except TypeError as exc:
        raise MessagePathError(f"{operation} requires a numeric array") from exc
    if isinstance(obj, (str, Mapping)):
        raise MessagePathError(f"{operation} requires a numeric array")
    values: list[int | float] = []
    for value in iterator:
        if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(float(value)):
            raise MessagePathError(f"{operation} requires a numeric array without NaN values")
        values.append(int(value) if isinstance(value, Integral) else float(value))
    return values


def _array_min(obj: Any) -> int | float | None:
    """Return the minimum array element, or None for an empty array."""
    values = _numeric_array(obj, "min")
    return min(values) if values else None


def _array_max(obj: Any) -> int | float | None:
    """Return the maximum array element, or None for an empty array."""
    values = _numeric_array(obj, "max")
    return max(values) if values else None


def _argument_min(values: list[int | float]) -> float:
    return float(min(values))


def _argument_max(values: list[int | float]) -> float:
    return float(max(values))


def _argument_sum(values: list[int | float]) -> float:
    return math.fsum(values)


def _argument_mean(values: list[int | float]) -> float:
    return math.fsum(values) / len(values)


def _argument_rms(values: list[int | float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values) / len(values))


def _argument_product(values: list[int | float]) -> float:
    return float(math.prod(values))


@modifier(
    "min",
    array_reducer=_array_min,
    argument_reducer=_argument_min,
    preserves_element_type=True,
    min_args=0,
    max_args=None,
)
def _min(*args: float) -> float:
    """Return minimum of a scalar and its arguments, or reduce a bare array."""
    return min(args)


@modifier(
    "max",
    array_reducer=_array_max,
    argument_reducer=_argument_max,
    preserves_element_type=True,
    min_args=0,
    max_args=None,
)
def _max(*args: float) -> float:
    """Return maximum of a scalar and its arguments, or reduce a bare array."""
    return max(args)


@modifier(
    "sum",
    kind="aggregate",
    argument_reducer=_argument_sum,
    return_type=_FLOAT64_TYPE,
    min_args=0,
    max_args=None,
)
def _aggregate_sum(obj: Any) -> float:
    """Return the floating-point sum of an array."""
    return math.fsum(_numeric_array(obj, "sum"))


@modifier(
    "mean",
    kind="aggregate",
    argument_reducer=_argument_mean,
    return_type=_FLOAT64_TYPE,
    min_args=0,
    max_args=None,
)
def _aggregate_mean(obj: Any) -> float | None:
    """Return the arithmetic mean, or None for an empty array."""
    values = _numeric_array(obj, "mean")
    return math.fsum(values) / len(values) if values else None


@modifier(
    "rms",
    kind="aggregate",
    argument_reducer=_argument_rms,
    return_type=_FLOAT64_TYPE,
    min_args=0,
    max_args=None,
)
def _aggregate_rms(obj: Any) -> float | None:
    """Return the root mean square, or None for an empty array."""
    values = _numeric_array(obj, "rms")
    return math.sqrt(math.fsum(value * value for value in values) / len(values)) if values else None


@modifier(
    "product",
    kind="object",
    argument_reducer=_argument_product,
    return_type=_FLOAT64_TYPE,
    min_args=1,
    max_args=None,
)
def _product(_obj: Any, *values: float) -> float:
    """Multiply numeric field references, variables, and literal arguments."""
    return float(math.prod(values))


@modifier("wrap_angle", min_args=0, max_args=0)
def _wrap_angle(value: float) -> float:
    """Wrap angle to [-pi, pi] range."""
    return (value + math.pi) % (2 * math.pi) - math.pi


@modifier("sign", min_args=0, max_args=0)
def _sign(value: float) -> int:
    """Return the sign of a numeric value: 1, -1, or 0."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@modifier("clamp", min_args=2, max_args=2)
def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to the inclusive [lo, hi] range."""
    if lo > hi:
        raise MessagePathError(f"clamp requires lo <= hi, got lo={lo}, hi={hi}")
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# Stdlib numeric functions have no extra metadata; register them in bulk.
for _name, _builtin in {
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
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "trunc": math.trunc,
    "degrees": math.degrees,
    "radians": math.radians,
}.items():
    modifier(_name, min_args=0, max_args=0)(_builtin)


@modifier(
    "length",
    kind="object",
    requires_array=True,
    return_type=_INT64_TYPE,
    min_args=0,
    max_args=0,
)
def _length(obj: Any) -> int:
    """Return the number of elements in an array or typed array."""
    return len(obj)


@modifier(
    "norm",
    kind="object",
    requires_fields=("x", "y"),
    accepts_array=True,
    return_type=_FLOAT64_TYPE,
    min_args=0,
    max_args=0,
    numeric_fields=("x", "y", "z"),
)
def _norm(obj: Any) -> float:
    """Euclidean norm of a numeric array or object with x/y and optional z fields."""
    try:
        values = _numeric_array(obj, "norm")
        return math.sqrt(sum(value * value for value in values))
    except MessagePathError:
        pass

    try:
        x = _get_field(obj, "x")
        y = _get_field(obj, "y")
    except MessagePathError:
        raise MessagePathError(
            "norm requires a numeric array or object with x, y and optional z fields"
        ) from None
    z = _lookup_field(obj, "z")
    if z is _MISSING:
        z = 0
    return math.sqrt(x * x + y * y + z * z)


@modifier(
    "rpy",
    kind="object",
    requires_fields=("x", "y", "z", "w"),
    return_def=_EULER_RETURN_DEF,
    min_args=0,
    max_args=0,
    numeric_fields=("x", "y", "z", "w"),
)
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


@modifier(
    "quat",
    kind="object",
    requires_fields=("roll", "pitch", "yaw"),
    return_def=_QUAT_RETURN_DEF,
    min_args=0,
    max_args=0,
    numeric_fields=("roll", "pitch", "yaw"),
)
def _euler_to_quaternion(obj: Any) -> Quaternion:
    """Convert roll/pitch/yaw fields to Quaternion(x, y, z, w)."""
    try:
        roll = _get_field(obj, "roll")
        pitch = _get_field(obj, "pitch")
        yaw = _get_field(obj, "yaw")
    except MessagePathError:
        raise MessagePathError("quat requires an object with roll, pitch, yaw fields") from None

    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return Quaternion(x=qx, y=qy, z=qz, w=qw)


@modifier(
    "magnitude",
    kind="object",
    requires_array=True,
    requires_numeric_array=True,
    return_type=_FLOAT64_TYPE,
    min_args=0,
    max_args=0,
)
def _magnitude(obj: Any) -> float:
    """L2 norm of a list/array/sequence of numbers."""
    values = _numeric_array(obj, "magnitude")
    return math.sqrt(sum(value * value for value in values))


@modifier(
    "to_sec",
    kind="object",
    requires_fields=("sec", "nanosec"),
    return_type=_FLOAT64_TYPE,
    min_args=0,
    max_args=0,
    numeric_fields=("sec", "nanosec"),
)
def _to_sec(obj: Any) -> float:
    """Convert a Time/Duration {sec, nanosec} to float seconds."""
    try:
        sec = _get_field(obj, "sec")
        nanosec = _get_field(obj, "nanosec")
    except MessagePathError:
        raise MessagePathError("to_sec requires an object with sec, nanosec fields") from None
    return sec + nanosec * 1e-9


@modifier(
    "to_nsec",
    kind="object",
    requires_fields=("sec", "nanosec"),
    return_type=_INT64_TYPE,
    min_args=0,
    max_args=0,
    numeric_fields=("sec", "nanosec"),
)
def _to_nsec(obj: Any) -> int:
    """Convert a Time/Duration {sec, nanosec} to int nanoseconds."""
    try:
        sec = _get_field(obj, "sec")
        nanosec = _get_field(obj, "nanosec")
    except MessagePathError:
        raise MessagePathError("to_nsec requires an object with sec, nanosec fields") from None
    return sec * 1_000_000_000 + nanosec
