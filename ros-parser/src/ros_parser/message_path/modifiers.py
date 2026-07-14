"""Math modifier implementations for message-path ``.@op`` segments.

Each function registers itself — together with its dispatch ``kind`` and input
requirements — with the modifier registry in
:mod:`ros_parser.message_path.models` via the ``@modifier`` decorator. Importing
this module populates that registry as a side effect; ``models`` imports it so
the registry is always complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from operator import neg
from typing import Any

from ros_parser.message_path.models import (
    _ARRAY_TYPES,
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


@modifier("add")
def _add(value: float, *args: float) -> float:
    """Add multiple values."""
    return value + sum(args)


@modifier("sub")
def _sub(value: float, *args: float) -> float:
    """Subtract multiple values from the initial value."""
    return value - sum(args)


@modifier("mul")
def _mul(value: float, *args: float) -> float:
    """Multiply by multiple values."""
    result = value
    for arg in args:
        result *= arg
    return result


@modifier("div")
def _div(value: float, divisor: float) -> float:
    """Divide by multiple values with zero check."""
    if divisor == 0:
        raise ZeroDivisionError("Division by zero")
    return value / divisor


@modifier("round")
def _round_with_arg(value: float, precision: float | None = None) -> int | float:
    """Round with optional precision argument."""
    if precision is None:
        return round(value)
    return round(value, int(precision))


@modifier("min")
def _min(*args: float) -> float:
    """Return minimum of values."""
    return min(args)


@modifier("max")
def _max(*args: float) -> float:
    """Return maximum of values."""
    return max(args)


@modifier("wrap_angle")
def _wrap_angle(value: float) -> float:
    """Wrap angle to [-pi, pi] range."""
    return (value + math.pi) % (2 * math.pi) - math.pi


@modifier("sign")
def _sign(value: float) -> int:
    """Return the sign of a numeric value: 1, -1, or 0."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@modifier("clamp")
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
    modifier(_name)(_builtin)


@modifier("delta", kind="timeseries")
@modifier("derivative", kind="timeseries")
@modifier("timedelta", kind="timeseries")
def _timeseries_sentinel(value: float) -> float:  # noqa: ARG001
    """Sentinel for time-series functions. Raises when called without TransformContext."""
    raise MessagePathError(
        "Time-series function requires TransformContext. "
        "Use digitalis transforms.apply_with_history() instead."
    )


@modifier("length", kind="object", requires_array=True, return_type=_INT64_TYPE)
def _length(obj: Any) -> int:
    """Return the number of elements in an array or typed array."""
    return len(obj)


@modifier(
    "norm",
    kind="object",
    requires_fields=("x", "y"),
    accepts_array=True,
    return_type=_FLOAT64_TYPE,
)
def _norm(obj: Any) -> float:
    """Euclidean norm of a numeric array or object with x/y and optional z fields."""
    if isinstance(obj, _ARRAY_TYPES):
        return math.sqrt(sum(value * value for value in obj))

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


@modifier("rpy", kind="object", requires_fields=("x", "y", "z", "w"), return_def=_EULER_RETURN_DEF)
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


@modifier("magnitude", kind="object", requires_array=True, return_type=_FLOAT64_TYPE)
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


@modifier("to_sec", kind="object", requires_fields=("sec", "nanosec"), return_type=_FLOAT64_TYPE)
def _to_sec(obj: Any) -> float:
    """Convert a Time/Duration {sec, nanosec} to float seconds."""
    try:
        sec = _get_field(obj, "sec")
        nanosec = _get_field(obj, "nanosec")
    except MessagePathError:
        raise MessagePathError("to_sec requires an object with sec, nanosec fields") from None
    return sec + nanosec * 1e-9


@modifier("to_nsec", kind="object", requires_fields=("sec", "nanosec"), return_type=_INT64_TYPE)
def _to_nsec(obj: Any) -> int:
    """Convert a Time/Duration {sec, nanosec} to int nanoseconds."""
    try:
        sec = _get_field(obj, "sec")
        nanosec = _get_field(obj, "nanosec")
    except MessagePathError:
        raise MessagePathError("to_nsec requires an object with sec, nanosec fields") from None
    return sec * 1_000_000_000 + nanosec
