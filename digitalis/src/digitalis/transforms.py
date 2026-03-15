"""Time-series transform context for stateful @ functions."""

from dataclasses import dataclass
from typing import Any

from ros_parser.message_path import MathModifier, MessagePath

from digitalis.utilities import NANOSECONDS_PER_SECOND

TIMESERIES_OPS = {"delta", "derivative", "timedelta"}


@dataclass
class TransformContext:
    """Holds state for time-series functions (delta, derivative, timedelta)."""

    prev_value: float | None = None
    prev_timestamp_ns: int | None = None


def apply_with_history(
    path: MessagePath,
    message: Any,
    timestamp_ns: int,
    ctx: TransformContext,
) -> float | None:
    """Apply message path with time-series transform support.

    Splits the path at the first time-series MathModifier, applies pre-segments
    normally, then computes the stateful transform.

    Returns None if insufficient history (first sample) or non-numeric result.
    """
    # Find the first time-series operation
    ts_idx: int | None = None
    for i, seg in enumerate(path.segments):
        if isinstance(seg, MathModifier) and seg.operation in TIMESERIES_OPS:
            ts_idx = i
            break

    if ts_idx is None:
        # No time-series op — apply normally
        result = path.apply(message)
        if isinstance(result, (int, float)):
            ctx.prev_value = float(result)
            ctx.prev_timestamp_ns = timestamp_ns
            return float(result)
        return None

    # Apply segments before the time-series op
    pre_path = MessagePath(topic=path.topic, segments=path.segments[:ts_idx])
    raw_value = pre_path.apply(message) if pre_path.segments else path.apply(message)

    if not isinstance(raw_value, (int, float)):
        return None

    current_value = float(raw_value)
    ts_op = path.segments[ts_idx]
    assert isinstance(ts_op, MathModifier)

    result: float | None = None

    if ctx.prev_value is not None and ctx.prev_timestamp_ns is not None:
        if ts_op.operation == "delta":
            result = current_value - ctx.prev_value

        elif ts_op.operation == "timedelta":
            result = (timestamp_ns - ctx.prev_timestamp_ns) / NANOSECONDS_PER_SECOND

        elif ts_op.operation == "derivative":
            dt = (timestamp_ns - ctx.prev_timestamp_ns) / NANOSECONDS_PER_SECOND
            if dt > 0:
                result = (current_value - ctx.prev_value) / dt

    # Update context
    ctx.prev_value = current_value
    ctx.prev_timestamp_ns = timestamp_ns

    if result is None:
        return None

    # Apply any remaining segments after the time-series op
    remaining = path.segments[ts_idx + 1 :]
    if remaining:
        post_path = MessagePath(topic=path.topic, segments=remaining)
        result = post_path.apply(result)
        if not isinstance(result, (int, float)):
            return None
        return float(result)

    return result
