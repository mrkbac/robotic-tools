"""Stateful MessagePath evaluation across a stream of messages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from .models import (
    _ARRAY_TYPES,
    _EMPTY_VARIABLES,
    _STREAM_REDUCERS,
    _STREAM_TRANSFORMS,
    Action,
    MessagePath,
    MessagePathError,
    MessagePathVariables,
    StreamModifier,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

NO_OUTPUT: Final = object()


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if type(value) in (bool, int, float, str, dict):
        return False
    return isinstance(value, _ARRAY_TYPES) and len(value) == 0


@dataclass
class _TransformState:
    previous_value: float | None = None
    previous_timestamp_ns: int | None = None
    unchanged_value: Any = NO_OUTPUT
    unchanged_since_ns: int | None = None


@dataclass
class _ReducerState:
    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    mean: float = 0.0
    squared_deviations: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    first: Any = NO_OUTPUT
    last: Any = NO_OUTPUT


class MessagePathEvaluator:
    """Evaluate one parsed path with state isolated to one message stream."""

    def __init__(self, path: MessagePath) -> None:
        self.path = path
        self._prefix, self._pipeline, self._suffix, self._reducer = self._split(path)
        self._transform_states = [_TransformState() for _ in self._pipeline]
        self._reducer_state = _ReducerState()

    @staticmethod
    def _split(
        path: MessagePath,
    ) -> tuple[MessagePath, Sequence[Action], MessagePath, StreamModifier | None]:
        indexes = [
            index
            for index, segment in enumerate(path.segments)
            if isinstance(segment, StreamModifier)
        ]
        if not indexes:
            return path, [], MessagePath(path.topic, []), None
        first = indexes[0]
        stream = [segment for segment in path.segments if isinstance(segment, StreamModifier)]
        reducers = [segment for segment in stream if segment.operation in _STREAM_REDUCERS]
        if len(reducers) > 1:
            raise MessagePathError("A MessagePath may contain at most one stream reducer")
        for segment in stream:
            if segment.operation not in _STREAM_TRANSFORMS | _STREAM_REDUCERS:
                raise MessagePathError(f"Unknown stream modifier '{segment.operation}'")
        reducer = reducers[0] if reducers else None
        reducer_index = path.segments.index(reducer) if reducer is not None else None
        if reducer_index is not None and any(index > reducer_index for index in indexes):
            raise MessagePathError("Stream transforms must precede the stream reducer")
        pipeline_end = reducer_index + 1 if reducer_index is not None else len(path.segments)
        return (
            MessagePath(path.topic, path.segments[:first]),
            path.segments[first:pipeline_end],
            MessagePath(path.topic, path.segments[pipeline_end:]),
            reducer,
        )

    def observe(
        self,
        message: Any,
        timestamp_ns: int,
        variables: MessagePathVariables | None = None,
    ) -> Any:
        """Observe one message, returning a value or ``NO_OUTPUT``."""
        variable_store = variables if variables is not None else _EMPTY_VARIABLES
        value = self._prefix.apply(message, variable_store)
        if not self._pipeline:
            return value
        if _is_empty(value):
            return NO_OUTPUT
        for index, segment in enumerate(self._pipeline):
            if type(segment) is StreamModifier:
                if segment.operation in _STREAM_TRANSFORMS:
                    value = self._transform(segment.operation, value, timestamp_ns, index)
                    if value is NO_OUTPUT:
                        return NO_OUTPUT
                    continue
                self._reduce(segment.operation, value)
                return NO_OUTPUT
            value = segment.apply(value, variable_store)
            if _is_empty(value):
                return value if index == len(self._pipeline) - 1 else NO_OUTPUT
        return value

    def finalize(self, variables: MessagePathVariables | None = None) -> Any:
        """Finalize a terminal stream reducer and apply the remaining suffix."""
        if self._reducer is None:
            return NO_OUTPUT
        if self._reducer_state.count == 0 and self._reducer.operation != "count":
            return NO_OUTPUT
        value = self._reduced_value(self._reducer.operation)
        return self._suffix.apply(value, variables)

    def _transform(self, operation: str, value: Any, timestamp_ns: int, index: int) -> Any:
        # Backward time means the stream restarted (e.g. a playback seek):
        # re-baseline from the current message instead of erroring.
        state = self._transform_states[index]
        if operation == "unchanged_for":
            if type(value) not in (bool, int, float, str):
                raise MessagePathError(
                    "Stream modifier 'unchanged_for' requires a primitive scalar"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise MessagePathError(
                    "Stream modifier 'unchanged_for' received a non-finite value"
                )
            went_backward = (
                state.previous_timestamp_ns is not None
                and timestamp_ns < state.previous_timestamp_ns
            )
            state.previous_timestamp_ns = timestamp_ns
            if (
                went_backward
                or state.unchanged_value is NO_OUTPUT
                or value != state.unchanged_value
            ):
                state.unchanged_value = value
                state.unchanged_since_ns = timestamp_ns
                return 0.0
            assert state.unchanged_since_ns is not None
            return (timestamp_ns - state.unchanged_since_ns) / 1_000_000_000
        if operation == "timedelta":
            previous_timestamp_ns = state.previous_timestamp_ns
            state.previous_timestamp_ns = timestamp_ns
            if previous_timestamp_ns is None or timestamp_ns < previous_timestamp_ns:
                return NO_OUTPUT
            return (timestamp_ns - previous_timestamp_ns) / 1_000_000_000
        if type(value) not in (int, float) or not math.isfinite(value):
            raise MessagePathError(
                f"Stream modifier '{operation}' requires a finite numeric scalar"
            )
        current = float(value)
        previous_value = state.previous_value
        previous_timestamp_ns = state.previous_timestamp_ns
        state.previous_value = current
        state.previous_timestamp_ns = timestamp_ns
        if previous_value is None or previous_timestamp_ns is None:
            return NO_OUTPUT
        if operation == "delta":
            return current - previous_value
        elapsed = (timestamp_ns - previous_timestamp_ns) / 1_000_000_000
        if elapsed <= 0:
            return NO_OUTPUT
        return (current - previous_value) / elapsed

    def _reduce(self, operation: str, value: Any) -> None:
        state = self._reducer_state
        if operation == "count":
            state.count += 1
            return
        if operation in {"first", "last"}:
            if state.first is NO_OUTPUT:
                state.first = value
            state.last = value
            state.count += 1
            return
        if type(value) not in (int, float) or not math.isfinite(value):
            raise MessagePathError(f"Stream modifier '{operation}' requires finite numeric values")
        numeric = float(value)
        state.count += 1
        if operation == "min":
            state.minimum = numeric if state.minimum is None else min(state.minimum, numeric)
        elif operation == "max":
            state.maximum = numeric if state.maximum is None else max(state.maximum, numeric)
        elif operation == "sum":
            state.total += numeric
        elif operation == "rms":
            state.total_squares += numeric * numeric
        elif operation == "mean":
            state.mean += (numeric - state.mean) / state.count
        else:
            delta = numeric - state.mean
            state.mean += delta / state.count
            state.squared_deviations += delta * (numeric - state.mean)

    def _reduced_value(self, operation: str) -> Any:
        state = self._reducer_state
        if operation == "count":
            return state.count
        if operation == "first":
            return state.first
        if operation == "last":
            return state.last
        if operation == "min":
            return state.minimum
        if operation == "max":
            return state.maximum
        if operation == "sum":
            return state.total
        if operation == "mean":
            return state.mean
        if operation == "variance":
            return state.squared_deviations / state.count
        if operation == "stddev":
            return math.sqrt(state.squared_deviations / state.count)
        return math.sqrt(state.total_squares / state.count)
