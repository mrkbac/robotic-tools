from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimeSeriesBuffer:
    """Circular buffer for time-series data, pre-allocated with numpy arrays."""

    max_size: int = 10_000
    _timestamps: np.ndarray = field(init=False)
    _values: np.ndarray = field(init=False)
    _count: int = field(init=False, default=0)
    _head: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._timestamps = np.empty(self.max_size, dtype=np.float64)
        self._values = np.empty(self.max_size, dtype=np.float64)

    def append(self, timestamp_ns: int, value: float) -> None:
        """Append a data point to the buffer."""
        idx = self._head % self.max_size
        self._timestamps[idx] = timestamp_ns
        self._values[idx] = value
        self._head += 1
        if self._count < self.max_size:
            self._count += 1

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, values) in chronological order."""
        if self._count == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        if self._count < self.max_size:
            return self._timestamps[: self._count].copy(), self._values[: self._count].copy()

        # Buffer has wrapped - reorder chronologically
        start = self._head % self.max_size
        timestamps = np.concatenate([self._timestamps[start:], self._timestamps[:start]])
        values = np.concatenate([self._values[start:], self._values[:start]])
        return timestamps, values

    def clear(self) -> None:
        """Clear the buffer."""
        self._count = 0
        self._head = 0

    @property
    def time_range(self) -> tuple[float, float] | None:
        """Return (min_time, max_time) or None if empty."""
        if self._count == 0:
            return None
        timestamps, _ = self.get_data()
        return float(timestamps[0]), float(timestamps[-1])

    @property
    def value_range(self) -> tuple[float, float] | None:
        """Return (min_value, max_value) or None if empty."""
        if self._count == 0:
            return None
        _, values = self.get_data()
        return float(np.min(values)), float(np.max(values))

    def __len__(self) -> int:
        return self._count
