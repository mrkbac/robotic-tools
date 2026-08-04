"""Tests for the time-series circular buffer."""

from digitalis.ui.panels.plot_buffer import TimeSeriesBuffer


def test_time_series_buffer_empty_partial_and_clear() -> None:
    buffer = TimeSeriesBuffer(max_size=3)

    timestamps, values = buffer.get_data()
    assert timestamps.tolist() == []
    assert values.tolist() == []
    assert buffer.time_range is None
    assert buffer.value_range is None

    buffer.append(10, 2.5)
    buffer.append(20, -1.0)

    timestamps, values = buffer.get_data()
    assert timestamps.tolist() == [10.0, 20.0]
    assert values.tolist() == [2.5, -1.0]
    assert len(buffer) == 2
    assert buffer.time_range == (10.0, 20.0)
    assert buffer.value_range == (-1.0, 2.5)

    buffer.clear()
    assert len(buffer) == 0
    assert buffer.time_range is None


def test_time_series_buffer_wraps_in_chronological_order() -> None:
    buffer = TimeSeriesBuffer(max_size=3)
    for timestamp, value in [(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0)]:
        buffer.append(timestamp, value)

    timestamps, values = buffer.get_data()

    assert timestamps.tolist() == [20.0, 30.0, 40.0]
    assert values.tolist() == [2.0, 3.0, 4.0]
    assert len(buffer) == 3
