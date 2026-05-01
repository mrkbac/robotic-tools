"""Regression tests for ``_to_plain`` — decoded ROS2 value → DuckDB-ready dict.

The bug this guards against: the ROS2 decoder returns ``memoryview.cast('d')``
(or 'f', 'i', 'h' …) for primitive fixed-size arrays. Those memoryviews have
``len == N`` (number of elements) but ``.tobytes()`` returns ``N * itemsize``
raw bytes. An earlier version of ``_to_plain`` unconditionally called
``.tobytes()`` on every memoryview, silently turning ``float64[9]`` into 72
bytes and making pyarrow reject the column with "expected 9 but got 72".
"""

from __future__ import annotations

import array

from pymcap_cli.types.to_plain import to_plain as _to_plain


def _float64_mv(values: list[float]) -> memoryview:
    """Build a typed memoryview with format='d' matching the decoder's output."""
    buf = array.array("d", values)
    return memoryview(buf)


def test_typed_float_memoryview_preserves_element_count() -> None:
    # float64[9] — the exact pattern that triggered the original crash.
    mv = _float64_mv([float(i) for i in range(9)])
    assert mv.format == "d"
    assert len(mv) == 9
    assert len(mv.tobytes()) == 72  # what the broken code used to emit

    plain = _to_plain(mv)
    assert plain == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert len(plain) == 9  # NOT 72


def test_typed_int32_memoryview_preserves_element_count() -> None:
    mv = memoryview(array.array("i", [10, 20, 30]))
    plain = _to_plain(mv)
    assert plain == [10, 20, 30]


def test_byte_format_memoryview_collapses_to_bytes() -> None:
    # uint8[] fields arrive as memoryview with format='B'. These *should*
    # become bytes (one blob per field), not a list of 256 ints per pixel.
    mv = memoryview(b"\x01\x02\x03\x04")
    plain = _to_plain(mv)
    assert plain == b"\x01\x02\x03\x04"
    assert isinstance(plain, bytes)


def test_nested_message_with_memoryview_array() -> None:
    # A "message" carrying a typed memoryview inside a nested struct —
    # _to_plain should recurse and produce a real list for the inner field.
    class _Imu:
        __slots__ = ("cov", "temp")

        def __init__(self) -> None:
            self.cov = _float64_mv([0.1, 0.2, 0.3])
            self.temp = 1.5

    plain = _to_plain(_Imu())
    assert plain == {"cov": [0.1, 0.2, 0.3], "temp": 1.5}


def test_time_slots_object_collapses_to_int_nanoseconds() -> None:
    # builtin_interfaces/Time — a slots-based dataclass-style object.
    class _Time:
        __slots__ = ("nanosec", "sec")

        def __init__(self, sec: int, nanosec: int) -> None:
            self.sec = sec
            self.nanosec = nanosec

    assert _to_plain(_Time(1, 500_000_000)) == 1_500_000_000
    assert _to_plain(_Time(0, 0)) == 0


def test_time_dict_collapses_to_int_nanoseconds() -> None:
    # The CompressedPointCloud2 decompress factory emits dict-shaped messages,
    # so the dict path must collapse too.
    assert _to_plain({"sec": 2, "nanosec": 750_000_000}) == 2_750_000_000


def test_nested_header_stamp_collapses_inside_message() -> None:
    class _Stamp:
        __slots__ = ("nanosec", "sec")

        def __init__(self) -> None:
            self.sec = 12
            self.nanosec = 345

    class _Header:
        __slots__ = ("frame_id", "stamp")

        def __init__(self) -> None:
            self.stamp = _Stamp()
            self.frame_id = "base"

    class _Msg:
        __slots__ = ("header", "value")

        def __init__(self) -> None:
            self.header = _Header()
            self.value = 42

    assert _to_plain(_Msg()) == {
        "header": {"stamp": 12_000_000_345, "frame_id": "base"},
        "value": 42,
    }
