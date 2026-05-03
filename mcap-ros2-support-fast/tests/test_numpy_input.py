"""Test that the encoder accepts numpy ndarrays and other duck-typed array-likes.

Mirrors the upstream Foxglove fix for issue #1469 / PR #1619, but adapted to the
code-generated encoder in `_dynamic_encoder.py`. Numpy stays an optional dep —
these tests are skipped when it is not installed.
"""

from collections.abc import Sequence
from io import BytesIO

import pytest
from mcap_ros2_support_fast import ROS2EncoderFactory
from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap.reader import read_message_decoded
from small_mcap.writer import McapWriter

np = pytest.importorskip("numpy")


def _read(stream: BytesIO):
    return list(read_message_decoded(stream, decoder_factories=[DecoderFactory()]))


def _write_one(schema_data: bytes, schema_name: str, payload: dict) -> BytesIO:
    output = BytesIO()
    writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    writer.start()
    writer.add_schema(1, schema_name, "ros2msg", schema_data)
    writer.add_channel(1, "/test", "cdr", 1)
    writer.add_message_encode(
        channel_id=1,
        log_time=0,
        publish_time=0,
        sequence=0,
        data=payload,
    )
    writer.finish()
    output.seek(0)
    return output


def test_imu_reproducer_fixed_float64_9_as_ndarray() -> None:
    """foxglove/mcap#1469 reproducer: float64[9] given as a 1-D ndarray."""
    schema = b"float64[9] cov"
    cov = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)

    msgs = _read(_write_one(schema, "test_msgs/Cov", {"cov": cov}))
    assert len(msgs) == 1
    assert list(msgs[0].decoded_message.cov) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_fixed_float64_9_as_2d_ndarray_flattens() -> None:
    """A 3x3 ndarray for float64[9] is flattened in row-major order."""
    schema = b"float64[9] cov"
    cov = np.arange(9, dtype=np.float64).reshape(3, 3)

    msgs = _read(_write_one(schema, "test_msgs/Cov", {"cov": cov}))
    assert list(msgs[0].decoded_message.cov) == list(range(9))


def test_fixed_array_byte_equivalence_ndarray_vs_list() -> None:
    """The serialized bytes must match list-input bytes exactly."""
    schema = b"float64[9] cov"
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    out_list = _write_one(schema, "test_msgs/Cov", {"cov": values}).getvalue()
    out_np = _write_one(
        schema, "test_msgs/Cov", {"cov": np.array(values, dtype=np.float64)}
    ).getvalue()
    assert out_list == out_np


def test_dynamic_float32_ndarray() -> None:
    schema = b"float32[] xs"
    xs = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)

    msgs = _read(_write_one(schema, "test_msgs/Xs", {"xs": xs}))
    out = list(msgs[0].decoded_message.xs)
    assert len(out) == 5
    for got, want in zip(out, [1.5, 2.5, 3.5, 4.5, 5.5], strict=True):
        assert abs(got - want) < 1e-6


def test_dynamic_ndarray_uses_flattened_element_count() -> None:
    schema = b"""int32[] xs
int32 tail"""
    xs = np.array([[1, 2], [3, 4]], dtype=np.int32)

    msgs = _read(_write_one(schema, "test_msgs/Xs", {"xs": xs, "tail": 99}))
    decoded = msgs[0].decoded_message
    assert list(decoded.xs) == [1, 2, 3, 4]
    assert decoded.tail == 99


def test_bounded_ndarray_truncates_flattened_elements() -> None:
    schema = b"""int32[<=3] xs
int32 tail"""
    xs = np.array([[1, 2], [3, 4]], dtype=np.int32)

    msgs = _read(_write_one(schema, "test_msgs/Xs", {"xs": xs, "tail": 99}))
    decoded = msgs[0].decoded_message
    assert list(decoded.xs) == [1, 2, 3]
    assert decoded.tail == 99


def test_dtype_mismatch_int32_field_with_int64_ndarray() -> None:
    """An int64 ndarray for an int32[] field should be cast element-wise."""
    schema = b"int32[] ints"
    ints = np.array([10, -20, 30, -40], dtype=np.int64)

    msgs = _read(_write_one(schema, "test_msgs/Ints", {"ints": ints}))
    assert list(msgs[0].decoded_message.ints) == [10, -20, 30, -40]


def test_byte_array_safety_int32_ndarray_for_uint8_field() -> None:
    """uint8[] given an int32 ndarray must encode 1 byte/value (not raw int32 bytes)."""
    schema = b"uint8[] bs"
    bs = np.array([1, 2, 3, 4, 250], dtype=np.int32)

    msgs = _read(_write_one(schema, "test_msgs/Bs", {"bs": bs}))
    assert list(msgs[0].decoded_message.bs) == [1, 2, 3, 4, 250]


def test_byte_array_uint8_ndarray() -> None:
    schema = b"uint8[] bs"
    bs = np.array([10, 20, 30], dtype=np.uint8)

    msgs = _read(_write_one(schema, "test_msgs/Bs", {"bs": bs}))
    assert list(msgs[0].decoded_message.bs) == [10, 20, 30]


def test_bounded_array_truncates_ndarray() -> None:
    schema = b"int32[<=3] bounded"
    bounded = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    msgs = _read(_write_one(schema, "test_msgs/B", {"bounded": bounded}))
    assert list(msgs[0].decoded_message.bounded) == [1, 2, 3]


def test_fixed_uint8_5_ndarray() -> None:
    """Fixed-size byte arrays accept ndarrays."""
    schema = b"uint8[5] bs"
    bs = np.array([10, 11, 12, 13, 14], dtype=np.uint8)

    msgs = _read(_write_one(schema, "test_msgs/Bs", {"bs": bs}))
    assert list(msgs[0].decoded_message.bs) == [10, 11, 12, 13, 14]


def test_fixed_int32_3_ndarray() -> None:
    """Fixed-size primitive arrays accept ndarrays (dispatched to helper)."""
    schema = b"int32[3] xs"
    xs = np.array([100, -200, 300], dtype=np.int32)

    msgs = _read(_write_one(schema, "test_msgs/Xs", {"xs": xs}))
    assert list(msgs[0].decoded_message.xs) == [100, -200, 300]


def test_fixed_ndarray_rejects_extra_elements() -> None:
    schema = b"""float64[3] xs
int32 tail"""
    xs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    with pytest.raises(ValueError, match="fixed array expected 3 elements, got 4"):
        _write_one(schema, "test_msgs/Fixed", {"xs": xs, "tail": 99})


class _ListBackedSeq(Sequence):
    """Custom Sequence subclass used to verify duck-typed array-like support."""

    def __init__(self, items: list) -> None:
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx):  # type: ignore[override]
        return self._items[idx]


def test_custom_sequence_subclass_dynamic_array() -> None:
    schema = b"int32[] xs"
    xs = _ListBackedSeq([1, 2, 3, 4])

    msgs = _read(_write_one(schema, "test_msgs/Xs", {"xs": xs}))
    assert list(msgs[0].decoded_message.xs) == [1, 2, 3, 4]


def test_custom_sequence_subclass_fixed_array() -> None:
    schema = b"float64[3] xs"
    xs = _ListBackedSeq([1.5, 2.5, 3.5])

    msgs = _read(_write_one(schema, "test_msgs/Xs", {"xs": xs}))
    assert list(msgs[0].decoded_message.xs) == [1.5, 2.5, 3.5]
