"""Round-trip and byte-equivalence tests for byte-array input types.

The byte branch in ``_dynamic_encoder.py`` (TypeId.UINT8 / BYTE / CHAR) accepts:

- ``bytes`` / ``bytearray`` / ``memoryview`` (zero-copy buffer concat)
- ``list`` / ``tuple`` of ints in [0, 255] (``bytes(value)`` constructor)
- numpy ``ndarray`` (cast to uint8 via ``_to_packed_bytes``)
- any other iterable of ints (``array.array`` fallback)

All input types are required to produce *byte-identical* CDR output for the same
logical values — that is what these tests verify, in addition to round-trip
correctness through ``ROS2EncoderFactory`` + ``DecoderFactory``.
"""

from array import array
from collections.abc import Sequence
from io import BytesIO

import pytest
from mcap_ros2_support_fast import ROS2EncoderFactory
from mcap_ros2_support_fast._dynamic_encoder import create_encoder
from mcap_ros2_support_fast._planner import generate_plans, optimize_plan
from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap.reader import read_message_decoded
from small_mcap.writer import McapWriter


def _round_trip(schema_data: bytes, schema_name: str, payload: dict):
    """Encode ``payload`` through the public writer, decode, return decoded message."""
    output = BytesIO()
    writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    writer.start()
    writer.add_schema(1, schema_name, "ros2msg", schema_data)
    writer.add_channel(1, "/test", "cdr", 1)
    writer.add_message_encode(channel_id=1, log_time=0, publish_time=0, sequence=0, data=payload)
    writer.finish()
    output.seek(0)
    msgs = list(read_message_decoded(output, decoder_factories=[DecoderFactory()]))
    assert len(msgs) == 1
    return msgs[0].decoded_message


def _encode_cdr(schema_name: str, schema_text: str, payload: dict) -> bytes:
    """Run the encoder directly to compare raw CDR output without MCAP framing."""
    plan = generate_plans(schema_name, schema_text)
    encoder = create_encoder(optimize_plan(plan))
    return bytes(encoder(payload))


class _IntSequence(Sequence):
    """Custom Sequence that is neither list nor tuple — exercises the helper fallback."""

    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


_DYNAMIC_INPUTS = pytest.mark.parametrize(
    "value",
    [
        b"\x01\x02\x03\x04\x05",
        bytearray([1, 2, 3, 4, 5]),
        memoryview(b"\x01\x02\x03\x04\x05"),
        [1, 2, 3, 4, 5],
        (1, 2, 3, 4, 5),
        array("B", [1, 2, 3, 4, 5]),
        _IntSequence([1, 2, 3, 4, 5]),
    ],
    ids=["bytes", "bytearray", "memoryview", "list", "tuple", "array_B", "Sequence"],
)


@_DYNAMIC_INPUTS
def test_dynamic_uint8_round_trip(value):
    decoded = _round_trip(b"uint8[] xs", "test_msgs/U8", {"xs": value})
    assert list(decoded.xs) == [1, 2, 3, 4, 5]


@_DYNAMIC_INPUTS
def test_dynamic_byte_round_trip(value):
    decoded = _round_trip(b"byte[] xs", "test_msgs/B", {"xs": value})
    assert list(decoded.xs) == [1, 2, 3, 4, 5]


@_DYNAMIC_INPUTS
def test_dynamic_char_round_trip(value):
    decoded = _round_trip(b"char[] xs", "test_msgs/C", {"xs": value})
    assert list(decoded.xs) == [1, 2, 3, 4, 5]


@_DYNAMIC_INPUTS
def test_fixed_uint8_round_trip(value):
    decoded = _round_trip(b"uint8[5] xs", "test_msgs/U8F", {"xs": value})
    assert list(decoded.xs) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "value",
    [
        b"\x01\x02\x03",
        bytearray([1, 2, 3]),
        memoryview(b"\x01\x02\x03"),
        [1, 2, 3],
        (1, 2, 3),
        array("B", [1, 2, 3]),
    ],
    ids=["bytes", "bytearray", "memoryview", "list", "tuple", "array_B"],
)
def test_bounded_uint8_truncates(value):
    """Bounded uint8[<=3] accepts the same input types and reaches the bound exactly."""
    decoded = _round_trip(b"uint8[<=3] xs", "test_msgs/U8B", {"xs": value})
    assert list(decoded.xs) == [1, 2, 3]


def test_bounded_uint8_truncates_oversized_inputs():
    """Bounded uint8[<=3] truncates inputs that exceed the bound."""
    for value in [
        b"\x01\x02\x03\x04\x05",
        [1, 2, 3, 4, 5],
        (1, 2, 3, 4, 5),
    ]:
        decoded = _round_trip(b"uint8[<=3] xs", "test_msgs/U8BT", {"xs": value})
        assert list(decoded.xs) == [1, 2, 3]


def test_byte_array_input_types_produce_identical_cdr_bytes():
    """All supported input types must produce byte-identical CDR encoding."""
    schema = "uint8[5] xs\nuint16 tail"
    inputs = [
        b"\x01\x02\x03\x04\x05",
        bytearray([1, 2, 3, 4, 5]),
        memoryview(b"\x01\x02\x03\x04\x05"),
        [1, 2, 3, 4, 5],
        (1, 2, 3, 4, 5),
        array("B", [1, 2, 3, 4, 5]),
        _IntSequence([1, 2, 3, 4, 5]),
    ]
    cdrs = [_encode_cdr("test_msgs/U8E", schema, {"xs": v, "tail": 0xCAFE}) for v in inputs]
    assert all(c == cdrs[0] for c in cdrs[1:])


def test_dynamic_byte_array_input_types_produce_identical_cdr_bytes():
    schema = "uint8[] xs\nuint16 tail"
    inputs = [
        b"\x01\x02\x03",
        bytearray([1, 2, 3]),
        memoryview(b"\x01\x02\x03"),
        [1, 2, 3],
        (1, 2, 3),
        array("B", [1, 2, 3]),
    ]
    cdrs = [_encode_cdr("test_msgs/U8DE", schema, {"xs": v, "tail": 0xBEEF}) for v in inputs]
    assert all(c == cdrs[0] for c in cdrs[1:])


def test_empty_dynamic_byte_array():
    """Empty byte array round-trips cleanly and the trailing field decodes correctly."""
    decoded = _round_trip(
        b"uint8[] xs\nint32 tail",
        "test_msgs/Empty",
        {"xs": [], "tail": 42},
    )
    assert list(decoded.xs) == []
    assert decoded.tail == 42


def test_fixed_byte_array_rejects_wrong_count_for_list():
    with pytest.raises(ValueError, match="fixed array expected 3 elements, got 5"):
        _round_trip(b"uint8[3] xs", "test_msgs/Wrong", {"xs": [1, 2, 3, 4, 5]})


def test_fixed_byte_array_rejects_wrong_count_for_bytes():
    with pytest.raises(ValueError, match="fixed array expected 3 elements, got 5"):
        _round_trip(b"uint8[3] xs", "test_msgs/Wrong", {"xs": b"\x01\x02\x03\x04\x05"})


@pytest.mark.parametrize("bad_value", [[-1, 2, 3], [1, 2, 256], (1, 2, 1000)])
def test_byte_array_list_rejects_out_of_range_value(bad_value):
    """``bytes()`` rejects ints outside [0, 255] — the error must propagate."""
    with pytest.raises((ValueError, OverflowError)):
        _round_trip(b"uint8[] xs", "test_msgs/Bad", {"xs": bad_value})


# ---------------------------------------------------------------------------
# Non-byte primitive byte-equivalence (sanity check that struct.pack /
# _to_packed_bytes / array.array all agree on bytes for matching values).
# ---------------------------------------------------------------------------


def test_int32_array_input_types_produce_identical_cdr_bytes():
    schema = "int32[] xs\nuint16 tail"
    inputs = [
        [1, 2, 3, -4, 5],
        (1, 2, 3, -4, 5),
        array("i", [1, 2, 3, -4, 5]),
    ]
    np = pytest.importorskip("numpy")
    inputs.extend(
        [
            np.array([1, 2, 3, -4, 5], dtype=np.int32),
            np.array([1, 2, 3, -4, 5], dtype=np.int64),  # cast path
        ]
    )
    cdrs = [_encode_cdr("test_msgs/I32E", schema, {"xs": v, "tail": 7}) for v in inputs]
    assert all(c == cdrs[0] for c in cdrs[1:])


def test_fixed_float64_array_input_types_produce_identical_cdr_bytes():
    schema = "float64[3] xs"
    inputs = [
        [1.5, 2.5, 3.5],
        (1.5, 2.5, 3.5),
        array("d", [1.5, 2.5, 3.5]),
    ]
    np = pytest.importorskip("numpy")
    inputs.extend(
        [
            np.array([1.5, 2.5, 3.5], dtype=np.float64),
            np.array([1.5, 2.5, 3.5], dtype=np.float32),  # cast path
        ]
    )
    cdrs = [_encode_cdr("test_msgs/F64E", schema, {"xs": v}) for v in inputs]
    assert all(c == cdrs[0] for c in cdrs[1:])
