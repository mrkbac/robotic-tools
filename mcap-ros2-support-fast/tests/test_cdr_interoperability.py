"""Independent CDR interoperability and boundary tests."""

from __future__ import annotations

import os
import subprocess
import sys
from array import array
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from mcap_ros2._cdr import CdrReader, CdrWriter, EncapsulationKind
from mcap_ros2_support_fast._dynamic_encoder import EncoderGeneratorFactory
from mcap_ros2_support_fast._planner import (
    create_decoder_function,
    create_encoder_function,
    generate_plans,
    optimize_plan,
)
from mcap_ros2_support_fast._plans import McapROS2DecodeError

ReferenceWriter = Callable[[CdrWriter], None]
MessageEncoder = Callable[[SimpleNamespace], bytes | memoryview]


def _reference_payload(kind: EncapsulationKind, write: ReferenceWriter) -> bytes:
    output = BytesIO()
    writer = CdrWriter(output, kind=kind)
    write(writer)
    return output.getvalue()


def _assert_exact_cdr(payload: bytes, expected_hex: str) -> None:
    assert payload == bytes.fromhex(expected_hex)


def _generated_encoder(schema_name: str, schema: str, endianness: str) -> MessageEncoder:
    plan = optimize_plan(generate_plans(schema_name, schema))
    factory = EncoderGeneratorFactory(plan, endianness=endianness)
    code = factory.generate_encoder_code("encode")
    namespace = factory.create_namespace()
    exec(code, namespace)  # noqa: S102
    return cast("MessageEncoder", namespace["encode"])


REPRESENTATIVE_SCHEMA = """\
int32 scalar
float64 aligned
Nested child
int32[] values
uint8 tail
================================================================================
MSG: example_msgs/Nested
int8 lead
float64 value
"""


REPRESENTATIVE_MESSAGE = SimpleNamespace(
    scalar=-7,
    aligned=1.25,
    child=SimpleNamespace(lead=-3, value=-2.5),
    values=[-100, 200],
    tail=0xA5,
)


def _write_representative(writer: CdrWriter) -> None:
    writer.write_int32(-7)
    writer.write_float64(1.25)
    writer.write_int8(-3)
    writer.write_float64(-2.5)
    writer.write_uint32(2)
    writer.write_int32(-100)
    writer.write_int32(200)
    writer.write_uint8(0xA5)


@pytest.mark.parametrize(
    ("kind", "endianness"),
    [(EncapsulationKind.CDR_LE, "<"), (EncapsulationKind.CDR_BE, ">")],
    ids=["little-endian", "big-endian"],
)
def test_decoder_matches_reference_cdr_for_scalar_alignment_array_and_nested(
    kind: EncapsulationKind, endianness: str
) -> None:
    payload = _reference_payload(kind, _write_representative)
    _assert_exact_cdr(
        payload,
        (
            "00010000f9ffffff00000000000000000000f43ffd0000000000000000000000000004"
            "c0020000009cffffffc8000000a5"
            if kind == EncapsulationKind.CDR_LE
            else (
                "00000000fffffff9000000003ff4000000000000fd00000000000000c004"
                "00000000000000000002ffffff9c000000c8a5"
            )
        ),
    )
    reference_reader = CdrReader(payload)

    assert reference_reader.kind() == kind
    assert reference_reader.int32() == -7
    assert reference_reader.float64() == 1.25
    assert reference_reader.int8() == -3
    assert reference_reader.float64() == -2.5
    assert reference_reader.sequence_length() == 2
    assert reference_reader.int32_array(2) == [-100, 200]
    assert reference_reader.uint8() == 0xA5

    decoder = create_decoder_function("example_msgs/Representative", REPRESENTATIVE_SCHEMA)
    decoded = decoder(payload)

    assert decoded.scalar == -7
    assert decoded.aligned == 1.25
    assert decoded.child.lead == -3
    assert decoded.child.value == -2.5
    assert list(decoded.values) == [-100, 200]
    assert decoded.tail == 0xA5
    assert payload[:2] == (b"\x00\x01" if endianness == "<" else b"\x00\x00")


@pytest.mark.parametrize(
    ("kind", "endianness"),
    [(EncapsulationKind.CDR_LE, "<"), (EncapsulationKind.CDR_BE, ">")],
    ids=["little-endian", "big-endian"],
)
def test_generated_encoder_matches_reference_cdr_for_both_endiannesses(
    kind: EncapsulationKind, endianness: str
) -> None:
    expected = _reference_payload(kind, _write_representative)
    encoded = _generated_encoder("example_msgs/Representative", REPRESENTATIVE_SCHEMA, endianness)(
        REPRESENTATIVE_MESSAGE
    )

    assert bytes(encoded) == expected


@pytest.mark.parametrize(
    ("base_kind", "pl_kind"),
    [(EncapsulationKind.CDR_LE, 3), (EncapsulationKind.CDR_BE, 2)],
    ids=["pl-cdr-little-endian", "pl-cdr-big-endian"],
)
def test_decoder_rejects_ros2_pl_cdr_encapsulation_kinds(
    base_kind: EncapsulationKind, pl_kind: int
) -> None:
    payload = bytearray(_reference_payload(base_kind, _write_representative))
    payload[1] = pl_kind

    decoder = create_decoder_function("example_msgs/Representative", REPRESENTATIVE_SCHEMA)

    with pytest.raises(McapROS2DecodeError, match="unsupported CDR encapsulation kind"):
        decoder(payload)


@pytest.mark.parametrize(
    ("schema", "message", "write", "expected_hex"),
    [
        (
            "float32[] values\nuint8 tail",
            SimpleNamespace(values=[], tail=0xA5),
            lambda writer: (writer.write_uint32(0), writer.write_uint8(0xA5)),
            "0001000000000000a5",
        ),
        (
            "float64[] values\nuint32 tail",
            SimpleNamespace(values=[], tail=0x12345678),
            lambda writer: (writer.write_uint32(0), writer.write_uint32(0x12345678)),
            "000100000000000078563412",
        ),
    ],
    ids=["float32-followed-by-uint8", "float64-followed-by-uint32"],
)
def test_empty_dynamic_aligned_array_matches_reference_without_element_padding(
    schema: str, message: SimpleNamespace, write: ReferenceWriter, expected_hex: str
) -> None:
    schema_name = "example_msgs/EmptyArray"
    expected = _reference_payload(EncapsulationKind.CDR_LE, write)
    _assert_exact_cdr(expected, expected_hex)
    encoder = create_encoder_function(schema_name, schema)
    decoder = create_decoder_function(schema_name, schema)

    encoded = bytes(encoder(message))
    decoded = decoder(expected)

    assert encoded == expected
    assert list(decoded.values) == []
    assert decoded.tail == message.tail


@pytest.mark.parametrize(
    ("schema_name", "schema", "message", "write", "expected_hex"),
    [
        (
            "example_msgs/FixedPrimitive",
            "int32[2] values\nuint8 tail",
            SimpleNamespace(values=[-2, 9], tail=0x7E),
            lambda writer: (
                writer.write_int32(-2),
                writer.write_int32(9),
                writer.write_uint8(0x7E),
            ),
            "00010000feffffff090000007e",
        ),
        (
            "example_msgs/FixedString",
            "string[2] values\nuint8 tail",
            SimpleNamespace(values=["one", "two"], tail=0x7E),
            lambda writer: (
                writer.write_string("one"),
                writer.write_string("two"),
                writer.write_uint8(0x7E),
            ),
            "00010000040000006f6e65000400000074776f007e",
        ),
        (
            "example_msgs/FixedComplex",
            "Nested[2] values\nuint8 tail\n============\nMSG: example_msgs/Nested\nint32 value",
            SimpleNamespace(values=[{"value": -2}, {"value": 9}], tail=0x7E),
            lambda writer: (
                writer.write_int32(-2),
                writer.write_int32(9),
                writer.write_uint8(0x7E),
            ),
            "00010000feffffff090000007e",
        ),
    ],
    ids=["primitive", "string", "complex"],
)
def test_fixed_arrays_match_reference_and_reject_short_or_long_values(
    schema_name: str,
    schema: str,
    message: SimpleNamespace,
    write: ReferenceWriter,
    expected_hex: str,
) -> None:
    expected = _reference_payload(EncapsulationKind.CDR_LE, write)
    _assert_exact_cdr(expected, expected_hex)
    encoder = create_encoder_function(schema_name, schema)
    decoder = create_decoder_function(schema_name, schema)

    assert bytes(encoder(message)) == expected
    assert decoder(expected).tail == 0x7E

    for count in (1, 3):
        bad_values = message.values[:count]
        if count == 3:
            bad_values = [*message.values, message.values[-1]]
        bad_message = SimpleNamespace(values=bad_values, tail=0x7E)
        with pytest.raises(ValueError, match="fixed array expected 2 elements"):
            encoder(bad_message)


CHAR_SCHEMA = "char[] values\nchar[3] fixed\nuint8 tail"


def _write_char_values(writer: CdrWriter) -> None:
    writer.write_uint32(4)
    writer.write_uint8_array([0, 127, 128, 255])
    writer.write_uint8_array([0, 128, 255])
    writer.write_uint8(0xCC)


def test_char_arrays_match_ros2_unsigned_boundaries_for_lists() -> None:
    message = SimpleNamespace(
        values=[0, 127, 128, 255],
        fixed=[0, 128, 255],
        tail=0xCC,
    )
    expected = _reference_payload(EncapsulationKind.CDR_LE, _write_char_values)
    _assert_exact_cdr(expected, "0001000004000000007f80ff0080ffcc")
    encoder = create_encoder_function("example_msgs/CharArrays", CHAR_SCHEMA)
    decoder = create_decoder_function("example_msgs/CharArrays", CHAR_SCHEMA)

    assert bytes(encoder(message)) == expected
    decoded = decoder(expected)
    assert list(decoded.values) == [0, 127, 128, 255]
    assert list(decoded.fixed) == [0, 128, 255]
    assert decoded.tail == 0xCC


def test_char_arrays_match_reference_for_numpy_uint8_values() -> None:
    np = pytest.importorskip("numpy")
    message = SimpleNamespace(
        values=np.array([0, 127, 128, 255], dtype=np.uint8),
        fixed=np.array([0, 128, 255], dtype=np.uint8),
        tail=0xCC,
    )
    expected = _reference_payload(EncapsulationKind.CDR_LE, _write_char_values)
    encoder = create_encoder_function("example_msgs/CharArrays", CHAR_SCHEMA)

    assert bytes(encoder(message)) == expected


def test_standalone_fixed_char_array_matches_ros2_unsigned_values() -> None:
    schema = "char[3] values"
    expected = _reference_payload(
        EncapsulationKind.CDR_LE, lambda writer: writer.write_uint8_array([0, 128, 255])
    )
    _assert_exact_cdr(expected, "000100000080ff")
    encoder = create_encoder_function("example_msgs/FixedChars", schema)
    decoder = create_decoder_function("example_msgs/FixedChars", schema)

    assert bytes(encoder({"values": [0, 128, 255]})) == expected
    assert list(decoder(expected).values) == [0, 128, 255]


def test_fixed_byte_array_after_dynamic_prefix_matches_reference() -> None:
    schema = "uint8[] prefix\nuint8[2] values\nuint8 tail"
    expected = _reference_payload(
        EncapsulationKind.CDR_LE,
        lambda writer: (
            writer.write_uint32(1),
            writer.write_uint8(7),
            writer.write_uint8_array([2, 3]),
            writer.write_uint8(0xA5),
        ),
    )
    _assert_exact_cdr(expected, "0001000001000000070203a5")
    encoder = create_encoder_function("example_msgs/FixedBytes", schema)
    decoder = create_decoder_function("example_msgs/FixedBytes", schema)

    assert bytes(encoder({"prefix": [7], "values": [2, 3], "tail": 0xA5})) == expected
    decoded = decoder(expected)
    assert list(decoded.values) == [2, 3]
    assert decoded.tail == 0xA5


@pytest.mark.parametrize(
    "value",
    [
        memoryview(array("H", [1, 2])),
        memoryview(b"\x01\x02\x03")[::2],
        memoryview(b"\x01\x02\x03\x04").cast("B", shape=(2, 2)),
    ],
    ids=["multi-byte-items", "strided", "multidimensional"],
)
def test_byte_array_rejects_memoryviews_without_one_dimensional_byte_cardinality(
    value: memoryview,
) -> None:
    encoder = create_encoder_function("example_msgs/Bytes", "uint8[2] values\nuint8 tail")

    with pytest.raises(ValueError, match="memoryview must be contiguous with itemsize 1"):
        encoder({"values": value, "tail": 0xA5})


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ("bool[] values", b"\x02"),
        ("int32[] values", b"\x01\x00\x00\x00"),
        ("int32[] values", memoryview(array("i", [1, 2]))),
    ],
    ids=["bool-bytes", "int32-bytes", "int32-memoryview"],
)
def test_non_byte_primitive_arrays_reject_raw_byte_buffers(schema: str, value) -> None:
    encoder = create_encoder_function("example_msgs/RawBuffer", schema)

    with pytest.raises(ValueError, match="raw byte buffers are only supported"):
        encoder({"values": value})


def test_fixed_numeric_array_after_dynamic_prefix_matches_reference() -> None:
    schema = "uint8[] prefix\nint32[2] values\nuint8 tail"
    expected = _reference_payload(
        EncapsulationKind.CDR_LE,
        lambda writer: (
            writer.write_uint32(1),
            writer.write_uint8(7),
            writer.write_int32_array([-2, 9]),
            writer.write_uint8(0xA5),
        ),
    )
    _assert_exact_cdr(expected, "000100000100000007000000feffffff09000000a5")
    encoder = create_encoder_function("example_msgs/FixedInts", schema)
    decoder = create_decoder_function("example_msgs/FixedInts", schema)

    assert bytes(encoder({"prefix": [7], "values": [-2, 9], "tail": 0xA5})) == expected
    decoded = decoder(expected)
    assert list(decoded.values) == [-2, 9]
    assert decoded.tail == 0xA5


BOOL_SCHEMA = "bool[] flags\nuint32 tail"


def _write_bool_values(writer: CdrWriter) -> None:
    writer.write_uint32(3)
    writer.write_boolean_array([True, False, True])
    writer.write_uint32(0x12345678)


def test_bool_array_list_matches_reference_cdr() -> None:
    expected = _reference_payload(EncapsulationKind.CDR_LE, _write_bool_values)
    _assert_exact_cdr(expected, "00010000030000000100010078563412")
    encoder = create_encoder_function("example_msgs/BoolArray", BOOL_SCHEMA)
    decoder = create_decoder_function("example_msgs/BoolArray", BOOL_SCHEMA)

    encoded = bytes(encoder({"flags": [True, False, True], "tail": 0x12345678}))
    decoded = decoder(expected)

    assert encoded == expected
    assert list(decoded.flags) == [True, False, True]
    assert decoded.tail == 0x12345678


def test_bool_array_list_works_without_numpy() -> None:
    script = """
import sys
sys.modules["numpy"] = None
from mcap_ros2_support_fast._planner import create_decoder_function, create_encoder_function

schema = "bool[] flags\\nuint32 tail"
encoder = create_encoder_function("example_msgs/BoolArray", schema)
decoder = create_decoder_function("example_msgs/BoolArray", schema)
encoded = bytes(encoder({"flags": [True, False, True], "tail": 0x12345678}))
decoded = decoder(encoded)
assert list(decoded.flags) == [True, False, True]
assert decoded.tail == 0x12345678
print(encoded.hex())
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    expected = _reference_payload(EncapsulationKind.CDR_LE, _write_bool_values)

    assert result.stdout.strip() == expected.hex()


def test_empty_nested_message_after_dynamic_prefix_matches_reference() -> None:
    schema = """\
uint8[] prefix
Empty empty
============
MSG: example_msgs/Empty
"""
    expected = _reference_payload(
        EncapsulationKind.CDR_LE,
        lambda writer: (
            writer.write_uint32(1),
            writer.write_uint8(7),
            writer.write_uint8(0),
        ),
    )
    _assert_exact_cdr(expected, "00010000010000000700")
    encoder = create_encoder_function("example_msgs/WithEmpty", schema)
    decoder = create_decoder_function("example_msgs/WithEmpty", schema)

    encoded = bytes(encoder({"prefix": [7], "empty": {}}))
    decoded = decoder(expected)

    assert encoded == expected
    assert list(decoded.prefix) == [7]
    assert decoded.empty == type(decoded.empty)()


DEFAULT_SCHEMA = 'int32 count 42\nstring name "default"\nuint8 tail'


def _write_defaults(writer: CdrWriter) -> None:
    writer.write_int32(42)
    writer.write_string("default")
    writer.write_uint8(0xDD)


def test_dictionary_none_uses_the_same_defaults_as_attribute_objects() -> None:
    expected = _reference_payload(EncapsulationKind.CDR_LE, _write_defaults)
    _assert_exact_cdr(expected, "000100002a0000000800000064656661756c7400dd")
    encoder = create_encoder_function("example_msgs/Defaults", DEFAULT_SCHEMA)

    dictionary_encoded = bytes(encoder({"count": None, "name": None, "tail": 0xDD}))
    object_encoded = bytes(encoder(SimpleNamespace(count=None, name=None, tail=0xDD)))

    assert dictionary_encoded == object_encoded == expected


@pytest.mark.parametrize(
    "payload",
    [
        b"\x01\x01\x00\x00\x00",
        b"\x00\x04\x00\x00\x00",
    ],
    ids=[
        "wrong-first-byte",
        "unknown-kind",
    ],
)
def test_decoder_rejects_unsupported_cdr_encapsulation(payload: bytes) -> None:
    decoder = create_decoder_function("example_msgs/Empty", "")

    with pytest.raises(McapROS2DecodeError):
        decoder(payload)


@pytest.mark.parametrize("payload", [b"", b"\x00"], ids=["empty", "one-byte"])
def test_decoder_short_envelope_retains_native_index_error(payload: bytes) -> None:
    decoder = create_decoder_function("example_msgs/Empty", "")

    with pytest.raises(IndexError):
        decoder(payload)


def test_empty_message_decodes_the_reference_sentinel() -> None:
    payload = _reference_payload(EncapsulationKind.CDR_LE, lambda writer: writer.write_uint8(0))
    _assert_exact_cdr(payload, "0001000000")
    reader = CdrReader(payload)
    decoder = create_decoder_function("example_msgs/Empty", "")

    assert reader.kind() == EncapsulationKind.CDR_LE
    assert reader.uint8() == 0
    decoded = decoder(payload)
    assert decoded == type(decoded)()
