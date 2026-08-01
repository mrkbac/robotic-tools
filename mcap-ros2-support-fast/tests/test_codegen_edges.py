import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import mcap_ros2_support_fast._dynamic_encoder as dynamic_encoder
import pytest
from mcap_ros2_support_fast._dynamic_decoder import DecoderGeneratorFactory
from mcap_ros2_support_fast._dynamic_encoder import EncoderGeneratorFactory
from mcap_ros2_support_fast._planner import (
    create_decoder_function,
    create_encoder_function,
)
from mcap_ros2_support_fast._plans import (
    PlanAction,
    PrimitiveGroupAction,
    TypeId,
)
from small_mcap.writer import McapWriter

EMPTY_PLAN = (SimpleNamespace, [])


def test_decoder_rejects_unknown_primitive_type() -> None:
    generator = DecoderGeneratorFactory(EMPTY_PLAN)

    with pytest.raises(NotImplementedError, match="Unsupported type"):
        generator.generate_primitive_reader("value", cast("TypeId", 999))


def test_decoder_generates_dynamic_fixed_byte_array() -> None:
    decoder = create_decoder_function("example_msgs/Bytes", "uint8[2] value\nuint8 tail")

    decoded = decoder(bytes.fromhex("00010000010203"))

    assert list(decoded.value) == [1, 2]
    assert decoded.tail == 3


@pytest.mark.parametrize("is_native_endianness", [True, False])
def test_decoder_generates_dynamic_fixed_numeric_array(is_native_endianness: bool) -> None:
    endian = "little" if is_native_endianness else "big"
    header = "00010000" if is_native_endianness else "00000000"
    values = "0100000002000000" if is_native_endianness else "0000000100000002"
    decoder = create_decoder_function("example_msgs/Ints", "int32[2] value")

    decoded = decoder(bytes.fromhex(header + values))

    assert list(decoded.value) == [1, 2], endian


def test_decoder_rejects_unknown_action() -> None:
    generator = DecoderGeneratorFactory(EMPTY_PLAN)
    action = cast("PlanAction", SimpleNamespace(type=999))

    with pytest.raises(ValueError, match="Unknown action type"):
        generator.generate_type(action)


def test_decoder_generates_static_tail_group() -> None:
    generator = DecoderGeneratorFactory(EMPTY_PLAN)
    generator.static_offset = 4
    group = PrimitiveGroupAction([("value", TypeId.INT32, None)])

    names = generator._generate_tail_group(group)

    assert names == ["_v1"]
    assert "_d_le_i(_data, 4)" in str(generator.code)
    assert generator.static_offset == 8


def test_decoder_generates_dynamic_empty_message() -> None:
    decoder = create_decoder_function("example_msgs/Empty", "")

    decoded = decoder(bytes.fromhex("0001000000"))

    assert decoded == type(decoded)()


def test_encoder_fallback_helpers_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "numpy", None)
        no_numpy_encoder = importlib.reload(dynamic_encoder)

        assert no_numpy_encoder._to_packed_bytes([1, 2], "B") == b"\x01\x02"
        assert no_numpy_encoder._to_packed_bytes([True, False], "?") == b"\x01\x00"
        assert no_numpy_encoder._to_packed_bytes([0x0102], "H", ">") == b"\x01\x02"
        assert no_numpy_encoder._array_length((1, 2, 3)) == 3
        with pytest.raises(ValueError, match="expected 2 elements, got 1"):
            no_numpy_encoder._to_packed_bytes([True], "?", expected_length=2)
        with pytest.raises(ValueError, match="expected 2 elements, got 1"):
            no_numpy_encoder._to_packed_bytes([1], "H", expected_length=2)

    importlib.reload(dynamic_encoder)


@pytest.mark.parametrize("schema", ["bool[2] value", "int32[2] value"])
def test_fixed_primitive_array_rejects_short_single_pass_iterable(schema: str) -> None:
    encoder = create_encoder_function("example_msgs/Values", schema)

    with pytest.raises(ValueError, match="expected 2 elements, got 1"):
        encoder({"value": iter([1])})


def test_encoder_generates_static_alignment_padding() -> None:
    generator = EncoderGeneratorFactory(EMPTY_PLAN)
    generator.static_offset = 5
    generator.current_alignment = 1

    generator.generate_alignment(4)

    assert "_buffer += b'\\x00\\x00\\x00'" in str(generator.code)
    assert generator.static_offset == 8


def test_encoder_generates_wstring_errors() -> None:
    for schema, message in [
        ("wstring value", {"value": "text"}),
        ("wstring[] values", {"values": []}),
    ]:
        encoder = create_encoder_function("example_msgs/WString", schema)
        with pytest.raises(NotImplementedError, match="wstring not implemented"):
            encoder(message)


def test_encoder_rejects_unknown_primitive_type() -> None:
    generator = EncoderGeneratorFactory(EMPTY_PLAN)

    with pytest.raises(NotImplementedError, match="Unsupported type"):
        generator.generate_primitive_writer("value", cast("TypeId", 999))


def test_encoder_rejects_unknown_action() -> None:
    generator = EncoderGeneratorFactory(EMPTY_PLAN)
    action = cast("PlanAction", SimpleNamespace(type=999))

    with pytest.raises(ValueError, match="Unknown action type"):
        generator.generate_type_writer("message", action)


def test_encoder_generates_dynamic_empty_message() -> None:
    encoder = create_encoder_function("example_msgs/Empty", "")

    assert bytes(encoder({})) == bytes.fromhex("0001000000")


def test_generated_module_matches_runtime_and_rejects_pl_cdr(
    tmp_path,
) -> None:
    schema_name = "example_msgs/Generated"
    schema = "int32[2] values\nint32 count 42"
    mcap_path = tmp_path / "schema.mcap"
    generated_path = tmp_path / "generated.py"

    with mcap_path.open("wb") as output:
        writer = McapWriter(output)
        writer.start()
        writer.add_schema(1, schema_name, "ros2msg", schema.encode())
        writer.finish()

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "generate_code.py"),
            str(mcap_path),
            "--output",
            str(generated_path),
        ],
        check=True,
    )

    message = {"values": [1, 2], "count": None}
    runtime_encoded = bytes(create_encoder_function(schema_name, schema)(message))
    assert runtime_encoded == bytes.fromhex("0001000001000000020000002a000000")

    isolated_check = f"""
import importlib.util

spec = importlib.util.spec_from_file_location("generated_ros2", {str(generated_path)!r})
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

expected_le = bytes.fromhex("0001000001000000020000002a000000")
expected_be = bytes.fromhex("0000000000000001000000020000002a")
encoded = module.encode_example_msgs_Generated({{"values": [1, 2], "count": None}})
assert bytes(encoded) == expected_le
assert list(module.decode_example_msgs_Generated(expected_le).values) == [1, 2]
assert list(module.decode_example_msgs_Generated(expected_be).values) == [1, 2]

for kind in (2, 3):
    payload = bytearray(expected_le)
    payload[1] = kind
    try:
        module.decode_example_msgs_Generated(payload)
    except module.McapROS2DecodeError as exc:
        assert "unsupported CDR encapsulation kind" in str(exc)
    else:
        raise AssertionError(f"PL-CDR kind {{kind}} was accepted")
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", isolated_check], check=True)
