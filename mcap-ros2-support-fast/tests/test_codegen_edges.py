import importlib
import sys
from types import SimpleNamespace
from typing import cast

import mcap_ros2_support_fast._dynamic_encoder as dynamic_encoder
import pytest
from mcap_ros2_support_fast._dynamic_decoder import DecoderGeneratorFactory
from mcap_ros2_support_fast._dynamic_encoder import EncoderGeneratorFactory
from mcap_ros2_support_fast._plans import (
    PlanAction,
    PrimitiveGroupAction,
    TypeId,
)

EMPTY_PLAN = (SimpleNamespace, [])


def test_decoder_rejects_unknown_primitive_type() -> None:
    generator = DecoderGeneratorFactory(EMPTY_PLAN)

    with pytest.raises(NotImplementedError, match="Unsupported type"):
        generator.generate_primitive_reader("value", cast("TypeId", 999))


def test_decoder_generates_dynamic_fixed_byte_array() -> None:
    generator = DecoderGeneratorFactory(EMPTY_PLAN)

    generator.generate_primitive_array("value", TypeId.UINT8, 2)

    assert "value = _data[_offset : _offset + 2]" in str(generator.code)
    assert "_offset += 2" in str(generator.code)


@pytest.mark.parametrize("is_native_endianness", [True, False])
def test_decoder_generates_dynamic_fixed_numeric_array(is_native_endianness: bool) -> None:
    native_endianness = "<" if sys.byteorder == "little" else ">"
    opposite_endianness = ">" if native_endianness == "<" else "<"
    generator = DecoderGeneratorFactory(
        EMPTY_PLAN,
        endianness=native_endianness if is_native_endianness else opposite_endianness,
    )

    generator.generate_primitive_array("value", TypeId.INT32, 2)

    code = str(generator.code)
    assert "_offset += 8" in code
    if is_native_endianness:
        assert ".cast('i')" in code
    else:
        assert "value.byteswap()" in code


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
    generator = DecoderGeneratorFactory(EMPTY_PLAN)

    generator.generate_plan("value", EMPTY_PLAN)

    assert "_offset += 1" in str(generator.code)
    assert "value = SimpleNamespace()" in str(generator.code)


def test_encoder_fallback_helpers_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "numpy", None)
        no_numpy_encoder = importlib.reload(dynamic_encoder)

        assert no_numpy_encoder._to_packed_bytes([1, 2], "B") == b"\x01\x02"
        assert no_numpy_encoder._array_length((1, 2, 3)) == 3

    importlib.reload(dynamic_encoder)


def test_encoder_generates_static_alignment_padding() -> None:
    generator = EncoderGeneratorFactory(EMPTY_PLAN)
    generator.static_offset = 5
    generator.current_alignment = 1

    generator.generate_alignment(4)

    assert "_buffer += b'\\x00\\x00\\x00'" in str(generator.code)
    assert generator.static_offset == 8


def test_encoder_generates_wstring_errors() -> None:
    generator = EncoderGeneratorFactory(EMPTY_PLAN)

    generator.generate_primitive_writer("value", TypeId.WSTRING)
    generator.generate_primitive_array_writer("values", TypeId.WSTRING, None)

    assert str(generator.code).count("wstring not implemented") == 2


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
    generator = EncoderGeneratorFactory(EMPTY_PLAN)

    generator.generate_plan_writer("message", EMPTY_PLAN)

    assert "_offset += 1" in str(generator.code)
