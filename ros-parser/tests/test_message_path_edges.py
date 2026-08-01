from array import array
from collections import UserDict
from collections.abc import Sequence
from typing import cast

import pytest
import ros_parser.message_path.models as message_path_models
from ros_parser.message_path import (
    NO_OUTPUT,
    ArraySlice,
    Comparison,
    ComparisonOperator,
    CompoundFilter,
    FieldAccess,
    Filter,
    FilterFieldRef,
    MathModifier,
    MessagePath,
    MessagePathError,
    MessagePathEvaluator,
    ModifierFieldRef,
    StreamModifier,
    ValidationError,
    modifiers,
    parse_message_path,
)
from ros_parser.message_path.models import (
    _FLOAT64_TYPE,
    _compare,
    _extract_package_name,
    _get_message_definition,
    _Modifier,
    _resolve_field_path,
)
from ros_parser.models import Constant, Field, MessageDefinition, Type


class _BrokenSequence(Sequence[int]):
    def __getitem__(self, index: int | slice) -> int:
        raise IndexError(index)

    def __len__(self) -> int:
        return 1


def _message_definition() -> MessageDefinition:
    return MessageDefinition(
        "example_msgs/Pair",
        [
            Constant(Type("int32"), "ACTIVE", 1),
            Field(Type("int32"), "left"),
            Field(Type("int32"), "right"),
        ],
    )


def test_field_lookup_errors_cover_mapping_and_object_paths() -> None:
    assert FieldAccess("value").apply(UserDict({"value": 4}), {}) == 4

    class Value:
        value = 5

    assert FieldAccess("value").apply(Value(), {}) == 5

    with pytest.raises(MessagePathError, match="Available fields: other"):
        FieldAccess("missing").apply(UserDict({"other": 1}), {})
    with pytest.raises(MessagePathError, match="object of type 'Value'"):
        FieldAccess("missing").apply(Value(), {})


def test_slice_wraps_sequence_index_errors() -> None:
    with pytest.raises(MessagePathError, match="out of range"):
        ArraySlice(0, 1).apply(_BrokenSequence(), {})


def test_filter_field_references_bind_to_fields_and_missing_values_are_filtered() -> None:
    definition = _message_definition()
    ref = FilterFieldRef("right")
    expression = CompoundFilter(
        "and",
        [
            Comparison("left", ComparisonOperator.EQUAL, ref),
            Comparison("left", ComparisonOperator.GREATER_THAN, 0),
        ],
    )
    filter_ = Filter(expression)
    current_type = Type("Pair", package_name="example_msgs")

    assert filter_.validate(current_type, definition, {}) == (current_type, definition)
    assert ref.resolve({"right": 3}) == 3
    assert filter_.apply({"left": 3, "right": 3}, {}) == {"left": 3, "right": 3}

    missing = Filter(Comparison("missing", ComparisonOperator.EQUAL, 1))
    assert missing.apply({"left": 1}, {}) is None
    assert _resolve_field_path({"nested": {"value": 2}}, "nested.value") == 2


def test_comparison_rejects_unknown_operator() -> None:
    operator = cast("ComparisonOperator", "approximately")
    with pytest.raises(MessagePathError, match="Unsupported comparison operator"):
        _compare(1, operator, 1)


def test_math_modifier_wraps_registered_operation_errors(monkeypatch) -> None:
    def fail(_values: list[int | float]) -> float:
        raise RuntimeError("broken reducer")

    monkeypatch.setitem(
        message_path_models._MODIFIERS,
        "broken_args",
        _Modifier(
            func=lambda value: value,
            argument_reducer=fail,
        ),
    )
    with pytest.raises(MessagePathError, match="broken reducer"):
        MathModifier("broken_args", [ModifierFieldRef("value")]).apply({"value": 1}, {})

    monkeypatch.setitem(
        message_path_models._MODIFIERS,
        "broken_array",
        _Modifier(
            func=lambda value: value,
            array_reducer=fail,
        ),
    )
    with pytest.raises(MessagePathError, match="broken reducer"):
        MathModifier("broken_array", []).apply([1], {})

    def broken_object(_value: dict[str, int]) -> float:
        raise RuntimeError("broken object")

    monkeypatch.setitem(
        message_path_models._MODIFIERS,
        "broken_object",
        _Modifier(func=broken_object, kind="object"),
    )
    with pytest.raises(MessagePathError, match="broken object"):
        MathModifier("broken_object", []).apply({}, {})


def test_modifier_error_paths_and_non_list_iterables() -> None:
    with pytest.raises(MessagePathError, match="numeric array"):
        modifiers._numeric_array(1, "sum")
    for non_numeric_iterable in ("123", {"value": 1}):
        with pytest.raises(MessagePathError, match="numeric array"):
            modifiers._numeric_array(non_numeric_iterable, "sum")

    assert (
        MathModifier(
            "product",
            [ModifierFieldRef("x"), ModifierFieldRef("y")],
        ).apply({"x": 2, "y": 3}, {})
        == 6
    )
    assert MathModifier("product", [2, 3]).apply({}, {}) == 6
    assert modifiers._magnitude(array("d", [3, 4])) == 5
    assert modifiers._magnitude(range(3, 5)) == 5
    assert parse_message_path("/topic.values.@norm").apply({"values": range(3, 5)}) == 5
    with pytest.raises(MessagePathError, match="numeric array"):
        MathModifier("magnitude", []).apply(UserDict({3: "x", 4: "y"}), {})

    with pytest.raises(MessagePathError, match="magnitude requires"):
        modifiers._magnitude(1)
    with pytest.raises(MessagePathError, match="accepts at most 0 arguments"):
        MathModifier("abs", [1]).apply(1, {})
    with pytest.raises(MessagePathError, match="to_sec requires"):
        modifiers._to_sec({"sec": 1})
    with pytest.raises(MessagePathError, match="to_nsec requires"):
        modifiers._to_nsec({"sec": 1})


def test_magnitude_accepts_numpy_numeric_scalars() -> None:
    np = pytest.importorskip("numpy")

    assert MathModifier("magnitude", []).apply(np.array([3, 4], dtype=np.int64), {}) == 5
    assert MathModifier("magnitude", []).apply(np.array([3.0, 4.0]), {}) == 5


def test_math_modifier_validation_edges(monkeypatch) -> None:
    pair = _message_definition()
    pair_type = Type("Pair", package_name="example_msgs")

    with pytest.raises(ValidationError, match="Unknown math modifier"):
        MathModifier("unknown", []).validate(Type("int32"), None, {})
    with pytest.raises(ValidationError, match="does not accept field references"):
        MathModifier("add", [ModifierFieldRef("left")]).validate(pair_type, pair, {})
    with pytest.raises(ValidationError, match="accepts at most 0 arguments"):
        MathModifier("abs", [ModifierFieldRef("left")]).validate(pair_type, pair, {})
    with pytest.raises(ValidationError, match="requires a numeric array"):
        MathModifier("min", []).validate(Type("string", is_array=True), None, {})
    with pytest.raises(ValidationError, match="requires a numeric array"):
        MathModifier("norm", []).validate(
            Type("Point", package_name="geometry_msgs", is_array=True),
            None,
            {},
        )
    with pytest.raises(ValidationError, match="cannot be applied to array"):
        MathModifier("rpy", []).validate(Type("float64", is_array=True), None, {})
    with pytest.raises(ValidationError, match="does not accept arguments"):
        MathModifier("sum", [1]).validate(Type("float64", is_array=True), None, {})

    monkeypatch.setitem(
        message_path_models._MODIFIERS,
        "array_float",
        _Modifier(
            func=lambda value: value,
            array_reducer=lambda values: values[0],
            return_type=_FLOAT64_TYPE,
        ),
    )
    assert MathModifier("array_float", []).validate(Type("int32", is_array=True), None, {}) == (
        _FLOAT64_TYPE,
        None,
    )

    monkeypatch.setitem(
        message_path_models._MODIFIERS,
        "aggregate_first",
        _Modifier(
            func=lambda values: values[0],
            kind="aggregate",
            preserves_element_type=True,
        ),
    )
    result_type, result_definition = MathModifier("aggregate_first", []).validate(
        Type("int32", is_array=True),
        None,
        {},
    )
    assert result_type == Type("int32")
    assert result_definition is None


@pytest.mark.parametrize(
    ("path_text", "message_definition"),
    [
        (
            "/topic.value.@abs(1)",
            MessageDefinition("pkg/Scalar", [Field(Type("float64"), "value")]),
        ),
        (
            "/topic.value.@div",
            MessageDefinition("pkg/Scalar", [Field(Type("float64"), "value")]),
        ),
        (
            "/topic.value.@clamp(1)",
            MessageDefinition("pkg/Scalar", [Field(Type("float64"), "value")]),
        ),
        (
            "/topic.@norm(1)",
            MessageDefinition(
                "pkg/Point",
                [Field(Type("float64"), "x"), Field(Type("float64"), "y")],
            ),
        ),
        (
            "/topic.values.@length(1)",
            MessageDefinition("pkg/Values", [Field(Type("string", is_array=True), "values")]),
        ),
    ],
)
def test_validate_rejects_modifier_arity(path_text, message_definition) -> None:
    with pytest.raises(ValidationError):
        parse_message_path(path_text).validate(message_definition, {})


@pytest.mark.parametrize(
    ("path_text", "message_definition", "message", "expected"),
    [
        (
            "/topic.value.@abs",
            MessageDefinition("pkg/Scalar", [Field(Type("float64"), "value")]),
            {"value": -3.0},
            3.0,
        ),
        (
            "/topic.value.@div(2)",
            MessageDefinition("pkg/Scalar", [Field(Type("float64"), "value")]),
            {"value": 6.0},
            3.0,
        ),
        (
            "/topic.@norm",
            MessageDefinition(
                "pkg/Point",
                [Field(Type("float64"), "x"), Field(Type("float64"), "y")],
            ),
            {"x": 3.0, "y": 4.0},
            5.0,
        ),
        (
            "/topic.values.@length",
            MessageDefinition("pkg/Values", [Field(Type("string", is_array=True), "values")]),
            {"values": ["a", "b"]},
            2,
        ),
        (
            "/topic.values.@sum",
            MessageDefinition("pkg/Values", [Field(Type("int32", is_array=True), "values")]),
            {"values": [1, 2, 3]},
            6.0,
        ),
    ],
)
def test_valid_modifier_categories_validate_and_apply(
    path_text, message_definition, message, expected
) -> None:
    path = parse_message_path(path_text)
    path.validate(message_definition, {})
    assert path.apply(message) == expected


@pytest.mark.parametrize(
    "type_name",
    [
        "byte",
        "char",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    ],
)
def test_magnitude_accepts_numeric_primitive_arrays(type_name: str) -> None:
    message_definition = MessageDefinition(
        "pkg/Values", [Field(Type(type_name, is_array=True), "values")]
    )
    path = parse_message_path("/topic.values.@magnitude")

    path.validate(message_definition, {})
    assert path.apply({"values": [3, 4]}) == 5


def test_validate_magnitude_requires_numeric_array() -> None:
    message_definition = MessageDefinition(
        "pkg/Values", [Field(Type("string", is_array=True), "values")]
    )

    with pytest.raises(ValidationError, match="numeric array"):
        parse_message_path("/topic.values.@magnitude").validate(message_definition, {})


@pytest.mark.parametrize(
    ("operation", "fields"),
    [
        ("norm", [("x", "string"), ("y", "float64")]),
        ("rpy", [("x", "string"), ("y", "float64"), ("z", "float64"), ("w", "float64")]),
        ("quat", [("roll", "string"), ("pitch", "float64"), ("yaw", "float64")]),
        ("to_sec", [("sec", "string"), ("nanosec", "uint32")]),
        ("to_nsec", [("sec", "string"), ("nanosec", "uint32")]),
    ],
)
def test_validate_object_modifiers_require_numeric_members(operation, fields) -> None:
    message_definition = MessageDefinition(
        "pkg/Object",
        [Field(Type(type_name), name) for name, type_name in fields],
    )

    with pytest.raises(ValidationError, match="numeric"):
        parse_message_path(f"/topic.@{operation}").validate(message_definition, {})


def test_stream_modifier_validation_and_evaluator_errors() -> None:
    scalar = Type("float64")
    array_type = Type("float64", is_array=True)
    message_type = Type("Pair", package_name="example_msgs")

    with pytest.raises(MessagePathError, match="requires MessagePathEvaluator"):
        StreamModifier("delta").apply(1, {})
    with pytest.raises(ValidationError, match="Unknown stream modifier"):
        StreamModifier("unknown").validate(scalar, None, {})

    assert StreamModifier("count").validate(scalar, None, {}) == (Type("int64"), None)
    assert StreamModifier("first").validate(scalar, None, {}) == (scalar, None)
    assert StreamModifier("last").validate(scalar, None, {}) == (scalar, None)
    assert StreamModifier("timedelta").validate(scalar, None, {}) == (_FLOAT64_TYPE, None)
    with pytest.raises(ValidationError, match="one value per message"):
        StreamModifier("timedelta").validate(array_type, None, {})
    assert StreamModifier("unchanged_for").validate(scalar, None, {}) == (_FLOAT64_TYPE, None)
    with pytest.raises(ValidationError, match="primitive scalar"):
        StreamModifier("unchanged_for").validate(message_type, None, {})
    with pytest.raises(ValidationError, match="numeric scalar"):
        StreamModifier("delta").validate(Type("string"), None, {})
    assert StreamModifier("min").validate(scalar, None, {}) == (scalar, None)
    assert StreamModifier("sum").validate(scalar, None, {}) == (_FLOAT64_TYPE, None)

    plain = MessagePathEvaluator(MessagePath("/topic", []))
    assert plain.observe({"value": 1}, 0) == {"value": 1}
    assert plain.finalize() is NO_OUTPUT

    with pytest.raises(MessagePathError, match="at most one stream reducer"):
        MessagePathEvaluator(
            MessagePath("/topic", [StreamModifier("sum"), StreamModifier("count")])
        )
    with pytest.raises(MessagePathError, match="Unknown stream modifier"):
        MessagePathEvaluator(MessagePath("/topic", [StreamModifier("unknown")]))

    unchanged = MessagePathEvaluator(MessagePath("/topic", [StreamModifier("unchanged_for")]))
    with pytest.raises(MessagePathError, match="primitive scalar"):
        unchanged.observe([1], 0)
    with pytest.raises(MessagePathError, match="non-finite"):
        unchanged.observe(float("nan"), 0)

    delta = MessagePathEvaluator(MessagePath("/topic", [StreamModifier("delta")]))
    with pytest.raises(MessagePathError, match="finite numeric scalar"):
        delta.observe("not numeric", 0)


def test_message_definition_lookup_and_field_path_errors() -> None:
    definition = _message_definition()
    pair_type = Type("Pair", package_name="example_msgs")

    assert _get_message_definition(pair_type, {"Pair": definition}) is definition
    assert _extract_package_name(None) is None
    assert _extract_package_name("Pair") is None

    with pytest.raises(ValidationError, match="primitive type"):
        _get_message_definition(Type("int32"), {})
    with pytest.raises(ValidationError, match="not found"):
        _get_message_definition(pair_type, {})

    primitive_filter = Filter(Comparison("value", ComparisonOperator.EQUAL, 1))
    with pytest.raises(ValidationError, match="primitive type"):
        primitive_filter.validate(Type("int32"), None, {})

    array_filter = Filter(Comparison("value", ComparisonOperator.EQUAL, 1))
    with pytest.raises(ValidationError, match="array type"):
        array_filter._validate_expression(
            array_filter.expression,
            Type("Pair", package_name="example_msgs", is_array=True),
            None,
            {},
        )

    assert Filter(Comparison("left", ComparisonOperator.EQUAL, 1)).validate(
        pair_type,
        None,
        {"Pair": definition},
    ) == (pair_type, None)
