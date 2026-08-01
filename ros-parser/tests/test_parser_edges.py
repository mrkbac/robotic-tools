from pathlib import Path

import pytest
from ros_parser._lark_standalone_runtime import Token
from ros_parser.models import Constant, Field, Type
from ros_parser.ros1_msg import parser as ros1_parser
from ros_parser.ros1_msg import schema_parser as ros1_schema
from ros_parser.ros2_msg import parser as ros2_parser
from ros_parser.ros2_msg import schema_parser as ros2_schema


def test_ros1_transformer_rejects_invalid_callback_values() -> None:
    transformer = ros1_parser.Ros1MessageTransformer("example")

    assert transformer.start([None]).fields_all == []
    assert transformer.content([]) is None
    with pytest.raises(TypeError, match="Expected Field"):
        transformer.content([1])
    with pytest.raises(TypeError, match="Expected bool"):
        transformer.constant_tail([[]])
    assert transformer.constant_tail([]) == ""
    with pytest.raises(ValueError, match="consecutive underscores"):
        transformer.identifier([Token("IDENTIFIER", "bad__name")])
    with pytest.raises(TypeError, match="tuple of length 2"):
        transformer.array_spec([1])
    with pytest.raises(TypeError, match="Expected bool"):
        transformer.constant_value([[]])
    assert transformer.quoted_string([Token("STRING", "unquoted")]) == "unquoted"
    assert transformer.line([]) is None
    with pytest.raises(TypeError, match="Expected Field"):
        transformer.line([1])


def test_ros1_transformer_rejects_complex_and_array_constants() -> None:
    transformer = ros1_parser.Ros1MessageTransformer()

    with pytest.raises(ValueError, match="complex type"):
        transformer.field_or_constant([Type("Point", package_name="geometry_msgs"), "POINT", 1])
    with pytest.raises(ValueError, match="array type"):
        transformer.field_or_constant([Type("int32", is_array=True), "VALUES", 1])


def test_ros1_file_and_schema_helpers(tmp_path: Path) -> None:
    message_path = tmp_path / "example_msgs" / "msg" / "Reading.msg"
    message_path.parent.mkdir(parents=True)
    message_path.write_text("float64 value\n", encoding="utf-8")
    message = ros1_parser.parse_file(message_path)

    assert message.fields == [Field(Type("float64"), "value")]
    assert ros1_parser._infer_package_name(message_path) == "example_msgs"
    assert ros1_parser._infer_package_name(tmp_path / "custom" / "Reading.msg") == "custom"

    service_path = tmp_path / "example_msgs" / "srv" / "Add.srv"
    service_path.parent.mkdir(parents=True)
    service_path.write_text("int32 a\n---\nint32 sum\n", encoding="utf-8")
    service = ros1_parser.parse_service_file(service_path)
    assert service.name == "example_msgs/Add"

    schema = (
        b"float64 value\n"
        b"================================================================================\n"
        b"MSG: example_msgs/Nested\n"
        b"int32 count\n"
    )
    definitions = ros1_schema.parse_schema_to_definitions("example_msgs/Reading", schema)
    assert "std_msgs/Header" in definitions
    assert definitions["example_msgs/Nested"].fields[0].name == "count"

    names: list[tuple[str, str]] = []
    ros1_schema.for_each_msgdef(
        "example_msgs/Reading",
        "float64 value",
        lambda full, short, _definition: names.append((full, short)),
    )
    assert names == [("example_msgs/Reading", "example_msgs/Reading")]


def test_ros2_transformer_rejects_invalid_callback_values() -> None:
    transformer = ros2_parser.MessageTransformer("example")

    assert transformer.start([None]).fields_all == []
    assert transformer.content([]) is None
    with pytest.raises(TypeError, match="Expected Field"):
        transformer.content([1])
    assert transformer.field_or_const_tail([]) == (False, None)
    with pytest.raises(TypeError, match="tuple of length 2"):
        transformer.field_or_const_tail([1])
    assert transformer.constant_tail([]) == (True, None)
    assert transformer.default_tail([]) == (False, None)
    with pytest.raises(ValueError, match="consecutive underscores"):
        transformer.identifier([Token("IDENTIFIER", "bad__name")])
    assert transformer.local_type([Token("TYPE", "Nested")]) == Type("Nested", "example")
    with pytest.raises(TypeError, match="tuple of length 3"):
        transformer.array_spec([1])
    with pytest.raises(TypeError, match="Expected bool"):
        transformer.default_value([()])
    with pytest.raises(TypeError, match="Expected bool"):
        transformer.constant_value([[]])
    with pytest.raises(TypeError, match="Expected bool"):
        transformer.primitive_literal([()])
    assert transformer.quoted_string([Token("STRING", "unquoted")]) == "unquoted"
    assert transformer.numeric_literal([Token("NUMBER", "0x10")]) == 16
    assert transformer.numeric_literal([Token("NUMBER", "0b10")]) == 2
    assert transformer.numeric_literal([Token("NUMBER", "0o10")]) == 8
    assert transformer.unquoted_string([Token("STRING", r"line\nnext")]) == "line\nnext"
    assert transformer.line([]) is None
    with pytest.raises(TypeError, match="Expected Field"):
        transformer.line([1])


def test_ros2_transformer_rejects_complex_and_array_constants() -> None:
    transformer = ros2_parser.MessageTransformer()

    with pytest.raises(ValueError, match="complex type"):
        transformer.field_or_constant(
            [Type("Point", package_name="geometry_msgs"), "POINT", (True, 1)]
        )
    with pytest.raises(ValueError, match="array type"):
        transformer.field_or_constant([Type("int32", is_array=True), "VALUES", (True, 1)])


def test_ros2_file_and_schema_helpers(tmp_path: Path) -> None:
    message_path = tmp_path / "example_msgs" / "msg" / "Reading.msg"
    message_path.parent.mkdir(parents=True)
    message_path.write_text("float64 value\n", encoding="utf-8")
    message = ros2_parser.parse_file(message_path)

    assert message.fields == [Field(Type("float64"), "value")]
    assert ros2_parser._infer_package_name(message_path) == "example_msgs"
    assert ros2_parser._infer_package_name(tmp_path / "custom" / "Reading.msg") == "custom"

    service_path = tmp_path / "example_msgs" / "srv" / "Add.srv"
    service_path.parent.mkdir(parents=True)
    service_path.write_text("int32 a\n---\nint32 sum\n", encoding="utf-8")
    service = ros2_parser.parse_service_file(service_path)
    assert service.name == "example_msgs/Add"

    action_path = tmp_path / "example_msgs" / "action" / "Move.action"
    action_path.parent.mkdir(parents=True)
    action_path.write_text(
        "int32 target\n---\nbool success\n---\nfloat32 progress\n",
        encoding="utf-8",
    )
    action = ros2_parser.parse_action_file(action_path)
    assert action.name == "example_msgs/Move"

    names: list[tuple[str, str]] = []
    ros2_schema.for_each_msgdef(
        "example_msgs/msg/Reading",
        "float64 value",
        lambda full, short, _definition: names.append((full, short)),
    )
    assert names == [("example_msgs/msg/Reading", "example_msgs/Reading")]


def test_transformer_content_accepts_fields_and_constants() -> None:
    field = Field(Type("int32"), "value")
    constant = Constant(Type("int32"), "DEFAULT", 1)

    assert ros1_parser.Ros1MessageTransformer().content([field]) is field
    assert ros2_parser.MessageTransformer().content([constant]) is constant
