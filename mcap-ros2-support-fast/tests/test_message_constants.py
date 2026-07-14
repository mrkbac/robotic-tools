from dataclasses import fields
from sys import getsizeof

import pytest
from mcap_ros2_support_fast._planner import create_decoder_function, create_encoder_function

STATE_SCHEMA = """\
uint8 STATE_UNKNOWN=255
uint8 STATE_ACTIVE=4
bool ENABLED=true
float64 SCALE=1.25
string LABEL='example'
uint8 state
"""


def test_decoded_message_exposes_constants_like_ros2() -> None:
    payload = bytes.fromhex("0001000004")
    decoder = create_decoder_function("example_msgs/State", STATE_SCHEMA)

    message = decoder(payload)
    message_class = type(message)

    assert message_class.STATE_UNKNOWN == 255
    assert message.STATE_UNKNOWN == 255
    assert message_class.STATE_ACTIVE == 4
    assert message.STATE_ACTIVE == 4
    assert message.ENABLED is True
    assert message.SCALE == 1.25
    assert message.LABEL == "example"


def test_constants_are_read_only_on_class_and_instance() -> None:
    decoder = create_decoder_function("example_msgs/State", STATE_SCHEMA)
    message = decoder(bytes.fromhex("0001000004"))
    message_class = type(message)

    with pytest.raises(AttributeError):
        message_class.STATE_ACTIVE = 9
    with pytest.raises(AttributeError):
        del message_class.STATE_ACTIVE
    with pytest.raises(AttributeError):
        message.STATE_ACTIVE = 9
    with pytest.raises(AttributeError):
        del message.STATE_ACTIVE

    assert message_class.STATE_ACTIVE == 4
    assert message.STATE_ACTIVE == 4


def test_constants_are_not_message_fields_or_wire_data() -> None:
    payload = bytes.fromhex("0001000004")
    decoder = create_decoder_function("example_msgs/State", STATE_SCHEMA)
    encoder = create_encoder_function("example_msgs/State", STATE_SCHEMA)

    message = decoder(payload)
    same_message = decoder(payload)

    assert [field.name for field in fields(message)] == ["state"]
    assert type(message).get_fields_and_field_types() == {"state": "uint8"}
    assert repr(message) == "example_msgs_State(state=4)"
    assert message == same_message
    assert encoder(message) == payload
    with pytest.raises(TypeError):
        type(message)(state=4, STATE_ACTIVE=4)


def test_nested_and_constant_only_messages_expose_their_own_constants() -> None:
    nested_schema = """\
Nested nested
================================================================================
MSG: example_msgs/Nested
uint8 READY=1
uint8 state
"""
    nested_decoder = create_decoder_function("example_msgs/Wrapper", nested_schema)
    wrapper = nested_decoder(bytes.fromhex("0001000001"))

    assert type(wrapper.nested).READY == 1
    assert wrapper.nested.READY == 1

    empty_decoder = create_decoder_function("example_msgs/ConstantsOnly", "uint8 UNKNOWN=255")
    constants_only = empty_decoder(bytes.fromhex("0001000000"))

    assert type(constants_only).UNKNOWN == 255
    assert constants_only.UNKNOWN == 255
    assert fields(constants_only) == ()


def test_messages_without_constants_keep_plain_class_layout() -> None:
    decoder = create_decoder_function("example_msgs/PlainState", "uint8 state")
    message = decoder(bytes.fromhex("0001000001"))

    assert type(message).__bases__ == (object,)
    assert type(type(message)) is type


def test_constants_add_no_codec_hot_path_work() -> None:
    plain_decoder = create_decoder_function("example_msgs/State", "uint8 state")
    constant_decoder = create_decoder_function("example_msgs/State", "uint8 READY=1\nuint8 state")
    plain_encoder = create_encoder_function("example_msgs/State", "uint8 state")
    constant_encoder = create_encoder_function("example_msgs/State", "uint8 READY=1\nuint8 state")
    payload = bytes.fromhex("0001000001")

    plain_message = plain_decoder(payload)
    constant_message = constant_decoder(payload)

    assert plain_decoder.__code__.co_code == constant_decoder.__code__.co_code
    assert plain_decoder.__code__.co_consts == constant_decoder.__code__.co_consts
    assert plain_decoder.__code__.co_names == constant_decoder.__code__.co_names
    assert plain_encoder.__code__.co_code == constant_encoder.__code__.co_code
    assert plain_encoder.__code__.co_consts == constant_encoder.__code__.co_consts
    assert plain_encoder.__code__.co_names == constant_encoder.__code__.co_names
    assert plain_encoder(plain_message) == payload
    assert constant_encoder(constant_message) == payload
    assert getsizeof(plain_message) == getsizeof(constant_message)
    assert type(constant_message).__bases__ == (object,)
