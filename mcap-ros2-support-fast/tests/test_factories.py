from dataclasses import dataclass

import pytest
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import McapROS2WriteError, ROS2EncoderFactory


@dataclass
class Schema:
    id: int = 1
    name: str = "example_msgs/Value"
    encoding: str = "ros2msg"
    data: bytes = b"uint32 value"


def test_decoder_factory_rejects_unsupported_inputs() -> None:
    factory = DecoderFactory()

    assert factory.decoder_for("json", Schema()) is None
    assert factory.decoder_for("cdr", None) is None
    assert factory.decoder_for("cdr", Schema(encoding="ros1msg")) is None


def test_encoder_factory_returns_none_without_schema() -> None:
    assert ROS2EncoderFactory().encoder_for(None) is None


def test_encoder_factory_rejects_unsupported_schema() -> None:
    with pytest.raises(McapROS2WriteError, match='encoding "ros1msg"'):
        ROS2EncoderFactory().encoder_for(Schema(encoding="ros1msg"))
