from mcap.reader import make_reader
from mcap_ros2_support_fast._dynamic import create_decoder
from mcap_ros2_support_fast.decoder import DecoderFactory

from .generate import generate_sample_data


def test_ros2_decoder() -> None:
    with generate_sample_data() as m:
        reader = make_reader(m, decoder_factories=[DecoderFactory(create_decoder)])
        count = 0
        for index, (_, _, _, ros_msg) in enumerate(reader.iter_decoded_messages("/chatter")):
            assert ros_msg.data == f"string message {index}"
            assert ros_msg._type == "std_msgs/String"
            assert ros_msg._full_text == "# std_msgs/String\nstring data"
            count += 1
        assert count == 10

        count = 0
        for _, _, _, ros_msg in reader.iter_decoded_messages("/empty"):
            assert ros_msg._type == "std_msgs/Empty"
            assert ros_msg._full_text == "# std_msgs/Empty"
            count += 1
        assert count == 10


def test_ros2_decoder_msg_eq() -> None:
    with generate_sample_data() as m:
        reader = make_reader(m, decoder_factories=[DecoderFactory(create_decoder)])

        decoded_messages = reader.iter_decoded_messages("/chatter")
        _, _, _, msg0 = next(decoded_messages)
        _, _, _, msg1 = next(decoded_messages)
        assert msg0.data == "string message 0"
        assert msg1.data == "string message 1"
        assert msg0 != msg1
