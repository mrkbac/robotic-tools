from array import array
from io import BytesIO

from mcap_ros2_support_fast import ROS2EncoderFactory
from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap.reader import read_message_decoded
from small_mcap.writer import McapWriter


def read_ros2_messages(stream: BytesIO):
    return read_message_decoded(stream, decoder_factories=[DecoderFactory()])


def test_write_messages() -> None:
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()
    schema_name = "test_msgs/TestData"
    schema_data = b"string a\nint32 b"
    for i in range(10):
        ros_writer.add_message_object(
            topic="/test",
            schema_name=schema_name,
            schema_data=schema_data,
            message_obj={"a": f"string message {i}", "b": i},
            log_time=i,
            publish_time=i,
            sequence=i,
        )
    ros_writer.finish()

    output.seek(0)
    for index, msg in enumerate(read_ros2_messages(output)):
        assert msg.channel.topic == "/test"
        assert msg.schema.name == "test_msgs/TestData"
        assert msg.decoded_message.a == f"string message {index}"
        assert msg.decoded_message.b == index
        assert msg.message.log_time == index
        assert msg.message.publish_time == index
        assert msg.message.sequence == index


def test_write_std_msgs_empty_messages() -> None:
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()
    schema_name = "std_msgs/msg/Empty"
    schema_data = b""
    for i in range(10):
        ros_writer.add_message_object(
            topic="/test",
            schema_name=schema_name,
            schema_data=schema_data,
            message_obj={},
            log_time=i,
            publish_time=i,
            sequence=i,
        )
    ros_writer.finish()

    output.seek(0)
    for index, msg in enumerate(read_ros2_messages(output)):
        assert msg.channel.topic == "/test"
        assert msg.schema.name == "std_msgs/msg/Empty"
        assert msg.message.log_time == index
        assert msg.message.publish_time == index
        assert msg.message.sequence == index


def test_write_uint8_array_with_py_array() -> None:
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()
    schema_name = "test_msgs/ByteArray"
    schema_data = b"uint8[] data"

    for i in range(10):
        byte_array = array("B", [i] * 5)
        ros_writer.add_message_object(
            topic="/image",
            schema_name=schema_name,
            schema_data=schema_data,
            message_obj={"data": byte_array},
            log_time=i,
            publish_time=i,
            sequence=i,
        )

    ros_writer.finish()

    output.seek(0)
    for i, msg in enumerate(read_ros2_messages(output)):
        assert msg.channel.topic == "/image"
        assert msg.schema.name == "test_msgs/ByteArray"
        assert list(msg.decoded_message.data) == [i] * 5
        assert msg.message.log_time == i
        assert msg.message.publish_time == i
        assert msg.message.sequence == i
