"""Runtime checks for the public decoded ROS message classes."""

from dataclasses import is_dataclass

from mcap_codec_support._messages import Header, Time
from mcap_codec_support.pointcloud import PointCloud2, PointField
from mcap_codec_support.video import CompressedImage, Image


def test_decoded_message_types_are_slotted_ros_like_dataclasses() -> None:
    stamp = Time(sec=1, nanosec=2)
    header = Header(stamp=stamp, frame_id="lidar")
    compressed = CompressedImage(header=header, format="jpeg", data=b"jpeg")
    image = Image(
        header=header,
        height=1,
        width=1,
        encoding="rgb8",
        is_bigendian=0,
        step=3,
        data=b"rgb",
    )
    cloud = PointCloud2(
        header=header,
        height=1,
        width=1,
        fields=[PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1)],
        is_bigendian=False,
        point_step=4,
        row_step=4,
        data=b"\0\0\0\0",
        is_dense=True,
    )

    assert all(is_dataclass(message) for message in (stamp, header, compressed, image, cloud))
    assert not hasattr(cloud, "__dict__")
    assert compressed.format == "jpeg"
    assert image.encoding == "rgb8"
    assert cloud.header.frame_id == "lidar"
    assert cloud.fields[0].name == "x"


def test_decoded_message_types_expose_ros_metadata() -> None:
    assert Time._type == "builtin_interfaces/msg/Time"
    assert Header.get_fields_and_field_types() == {
        "stamp": "builtin_interfaces/Time",
        "frame_id": "string",
    }
    assert PointField.get_fields_and_field_types() == {
        "name": "string",
        "offset": "uint32",
        "datatype": "uint8",
        "count": "uint32",
    }
    assert PointCloud2._type == "sensor_msgs/msg/PointCloud2"
    assert Image._type == "sensor_msgs/msg/Image"
    assert CompressedImage._type == "sensor_msgs/msg/CompressedImage"

    fields = Image.get_fields_and_field_types()
    fields["unexpected"] = "uint8"
    assert "unexpected" not in Image.get_fields_and_field_types()
