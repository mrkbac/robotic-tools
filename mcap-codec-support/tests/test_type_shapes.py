"""Runtime checks for the public decoded-message TypedDict shapes."""

from mcap_codec_support._messages import Header, Stamp
from mcap_codec_support.pointcloud._messages import Pointcloud2Dict
from mcap_codec_support.video._messages import CompressedImageDict, ImageDict


def test_decoded_message_typed_dict_shapes_are_constructible() -> None:
    stamp: Stamp = Stamp(sec=1, nanosec=2)
    header: Header = Header(stamp=stamp, frame_id="lidar")
    compressed: CompressedImageDict = CompressedImageDict(
        header=header,
        format="jpeg",
        data=b"jpeg",
    )
    image: ImageDict = ImageDict(
        header=header,
        height=1,
        width=1,
        encoding="rgb8",
        is_bigendian=0,
        step=3,
        data=b"rgb",
    )
    cloud: Pointcloud2Dict = Pointcloud2Dict(
        header=header,
        height=1,
        width=0,
        fields=[],
        is_bigendian=False,
        point_step=0,
        row_step=0,
        data=b"",
        is_dense=True,
    )

    assert compressed["format"] == "jpeg"
    assert image["encoding"] == "rgb8"
    assert cloud["header"] == header
