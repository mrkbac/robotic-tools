"""Point cloud cleanup + topic exclusion in roscompress."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mcap_codec_support.pointcloud.factories import CloudiniPointCloudDecompressFactory
from mcap_codec_support.pointcloud.schemas import POINTCLOUD2
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cmd.roscompress_cmd import roscompress
from small_mcap import CompressionType, McapWriter, read_message, read_message_decoded

if TYPE_CHECKING:
    from pathlib import Path

_STRING_SCHEMA = "string data"

# Fixed-width lidar point layout: xyz float32 + line ring index.
_FIELDS = [
    {"name": "x", "offset": 0, "datatype": 7, "count": 1},
    {"name": "y", "offset": 4, "datatype": 7, "count": 1},
    {"name": "z", "offset": 8, "datatype": 7, "count": 1},
    {"name": "line", "offset": 12, "datatype": 2, "count": 1},
]
_POINT_STEP = 16


def _cloud_payload(n_valid: int, n_zero: int) -> tuple[bytes, int]:
    """Build a point buffer with ``n_valid`` real points and ``n_zero`` (0,0,0) pads."""
    dtype = np.dtype(
        {"names": ["x", "y", "z", "line"], "formats": ["<f4", "<f4", "<f4", "u1"], "itemsize": 16}
    )
    total = n_valid + n_zero
    pts = np.zeros(total, dtype=dtype)
    pts["x"][:n_valid] = np.arange(1, n_valid + 1, dtype=np.float32)
    pts["y"][:n_valid] = np.arange(1, n_valid + 1, dtype=np.float32)
    pts["z"][:n_valid] = np.arange(1, n_valid + 1, dtype=np.float32)
    pts["line"][:n_valid] = np.arange(n_valid, dtype=np.uint8) % 4
    return pts.tobytes(), total


def _pointcloud_message(data: bytes, width: int) -> dict:
    return {
        "header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": "lidar"},
        "height": 1,
        "width": width,
        "fields": _FIELDS,
        "is_bigendian": False,
        "point_step": _POINT_STEP,
        "row_step": _POINT_STEP * width,
        "data": data,
        "is_dense": True,
    }


def _write_input(path: Path) -> None:
    with path.open("wb") as f:
        writer = McapWriter(
            f, encoder_factory=ROS2EncoderFactory(), compression=CompressionType.ZSTD
        )
        writer.start(profile="ros2")
        writer.add_schema(1, "sensor_msgs/msg/PointCloud2", "ros2msg", POINTCLOUD2.encode())
        writer.add_channel(1, "/lidar/points", "cdr", 1)
        writer.add_schema(2, "std_msgs/msg/String", "ros2msg", _STRING_SCHEMA.encode())
        writer.add_channel(2, "/lidar/points/secondary", "cdr", 1)
        writer.add_channel(3, "/status", "cdr", 2)

        for i in range(3):
            data, width = _cloud_payload(n_valid=6, n_zero=4)
            writer.add_message_encode(1, 1000 + i, _pointcloud_message(data, width), 1000 + i)
            data2, width2 = _cloud_payload(n_valid=2, n_zero=2)
            writer.add_message_encode(2, 1001 + i, _pointcloud_message(data2, width2), 1001 + i)
            writer.add_message_encode(3, 1002 + i, {"data": f"ok {i}"}, 1002 + i)
        writer.finish()


def test_roscompress_cleans_pointclouds_and_excludes_by_glob(tmp_path: Path):
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _write_input(src)

    rc = roscompress(
        str(src),
        out,
        force=True,
        image_format="none",
        exclude_topic_glob=["*/secondary"],
    )
    assert rc == 0

    topics = [channel.topic for _s, channel, _m in _iter_raw(out)]
    assert "/lidar/points/secondary" not in topics  # excluded before decode
    assert topics.count("/status") == 3  # non-cloud topics copied verbatim
    assert topics.count("/lidar/points") == 3

    # The compressed cloud decodes back with the (0,0,0) pads gone.
    clouds = [
        m.decoded_message
        for m in read_message_decoded(
            out.open("rb"), decoder_factories=[CloudiniPointCloudDecompressFactory()]
        )
        if m.channel.topic == "/lidar/points"
    ]
    assert len(clouds) == 3
    for cloud in clouds:
        n = int(cloud["width"]) * int(cloud["height"])
        assert n == 6  # 6 valid, 4 zeros dropped
        buf = np.frombuffer(bytes(cloud["data"]), np.uint8).reshape(n, int(cloud["point_step"]))
        xyz = np.ascontiguousarray(buf[:, :12]).view(np.float32).reshape(n, 3)
        assert int((xyz == 0).all(axis=1).sum()) == 0


def _iter_raw(path: Path):
    with path.open("rb") as f:
        yield from list(read_message(f))
