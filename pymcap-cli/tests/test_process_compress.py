"""`process --compress-*`: transcode composes with drop/rechunk in one pass.

Exercises the CLI wiring (flags → processors) on a small mixed MCAP: cameras,
point clouds, and telemetry, plus a topic to drop.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cmd.process_cmd import process
from small_mcap import McapWriter, get_summary, read_message

from tests.fixtures.image_mcap_generator import (
    SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA,
    create_jpeg_frame,
)

if TYPE_CHECKING:
    from pathlib import Path

_PC_SCHEMA = """std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: sensor_msgs/PointField
string name
uint32 offset
uint8 datatype
uint32 count"""

_STRING_SCHEMA = "string data"
_FIELDS = [
    {"name": "x", "offset": 0, "datatype": 7, "count": 1},
    {"name": "y", "offset": 4, "datatype": 7, "count": 1},
    {"name": "z", "offset": 8, "datatype": 7, "count": 1},
]


def _pc_msg() -> dict:
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4", "<f4", "<f4"], "itemsize": 12})
    pts = np.zeros(8, dtype=dtype)
    pts["x"] = np.arange(1, 9, dtype=np.float32)
    return {
        "header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": "lidar"},
        "height": 1,
        "width": 8,
        "fields": _FIELDS,
        "is_bigendian": False,
        "point_step": 12,
        "row_step": 96,
        "data": pts.tobytes(),
        "is_dense": True,
    }


def _write_mixed(path: Path, n: int = 8) -> None:
    buf = io.BytesIO()
    w = McapWriter(buf, chunk_size=1 << 20, encoder_factory=ROS2EncoderFactory())
    w.start(profile="ros2")
    w.add_schema(
        1,
        "sensor_msgs/msg/CompressedImage",
        "ros2msg",
        SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA.encode(),
    )
    w.add_schema(2, "sensor_msgs/msg/PointCloud2", "ros2msg", _PC_SCHEMA.encode())
    w.add_schema(3, "std_msgs/msg/String", "ros2msg", _STRING_SCHEMA.encode())
    w.add_channel(1, "/cam/front", "cdr", 1)
    w.add_channel(2, "/lidar/points", "cdr", 2)
    w.add_channel(3, "/status", "cdr", 3)
    w.add_channel(4, "/debug/noise", "cdr", 3)
    step = 1_000_000
    for i in range(n):
        t = i * step
        w.add_message_encode(
            1, t, {"header": {"stamp": {"sec": i, "nanosec": 0}, "frame_id": "c"},
                   "format": "jpeg", "data": create_jpeg_frame(160, 120, i)}, t)
        w.add_message_encode(2, t + 1, _pc_msg(), t + 1)
        w.add_message_encode(3, t + 2, {"data": f"ok {i}"}, t + 2)
        w.add_message_encode(4, t + 3, {"data": f"noise {i}"}, t + 3)
    w.finish()
    path.write_bytes(buf.getvalue())


def _schemas_by_topic(path: Path) -> dict[str, str]:
    with path.open("rb") as f:
        s = get_summary(f)
    assert s is not None
    return {c.topic: s.schemas[c.schema_id].name for c in s.channels.values()}


def _counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("rb") as f:
        for _s, ch, _m in read_message(f):
            counts[ch.topic] = counts.get(ch.topic, 0) + 1
    return counts


def test_process_compress_video_and_pointcloud_with_drop(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_mixed(src)

    rc = process(
        file=[str(src)],
        output=out,
        compress_video=True,
        compress_pointcloud=True,
        video_codec="h264",
        exclude_topic_regex=[r"/debug/.*"],
        force=True,
    )
    assert rc == 0

    schemas = _schemas_by_topic(out)
    assert schemas["/cam/front"] == "foxglove_msgs/msg/CompressedVideo"
    assert schemas["/lidar/points"] == "point_cloud_interfaces/msg/CompressedPointCloud2"
    assert schemas["/status"] == "std_msgs/msg/String"  # untouched telemetry
    assert "/debug/noise" not in schemas  # dropped in the same pass

    counts = _counts(out)
    assert counts["/cam/front"] == 8
    assert counts["/lidar/points"] == 8
    assert counts["/status"] == 8


def test_process_compress_pointcloud_only_leaves_video_untouched(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_mixed(src)

    rc = process(
        file=[str(src)],
        output=out,
        compress_pointcloud=True,
        force=True,
    )
    assert rc == 0
    schemas = _schemas_by_topic(out)
    assert schemas["/lidar/points"] == "point_cloud_interfaces/msg/CompressedPointCloud2"
    # Video left as CompressedImage (not requested).
    assert schemas["/cam/front"] == "sensor_msgs/msg/CompressedImage"
