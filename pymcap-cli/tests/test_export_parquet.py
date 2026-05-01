"""End-to-end tests for ``export-parquet`` — MCAP → per-topic Parquet files."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pointcloud2 import PointField, create_cloud
from pymcap_cli.cmd.export_parquet_cmd import export_parquet
from pymcap_cli.exporters._common import unique_topic_filename as _unique_topic_filename
from small_mcap import McapWriter

_POINTCLOUD2_SCHEMA = """\
std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense

================================================================================
MSG: sensor_msgs/PointField
uint8 INT8    = 1
uint8 UINT8   = 2
uint8 INT16   = 3
uint8 UINT16  = 4
uint8 INT32   = 5
uint8 UINT32  = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8
string name
uint32 offset
uint8  datatype
uint32 count

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""


def test_unique_topic_filename_handles_repeated_collisions() -> None:
    used = {"a", "a_2"}
    assert _unique_topic_filename("/a!", used) == "a_3"
    assert _unique_topic_filename("/123", set()) == "t_123"
    assert _unique_topic_filename("///", set()) == "topic"


def _make_cloud_dict(n: int) -> dict:
    pts = np.array(
        [(float(i), float(2 * i), float(3 * i), float(i)) for i in range(n)],
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")]),
    )
    fields = [
        PointField("x", 0, PointField.FLOAT32),
        PointField("y", 4, PointField.FLOAT32),
        PointField("z", 8, PointField.FLOAT32),
        PointField("intensity", 12, PointField.FLOAT32),
    ]
    cloud = create_cloud(header=None, fields=fields, points=pts)
    return {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "base"},
        "height": cloud.height,
        "width": cloud.width,
        "fields": [
            {"name": f.name, "offset": f.offset, "datatype": f.datatype, "count": f.count}
            for f in cloud.fields
        ],
        "is_bigendian": cloud.is_bigendian,
        "point_step": cloud.point_step,
        "row_step": cloud.row_step,
        "data": cloud.data,
        "is_dense": cloud.is_dense,
    }


@pytest.fixture
def pointcloud_mcap(tmp_path: Path) -> Path:
    """Write a small MCAP file with one PointCloud2 topic."""
    out = tmp_path / "pointcloud.mcap"
    with out.open("wb") as f:
        writer = McapWriter(f, encoder_factory=ROS2EncoderFactory())
        writer.start()
        sid = 1
        cid = 1
        writer.add_schema(
            sid, "sensor_msgs/msg/PointCloud2", "ros2msg", _POINTCLOUD2_SCHEMA.encode()
        )
        writer.add_channel(
            channel_id=cid, topic="/lidar/points", message_encoding="cdr", schema_id=sid
        )
        for i in range(3):
            writer.add_message_encode(
                channel_id=cid,
                log_time=i * 1_000_000,
                publish_time=i * 1_000_000,
                data=_make_cloud_dict(n=4),
            )
        writer.finish()
    return out


_IMU_LIKE_SCHEMA = """\
std_msgs/Header header
float64[9] covariance
float32 temperature

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""


@pytest.fixture
def fixed_array_mcap(tmp_path: Path) -> Path:
    """MCAP with a ``float64[9]`` field — guards against the 9/72 memoryview bug."""
    out = tmp_path / "imu_like.mcap"
    with out.open("wb") as f:
        writer = McapWriter(f, encoder_factory=ROS2EncoderFactory())
        writer.start()
        writer.add_schema(1, "pkg/msg/ImuLike", "ros2msg", _IMU_LIKE_SCHEMA.encode())
        writer.add_channel(channel_id=1, topic="/imu", message_encoding="cdr", schema_id=1)
        for i in range(3):
            writer.add_message_encode(
                channel_id=1,
                log_time=i * 1_000_000,
                publish_time=i * 1_000_000,
                data={
                    "header": {"stamp": {"sec": i, "nanosec": 0}, "frame_id": "imu"},
                    "covariance": [float(i * 10 + j) for j in range(9)],
                    "temperature": 20.0 + i,
                },
            )
        writer.finish()
    return out


def test_export_fixed_size_float_array_keeps_exact_length(
    fixed_array_mcap: Path, tmp_path: Path
) -> None:
    # Regression: _to_plain used to call .tobytes() on typed memoryviews,
    # turning float64[9] into 72 bytes and breaking pyarrow FixedSizeList.
    out_dir = tmp_path / "out"
    rc = export_parquet(str(fixed_array_mcap), out_dir, force=True)
    assert rc == 0

    table = pq.read_table(out_dir / "imu.parquet")
    schema = table.schema

    cov_type = schema.field("covariance").type
    temp_type = schema.field("temperature").type
    assert cov_type == pa.list_(pa.float64(), list_size=9)
    assert temp_type == pa.float32()

    cov = table.column("covariance").to_pylist()
    assert cov[0] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert len(cov[0]) == 9
    assert table.column("temperature").to_pylist()[0] == pytest.approx(20.0)

    # Header.stamp is collapsed from struct<sec, nanosec> to timestamp(ns).
    header_type = schema.field("header").type
    stamp_type = header_type.field("stamp").type
    assert stamp_type == pa.timestamp("ns")

    # Values round-trip: sec=i, nanosec=0 → i seconds past epoch.
    stamps_ns = [s.value for s in table.column("header").combine_chunks().field("stamp")]
    assert stamps_ns == [0, 1_000_000_000, 2_000_000_000]


def test_export_expands_pointcloud_to_list_struct(pointcloud_mcap: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = export_parquet(str(pointcloud_mcap), out_dir, force=True)
    assert rc == 0

    table = pq.read_table(out_dir / "lidar_points.parquet")
    assert table.num_rows == 3

    # points is list<struct<x,y,z,intensity>> — length matches, dtypes preserved.
    points_type = table.schema.field("points").type
    assert pa.types.is_list(points_type)
    struct_type = points_type.value_type
    assert struct_type.field("x").type == pa.float32()
    assert struct_type.field("intensity").type == pa.float32()

    first_row_points = table.column("points").to_pylist()[0]
    assert len(first_row_points) == 4
    assert first_row_points[0]["x"] == pytest.approx(0.0)
    assert first_row_points[0]["intensity"] == pytest.approx(0.0)

    # _log_time_ns is timestamp(ns).
    assert table.schema.field("_log_time_ns").type == pa.timestamp("ns")
    times_ns = [s.value for s in table.column("_log_time_ns")]
    assert times_ns == [0, 1_000_000, 2_000_000]


def test_export_writes_topics_index(pointcloud_mcap: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    assert export_parquet(str(pointcloud_mcap), out_dir, force=True) == 0

    index = pq.read_table(out_dir / "_topics.parquet").to_pylist()
    assert index == [
        {
            "topic": "/lidar/points",
            "file": "lidar_points.parquet",
            "schema": "sensor_msgs/msg/PointCloud2",
            "message_count": 3,
        }
    ]
