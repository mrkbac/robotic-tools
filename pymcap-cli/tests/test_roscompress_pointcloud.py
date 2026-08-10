"""Point cloud cleanup + topic exclusion in roscompress."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pymcap_cli.cmd.roscompress_cmd as roscompress_module
from mcap_codec_support.pointcloud.factories import (
    CloudiniPointCloudDecompressFactory,
    CompressedPointCloudDecompressFactory,
)
from mcap_codec_support.pointcloud.schemas import (
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    POINTCLOUD2,
)
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cmd.roscompress_cmd import roscompress
from small_mcap import CompressionType, McapWriter, get_summary, read_message, read_message_decoded

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from pymcap_cli.cmd._roscompress import RoscompressConfig
    from pymcap_cli.core.message_filter import TopicSelection
    from pymcap_cli.core.processors.message_transform import MessageTransformProcessor

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


def test_roscompress_cleans_pointclouds_and_excludes_by_regex(tmp_path: Path):
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _write_input(src)

    rc = roscompress(
        str(src),
        out,
        force=True,
        image_format="none",
        exclude_topic=[r".*/secondary"],
    )
    assert rc == 0

    topics = [channel.topic for _s, channel, _m in _iter_raw(out)]
    assert "/lidar/points/secondary" not in topics  # excluded before decode
    assert topics.count("/status") == 3  # non-cloud topics copied verbatim
    assert topics.count("/lidar/points") == 3

    # The compressed cloud decodes back with the (0,0,0) pads gone.
    with out.open("rb") as f:
        clouds = [
            m.decoded_message
            for m in read_message_decoded(
                f, decoder_factories=[CloudiniPointCloudDecompressFactory()]
            )
            if m.channel.topic == "/lidar/points"
        ]
    assert len(clouds) == 3
    for cloud in clouds:
        n = int(cloud.width) * int(cloud.height)
        assert n == 6  # 6 valid, 4 zeros dropped
        buf = np.frombuffer(bytes(cloud.data), np.uint8).reshape(n, int(cloud.point_step))
        xyz = np.ascontiguousarray(buf[:, :12]).view(np.float32).reshape(n, 3)
        assert int((xyz == 0).all(axis=1).sum()) == 0


def test_roscompress_clean_pointcloud_without_compression(tmp_path: Path):
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _write_input(src)

    rc = roscompress(
        str(src),
        out,
        force=True,
        image_format="none",
        pointcloud=False,
        pointcloud_drop_invalid=True,
    )
    assert rc == 0

    with out.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    schema_names = {schema.name for schema in summary.schemas.values()}
    assert "sensor_msgs/msg/PointCloud2" in schema_names
    assert all("CompressedPointCloud" not in name for name in schema_names)

    with out.open("rb") as f:
        clouds = [
            m.decoded_message
            for m in read_message_decoded(f, decoder_factories=[DecoderFactory()])
            if m.channel.topic == "/lidar/points"
        ]
    assert [int(cloud.width) * int(cloud.height) for cloud in clouds] == [6, 6, 6]


def _iter_raw(path: Path):
    with path.open("rb") as f:
        yield from list(read_message(f))


def _schema_names_by_topic(path: Path) -> dict[str, str]:
    with path.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    return {
        channel.topic: summary.schemas[channel.schema_id].name
        for channel in summary.channels.values()
    }


def test_roscompress_applies_a_pointcloud_profile_to_matching_topics_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """One topic gets Draco/Foxglove, the other inherits the Cloudini defaults."""
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)
    worker_counts: list[int] = []
    create_processor = roscompress_module.create_pointcloud_compress_processor

    def create_counted_processor(
        config: RoscompressConfig,
        *,
        workers: int = 0,
        topics: TopicSelection,
    ) -> MessageTransformProcessor:
        worker_counts.append(workers)
        return create_processor(config, workers=workers, topics=topics)

    monkeypatch.setattr(roscompress_module, "pointcloud_worker_count", lambda: 4)
    monkeypatch.setattr(
        roscompress_module, "create_pointcloud_compress_processor", create_counted_processor
    )

    rc = roscompress(
        str(src),
        out,
        force=True,
        image_format="none",
        pointcloud=True,
        pointcloud_topic_options=["/lidar/points:pc-format=draco,pc-schema=foxglove"],
    )
    assert rc == 0

    schemas = _schema_names_by_topic(out)
    assert schemas["/lidar/points"] == FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA
    assert schemas["/lidar/points/secondary"] == COMPRESSED_POINTCLOUD2_SCHEMA
    assert worker_counts == [4, 4]


def test_roscompress_pointcloud_profile_regex_covers_every_matching_topic(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)

    rc = roscompress(
        str(src),
        out,
        force=True,
        image_format="none",
        pointcloud=True,
        pointcloud_topic_options=[r"/lidar/points.*:pc-format=draco,pc-schema=foxglove"],
    )
    assert rc == 0

    schemas = _schema_names_by_topic(out)
    assert schemas["/lidar/points"] == FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA
    assert schemas["/lidar/points/secondary"] == FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA


def test_roscompress_cleans_profiled_and_default_pointcloud_topics(tmp_path: Path):
    """Cleanup still applies to every cloud when one topic has its own profile."""
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)

    rc = roscompress(
        str(src),
        out,
        force=True,
        image_format="none",
        pointcloud=True,
        pointcloud_drop_invalid=True,
        pointcloud_topic_options=["/lidar/points:pc-format=draco,pc-schema=foxglove"],
    )
    assert rc == 0

    factories = [CompressedPointCloudDecompressFactory(), DecoderFactory()]
    with out.open("rb") as f:
        counts = {
            m.channel.topic: int(m.decoded_message.width) * int(m.decoded_message.height)
            for m in read_message_decoded(f, decoder_factories=factories)
            if m.channel.topic.startswith("/lidar/points")
        }
    # All valid points survive: 6 on the primary topic and 2 on the secondary.
    assert counts["/lidar/points"] == 6
    assert counts["/lidar/points/secondary"] == 2


def test_roscompress_warns_when_a_pointcloud_profile_matches_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)

    with caplog.at_level(logging.WARNING):
        rc = roscompress(
            str(src),
            out,
            force=True,
            image_format="none",
            pointcloud=True,
            pointcloud_topic_options=["/lidar/pionts:resolution=0.5"],
        )

    assert rc == 0
    assert any(
        "No point-cloud input topics matched" in record.message
        and "/lidar/pionts" in str(record.args)
        for record in caplog.records
    )
