"""PointcloudCompressProcessor: transcode in the pipeline, and compose with
topic-drop + per-schema compression split in a single pass."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from mcap_codec_support.pointcloud.factories import CloudiniPointCloudDecompressFactory
from mcap_codec_support.pointcloud.schemas import (
    COMPRESSED_POINTCLOUD2_SCHEMA,
    POINTCLOUD2,
)
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cmd._run_processor import run_processor
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from pymcap_cli.core.processors.chunk_groupers import SchemaCompressionGrouper
from pymcap_cli.core.processors.pointcloud_clean import PointcloudCleanProcessor
from pymcap_cli.core.processors.pointcloud_compress import PointcloudCompressProcessor
from small_mcap import CompressionType, McapWriter, get_summary, read_message, read_message_decoded

if TYPE_CHECKING:
    from pathlib import Path

_STRING_SCHEMA = "string data"
_FIELDS = [
    {"name": "x", "offset": 0, "datatype": 7, "count": 1},
    {"name": "y", "offset": 4, "datatype": 7, "count": 1},
    {"name": "z", "offset": 8, "datatype": 7, "count": 1},
    {"name": "line", "offset": 12, "datatype": 2, "count": 1},
]
_POINT_STEP = 16


def _cloud_payload(n_valid: int, n_zero: int) -> tuple[bytes, int]:
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


def _pc_msg(data: bytes, width: int) -> dict:
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
        writer.add_channel(2, "/debug/noise", "cdr", 2)
        writer.add_channel(3, "/status", "cdr", 2)
        for i in range(3):
            data, width = _cloud_payload(n_valid=6, n_zero=4)
            writer.add_message_encode(1, 1000 + i, _pc_msg(data, width), 1000 + i)
            writer.add_message_encode(2, 1001 + i, {"data": f"noise {i}"}, 1001 + i)
            writer.add_message_encode(3, 1002 + i, {"data": f"ok {i}"}, 1002 + i)
        writer.finish()


def _run(
    input_path: Path,
    output_path: Path,
    *,
    extra_processors,
    output_processors=None,
    exclude=None,
):
    run_processor(
        files=[str(input_path)],
        output=output_path,
        input_options=InputOptions.from_args(
            extra_processors=extra_processors,
            exclude_topic_regex=exclude or [],
        ),
        output_options=OutputOptions(
            overwrite_policy=OverwriteCollisionPolicy.OVERWRITE,
            output_processors=output_processors or [],
        ),
    )


def _topics(path: Path) -> list[str]:
    with path.open("rb") as f:
        return [channel.topic for _s, channel, _m in read_message(f)]


def _compressed_cloud_point_counts(path: Path) -> list[int]:
    with path.open("rb") as f:
        clouds = [
            m.decoded_message
            for m in read_message_decoded(
                f, decoder_factories=[CloudiniPointCloudDecompressFactory()]
            )
            if m.channel.topic == "/lidar/points"
        ]
    return [int(cloud["width"]) * int(cloud["height"]) for cloud in clouds]


def _pointcloud_x_values(path: Path) -> list[list[float]]:
    values: list[list[float]] = []
    with path.open("rb") as f:
        for msg in read_message_decoded(f, decoder_factories=[DecoderFactory()]):
            if msg.channel.topic != "/lidar/points":
                continue
            cloud = msg.decoded_message
            n = int(cloud.width) * int(cloud.height)
            buf = np.frombuffer(bytes(cloud.data), np.uint8).reshape(n, int(cloud.point_step))
            xyz = np.ascontiguousarray(buf[:, :12]).view(np.float32).reshape(n, 3)
            values.append([float(x) for x in xyz[:, 0]])
    return values


def test_pointcloud_processor_compresses_without_cleanup(tmp_path: Path):
    """PointCloud2 → CompressedPointCloud2 without dropping invalid pads."""
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)

    _run(src, out, extra_processors=[PointcloudCompressProcessor()])

    # Non-cloud topics pass through; cloud topic count preserved.
    topics = _topics(out)
    assert topics.count("/lidar/points") == 3
    assert topics.count("/status") == 3
    assert topics.count("/debug/noise") == 3

    # The cloud channel now carries the compressed schema, and there is exactly
    # one channel for the topic (the original PointCloud2 channel is not left
    # behind empty).
    with out.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    cloud_channels = [c for c in summary.channels.values() if c.topic == "/lidar/points"]
    assert len(cloud_channels) == 1
    assert summary.schemas[cloud_channels[0].schema_id].name == COMPRESSED_POINTCLOUD2_SCHEMA

    assert _compressed_cloud_point_counts(out) == [10, 10, 10]


def test_pointcloud_clean_then_compress_drops_invalid_points(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)

    _run(
        src,
        out,
        extra_processors=[PointcloudCleanProcessor(), PointcloudCompressProcessor()],
    )

    assert _compressed_cloud_point_counts(out) == [6, 6, 6]


def test_pointcloud_clean_only_keeps_pointcloud2_schema(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)

    _run(src, out, extra_processors=[PointcloudCleanProcessor()])

    with out.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    cloud_channels = [c for c in summary.channels.values() if c.topic == "/lidar/points"]
    assert len(cloud_channels) == 1
    assert summary.schemas[cloud_channels[0].schema_id].name == "sensor_msgs/msg/PointCloud2"
    assert _pointcloud_x_values(out) == [[1.0, 5.0, 2.0, 6.0, 3.0, 4.0]] * 3


def test_pointcloud_clean_sort_field_none_preserves_original_order(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)

    _run(src, out, extra_processors=[PointcloudCleanProcessor(sort_field=None)])

    assert _pointcloud_x_values(out) == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3


def test_pointcloud_compress_drop_topic_and_split_in_one_pass(tmp_path: Path):
    """One pipeline pass: compress clouds + drop /debug/* + uncompressed cloud chunks."""
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)

    _run(
        src,
        out,
        extra_processors=[PointcloudCleanProcessor(), PointcloudCompressProcessor()],
        exclude=[r"/debug/.*"],
        output_processors=[SchemaCompressionGrouper([re.compile("CompressedPointCloud2")])],
    )

    topics = _topics(out)
    # Dropped in the same pass.
    assert "/debug/noise" not in topics
    assert topics.count("/lidar/points") == 3
    assert topics.count("/status") == 3

    # Exactly one /lidar/points channel, on the compressed schema — no orphaned
    # empty channel left behind on the original PointCloud2 schema.
    with out.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    cloud_channels = [c for c in summary.channels.values() if c.topic == "/lidar/points"]
    assert len(cloud_channels) == 1
    assert summary.schemas[cloud_channels[0].schema_id].name == COMPRESSED_POINTCLOUD2_SCHEMA

    # The compressed-cloud chunk group is stored uncompressed; telemetry stays zstd.
    compressions = {
        ci.compression for ci in (summary.chunk_indexes or []) if ci.compression is not None
    }
    assert "" in compressions  # cloud group: no compression
    assert "zstd" in compressions  # telemetry group

    assert _compressed_cloud_point_counts(out) == [6, 6, 6]


def test_pointcloud_processor_parallel_matches_inline(tmp_path: Path):
    """workers>0 (parallel, thread-local compressor) yields the same output as inline."""
    src = tmp_path / "in.mcap"
    _write_input(src)
    out_inline = tmp_path / "inline.mcap"
    out_parallel = tmp_path / "parallel.mcap"

    _run(src, out_inline, extra_processors=[PointcloudCompressProcessor(workers=0)])
    _run(src, out_parallel, extra_processors=[PointcloudCompressProcessor(workers=3)])

    # Same per-channel message counts and — since clouds decode deterministically —
    # the same decompressed point counts in the same per-channel order.
    def _clouds(p: Path) -> list[tuple[str, int]]:
        result = []
        with p.open("rb") as f:
            for m in read_message_decoded(
                f, decoder_factories=[CloudiniPointCloudDecompressFactory()]
            ):
                if m.channel.topic == "/lidar/points":
                    d = m.decoded_message
                    result.append((m.channel.topic, int(d["width"]) * int(d["height"])))
        return result

    assert _clouds(out_parallel) == _clouds(out_inline)
    assert len(_clouds(out_parallel)) == 3


def test_pointcloud_processor_preserves_timestamps(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_input(src)

    _run(src, out, extra_processors=[PointcloudCompressProcessor()])

    with out.open("rb") as f:
        cloud_times = [m.log_time for _s, c, m in read_message(f) if c.topic == "/lidar/points"]
    assert cloud_times == [1000, 1001, 1002]
