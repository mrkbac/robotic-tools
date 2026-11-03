from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import TYPE_CHECKING

import pytest
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cmd import roscompress_cmd
from pymcap_cli.cmd.roscompress_cmd import RoscompressConfig, RoscompressOptions, compress
from small_mcap import McapWriter, read_message

from tests.fixtures.image_mcap_generator import (
    SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA,
    create_jpeg_frame,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_compress_returns_stats_and_empty_drops(simple_mcap: Path, tmp_path: Path) -> None:
    output = tmp_path / "output.mcap"

    result = compress(
        str(simple_mcap),
        output,
        RoscompressOptions(defaults=RoscompressConfig(image_format="none", pointcloud=False)),
    )

    assert result.stats.writer_statistics.message_count == 100
    assert result.stats.errors_encountered == 0
    assert result.dropped_frames == ()
    with output.open("rb") as stream:
        assert sum(1 for _ in read_message(stream)) == 100


def test_compress_collects_dropped_frames_across_video_profiles(tmp_path: Path) -> None:
    pytest.importorskip("av")
    pytest.importorskip("PIL.Image")
    source, output = tmp_path / "input.mcap", tmp_path / "output.mcap"
    images = [create_jpeg_frame(64, 48, index) for index in range(3)]
    scan = images[1].index(b"\xff\xda")
    entropy_start = scan + 2 + int.from_bytes(images[1][scan + 2 : scan + 4], "big")
    images[1] = images[1][: entropy_start + 4] + b"\xff\xd9"
    topics = ["/CAM_FRONT/image", "/CAM_BACK/image"]
    with source.open("wb") as stream:
        writer = McapWriter(stream, encoder_factory=ROS2EncoderFactory())
        writer.start(profile="ros2")
        writer.add_schema(
            1,
            "sensor_msgs/msg/CompressedImage",
            "ros2msg",
            SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA.encode(),
        )
        for channel_id, topic in enumerate(topics, start=1):
            writer.add_channel(channel_id, topic, "cdr", 1)
        for index, image in enumerate(images):
            for channel_id, topic in enumerate(topics, start=1):
                writer.add_message_encode(
                    channel_id,
                    1_000_000 + index * 100,
                    {
                        "header": {
                            "stamp": {"sec": index, "nanosec": index + 7},
                            "frame_id": topic,
                        },
                        "format": "jpeg",
                        "data": image,
                    },
                    2_000_000 + index * 100,
                )
        writer.finish()
    defaults = RoscompressConfig(encoder="libx264", backend="pyav", pointcloud=False)

    result = compress(
        str(source),
        output,
        RoscompressOptions(
            defaults=defaults,
            video_topic_settings={topics[0]: replace(defaults, quality=25)},
        ),
    )

    assert result.stats.writer_statistics.message_count == 4
    assert isinstance(result.dropped_frames, tuple)
    assert sorted(
        (
            drop.topic,
            drop.log_time,
            drop.publish_time,
            drop.stamp_sec,
            drop.stamp_nanosec,
            drop.frame_id,
            drop.sha256,
        )
        for drop in result.dropped_frames
    ) == sorted(
        (topic, 1_000_100, 2_000_100, 1, 8, topic, sha256(images[1]).hexdigest())
        for topic in topics
    )
    assert all(drop.reason for drop in result.dropped_frames)
    with output.open("rb") as stream:
        assert sorted(
            (channel.topic, message.log_time) for _, channel, message in read_message(stream)
        ) == sorted((topic, timestamp) for topic in topics for timestamp in [1_000_000, 1_000_200])


def test_compress_propagates_failure_and_removes_partial_output(
    simple_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.mcap"
    original = simple_mcap.read_bytes()

    def fail_processing(**_kwargs) -> None:
        output.write_bytes(b"partial output")
        raise RuntimeError("injected processing failure")

    monkeypatch.setattr(roscompress_cmd, "run_processor", fail_processing)

    with pytest.raises(
        roscompress_cmd.RoscompressError, match="injected processing failure"
    ) as failure:
        compress(
            str(simple_mcap),
            output,
            RoscompressOptions(defaults=RoscompressConfig(image_format="none", pointcloud=False)),
        )

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert simple_mcap.read_bytes() == original
    assert not output.exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_compress_rejects_output_aliasing_source(
    simple_mcap: Path, tmp_path: Path, symlink: bool
) -> None:
    source = tmp_path / "source.mcap"
    original = simple_mcap.read_bytes()
    source.write_bytes(original)
    output = source
    if symlink:
        output = tmp_path / "output.mcap"
        output.symlink_to(source)

    with pytest.raises(ValueError, match="same file as the input"):
        compress(str(source), output, force=True)

    assert source.read_bytes() == original


@pytest.mark.parametrize("error_type", [ImportError, ValueError])
def test_cli_processing_errors_keep_processing_diagnostic(
    simple_mcap: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
) -> None:
    def fail_processing(**_kwargs) -> None:
        raise error_type("injected processing failure")

    monkeypatch.setattr(roscompress_cmd, "run_processor", fail_processing)

    assert (
        roscompress_cmd.roscompress(
            str(simple_mcap), tmp_path / "output.mcap", image_format="none", pointcloud=False
        )
        == 1
    )
    assert any(
        record.getMessage() == "Error during compression" and record.exc_info
        for record in caplog.records
    )
    assert "Optional dependencies" not in caplog.text


def test_cli_unexpected_processor_setup_error_propagates(
    simple_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_setup(*_args, **_kwargs) -> None:
        raise RuntimeError("injected setup failure")

    monkeypatch.setattr(roscompress_cmd, "create_video_compress_processor", fail_setup)

    with pytest.raises(RuntimeError, match="injected setup failure"):
        roscompress_cmd.roscompress(str(simple_mcap), tmp_path / "output.mcap", pointcloud=False)
