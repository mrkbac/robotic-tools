from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import TYPE_CHECKING

import pytest
from mcap_codec_support.video import VideoEncoderError
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
    assert simple_mcap.exists()
    with output.open("rb") as stream:
        assert sum(1 for _ in read_message(stream)) == 100


def test_compress_deletes_source_after_validating_output(simple_mcap: Path, tmp_path: Path) -> None:
    source, output = tmp_path / "source.mcap", tmp_path / "output.mcap"
    source.write_bytes(simple_mcap.read_bytes())

    result = compress(
        str(source),
        output,
        RoscompressOptions(defaults=RoscompressConfig(image_format="none", pointcloud=False)),
        delete_source=True,
    )

    assert not source.exists()
    assert result.stats.writer_statistics.message_count == 100
    assert result.dropped_frames == ()
    with output.open("rb") as stream:
        assert sum(1 for _ in read_message(stream)) == 100


def test_compress_preserves_source_and_output_when_deletion_validation_fails(
    simple_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source.mcap", tmp_path / "output.mcap"
    original = simple_mcap.read_bytes()
    source.write_bytes(original)
    run_processor = roscompress_cmd.run_processor

    def lose_output_messages(**kwargs):
        result = run_processor(**kwargs)
        with output.open("wb") as stream:
            writer = McapWriter(stream)
            writer.start(profile="ros2")
            writer.finish()
        return result

    monkeypatch.setattr(roscompress_cmd, "run_processor", lose_output_messages)

    with pytest.raises(roscompress_cmd.RoscompressError):
        compress(
            str(source),
            output,
            RoscompressOptions(defaults=RoscompressConfig(image_format="none", pointcloud=False)),
            delete_source=True,
        )

    assert source.read_bytes() == original
    with output.open("rb") as stream:
        assert list(read_message(stream)) == []


@pytest.mark.parametrize("error_field", ["errors_encountered", "validation_errors"])
def test_compress_preserves_source_when_processing_reports_errors(
    simple_mcap: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_field: str,
) -> None:
    source, output = tmp_path / "source.mcap", tmp_path / "output.mcap"
    original = simple_mcap.read_bytes()
    source.write_bytes(original)
    run_processor = roscompress_cmd.run_processor

    def report_processing_error(**kwargs):
        result = run_processor(**kwargs)
        if error_field == "errors_encountered":
            result.stats.errors_encountered = 1
        else:
            result.stats.validation_errors = 1
        return result

    monkeypatch.setattr(roscompress_cmd, "run_processor", report_processing_error)

    with pytest.raises(roscompress_cmd.RoscompressError, match="Processing reported errors"):
        compress(
            str(source),
            output,
            RoscompressOptions(defaults=RoscompressConfig(image_format="none", pointcloud=False)),
            delete_source=True,
        )

    assert source.read_bytes() == original
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


@pytest.mark.parametrize("delete_source", [False, True])
def test_compress_propagates_failure_and_removes_partial_output(
    simple_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delete_source: bool
) -> None:
    source, output = tmp_path / "source.mcap", tmp_path / "output.mcap"
    original = simple_mcap.read_bytes()
    source.write_bytes(original)

    def fail_processing(**_kwargs) -> None:
        output.write_bytes(b"partial output")
        raise RuntimeError("injected processing failure")

    monkeypatch.setattr(roscompress_cmd, "run_processor", fail_processing)

    with pytest.raises(
        roscompress_cmd.RoscompressError, match="injected processing failure"
    ) as failure:
        compress(
            str(source),
            output,
            RoscompressOptions(defaults=RoscompressConfig(image_format="none", pointcloud=False)),
            delete_source=delete_source,
        )

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert source.read_bytes() == original
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


@pytest.mark.parametrize(
    ("error_type", "diagnostic"),
    [
        (ImportError, "Optional dependencies are required for this mode"),
        (VideoEncoderError, "injected setup failure"),
    ],
)
def test_compress_reports_processor_setup_errors(
    simple_mcap: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
    diagnostic: str,
) -> None:
    error = error_type("injected setup failure")

    def fail_setup(*_args, **_kwargs) -> None:
        raise error

    monkeypatch.setattr(roscompress_cmd, "create_video_compress_processor", fail_setup)
    output = tmp_path / "output.mcap"

    with pytest.raises(roscompress_cmd.RoscompressError, match=str(error)) as failure:
        compress(
            str(simple_mcap),
            output,
            RoscompressOptions(defaults=RoscompressConfig(pointcloud=False)),
        )

    assert failure.value.__cause__ is error
    caplog.clear()
    assert roscompress_cmd.roscompress(str(simple_mcap), output, pointcloud=False) == 1
    assert diagnostic in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert not output.exists()


def test_compress_reports_invalid_message_filter(
    simple_mcap: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "output.mcap"
    options = RoscompressOptions(
        defaults=RoscompressConfig(image_format="none", pointcloud=False), start="10", end="10"
    )

    with pytest.raises(roscompress_cmd.RoscompressError, match="must be less") as failure:
        compress(str(simple_mcap), output, options)

    assert isinstance(failure.value.__cause__, ValueError)
    caplog.clear()
    assert (
        roscompress_cmd.roscompress(
            str(simple_mcap), output, image_format="none", pointcloud=False, start="10", end="10"
        )
        == 1
    )
    assert str(failure.value.__cause__) in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert not output.exists()


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
