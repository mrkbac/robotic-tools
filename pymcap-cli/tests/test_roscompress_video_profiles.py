"""End-to-end ``roscompress`` video runs with per-topic profile overrides."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from mcap_codec_support.video.ffmpeg import check_encoder_cli, find_ffmpeg
from mcap_ros2_support_fast.decoder import DecoderFactory
from pymcap_cli.cmd._roscompress_topic_options import VideoTopicProfile
from pymcap_cli.cmd.roscompress_cmd import roscompress
from small_mcap import get_summary, read_message, read_message_decoded

from tests.fixtures.image_mcap_generator import write_camera_mcap

if TYPE_CHECKING:
    from pathlib import Path

# Every case drives the real ffmpeg CLI so the per-topic encoder choice is
# observable in the output codec.
pytestmark = pytest.mark.skipif(
    find_ffmpeg() is None or not check_encoder_cli("libx264"),
    reason="ffmpeg CLI with libx264 required",
)
requires_x265 = pytest.mark.skipif(
    not check_encoder_cli("libx265"), reason="ffmpeg CLI with libx265 required"
)

_CAMERAS = ["/CAM_FRONT/image", "/CAM_BACK/image"]


def _video_profiles(*values: str) -> list[VideoTopicProfile]:
    return [VideoTopicProfile.parse(value) for value in values]


def _codecs_by_topic(path: Path) -> dict[str, set[str]]:
    codecs: dict[str, set[str]] = {}
    with path.open("rb") as stream:
        for message in read_message_decoded(stream, decoder_factories=[DecoderFactory()]):
            codecs.setdefault(message.channel.topic, set()).add(message.decoded_message.format)
    return codecs


def _raw_records_for_topic(path: Path, topic: str) -> list[tuple]:
    with path.open("rb") as stream:
        return [
            (
                schema.id,
                schema.name,
                schema.encoding,
                schema.data,
                channel.id,
                channel.schema_id,
                channel.topic,
                channel.message_encoding,
                channel.metadata,
                message.sequence,
                message.log_time,
                message.publish_time,
                bytes(message.data),
            )
            for schema, channel, message in read_message(stream)
            if channel.topic == topic
        ]


def _schema_names_by_topic(path: Path) -> dict[str, str]:
    with path.open("rb") as stream:
        summary = get_summary(stream)
    assert summary is not None
    return {
        channel.topic: summary.schemas[channel.schema_id].name
        for channel in summary.channels.values()
    }


def test_video_mode_keep_copies_topic_unchanged(tmp_path: Path) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    write_camera_mcap(src, _CAMERAS, 4)
    expected = _raw_records_for_topic(src, "/CAM_FRONT/image")

    rc = roscompress(
        str(src),
        out,
        force=True,
        pointcloud=False,
        backend="ffmpeg-cli",
        encoder="libx264",
        video_topic_options=_video_profiles("/CAM_FRONT/image:mode=keep"),
    )

    assert rc == 0
    assert _raw_records_for_topic(out, "/CAM_FRONT/image") == expected
    schemas = _schema_names_by_topic(out)
    assert schemas["/CAM_FRONT/image"] == _schema_names_by_topic(src)["/CAM_FRONT/image"]
    assert "CompressedVideo" in schemas["/CAM_BACK/image"]


@requires_x265
def test_video_profile_applies_only_to_matching_topics(tmp_path: Path) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    write_camera_mcap(src, _CAMERAS, 6)

    rc = roscompress(
        str(src),
        out,
        force=True,
        pointcloud=False,
        backend="ffmpeg-cli",
        encoder="libx264",
        video_topic_options=_video_profiles("/CAM_FRONT/image:codec=h265,encoder=libx265"),
    )

    assert rc == 0
    assert _codecs_by_topic(out) == {
        "/CAM_FRONT/image": {"h265"},
        "/CAM_BACK/image": {"h264"},
    }


@requires_x265
def test_video_profile_regex_applies_to_every_matching_topic(tmp_path: Path) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    write_camera_mcap(src, [*_CAMERAS, "/LIDAR_TOP/image"], 4)

    rc = roscompress(
        str(src),
        out,
        force=True,
        pointcloud=False,
        backend="ffmpeg-cli",
        encoder="libx264",
        video_topic_options=_video_profiles(r"/CAM_.*/image:codec=h265,encoder=libx265"),
    )

    assert rc == 0
    assert _codecs_by_topic(out) == {
        "/CAM_FRONT/image": {"h265"},
        "/CAM_BACK/image": {"h265"},
        "/LIDAR_TOP/image": {"h264"},
    }


@requires_x265
def test_earlier_video_profile_wins_over_a_broader_one(tmp_path: Path) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    write_camera_mcap(src, _CAMERAS, 4)

    rc = roscompress(
        str(src),
        out,
        force=True,
        pointcloud=False,
        backend="ffmpeg-cli",
        encoder="libx264",
        video_topic_options=_video_profiles(
            "/CAM_FRONT/image:codec=h264,encoder=libx264",
            r"/CAM_.*/image:codec=h265,encoder=libx265",
        ),
    )

    assert rc == 0
    assert _codecs_by_topic(out) == {
        "/CAM_FRONT/image": {"h264"},
        "/CAM_BACK/image": {"h265"},
    }


def test_video_topic_ffmpeg_args_apply_per_topic(tmp_path: Path) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    write_camera_mcap(src, _CAMERAS, 4)

    rc = roscompress(
        str(src),
        out,
        force=True,
        pointcloud=False,
        backend="ffmpeg-cli",
        encoder="libx264",
        ffmpeg_args="-preset ultrafast",
        video_topic_ffmpeg_args=["/CAM_FRONT/image:-tune zerolatency"],
    )

    assert rc == 0
    assert _codecs_by_topic(out) == {
        "/CAM_FRONT/image": {"h264"},
        "/CAM_BACK/image": {"h264"},
    }


def test_warns_when_a_video_profile_matches_no_topic(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    write_camera_mcap(src, _CAMERAS, 2)

    with caplog.at_level(logging.WARNING):
        rc = roscompress(
            str(src),
            out,
            force=True,
            pointcloud=False,
            backend="ffmpeg-cli",
            encoder="libx264",
            video_topic_options=_video_profiles("/CAM_FRNT/image:quality=20"),
        )

    assert rc == 0
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert any("No video input topics matched" in message for message in warnings)
    assert any("/CAM_FRNT/image" in message for message in warnings)


def test_no_warning_when_every_video_profile_matches(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    write_camera_mcap(src, _CAMERAS, 2)

    with caplog.at_level(logging.WARNING):
        rc = roscompress(
            str(src),
            out,
            force=True,
            pointcloud=False,
            backend="ffmpeg-cli",
            encoder="libx264",
            video_topic_options=_video_profiles("/CAM_FRONT/image:quality=20"),
        )

    assert rc == 0
    assert not [
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    ]


def test_warns_when_a_video_profile_is_fully_shadowed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    write_camera_mcap(src, _CAMERAS, 2)

    with caplog.at_level(logging.WARNING):
        rc = roscompress(
            str(src),
            out,
            force=True,
            pointcloud=False,
            backend="ffmpeg-cli",
            encoder="libx264",
            video_topic_options=_video_profiles(
                r"/CAM_.*/image:quality=20",
                "/CAM_FRONT/image:quality=24",
            ),
        )

    assert rc == 0
    assert any(
        "video profile '/CAM_FRONT/image' was fully shadowed" in record.getMessage()
        for record in caplog.records
    )
