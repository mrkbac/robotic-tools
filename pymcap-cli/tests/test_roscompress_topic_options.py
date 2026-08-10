from __future__ import annotations

from dataclasses import replace

import pytest
from pymcap_cli.cmd.bridge._roscompress import RoscompressConfig
from pymcap_cli.cmd.roscompress_cmd import (
    _parse_ffmpeg_args,
    _resolve_pointcloud_topic_options,
    _resolve_video_topic_options,
)


def test_pointcloud_topic_options_override_selected_values_and_inherit_defaults() -> None:
    defaults = RoscompressConfig(
        pc_format="cloudini",
        pc_schema="auto",
        pc_encoding="lossy",
        pc_compression="zstd",
        resolution=0.01,
        draco_compression_level=7,
    )

    resolved = _resolve_pointcloud_topic_options(
        [
            "/sensors/lidar_top/points:resolution=0.02,pc-compression=lz4",
            "/sensors/lidar_top/points:pc-encoding=lossless",
        ],
        defaults,
    )

    assert resolved == {
        "/sensors/lidar_top/points": replace(
            defaults,
            pc_encoding="lossless",
            pc_compression="lz4",
            resolution=0.02,
        )
    }


def test_video_topic_options_override_selected_values_and_inherit_defaults() -> None:
    defaults = RoscompressConfig(
        codec="h264",
        quality=28,
        encoder=None,
        scale=None,
        backend="ffmpeg-cli",
    )

    resolved = _resolve_video_topic_options(
        [
            "/camera/front/image:quality=24,scale=1280,codec=h265",
            "/camera/front/image:encoder=auto,backend=pyav",
        ],
        defaults,
    )

    assert resolved == {
        "/camera/front/image": replace(
            defaults,
            codec="h265",
            quality=24,
            scale=1280,
            backend="pyav",
        )
    }


def test_video_topic_ffmpeg_args_append_to_global_args() -> None:
    defaults = RoscompressConfig(
        codec="h264",
        quality=28,
        encoder=None,
        scale=None,
        backend="ffmpeg-cli",
        ffmpeg_args=_parse_ffmpeg_args("-preset medium"),
    )

    resolved = _resolve_video_topic_options(
        None,
        defaults,
        ["/CAM_FRONT/image:-tune film -threads 4"],
    )

    assert resolved["/CAM_FRONT/image"].ffmpeg_args == (
        "-preset",
        "medium",
        "-tune",
        "film",
        "-threads",
        "4",
    )

    cleared = _resolve_video_topic_options(
        None,
        defaults,
        ["/CAM_BACK/image:none"],
    )
    assert cleared["/CAM_BACK/image"].ffmpeg_args == ()


def test_ffmpeg_args_reject_unbalanced_shell_quoting() -> None:
    with pytest.raises(ValueError, match="invalid FFmpeg arguments"):
        _parse_ffmpeg_args("-metadata 'unterminated")


@pytest.mark.parametrize(
    ("resolver", "specification", "match"),
    [
        (
            "pointcloud",
            "/sensors/lidar_top/points:unknown=1",
            "unknown point-cloud topic option 'unknown'",
        ),
        ("pointcloud", "/sensors/lidar_top/points:resolution=0", "must be positive"),
        ("video", "/camera/front/image:quality=52", "between 0 and 51"),
        ("video", "/camera/front/image:scale=0", "must be positive"),
        ("video", "quality=24", "TOPIC:key=value"),
    ],
)
def test_topic_options_reject_invalid_values(
    resolver: str,
    specification: str,
    match: str,
) -> None:
    if resolver == "pointcloud":
        defaults = RoscompressConfig(
            pc_format="cloudini",
            pc_schema="auto",
            pc_encoding="lossy",
            pc_compression="zstd",
            resolution=0.01,
            draco_compression_level=7,
        )
        with pytest.raises(ValueError, match=match):
            _resolve_pointcloud_topic_options([specification], defaults)
    else:
        defaults = RoscompressConfig(
            codec="h264",
            quality=28,
            encoder=None,
            scale=None,
            backend="ffmpeg-cli",
        )
        with pytest.raises(ValueError, match=match):
            _resolve_video_topic_options([specification], defaults)
