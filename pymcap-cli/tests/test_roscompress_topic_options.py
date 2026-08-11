from __future__ import annotations

from dataclasses import replace

import pytest
from pymcap_cli.cmd._roscompress import RoscompressConfig
from pymcap_cli.cmd._roscompress_topic_options import (
    PointcloudTopicProfile,
    VideoTopicProfile,
)
from pymcap_cli.cmd.roscompress_cmd import (
    _parse_ffmpeg_args,
    _profile_entries,
    _resolve_pointcloud_topic_options,
    _resolve_video_topic_options,
)


def _pointcloud_profiles(*values: str) -> list[PointcloudTopicProfile]:
    return [PointcloudTopicProfile.parse(value) for value in values]


def _video_profiles(*values: str) -> list[VideoTopicProfile]:
    return [VideoTopicProfile.parse(value) for value in values]


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
        _pointcloud_profiles(
            "/sensors/lidar_top/points:resolution=0.02,pc-compression=lz4",
            "/sensors/lidar_top/points:pc-encoding=lossless",
        ),
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
        _video_profiles(
            "/camera/front/image:quality=24,scale=1280,codec=h265",
            "/camera/front/image:encoder=auto,backend=pyav",
        ),
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


@pytest.mark.parametrize("kind", ["pointcloud", "video"])
def test_topic_mode_keep_bypasses_compression_and_default_resets_settings(kind: str) -> None:
    defaults = RoscompressConfig(
        image_format="video",
        quality=28,
        pointcloud=True,
        resolution=0.01,
    )

    if kind == "pointcloud":
        kept = _resolve_pointcloud_topic_options(
            _pointcloud_profiles("/lidar/points:mode=keep"), defaults
        )
        reset = _resolve_pointcloud_topic_options(
            _pointcloud_profiles(
                "/lidar/points:resolution=0.5",
                "/lidar/points:mode=default",
            ),
            defaults,
        )
        assert kept["/lidar/points"] == replace(defaults, pointcloud=False)
    else:
        kept = _resolve_video_topic_options(_video_profiles("/camera/image:mode=keep"), defaults)
        reset = _resolve_video_topic_options(
            _video_profiles(
                "/camera/image:quality=20",
                "/camera/image:mode=default",
            ),
            defaults,
        )
        assert kept["/camera/image"] == replace(defaults, image_format="none")

    assert next(iter(reset.values())) == defaults


@pytest.mark.parametrize(
    ("resolver", "specification", "match"),
    [
        ("pointcloud", "/lidar/points:mode=other", "mode must be one of"),
        (
            "pointcloud",
            "/lidar/points:mode=keep,resolution=0.02",
            "mode must be specified alone",
        ),
        ("video", "/camera/image:mode=other", "mode must be one of"),
        (
            "video",
            "/camera/image:mode=keep,quality=20",
            "mode must be specified alone",
        ),
    ],
)
def test_topic_mode_rejects_ambiguous_values(
    resolver: str,
    specification: str,
    match: str,
) -> None:
    load = PointcloudTopicProfile.parse if resolver == "pointcloud" else VideoTopicProfile.parse

    with pytest.raises(ValueError, match=match):
        load(specification)


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
        ("video", "quality=24", "PATTERN:key=value"),
    ],
)
def test_topic_options_reject_invalid_values(
    resolver: str,
    specification: str,
    match: str,
) -> None:
    if resolver == "pointcloud":
        with pytest.raises(ValueError, match=match):
            PointcloudTopicProfile.parse(specification)
    else:
        with pytest.raises(ValueError, match=match):
            VideoTopicProfile.parse(specification)


def test_topic_options_accept_a_regex_pattern_as_the_selector() -> None:
    defaults = RoscompressConfig(codec="h264", quality=28, backend="ffmpeg-cli")

    resolved = _resolve_video_topic_options(_video_profiles(r"/CAM_.*/image:quality=20"), defaults)

    assert resolved == {r"/CAM_.*/image": replace(defaults, quality=20)}


def test_topic_options_accept_a_non_capturing_regex_group() -> None:
    defaults = RoscompressConfig(codec="h264", quality=28, backend="ffmpeg-cli")

    resolved = _resolve_video_topic_options(
        _video_profiles(r"/(?:CAM_FRONT|CAM_BACK)/image:quality=20"), defaults
    )

    assert resolved == {r"/(?:CAM_FRONT|CAM_BACK)/image": replace(defaults, quality=20)}


def test_topic_ffmpeg_args_keep_colons_after_the_pattern_delimiter() -> None:
    defaults = RoscompressConfig(backend="ffmpeg-cli")

    resolved = _resolve_video_topic_options(
        None,
        defaults,
        [r"/(?:CAM_FRONT|CAM_BACK)/image:-vf scale=1280:-2"],
    )

    assert resolved[r"/(?:CAM_FRONT|CAM_BACK)/image"].ffmpeg_args == (
        "-vf",
        "scale=1280:-2",
    )


def test_topic_options_reject_an_invalid_regex_pattern() -> None:
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        PointcloudTopicProfile.parse("/LIDAR_(TOP/points:resolution=0.02")


def test_profile_entries_give_earlier_patterns_precedence() -> None:
    defaults = RoscompressConfig(quality=28)
    profiles = {
        "/CAM_FRONT/image": replace(defaults, quality=20),
        r"/CAM_.*/image": replace(defaults, quality=30),
    }

    entries = _profile_entries(profiles, defaults)

    assert [entry.pattern for entry in entries] == [
        "/CAM_FRONT/image",
        r"/CAM_.*/image",
        None,
    ]
    # The specific profile keeps /CAM_FRONT/image; the broader one takes the rest.
    assert entries[0].topics.selects("/CAM_FRONT/image")
    assert not entries[1].topics.selects("/CAM_FRONT/image")
    assert entries[1].topics.selects("/CAM_BACK/image")
    # The catch-all handles every topic no profile claimed.
    assert not entries[2].topics.selects("/CAM_BACK/image")
    assert entries[2].topics.selects("/LIDAR_TOP/points")
