import pytest
from cyclopts import App
from pymcap_cli.cmd._cli_options import PointCloudTopicOptionsOption
from pymcap_cli.cmd._roscompress_topic_options import (
    PointcloudTopicProfile,
    RawTopicProfile,
    VideoTopicProfile,
    parse_topic_profile,
)


def test_parse_topic_profile_is_cli_neutral() -> None:
    assert parse_topic_profile(
        "/lidar/points:resolution=0.02,pc-compression=lz4",
        option_name="point-cloud topic profile",
    ) == RawTopicProfile(
        pattern="/lidar/points",
        values={"resolution": "0.02", "pc_compression": "lz4"},
    )


def test_pointcloud_topic_profile_parse_returns_typed_values() -> None:
    assert PointcloudTopicProfile.parse(
        "/lidar/points:resolution=0.02,pc-compression=lz4"
    ) == PointcloudTopicProfile(
        pattern="/lidar/points",
        resolution=0.02,
        pc_compression="lz4",
    )


def test_video_topic_profile_parse_preserves_explicit_resets() -> None:
    assert VideoTopicProfile.parse(
        "/camera/image:encoder=auto,scale=original"
    ) == VideoTopicProfile(
        pattern="/camera/image",
        encoder="auto",
        scale="original",
    )


def test_topic_profile_constructors_enforce_mode_is_standalone() -> None:
    with pytest.raises(ValueError, match="mode must be specified alone"):
        PointcloudTopicProfile(
            pattern="/lidar/points",
            mode="keep",
            resolution=0.02,
        )
    with pytest.raises(ValueError, match="mode must be specified alone"):
        VideoTopicProfile(
            pattern="/camera/image",
            mode="default",
            quality=20,
        )


def test_cyclopts_converts_each_repeatable_value_to_a_profile() -> None:
    captured: list[PointcloudTopicProfile] = []
    app = App()

    @app.default
    def command(profiles: PointCloudTopicOptionsOption = None) -> None:
        captured.extend(profiles or [])

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "--pointcloud-topic-options",
                "/lidar/front:resolution=0.02",
                "--pointcloud-topic-options",
                "/lidar/rear:mode=keep",
            ],
            exit_on_error=False,
        )

    assert exc_info.value.code == 0
    assert captured == [
        PointcloudTopicProfile(pattern="/lidar/front", resolution=0.02),
        PointcloudTopicProfile(pattern="/lidar/rear", mode="keep"),
    ]
