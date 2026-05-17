"""Tests for the rosdistro-based message resolver."""

import json
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest
import yaml
from pymcap_cli.core.msg_resolver import (
    _ARCHIVE_FALLBACK_ENV,
    ROS2Distro,
    _DistroIndex,
    _download,
    _download_text,
    _fetch_distro_index,
    _get_msg_def,
    _git_url_to_zip,
    _parse_distro_yaml,
    _RepoRelease,
    _RepoSource,
    _resolve_and_download_repo,
    _resolve_remote_msg_def,
    _ResponseTooLargeError,
)

SAMPLE_DISTRO_DATA = {
    "type": "distribution",
    "version": 2,
    "repositories": {
        "common_interfaces": {
            "source": {
                "type": "git",
                "url": "https://github.com/ros2/common_interfaces.git",
                "version": "humble",
            },
            "release": {
                "packages": [
                    "std_msgs",
                    "sensor_msgs",
                    "geometry_msgs",
                    "nav_msgs",
                ],
                "tags": {"release": "release/humble/{package}/{version}"},
                "url": "https://github.com/ros2-gbp/common_interfaces-release.git",
                "version": "4.9.1-1",
            },
        },
        "unique_identifier_msgs": {
            "source": {
                "type": "git",
                "url": "https://github.com/ros2/unique_identifier_msgs.git",
                "version": "humble",
            },
            "release": {
                "tags": {"release": "release/humble/{package}/{version}"},
                "url": "https://github.com/ros2-gbp/unique_identifier_msgs-release.git",
                "version": "2.3.2-1",
            },
        },
        "control_msgs": {
            "source": {
                "type": "git",
                "url": "https://github.com/ros-controls/control_msgs.git",
                "version": "humble",
            },
            "release": {
                "packages": ["control_msgs"],
                "tags": {"release": "release/humble/{package}/{version}"},
                "url": "https://github.com/ros2-gbp/control_msgs-release.git",
                "version": "5.0.0-1",
            },
        },
        "some_custom_msgs": {
            "source": {
                "type": "git",
                "url": "https://github.com/example/some_custom_msgs.git",
                "version": "main",
            },
        },
        "hg_repo": {
            "source": {
                "type": "hg",
                "url": "https://bitbucket.org/example/hg_repo",
                "version": "default",
            },
        },
        "release_only_pkg": {
            "release": {
                "tags": {"release": "release/humble/{package}/{version}"},
                "url": "https://github.com/ros2-gbp/release_only-release.git",
                "version": "1.0.0-1",
            },
        },
    },
}

SAMPLE_RAW_YAML = yaml.dump(SAMPLE_DISTRO_DATA).encode()


def _mock_response(
    data: bytes = b"",
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Context-manager mock for ``urlopen`` returning ``data``."""
    body = BytesIO(data)
    response = MagicMock(
        status=status,
        headers=headers if headers is not None else {"Content-Length": str(len(data))},
    )
    response.read = body.read
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    return response


class TestParseDistroYaml:
    def test_multi_package_repo(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        assert index.pkg_to_repo["std_msgs"] == "common_interfaces"
        assert index.pkg_to_repo["sensor_msgs"] == "common_interfaces"
        assert index.pkg_to_repo["geometry_msgs"] == "common_interfaces"
        assert index.pkg_to_repo["nav_msgs"] == "common_interfaces"

    def test_single_package_repo_with_no_packages_list(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        assert index.pkg_to_repo["unique_identifier_msgs"] == "unique_identifier_msgs"

    def test_single_package_repo_with_packages_list(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        assert index.pkg_to_repo["control_msgs"] == "control_msgs"

    def test_repo_with_source_but_no_release(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        assert index.pkg_to_repo["some_custom_msgs"] == "some_custom_msgs"

    def test_source_url_and_version(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        source = index.repo_to_source["common_interfaces"]
        assert source.url == "https://github.com/ros2/common_interfaces.git"
        assert source.version == "humble"

    def test_release_url_version_and_tag_template(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        release = index.repo_to_release["common_interfaces"]
        assert release.url == "https://github.com/ros2-gbp/common_interfaces-release.git"
        assert release.version == "4.9.1-1"
        assert release.tag_template == "release/humble/{package}/{version}"

    def test_non_git_source_skipped(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        assert "hg_repo" not in index.repo_to_source
        assert "hg_repo" not in index.pkg_to_repo

    def test_release_only_repo(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        assert "release_only_pkg" not in index.repo_to_source
        assert "release_only_pkg" in index.repo_to_release
        assert index.pkg_to_repo["release_only_pkg"] == "release_only_pkg"

    def test_empty_repositories(self) -> None:
        index = _parse_distro_yaml({"repositories": {}})
        assert index.pkg_to_repo == {}
        assert index.repo_to_source == {}

    def test_missing_repositories_key(self) -> None:
        index = _parse_distro_yaml({})
        assert index.pkg_to_repo == {}
        assert index.repo_to_source == {}

    def test_invalid_repositories_type(self) -> None:
        index = _parse_distro_yaml({"repositories": "not a dict"})
        assert index.pkg_to_repo == {}
        assert index.repo_to_source == {}


class TestGitUrlToZip:
    def test_with_git_suffix(self) -> None:
        result = _git_url_to_zip("https://github.com/ros2/common_interfaces.git", "humble")
        assert result == "https://github.com/ros2/common_interfaces/archive/refs/heads/humble.zip"

    def test_without_git_suffix(self) -> None:
        result = _git_url_to_zip("https://github.com/ros2/common_interfaces", "jazzy")
        assert result == "https://github.com/ros2/common_interfaces/archive/refs/heads/jazzy.zip"


class TestFetchDistroIndex:
    def test_caches_to_disk(self, tmp_path: Path) -> None:
        with patch("pymcap_cli.core.msg_resolver._download") as mock_dl:
            mock_dl.side_effect = lambda _url, buf: buf.write(SAMPLE_RAW_YAML)

            index = _fetch_distro_index(ROS2Distro.HUMBLE, tmp_path)

        assert index is not None
        assert index.pkg_to_repo["std_msgs"] == "common_interfaces"
        assert (tmp_path / "_distribution.yaml").exists()
        assert (tmp_path / "_distro_index.json").exists()
        assert (tmp_path / "_distro_index_timestamp").exists()

    def test_uses_cached_index_within_ttl(self, tmp_path: Path) -> None:
        index_data = {
            "pkg_to_repo": {"cached_pkg": "cached_repo"},
            "repo_to_source": {
                "cached_repo": {"url": "https://github.com/ex/r.git", "version": "v1"},
            },
            "repo_to_release": {
                "cached_repo": {
                    "url": "https://github.com/ex/r-release.git",
                    "version": "1.0.0-1",
                    "tag_template": "release/humble/{package}/{version}",
                },
            },
        }
        (tmp_path / "_distro_index.json").write_text(json.dumps(index_data))
        (tmp_path / "_distro_index_timestamp").write_text(str(time.time()))

        with patch("pymcap_cli.core.msg_resolver._download") as mock_dl:
            index = _fetch_distro_index(ROS2Distro.HUMBLE, tmp_path)

        mock_dl.assert_not_called()
        assert index is not None
        assert index.pkg_to_repo["cached_pkg"] == "cached_repo"
        assert index.repo_to_release["cached_repo"].version == "1.0.0-1"

    def test_refetches_when_ttl_expired(self, tmp_path: Path) -> None:
        (tmp_path / "_distro_index.json").write_text(
            json.dumps({"pkg_to_repo": {}, "repo_to_source": {}})
        )
        (tmp_path / "_distro_index_timestamp").write_text(str(time.time() - 999999))

        with patch("pymcap_cli.core.msg_resolver._download") as mock_dl:
            mock_dl.side_effect = lambda _url, buf: buf.write(SAMPLE_RAW_YAML)
            index = _fetch_distro_index(ROS2Distro.HUMBLE, tmp_path)

        mock_dl.assert_called_once()
        assert index is not None
        assert "std_msgs" in index.pkg_to_repo

    def test_returns_none_on_network_failure(self, tmp_path: Path) -> None:
        with patch(
            "pymcap_cli.core.msg_resolver._download", side_effect=ValueError("network error")
        ):
            index = _fetch_distro_index(ROS2Distro.HUMBLE, tmp_path)

        assert index is None

    def test_uses_stale_cache_on_network_failure(self, tmp_path: Path) -> None:
        cached_data = {
            "pkg_to_repo": {"stale_pkg": "stale_repo"},
            "repo_to_source": {
                "stale_repo": {"url": "https://github.com/ex/r.git", "version": "v1"}
            },
        }
        (tmp_path / "_distro_index.json").write_text(json.dumps(cached_data))
        (tmp_path / "_distro_index_timestamp").write_text(str(time.time() - 999999))

        with patch(
            "pymcap_cli.core.msg_resolver._download", side_effect=ValueError("network error")
        ):
            index = _fetch_distro_index(ROS2Distro.HUMBLE, tmp_path)

        assert index is not None
        assert index.pkg_to_repo["stale_pkg"] == "stale_repo"


class TestResolveAndDownloadRepo:
    def test_downloads_correct_repo(self, tmp_path: Path) -> None:
        index = _DistroIndex(
            pkg_to_repo={"sensor_msgs": "common_interfaces"},
            repo_to_source={
                "common_interfaces": _RepoSource(
                    url="https://github.com/ros2/common_interfaces.git",
                    version="humble",
                )
            },
        )

        with (
            patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index),
            patch("pymcap_cli.core.msg_resolver._download_and_extract") as mock_extract,
        ):
            result = _resolve_and_download_repo("sensor_msgs", ROS2Distro.HUMBLE, tmp_path)

        assert result == tmp_path / "common_interfaces"
        mock_extract.assert_called_once_with(
            "https://github.com/ros2/common_interfaces/archive/refs/heads/humble.zip",
            tmp_path / "common_interfaces",
        )

    def test_returns_none_for_unknown_package(self, tmp_path: Path) -> None:
        index = _DistroIndex(pkg_to_repo={}, repo_to_source={})

        with patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index):
            result = _resolve_and_download_repo("nonexistent_msgs", ROS2Distro.HUMBLE, tmp_path)

        assert result is None

    def test_returns_none_when_index_unavailable(self, tmp_path: Path) -> None:
        with patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=None):
            result = _resolve_and_download_repo("sensor_msgs", ROS2Distro.HUMBLE, tmp_path)

        assert result is None

    def test_returns_none_on_download_failure(self, tmp_path: Path) -> None:
        index = _DistroIndex(
            pkg_to_repo={"sensor_msgs": "common_interfaces"},
            repo_to_source={
                "common_interfaces": _RepoSource(
                    url="https://github.com/ros2/common_interfaces.git",
                    version="humble",
                )
            },
        )

        with (
            patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index),
            patch(
                "pymcap_cli.core.msg_resolver._download_and_extract",
                side_effect=ValueError("download error"),
            ),
        ):
            result = _resolve_and_download_repo("sensor_msgs", ROS2Distro.HUMBLE, tmp_path)

        assert result is None


class TestRemoteMsgResolution:
    def test_release_raw_file_is_used_before_source(self, tmp_path: Path) -> None:
        index = _DistroIndex(
            pkg_to_repo={"sensor_msgs": "common_interfaces"},
            repo_to_source={
                "common_interfaces": _RepoSource(
                    url="https://github.com/ros2/common_interfaces.git",
                    version="humble",
                )
            },
            repo_to_release={
                "common_interfaces": _RepoRelease(
                    url="https://github.com/ros2-gbp/common_interfaces-release.git",
                    version="4.9.1-1",
                    tag_template="release/humble/{package}/{version}",
                )
            },
        )

        with (
            patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index),
            patch(
                "pymcap_cli.core.msg_resolver._download_text",
                return_value="std_msgs/Header header\nuint8[] data\n",
            ) as mock_download,
        ):
            result = _resolve_remote_msg_def("sensor_msgs/msg/Image", ROS2Distro.HUMBLE, tmp_path)

        assert result == ("std_msgs/Header header\nuint8[] data\n", ["std_msgs/Header"])
        mock_download.assert_called_once_with(
            "https://raw.githubusercontent.com/ros2-gbp/common_interfaces-release/"
            "release/humble/sensor_msgs/4.9.1-1/msg/Image.msg"
        )
        assert (
            tmp_path
            / "_remote_msgs"
            / "common_interfaces"
            / "release_humble_sensor_msgs_4.9.1-1"
            / "sensor_msgs"
            / "msg"
            / "Image.msg"
        ).exists()

    def test_source_fast_paths_are_used_when_release_misses(self, tmp_path: Path) -> None:
        index = _DistroIndex(
            pkg_to_repo={"custom_msgs": "custom_msgs"},
            repo_to_source={
                "custom_msgs": _RepoSource(
                    url="https://github.com/acme/custom_msgs.git",
                    version="main",
                )
            },
            repo_to_release={
                "custom_msgs": _RepoRelease(
                    url="https://github.com/acme/custom_msgs-release.git",
                    version="1.0.0-1",
                    tag_template="release/humble/{package}/{version}",
                )
            },
        )

        with (
            patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index),
            patch(
                "pymcap_cli.core.msg_resolver._download_text",
                side_effect=[None, None, "float64 value\n"],
            ) as mock_download,
        ):
            result = _resolve_remote_msg_def("custom_msgs/Reading", ROS2Distro.HUMBLE, tmp_path)

        assert result == ("float64 value\n", [])
        assert [call.args[0] for call in mock_download.call_args_list] == [
            "https://raw.githubusercontent.com/acme/custom_msgs-release/"
            "release/humble/custom_msgs/1.0.0-1/msg/Reading.msg",
            "https://raw.githubusercontent.com/acme/custom_msgs/main/custom_msgs/msg/Reading.msg",
            "https://raw.githubusercontent.com/acme/custom_msgs/main/msg/Reading.msg",
        ]

    def test_oversized_response_propagates_instead_of_falling_through(self, tmp_path: Path) -> None:
        """An oversized .msg response is a real bug (e.g. HTML error page),
        not a tier-fallthrough miss — make sure we surface it."""
        index = _DistroIndex(
            pkg_to_repo={"sensor_msgs": "common_interfaces"},
            repo_to_source={
                "common_interfaces": _RepoSource(
                    url="https://github.com/ros2/common_interfaces.git",
                    version="humble",
                )
            },
            repo_to_release={
                "common_interfaces": _RepoRelease(
                    url="https://github.com/ros2-gbp/common_interfaces-release.git",
                    version="4.9.1-1",
                    tag_template="release/humble/{package}/{version}",
                )
            },
        )

        with (
            patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index),
            patch(
                "pymcap_cli.core.msg_resolver._download_text",
                side_effect=_ResponseTooLargeError("response is larger than 1048576 bytes"),
            ),
            pytest.raises(_ResponseTooLargeError, match="larger than"),
        ):
            _resolve_remote_msg_def("sensor_msgs/msg/Image", ROS2Distro.HUMBLE, tmp_path)

    def test_get_msg_def_does_not_use_archive_fallback_by_default(self, tmp_path: Path) -> None:
        index = _DistroIndex(
            pkg_to_repo={"sensor_msgs": "common_interfaces"},
            repo_to_source={
                "common_interfaces": _RepoSource(
                    url="https://github.com/ros2/common_interfaces.git",
                    version="humble",
                )
            },
        )

        with (
            patch("pymcap_cli.core.msg_resolver._get_cache_dir", return_value=tmp_path),
            patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index),
            patch("pymcap_cli.core.msg_resolver._download_text", return_value=None),
            patch("pymcap_cli.core.msg_resolver._resolve_and_download_repo") as mock_archive,
            patch.dict("os.environ", {_ARCHIVE_FALLBACK_ENV: ""}, clear=False),
        ):
            result = _get_msg_def("sensor_msgs/Image", ROS2Distro.HUMBLE, [])

        assert result is None
        mock_archive.assert_not_called()

    def test_get_msg_def_uses_archive_fallback_when_enabled(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "common_interfaces"

        def create_archive_cache(
            _pkg_name: str,
            _distro: ROS2Distro,
            _cache_dir: Path,
        ) -> Path:
            msg_dir = repo_dir / "sensor_msgs" / "msg"
            msg_dir.mkdir(parents=True)
            (msg_dir / "Image.msg").write_text("uint8[] data\n", encoding="utf-8")
            return repo_dir

        with (
            patch("pymcap_cli.core.msg_resolver._get_cache_dir", return_value=tmp_path),
            patch("pymcap_cli.core.msg_resolver._resolve_remote_msg_def", return_value=None),
            patch(
                "pymcap_cli.core.msg_resolver._resolve_and_download_repo",
                side_effect=create_archive_cache,
            ) as mock_archive,
            patch.dict("os.environ", {_ARCHIVE_FALLBACK_ENV: "1"}, clear=False),
        ):
            result = _get_msg_def("sensor_msgs/Image", ROS2Distro.HUMBLE, [])

        assert result == ("uint8[] data\n", [])
        mock_archive.assert_called_once_with("sensor_msgs", ROS2Distro.HUMBLE, tmp_path)


class TestDownload:
    def test_rejects_non_github_urls(self) -> None:
        buf = BytesIO()
        with pytest.raises(ValueError, match="URL must start with one of"):
            _download("https://evil.com/malware.zip", buf)

    def test_accepts_raw_githubusercontent(self) -> None:
        buf = BytesIO()
        with (
            patch(
                "pymcap_cli.core.msg_resolver.urlopen",
                side_effect=URLError("boom"),
            ),
            pytest.raises(ValueError, match="Network error"),
        ):
            _download("https://raw.githubusercontent.com/ros/rosdistro/master/test.yaml", buf)


class TestDownloadText:
    def test_rejects_oversized_content_length(self) -> None:
        with (
            patch(
                "pymcap_cli.core.msg_resolver.urlopen",
                return_value=_mock_response(headers={"Content-Length": str(1024 * 1024 + 1)}),
            ),
            pytest.raises(_ResponseTooLargeError, match="larger than"),
        ):
            _download_text("https://raw.githubusercontent.com/owner/repo/ref/msg/Foo.msg")

    def test_rejects_oversized_stream_without_content_length(self) -> None:
        with (
            patch(
                "pymcap_cli.core.msg_resolver.urlopen",
                return_value=_mock_response(data=b"x" * (1024 * 1024 + 1), headers={}),
            ),
            pytest.raises(_ResponseTooLargeError, match="larger than"),
        ):
            _download_text("https://raw.githubusercontent.com/owner/repo/ref/msg/Foo.msg")
