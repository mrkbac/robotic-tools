"""Tests for the rosdistro-based message resolver."""

import json
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pymcap_cli.core.msg_resolver import (
    ROS2Distro,
    _DistroIndex,
    _download,
    _extract_pkg_name,
    _fetch_distro_index,
    _git_url_to_zip,
    _parse_distro_yaml,
    _RepoSource,
    _resolve_and_download_repo,
)

# Minimal distribution.yaml structure for testing
SAMPLE_DISTRO_DATA: dict[str, object] = {
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
                "url": "https://github.com/ros2-gbp/control_msgs-release.git",
                "version": "5.0.0-1",
            },
        },
        # Repo with no release section but has source
        "some_custom_msgs": {
            "source": {
                "type": "git",
                "url": "https://github.com/example/some_custom_msgs.git",
                "version": "main",
            },
        },
        # Repo with non-git source (e.g., hg) — should be skipped
        "hg_repo": {
            "source": {
                "type": "hg",
                "url": "https://bitbucket.org/example/hg_repo",
                "version": "default",
            },
        },
        # Repo with no source — only release
        "release_only_pkg": {
            "release": {
                "url": "https://github.com/ros2-gbp/release_only-release.git",
                "version": "1.0.0-1",
            },
        },
    },
}

SAMPLE_RAW_YAML = yaml.dump(SAMPLE_DISTRO_DATA).encode()


class TestParseDistroYaml:
    def test_multi_package_repo(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        assert index.pkg_to_repo["std_msgs"] == "common_interfaces"
        assert index.pkg_to_repo["sensor_msgs"] == "common_interfaces"
        assert index.pkg_to_repo["geometry_msgs"] == "common_interfaces"
        assert index.pkg_to_repo["nav_msgs"] == "common_interfaces"

    def test_single_package_repo_with_no_packages_list(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        # unique_identifier_msgs has release but no packages list
        assert index.pkg_to_repo["unique_identifier_msgs"] == "unique_identifier_msgs"

    def test_single_package_repo_with_packages_list(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        assert index.pkg_to_repo["control_msgs"] == "control_msgs"

    def test_repo_with_source_but_no_release(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        # Should map repo key as package name
        assert index.pkg_to_repo["some_custom_msgs"] == "some_custom_msgs"

    def test_source_url_and_version(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        source = index.repo_to_source["common_interfaces"]
        assert source.url == "https://github.com/ros2/common_interfaces.git"
        assert source.version == "humble"

    def test_non_git_source_skipped(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        assert "hg_repo" not in index.repo_to_source
        assert "hg_repo" not in index.pkg_to_repo

    def test_release_only_repo(self) -> None:
        index = _parse_distro_yaml(SAMPLE_DISTRO_DATA)

        # No source, so not in repo_to_source
        assert "release_only_pkg" not in index.repo_to_source
        # Has release but no packages list — maps repo key as package name
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


class TestExtractPkgName:
    def test_two_parts(self) -> None:
        assert _extract_pkg_name("sensor_msgs/Image") == "sensor_msgs"

    def test_three_parts(self) -> None:
        assert _extract_pkg_name("sensor_msgs/msg/Image") == "sensor_msgs"

    def test_single_part(self) -> None:
        assert _extract_pkg_name("sensor_msgs") is None

    def test_empty(self) -> None:
        assert _extract_pkg_name("") is None


class TestFetchDistroIndex:
    def test_caches_to_disk(self, tmp_path: Path) -> None:
        """Verify fetched index is cached as JSON on disk."""
        with patch("pymcap_cli.core.msg_resolver._download") as mock_dl:
            mock_dl.side_effect = lambda _url, buf: buf.write(SAMPLE_RAW_YAML)

            index = _fetch_distro_index(ROS2Distro.HUMBLE, tmp_path)

        assert index is not None
        assert index.pkg_to_repo["std_msgs"] == "common_interfaces"

        # Check cached files exist
        assert (tmp_path / "_distribution.yaml").exists()
        assert (tmp_path / "_distro_index.json").exists()
        assert (tmp_path / "_distro_index_timestamp").exists()

    def test_uses_cached_index_within_ttl(self, tmp_path: Path) -> None:
        """Verify cached index is returned without network fetch when fresh."""
        cached_index = _DistroIndex(
            pkg_to_repo={"cached_pkg": "cached_repo"},
            repo_to_source={
                "cached_repo": _RepoSource(url="https://github.com/ex/r.git", version="v1")
            },
        )
        index_data = {
            "pkg_to_repo": cached_index.pkg_to_repo,
            "repo_to_source": {
                k: {"url": v.url, "version": v.version}
                for k, v in cached_index.repo_to_source.items()
            },
        }
        (tmp_path / "_distro_index.json").write_text(json.dumps(index_data))
        (tmp_path / "_distro_index_timestamp").write_text(str(time.time()))

        with patch("pymcap_cli.core.msg_resolver._download") as mock_dl:
            index = _fetch_distro_index(ROS2Distro.HUMBLE, tmp_path)

        mock_dl.assert_not_called()
        assert index is not None
        assert index.pkg_to_repo["cached_pkg"] == "cached_repo"

    def test_refetches_when_ttl_expired(self, tmp_path: Path) -> None:
        """Verify stale cache triggers a re-fetch."""
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
        """Verify graceful degradation on network failure with no cache."""
        with patch(
            "pymcap_cli.core.msg_resolver._download", side_effect=ValueError("network error")
        ):
            index = _fetch_distro_index(ROS2Distro.HUMBLE, tmp_path)

        assert index is None

    def test_uses_stale_cache_on_network_failure(self, tmp_path: Path) -> None:
        """Verify stale cache is used as last resort when network fails."""
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
        """Verify only the needed repo is downloaded."""
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


class TestDownload:
    def test_rejects_non_github_urls(self) -> None:
        buf = BytesIO()
        with pytest.raises(ValueError, match="URL must start with one of"):
            _download("https://evil.com/malware.zip", buf)

    def test_accepts_raw_githubusercontent(self) -> None:
        buf = BytesIO()
        # Should not raise ValueError for the URL prefix check
        # (will raise ValueError wrapping URLError for the actual download)
        with pytest.raises(ValueError, match="Network error"):
            _download("https://raw.githubusercontent.com/ros/rosdistro/master/test.yaml", buf)
