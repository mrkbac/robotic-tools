"""Tests for the rosdistro-based message resolver."""

import json
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError
from zipfile import ZipFile

import pytest
import yaml
from pymcap_cli.core.msg_resolver import (
    ROS2Distro,
    _DistroIndex,
    _download,
    _fetch_distro_index,
    _get_msg_def,
    _parse_distro_yaml,
    _RepoRelease,
    list_package_messages,
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


def _make_release_zip(top_dir: str, files: dict[str, str]) -> bytes:
    """Build an in-memory zip mimicking a GitHub release-branch archive."""
    buffer = BytesIO()
    with ZipFile(buffer, "w") as zf:
        for relative_path, content in files.items():
            zf.writestr(f"{top_dir}/{relative_path}", content)
    return buffer.getvalue()


class TestListPackageMessages:
    def test_lists_messages_from_extra_paths(self, tmp_path: Path) -> None:
        pkg_dir = tmp_path / "src" / "my_pkg" / "msg"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "Foo.msg").write_text("uint8 a\n")
        (pkg_dir / "Bar.msg").write_text("uint8 b\n")
        (pkg_dir / "ignore.txt").write_text("not a msg")

        assert list_package_messages(
            "my_pkg",
            distro=ROS2Distro.HUMBLE,
            extra_paths=(tmp_path,),
        ) == ["Bar", "Foo"]

    def test_unions_messages_across_overlay_roots(self, tmp_path: Path) -> None:
        base = tmp_path / "base" / "my_pkg" / "msg"
        overlay = tmp_path / "overlay" / "my_pkg" / "msg"
        base.mkdir(parents=True)
        overlay.mkdir(parents=True)
        (base / "Foo.msg").write_text("uint8 a\n")
        (overlay / "Bar.msg").write_text("uint8 b\n")

        result = list_package_messages(
            "my_pkg",
            extra_paths=(tmp_path / "base", tmp_path / "overlay"),
        )
        assert result == ["Bar", "Foo"]

    def test_release_zip_warms_per_message_cache(self, tmp_path: Path) -> None:
        """Listing via the release zip must populate _remote_msgs so a
        subsequent _get_msg_def hits disk with no further urlopen calls."""
        index = _DistroIndex(
            pkg_to_repo={"sensor_msgs": "common_interfaces"},
            repo_to_source={},
            repo_to_release={
                "common_interfaces": _RepoRelease(
                    url="https://github.com/ros2-gbp/common_interfaces-release.git",
                    version="4.9.1-1",
                    tag_template="release/humble/{package}/{version}",
                )
            },
        )
        zip_bytes = _make_release_zip(
            "common_interfaces-release-release-humble-sensor_msgs-4.9.1-1",
            {
                "msg/Image.msg": "uint32 height\nuint32 width\n",
                "msg/CompressedImage.msg": "string format\nuint8[] data\n",
                "package.xml": "<package/>\n",
            },
        )

        with (
            patch("pymcap_cli.core.msg_resolver._get_cache_dir", return_value=tmp_path),
            patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index),
            patch(
                "pymcap_cli.core.msg_resolver.urlopen",
                return_value=_mock_response(data=zip_bytes),
            ) as mock_urlopen,
        ):
            names = list_package_messages("sensor_msgs", distro=ROS2Distro.HUMBLE)

        assert names == ["CompressedImage", "Image"]
        assert mock_urlopen.call_count == 1
        zip_url = mock_urlopen.call_args.args[0]
        assert zip_url == (
            "https://github.com/ros2-gbp/common_interfaces-release/archive/refs/tags/"
            "release/humble/sensor_msgs/4.9.1-1.zip"
        )

        # Warming check: _get_msg_def now reads from disk, no extra urlopen.
        with (
            patch("pymcap_cli.core.msg_resolver._get_cache_dir", return_value=tmp_path),
            patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index),
            patch("pymcap_cli.core.msg_resolver.urlopen") as mock_after,
        ):
            result = _get_msg_def("sensor_msgs/Image", ROS2Distro.HUMBLE, [])

        assert result is not None
        msg_text, _deps = result
        assert msg_text == "uint32 height\nuint32 width\n"
        mock_after.assert_not_called()

    def test_returns_none_for_unknown_package(self, tmp_path: Path) -> None:
        index = _DistroIndex(pkg_to_repo={}, repo_to_source={}, repo_to_release={})
        with (
            patch("pymcap_cli.core.msg_resolver._get_cache_dir", return_value=tmp_path),
            patch("pymcap_cli.core.msg_resolver._get_distro_index", return_value=index),
            patch("pymcap_cli.core.msg_resolver.urlopen") as mock_urlopen,
        ):
            assert list_package_messages("not_a_package_msgs") is None

        mock_urlopen.assert_not_called()
