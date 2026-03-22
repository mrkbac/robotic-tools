"""ROS2 message definition resolver with dynamic rosdistro-based package lookup."""

import json
import logging
import os
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import platformdirs
import yaml
from ros_parser.ros2_msg import parse_message_file

logger = logging.getLogger(__name__)

_ALLOWED_URL_PREFIXES = ("https://github.com/", "https://raw.githubusercontent.com/")
_DISTRO_INDEX_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


class ROS2Distro(str, Enum):
    """Valid ROS2 distributions."""

    HUMBLE = "humble"
    IRON = "iron"  # Note: EOL
    JAZZY = "jazzy"
    KILTED = "kilted"
    ROLLING = "rolling"


@dataclass(frozen=True)
class _RepoSource:
    url: str  # e.g. "https://github.com/ros2/common_interfaces.git"
    version: str  # e.g. "humble"


@dataclass(frozen=True)
class _DistroIndex:
    pkg_to_repo: dict[str, str]  # "sensor_msgs" -> "common_interfaces"
    repo_to_source: dict[str, _RepoSource]  # "common_interfaces" -> _RepoSource(...)


def _download(url: str, buffer: BytesIO) -> None:
    """Download content from URL to buffer."""
    if not url.startswith(_ALLOWED_URL_PREFIXES):
        msg = f"URL must start with one of {_ALLOWED_URL_PREFIXES}"
        raise ValueError(msg)

    try:
        with urlopen(url) as response:  # noqa: S310
            if response.status != 200:
                msg = f"Failed to download {url}: {response.status} {response.reason}"
                raise ValueError(msg)
            buffer.write(response.read())
    except URLError as e:
        msg = f"Network error downloading {url}: {e}"
        raise ValueError(msg) from e


def _download_and_extract(url: str, target_dir: Path) -> None:
    """Download and extract zip file to target directory, flattening the structure."""
    # Assume already downloaded if not empty
    if target_dir.exists() and any(target_dir.iterdir()):
        return

    logger.info(f"Downloading {url} to {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract to temp directory first
    buffer = BytesIO()
    _download(url, buffer)

    # GitHub ZIPs extract to {repo}-{branch}/ subdirectory
    # We need to flatten this structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract to temp location
        with ZipFile(buffer) as zip_ref:
            zip_ref.extractall(temp_path)

        # Find the extracted subdirectory (should be only one)
        extracted_dirs = [d for d in temp_path.iterdir() if d.is_dir()]

        if len(extracted_dirs) == 1:
            # Move contents from extracted_dir/{repo}-{branch}/* to target_dir/*
            source_dir = extracted_dirs[0]
            for item in source_dir.iterdir():
                shutil.move(str(item), str(target_dir / item.name))
        else:
            # Fallback: just copy everything
            for item in temp_path.iterdir():
                if item.is_dir():
                    shutil.copytree(item, target_dir / item.name)
                else:
                    shutil.copy2(item, target_dir / item.name)


def _parse_distro_yaml(data: dict[str, object]) -> _DistroIndex:
    """Parse distribution.yaml data into a _DistroIndex."""
    pkg_to_repo: dict[str, str] = {}
    repo_to_source: dict[str, _RepoSource] = {}

    repositories = data.get("repositories", {})
    if not isinstance(repositories, dict):
        return _DistroIndex(pkg_to_repo={}, repo_to_source={})

    for repo_key, repo_data in repositories.items():
        if not isinstance(repo_key, str) or not isinstance(repo_data, dict):
            continue
        repo_data_dict: dict[str, object] = repo_data  # type: ignore[assignment]

        # Extract source info
        source_raw = repo_data_dict.get("source")
        if isinstance(source_raw, dict):
            source_dict: dict[str, object] = source_raw  # type: ignore[assignment]
            if source_dict.get("type") == "git":
                url = source_dict.get("url", "")
                version = source_dict.get("version", "")
                if isinstance(url, str) and isinstance(version, str) and url and version:
                    repo_to_source[repo_key] = _RepoSource(url=url, version=version)

        # Map packages to this repo
        release_raw = repo_data_dict.get("release")
        if isinstance(release_raw, dict):
            release_dict: dict[str, object] = release_raw  # type: ignore[assignment]
            packages = release_dict.get("packages")
            if isinstance(packages, list):
                for pkg in packages:
                    if isinstance(pkg, str):
                        pkg_to_repo[pkg] = repo_key
            else:
                # Single-package repo: repo name is the package name
                pkg_to_repo[repo_key] = repo_key
        elif repo_key in repo_to_source:
            # No release section but has source — assume repo name is package name
            pkg_to_repo[repo_key] = repo_key

    return _DistroIndex(pkg_to_repo=pkg_to_repo, repo_to_source=repo_to_source)


def _fetch_distro_index(distro: ROS2Distro, cache_dir: Path) -> _DistroIndex | None:
    """Fetch and cache the rosdistro distribution.yaml for a distro."""
    yaml_path = cache_dir / "_distribution.yaml"
    index_path = cache_dir / "_distro_index.json"
    timestamp_path = cache_dir / "_distro_index_timestamp"

    # Check if cached index is fresh enough
    if index_path.exists() and timestamp_path.exists():
        try:
            cached_ts = float(timestamp_path.read_text().strip())
            if (time.time() - cached_ts) < _DISTRO_INDEX_TTL_SECONDS:
                index_data = json.loads(index_path.read_text())
                return _DistroIndex(
                    pkg_to_repo=index_data["pkg_to_repo"],
                    repo_to_source={
                        k: _RepoSource(url=v["url"], version=v["version"])
                        for k, v in index_data["repo_to_source"].items()
                    },
                )
        except (ValueError, KeyError, json.JSONDecodeError):
            logger.debug("Cached distro index is corrupted, re-fetching")

    # Fetch distribution.yaml
    url = f"https://raw.githubusercontent.com/ros/rosdistro/master/{distro.value}/distribution.yaml"
    try:
        buffer = BytesIO()
        _download(url, buffer)
        raw_yaml = buffer.getvalue()
    except ValueError:
        logger.warning(f"Failed to fetch distribution.yaml for {distro.value}")
        # Try to use stale cache if available
        if index_path.exists():
            try:
                index_data = json.loads(index_path.read_text())
                return _DistroIndex(
                    pkg_to_repo=index_data["pkg_to_repo"],
                    repo_to_source={
                        k: _RepoSource(url=v["url"], version=v["version"])
                        for k, v in index_data["repo_to_source"].items()
                    },
                )
            except (KeyError, json.JSONDecodeError):
                pass
        return None

    # Parse YAML
    try:
        data = yaml.safe_load(raw_yaml)
    except yaml.YAMLError:
        logger.warning(f"Failed to parse distribution.yaml for {distro.value}")
        return None

    if not isinstance(data, dict):
        return None

    distro_index = _parse_distro_yaml(data)

    # Cache to disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.write_bytes(raw_yaml)
    index_data = {
        "pkg_to_repo": distro_index.pkg_to_repo,
        "repo_to_source": {
            k: {"url": v.url, "version": v.version} for k, v in distro_index.repo_to_source.items()
        },
    }
    index_path.write_text(json.dumps(index_data))
    timestamp_path.write_text(str(time.time()))

    return distro_index


@lru_cache
def _get_distro_index(distro: ROS2Distro, cache_dir: Path) -> _DistroIndex | None:
    """Get distro index with in-memory caching."""
    return _fetch_distro_index(distro, cache_dir)


def _git_url_to_zip(git_url: str, version: str) -> str:
    """Convert a git URL to a GitHub ZIP download URL."""
    # "https://github.com/ros2/common_interfaces.git" -> "https://github.com/ros2/common_interfaces"
    base = git_url.removesuffix(".git")
    return f"{base}/archive/refs/heads/{version}.zip"


def _ensure_repo_cached(cache_dir: Path, repo_name: str, zip_url: str) -> Path:
    """Download a single repo on demand if not already cached."""
    repo_dir = cache_dir / repo_name
    _download_and_extract(zip_url, repo_dir)
    return repo_dir


def _resolve_and_download_repo(pkg_name: str, distro: ROS2Distro, cache_dir: Path) -> Path | None:
    """Resolve a package to its source repo and download it."""
    index = _get_distro_index(distro, cache_dir)
    if index is None:
        return None

    repo_name = index.pkg_to_repo.get(pkg_name)
    if repo_name is None:
        return None

    source = index.repo_to_source.get(repo_name)
    if source is None:
        return None

    zip_url = _git_url_to_zip(source.url, source.version)
    try:
        return _ensure_repo_cached(cache_dir, repo_name, zip_url)
    except ValueError:
        logger.warning(f"Failed to download repo {repo_name} for package {pkg_name}")
        return None


def _get_cache_dir(distro: ROS2Distro) -> Path:
    """Get cache directory for a specific ROS2 distribution."""
    return platformdirs.user_cache_path(
        appname="pymcap_cli_msg_def",
        ensure_exists=True,
        version=distro.value,
    )


@lru_cache
def _get_ament_prefix_paths() -> tuple[Path, ...]:
    """Get AMENT_PREFIX_PATH directories from environment."""
    ament_prefix = os.environ.get("AMENT_PREFIX_PATH", "")
    if not ament_prefix:
        return ()

    separator = ";" if os.name == "nt" else ":"
    paths = [Path(p) for p in ament_prefix.split(separator) if p]
    return tuple(p for p in paths if p.exists())


@lru_cache
def _rglob_first(folder: tuple[Path, ...], pattern: str) -> Path | None:
    """Search for first file matching pattern in folder hierarchy."""
    for f in folder:
        if not f.exists():
            continue
        matches = list(f.rglob(pattern))
        if matches:
            return matches[0]
    return None


@lru_cache
def _get_msg_def_disk(msg_type: str, folders: tuple[Path, ...]) -> tuple[str, list[str]] | None:
    """Get message definition from disk by searching in folders."""
    parts = msg_type.split("/")

    # Handle both formats: "package/MessageName" and "package/msg/MessageName"
    if len(parts) == 3 and parts[1] == "msg":
        # Format: package/msg/MessageName -> extract package and MessageName
        pkg_name, _, msg_name = parts
    elif len(parts) == 2:
        # Format: package/MessageName
        pkg_name, msg_name = parts
    else:
        logger.warning(f"Invalid message type format: {msg_type}")
        return None

    # Search for .msg file
    msg_path = _rglob_first(folders, f"**/{pkg_name}/msg/{msg_name}.msg")
    if msg_path is None:
        return None

    logger.debug(f"Found {msg_type} at {msg_path}")

    # Parse message file
    try:
        msg_def = parse_message_file(msg_path)
    except Exception:
        logger.exception(f"Failed to parse {msg_path}")
        return None

    # Read raw text
    with msg_path.open(encoding="utf-8") as msg_file:
        msg_text = msg_file.read()

    # Extract dependencies
    dependencies = []
    for field in msg_def.fields:
        f_type = field.type
        if f_type.is_primitive:
            continue

        # builtin_interfaces are expected to be known by the parser
        if f_type.package_name == "builtin_interfaces":
            continue

        # Skip if no package name (local types)
        if not f_type.package_name:
            continue

        dependencies.append(f"{f_type.package_name}/{f_type.type_name}")

    return (msg_text, dependencies)


def _extract_pkg_name(msg_type: str) -> str | None:
    """Extract package name from a message type string."""
    parts = msg_type.split("/")
    if len(parts) in (2, 3):
        return parts[0]
    return None


def _get_msg_def(
    msg_type: str,
    distro: ROS2Distro,
    extra_paths: list[Path],
) -> tuple[str, list[str]] | None:
    """Get message definition with caching and multiple search paths.

    Search priority:
    1. User-provided extra paths + AMENT_PREFIX_PATH
    2. Already-cached repos in cache_dir
    3. Dynamic resolution via rosdistro index (downloads only the needed repo)
    """
    cache_dir = _get_cache_dir(distro)
    ament_paths = _get_ament_prefix_paths()

    # 1. Try local paths first (extra_paths + AMENT_PREFIX_PATH)
    result = _get_msg_def_disk(msg_type, (*extra_paths, *ament_paths))
    if result is not None:
        return result

    # 2. Try already-cached repos
    if cache_dir.exists():
        cached_repos = tuple(
            d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
        )
        if cached_repos:
            result = _get_msg_def_disk(msg_type, cached_repos)
            if result is not None:
                return result

    # 3. Dynamic resolution: resolve package -> repo, download just that repo
    pkg_name = _extract_pkg_name(msg_type)
    if pkg_name is not None:
        repo_dir = _resolve_and_download_repo(pkg_name, distro, cache_dir)
        if repo_dir is not None:
            return _get_msg_def_disk(msg_type, (repo_dir,))

    return None


@lru_cache
def get_message_definition(
    msg_type: str,
    distro: ROS2Distro = ROS2Distro.HUMBLE,
    extra_paths: tuple[Path, ...] | None = None,
) -> str | None:
    """Get complete message definition with all dependencies.

    Args:
        msg_type: Message type in format "package_name/MessageName"
        distro: ROS2 distribution to use for standard messages
        extra_paths: Additional paths to search for custom message definitions (as tuple)

    Returns:
        Complete message definition with dependencies, or None if not found

    """
    if extra_paths is None:
        extra_paths = ()

    root = _get_msg_def(msg_type, distro, list(extra_paths))
    if root is None:
        logger.warning(f"Could not find message definition for {msg_type}")
        return None

    msg_text, deps = root
    queue: deque[str] = deque(deps)
    added_types = set()

    while queue:
        dep = queue.popleft()
        if dep in added_types:
            continue

        msg_def = _get_msg_def(dep, distro, list(extra_paths))
        if msg_def is None:
            logger.error(f"Could not find dependency {dep} for {msg_type}")
            return None

        sub_text, sub_dep = msg_def
        # Ensure newline before separator
        if not msg_text.endswith("\n"):
            msg_text += "\n"
        msg_text += f"{'=' * 40}\nMSG: {dep}\n{sub_text}"
        added_types.add(dep)
        # Only enqueue unseen sub-deps
        queue.extend(d for d in sub_dep if d not in added_types)

    return msg_text
