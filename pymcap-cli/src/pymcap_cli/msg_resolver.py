"""ROS2 message definition resolver with cache and AMENT_PREFIX_PATH support."""

import logging
import os
import shutil
import tempfile
from collections import deque
from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import platformdirs
from ros_parser import parse_message_file

logger = logging.getLogger(__name__)


class ROS2Distro(str, Enum):
    """Valid ROS2 distributions."""

    HUMBLE = "humble"
    IRON = "iron"  # Note: EOL
    JAZZY = "jazzy"
    KILTED = "kilted"
    ROLLING = "rolling"


def _get_msg_def_repos(distro: ROS2Distro) -> list[tuple[str, str]]:
    """Get message definition repositories for a specific ROS2 distribution."""
    distro_str = distro.value

    return [
        (
            "rcl_interfaces",
            f"https://github.com/ros2/rcl_interfaces/archive/refs/heads/{distro_str}.zip",
        ),
        (
            "common_interfaces",
            f"https://github.com/ros2/common_interfaces/archive/refs/heads/{distro_str}.zip",
        ),
        (
            "geometry2",
            f"https://github.com/ros2/geometry2/archive/refs/heads/{distro_str}.zip",
        ),
        (
            "rosbag2",
            f"https://github.com/ros2/rosbag2/archive/refs/heads/{distro_str}.zip",
        ),
        (
            "unique_identifier_msgs",
            f"https://github.com/ros2/unique_identifier_msgs/archive/refs/heads/{distro_str}.zip",
        ),
    ]


def _download(url: str, buffer: BytesIO) -> None:
    """Download content from URL to buffer."""
    if not url.startswith("https://github.com/"):
        msg = "URL must start with 'https://github.com/'"
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


@lru_cache
def _update_cache(cache_dir: Path, distro: ROS2Distro) -> None:
    """Update cache with standard ROS2 message repositories."""
    msg_def_repos = _get_msg_def_repos(distro)
    for name, url in msg_def_repos:
        _download_and_extract(url, cache_dir / name)


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
        if len(matches) > 0:
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


def _get_msg_def(
    msg_type: str,
    distro: ROS2Distro,
    extra_paths: list[Path],
) -> tuple[str, list[str]] | None:
    """Get message definition with caching and multiple search paths.

    Search priority:
    1. User-provided extra paths
    2. AMENT_PREFIX_PATH
    3. Downloaded cache (only if not found locally)
    """
    cache_dir = _get_cache_dir(distro)
    ament_paths = _get_ament_prefix_paths()

    # Try local paths first (extra_paths + AMENT_PREFIX_PATH)
    result = _get_msg_def_disk(msg_type, (*extra_paths, *ament_paths))
    if result is not None:
        return result

    # Not found locally - update cache and search it
    _update_cache(cache_dir, distro)
    return _get_msg_def_disk(msg_type, (cache_dir,))


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
