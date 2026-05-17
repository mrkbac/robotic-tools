"""ROS2 message definition resolver with dynamic rosdistro-based package lookup."""

import json
import logging
import os
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import cast
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import platformdirs
import yaml
from ros_parser.ros2_msg import parse_message_string

logger = logging.getLogger(__name__)

_ALLOWED_URL_PREFIXES = (
    "https://github.com/",
    "https://raw.githubusercontent.com/",
)
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
    url: str
    version: str


@dataclass(frozen=True)
class _RepoRelease:
    url: str
    version: str
    tag_template: str


@dataclass(frozen=True)
class _DistroIndex:
    pkg_to_repo: dict[str, str]
    repo_to_source: dict[str, _RepoSource]
    repo_to_release: dict[str, _RepoRelease] = field(default_factory=dict)


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

    logger.debug(f"Downloading {url} to {target_dir}")
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
    repo_to_release: dict[str, _RepoRelease] = {}

    repositories = data.get("repositories", {})
    if not isinstance(repositories, dict):
        return _DistroIndex(pkg_to_repo={}, repo_to_source={}, repo_to_release={})

    for repo_key, repo_data in repositories.items():
        if not isinstance(repo_key, str) or not isinstance(repo_data, dict):
            continue
        repo_data_dict = cast("dict[str, object]", repo_data)

        source_raw = repo_data_dict.get("source")
        if isinstance(source_raw, dict):
            source_dict = cast("dict[str, object]", source_raw)
            if source_dict.get("type") == "git":
                url = source_dict.get("url", "")
                version = source_dict.get("version", "")
                if isinstance(url, str) and isinstance(version, str) and url and version:
                    repo_to_source[repo_key] = _RepoSource(url=url, version=version)

        # Release repositories expose package roots, which lets us fetch msg files
        # directly without knowing the source layout.
        release_raw = repo_data_dict.get("release")
        if isinstance(release_raw, dict):
            release_dict = cast("dict[str, object]", release_raw)
            url = release_dict.get("url", "")
            version = release_dict.get("version", "")
            tags_raw = release_dict.get("tags")
            tag_template = ""
            if isinstance(tags_raw, dict):
                tags_dict = cast("dict[str, object]", tags_raw)
                release_tag = tags_dict.get("release", "")
                if isinstance(release_tag, str):
                    tag_template = release_tag
            if (
                isinstance(url, str)
                and isinstance(version, str)
                and url
                and version
                and tag_template
            ):
                repo_to_release[repo_key] = _RepoRelease(
                    url=url,
                    version=version,
                    tag_template=tag_template,
                )

            packages = release_dict.get("packages")
            if isinstance(packages, list):
                for pkg in packages:
                    if isinstance(pkg, str):
                        pkg_to_repo[pkg] = repo_key
            else:
                pkg_to_repo[repo_key] = repo_key
        elif repo_key in repo_to_source:
            pkg_to_repo[repo_key] = repo_key

    return _DistroIndex(
        pkg_to_repo=pkg_to_repo,
        repo_to_source=repo_to_source,
        repo_to_release=repo_to_release,
    )


def _distro_index_from_json(index_data: dict[str, object]) -> _DistroIndex:
    """Build a _DistroIndex from cached JSON we wrote ourselves.

    Trusts the on-disk shape; ``_fetch_distro_index`` catches the
    broad set of errors (``KeyError``, ``TypeError``, ``ValueError``,
    ``json.JSONDecodeError``) and re-fetches on any mismatch.
    """
    sources = {
        repo: _RepoSource(url=v["url"], version=v["version"])
        for repo, v in cast("dict[str, dict[str, str]]", index_data["repo_to_source"]).items()
    }
    releases = {
        repo: _RepoRelease(url=v["url"], version=v["version"], tag_template=v["tag_template"])
        for repo, v in cast(
            "dict[str, dict[str, str]]", index_data.get("repo_to_release", {})
        ).items()
    }
    return _DistroIndex(
        pkg_to_repo=cast("dict[str, str]", index_data["pkg_to_repo"]),
        repo_to_source=sources,
        repo_to_release=releases,
    )


def _distro_index_to_json(index: _DistroIndex) -> dict[str, object]:
    return {
        "pkg_to_repo": index.pkg_to_repo,
        "repo_to_source": {
            k: {"url": v.url, "version": v.version} for k, v in index.repo_to_source.items()
        },
        "repo_to_release": {
            k: {"url": v.url, "version": v.version, "tag_template": v.tag_template}
            for k, v in index.repo_to_release.items()
        },
    }


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
                return _distro_index_from_json(index_data)
        except (ValueError, KeyError, json.JSONDecodeError, TypeError):
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
                return _distro_index_from_json(index_data)
            except (ValueError, KeyError, json.JSONDecodeError, TypeError):
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

    cache_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.write_bytes(raw_yaml)
    index_path.write_text(json.dumps(_distro_index_to_json(distro_index)))
    timestamp_path.write_text(str(time.time()))

    return distro_index


@lru_cache
def _get_distro_index(distro: ROS2Distro, cache_dir: Path) -> _DistroIndex | None:
    """Get distro index with in-memory caching."""
    return _fetch_distro_index(distro, cache_dir)


def _remote_msg_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / "_remote_msgs"


def _cache_component(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)


def _message_parts(msg_type: str) -> tuple[str, str] | None:
    parts = msg_type.split("/")
    if len(parts) == 3 and parts[1] == "msg":
        return parts[0], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1]

    logger.warning(f"Invalid message type format: {msg_type}")
    return None


def _github_repo_parts(git_url: str) -> tuple[str, str] | None:
    base = git_url.removesuffix(".git").removesuffix("/")
    prefix = "https://github.com/"
    if not base.startswith(prefix):
        return None

    parts = base[len(prefix) :].split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _repo_cache_dir(cache_dir: Path, repo_name: str, ref: str) -> Path:
    ref_key = _cache_component(ref)
    return _remote_msg_cache_dir(cache_dir) / _cache_component(repo_name) / ref_key


def _extract_dependencies(pkg_name: str, msg_text: str) -> list[str]:
    msg_def = parse_message_string(msg_text, context_package_name=pkg_name)
    return [
        f"{f.type.package_name}/{f.type.type_name}"
        for f in msg_def.fields
        if not f.type.is_primitive
        and f.type.package_name
        and f.type.package_name != "builtin_interfaces"
    ]


def _read_msg_path(msg_path: Path, pkg_name: str) -> tuple[str, list[str]] | None:
    try:
        msg_text = msg_path.read_text(encoding="utf-8")
        dependencies = _extract_dependencies(pkg_name, msg_text)
    except Exception:
        logger.exception(f"Failed to parse {msg_path}")
        return None

    return msg_text, dependencies


def _expanded_release_ref(release: _RepoRelease, pkg_name: str, distro: ROS2Distro) -> str:
    return (
        release.tag_template.replace("{package}", pkg_name)
        .replace("{version}", release.version)
        .replace("{distro}", distro.value)
    )


def _get_cache_dir(distro: ROS2Distro) -> Path:
    """Get cache directory for a specific ROS2 distribution.

    The ``version`` segment is the cache schema generation, bumped
    whenever the on-disk layout changes so legacy state from older
    releases doesn't poison a new one.
    """
    cache_dir = (
        platformdirs.user_cache_path(
            appname="pymcap_cli_msg_def",
            ensure_exists=True,
            version="v2",
        )
        / distro.value
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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
def _get_msg_def_disk(msg_type: str, folders: tuple[Path, ...]) -> tuple[str, list[str]] | None:
    """Get message definition from disk by ``rglob``-ing each folder.

    Used for user-supplied roots (``extra_paths`` + ``AMENT_PREFIX_PATH``)
    where we don't know how deep the workspace nests packages.
    """
    parts = _message_parts(msg_type)
    if parts is None:
        return None
    pkg_name, msg_name = parts

    pattern = f"**/{pkg_name}/msg/{msg_name}.msg"
    for f in folders:
        if not f.exists():
            continue
        msg_path = next(f.rglob(pattern), None)
        if msg_path is not None:
            logger.debug(f"Found {msg_type} at {msg_path}")
            return _read_msg_path(msg_path, pkg_name)
    return None


def _lookup_cached_remote_msg(
    cache_dir: Path,
    distro: ROS2Distro,
    pkg_name: str,
    msg_name: str,
) -> tuple[str, list[str]] | None:
    """Probe ``_remote_msgs/<repo>/<release-ref>/<pkg>/msg/<Name>.msg``.

    Uses the distro index to derive the release ref; returns ``None``
    if no release entry exists or the file isn't cached yet.
    """
    index = _get_distro_index(distro, cache_dir)
    if index is None:
        return None
    repo_name = index.pkg_to_repo.get(pkg_name)
    if repo_name is None:
        return None
    release = index.repo_to_release.get(repo_name)
    if release is None:
        return None

    ref = _expanded_release_ref(release, pkg_name, distro)
    pkg_dir = _repo_cache_dir(cache_dir, repo_name, ref) / pkg_name
    for subdir in _MSG_SUBDIRS:
        candidate = pkg_dir / subdir / f"{msg_name}.msg"
        if candidate.is_file():
            return _read_msg_path(candidate, pkg_name)
    return None


def _get_msg_def(
    msg_type: str,
    distro: ROS2Distro,
    extra_paths: list[Path],
) -> tuple[str, list[str]] | None:
    """Get message definition with caching and multiple search paths.

    Search priority:

    1. User-provided extra paths + ``AMENT_PREFIX_PATH`` (rglob).
    2. Per-message remote cache populated by an earlier release-zip fetch.
    3. Fresh release-zip download (also warms the per-message cache).
    """
    parts = _message_parts(msg_type)
    if parts is None:
        return None
    pkg_name, msg_name = parts

    cache_dir = _get_cache_dir(distro)
    ament_paths = _get_ament_prefix_paths()

    result = _get_msg_def_disk(msg_type, (*extra_paths, *ament_paths))
    if result is not None:
        return result

    result = _lookup_cached_remote_msg(cache_dir, distro, pkg_name, msg_name)
    if result is not None:
        return result

    if _ensure_release_pkg_cached(cache_dir, distro, pkg_name) is None:
        return None
    return _lookup_cached_remote_msg(cache_dir, distro, pkg_name, msg_name)


# Subdirectories that may hold a package's .msg files inside its source tree.
# Most packages use ``msg/``; dual-ROS sources like ``foxglove_msgs`` use ``ros2/``.
_MSG_SUBDIRS: tuple[str, ...] = ("msg", "ros2")


def _list_msgs_in_pkg_dir(pkg_dir: Path) -> list[str]:
    """Return sorted message stems found under any known msg subdirectory.

    Covers the conventional ``<pkg>/msg/`` layout and the dual-ROS
    ``<pkg>/ros2/`` layout (used by ``foxglove_msgs`` and other
    packages that ship ROS1 + ROS2 from a single repo).
    """
    names: set[str] = set()
    for subdir in _MSG_SUBDIRS:
        d = pkg_dir / subdir
        if not d.is_dir():
            continue
        names.update(p.stem for p in d.glob("*.msg") if p.is_file())
    return sorted(names)


def _release_tag_zip_url(
    release: _RepoRelease,
    pkg_name: str,
    distro: ROS2Distro,
) -> str | None:
    repo = _github_repo_parts(release.url)
    if repo is None:
        return None
    owner, github_repo = repo
    ref = _expanded_release_ref(release, pkg_name, distro)
    return f"https://github.com/{owner}/{github_repo}/archive/refs/tags/{ref}.zip"


def _ensure_release_pkg_cached(
    cache_dir: Path,
    distro: ROS2Distro,
    pkg_name: str,
) -> Path | None:
    """Download and extract the rosdistro release branch for ``pkg_name``.

    Returns the package-root directory
    (``_remote_msgs/<repo>/<ref>/<pkg>``) so callers can list
    ``msg/*.msg`` directly. Subsequent ``_lookup_cached_remote_msg``
    calls hit disk for free.

    Returns ``None`` if the package has no rosdistro release entry or
    the download fails.
    """
    index = _get_distro_index(distro, cache_dir)
    if index is None:
        return None
    repo_name = index.pkg_to_repo.get(pkg_name)
    if repo_name is None:
        return None
    release = index.repo_to_release.get(repo_name)
    if release is None:
        return None

    zip_url = _release_tag_zip_url(release, pkg_name, distro)
    if zip_url is None:
        return None

    ref = _expanded_release_ref(release, pkg_name, distro)
    pkg_dir = _repo_cache_dir(cache_dir, repo_name, ref) / pkg_name
    if pkg_dir.is_dir():
        return pkg_dir

    # Extract into a sibling temp dir, then atomically rename into place so
    # an interrupted run never leaves a half-populated pkg_dir behind.
    pkg_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=pkg_dir.parent, prefix=f".{pkg_name}.tmp.") as tmp:
        tmp_pkg = Path(tmp) / pkg_name
        try:
            _download_and_extract(zip_url, tmp_pkg)
        except ValueError:
            logger.debug(f"Failed to fetch release branch zip from {zip_url}", exc_info=True)
            return None
        tmp_pkg.replace(pkg_dir)
    return pkg_dir


def _find_pkg_dir_in_folders(folders: tuple[Path, ...], pkg_name: str) -> list[Path]:
    """Locate every ``<pkg_name>/msg`` directory under any of ``folders``.

    Returns the *parent* dir (the package root) for each match. Multiple
    matches let workspace overlays add messages to a package.
    """
    seen: list[Path] = []
    for f in folders:
        if not f.exists():
            continue
        for msg_dir in f.rglob(f"{pkg_name}/msg"):
            if msg_dir.is_dir():
                pkg_dir = msg_dir.parent
                if pkg_dir not in seen:
                    seen.append(pkg_dir)
    return seen


def list_package_messages(
    package_name: str,
    distro: ROS2Distro = ROS2Distro.HUMBLE,
    extra_paths: tuple[Path, ...] = (),
) -> list[str] | None:
    """List all ROS2 ``.msg`` types defined by ``package_name``.

    Returns the message names without the ``.msg`` suffix (e.g.
    ``["Image", "CompressedImage", ...]``), sorted and deduplicated.
    Returns ``None`` if the package can't be located, ``[]`` if it
    exists but defines no messages.

    Search priority:

    1. User-provided extra paths + ``AMENT_PREFIX_PATH`` (union across roots).
    2. Release-branch zip via rosdistro (also warms the per-message cache).
    """
    cache_dir = _get_cache_dir(distro)
    ament_paths = _get_ament_prefix_paths()

    local_pkg_dirs = _find_pkg_dir_in_folders((*extra_paths, *ament_paths), package_name)
    if local_pkg_dirs:
        names: set[str] = set()
        for pkg_dir in local_pkg_dirs:
            names.update(_list_msgs_in_pkg_dir(pkg_dir))
        return sorted(names)

    pkg_dir = _ensure_release_pkg_cached(cache_dir, distro, package_name)
    if pkg_dir is None:
        return None
    return _list_msgs_in_pkg_dir(pkg_dir)


def list_distro_packages(distro: ROS2Distro = ROS2Distro.HUMBLE) -> list[str] | None:
    """Return all package names known to the rosdistro index, sorted.

    Pure index lookup — no per-package downloads. Returns ``None`` if
    the distro index can't be fetched (offline, no cache).
    """
    cache_dir = _get_cache_dir(distro)
    index = _get_distro_index(distro, cache_dir)
    if index is None:
        return None
    return sorted(index.pkg_to_repo)


@dataclass(frozen=True)
class PackageInfo:
    """Rosdistro metadata for a single package, sourced from distribution.yaml."""

    name: str
    repo_name: str | None
    source_url: str | None
    source_version: str | None
    release_url: str | None
    release_version: str | None
    release_tag: str | None


def get_package_info(
    package_name: str,
    distro: ROS2Distro = ROS2Distro.HUMBLE,
) -> PackageInfo | None:
    """Return rosdistro metadata for ``package_name``.

    Pure index lookup — no per-package downloads. Returns ``None`` if
    the distro index can't be fetched or the package isn't listed.
    """
    cache_dir = _get_cache_dir(distro)
    index = _get_distro_index(distro, cache_dir)
    if index is None:
        return None
    repo_name = index.pkg_to_repo.get(package_name)
    if repo_name is None:
        return None

    source = index.repo_to_source.get(repo_name)
    release = index.repo_to_release.get(repo_name)
    return PackageInfo(
        name=package_name,
        repo_name=repo_name,
        source_url=source.url if source else None,
        source_version=source.version if source else None,
        release_url=release.url if release else None,
        release_version=release.version if release else None,
        release_tag=_expanded_release_ref(release, package_name, distro) if release else None,
    )


def get_message_text(
    msg_type: str,
    distro: ROS2Distro = ROS2Distro.HUMBLE,
    extra_paths: tuple[Path, ...] = (),
) -> tuple[str, list[str]] | None:
    """Resolve a message to its raw ``.msg`` text and direct dependencies.

    Unlike :func:`get_message_definition`, this does NOT recurse into
    dependencies — useful when callers want to display one message at
    a time (e.g. the `msg-serve` web view) and link to dependencies
    separately.
    """
    return _get_msg_def(msg_type, distro, list(extra_paths))


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
