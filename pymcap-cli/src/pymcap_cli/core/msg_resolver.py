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
from urllib.error import HTTPError, URLError
from urllib.parse import quote
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
_MAX_MSG_BYTES = 1024 * 1024
_ARCHIVE_FALLBACK_ENV = "PYMCAP_CLI_MSG_ARCHIVE_FALLBACK"


class _ResponseTooLargeError(Exception):
    """Raised when a remote .msg response exceeds ``_MAX_MSG_BYTES``.

    Distinct from network/HTTP errors so callers don't silently treat a
    suspiciously huge response (e.g. an HTML error page) as a missing
    file and fall through to the next resolution tier.
    """


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


def _download_text(url: str, *, max_bytes: int = _MAX_MSG_BYTES) -> str | None:
    """Download a small UTF-8 text file, returning None for 404s."""
    if not url.startswith(_ALLOWED_URL_PREFIXES):
        msg = f"URL must start with one of {_ALLOWED_URL_PREFIXES}"
        raise ValueError(msg)

    try:
        with urlopen(url) as response:  # noqa: S310
            if response.status != 200:
                return None

            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > max_bytes:
                msg = f"Refusing to download {url}: response is larger than {max_bytes} bytes"
                raise _ResponseTooLargeError(msg)

            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                msg = f"Refusing to download {url}: response is larger than {max_bytes} bytes"
                raise _ResponseTooLargeError(msg)
            return raw.decode("utf-8")
    except HTTPError as e:
        if e.code == 404:
            return None
        msg = f"HTTP error downloading {url}: {e.code} {e.reason}"
        raise ValueError(msg) from e
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


def _git_url_to_zip(git_url: str, version: str) -> str:
    """Convert a git URL to a GitHub ZIP download URL."""
    base = git_url.removesuffix(".git")
    return f"{base}/archive/refs/heads/{version}.zip"


def _ensure_repo_cached(cache_dir: Path, repo_name: str, zip_url: str) -> Path:
    """Download a single repo on demand if not already cached."""
    repo_dir = cache_dir / repo_name
    _download_and_extract(zip_url, repo_dir)
    return repo_dir


def _is_archive_fallback_enabled() -> bool:
    value = os.environ.get(_ARCHIVE_FALLBACK_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


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


def _github_raw_url(owner: str, repo: str, ref: str, path: str) -> str:
    quoted_path = quote(path, safe="/")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{quoted_path}"


def _repo_cache_dir(cache_dir: Path, repo_name: str, ref: str) -> Path:
    ref_key = _cache_component(ref)
    return _remote_msg_cache_dir(cache_dir) / _cache_component(repo_name) / ref_key


def _cache_remote_msg(
    cache_dir: Path,
    repo_name: str,
    ref: str,
    pkg_name: str,
    msg_name: str,
    msg_text: str,
) -> Path:
    msg_path = _repo_cache_dir(cache_dir, repo_name, ref) / pkg_name / "msg" / f"{msg_name}.msg"
    msg_path.parent.mkdir(parents=True, exist_ok=True)
    msg_path.write_text(msg_text, encoding="utf-8")
    return msg_path


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


def _fetch_raw_msg_to_cache(
    cache_dir: Path,
    repo_name: str,
    ref: str,
    pkg_name: str,
    msg_name: str,
    url: str,
) -> tuple[str, list[str]] | None:
    try:
        msg_text = _download_text(url)
    except ValueError:
        logger.debug(f"Failed to fetch remote message definition from {url}", exc_info=True)
        return None
    if msg_text is None:
        return None

    msg_path = _cache_remote_msg(cache_dir, repo_name, ref, pkg_name, msg_name, msg_text)
    return _read_msg_path(msg_path, pkg_name)


def _expanded_release_ref(release: _RepoRelease, pkg_name: str, distro: ROS2Distro) -> str:
    return (
        release.tag_template.replace("{package}", pkg_name)
        .replace("{version}", release.version)
        .replace("{distro}", distro.value)
    )


def _fetch_release_msg_def(
    cache_dir: Path,
    repo_name: str,
    release: _RepoRelease,
    pkg_name: str,
    msg_name: str,
    distro: ROS2Distro,
) -> tuple[str, list[str]] | None:
    repo = _github_repo_parts(release.url)
    if repo is None:
        return None

    owner, github_repo = repo
    ref = _expanded_release_ref(release, pkg_name, distro)
    url = _github_raw_url(owner, github_repo, ref, f"msg/{msg_name}.msg")
    return _fetch_raw_msg_to_cache(cache_dir, repo_name, ref, pkg_name, msg_name, url)


def _fetch_source_fast_path_msg_def(
    cache_dir: Path,
    repo_name: str,
    source: _RepoSource,
    pkg_name: str,
    msg_name: str,
) -> tuple[str, list[str]] | None:
    repo = _github_repo_parts(source.url)
    if repo is None:
        return None

    owner, github_repo = repo
    candidate_paths = (f"{pkg_name}/msg/{msg_name}.msg", f"msg/{msg_name}.msg")
    for path in candidate_paths:
        url = _github_raw_url(owner, github_repo, source.version, path)
        result = _fetch_raw_msg_to_cache(
            cache_dir,
            repo_name,
            source.version,
            pkg_name,
            msg_name,
            url,
        )
        if result is not None:
            return result
    return None


def _resolve_remote_msg_def(
    msg_type: str,
    distro: ROS2Distro,
    cache_dir: Path,
) -> tuple[str, list[str]] | None:
    parts = _message_parts(msg_type)
    if parts is None:
        return None
    pkg_name, msg_name = parts

    index = _get_distro_index(distro, cache_dir)
    if index is None:
        return None

    repo_name = index.pkg_to_repo.get(pkg_name)
    if repo_name is None:
        return None

    release = index.repo_to_release.get(repo_name)
    if release is not None:
        result = _fetch_release_msg_def(cache_dir, repo_name, release, pkg_name, msg_name, distro)
        if result is not None:
            return result

    source = index.repo_to_source.get(repo_name)
    if source is None:
        return None

    return _fetch_source_fast_path_msg_def(cache_dir, repo_name, source, pkg_name, msg_name)


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
    """Probe ``_remote_msgs/<repo>/<ref>/<pkg>/msg/<Name>.msg`` deterministically.

    Uses the distro index to derive each candidate ref (release tag and
    source version) — no full-tree scan needed.
    """
    index = _get_distro_index(distro, cache_dir)
    if index is None:
        return None
    repo_name = index.pkg_to_repo.get(pkg_name)
    if repo_name is None:
        return None

    refs: list[str] = []
    release = index.repo_to_release.get(repo_name)
    if release is not None:
        refs.append(_expanded_release_ref(release, pkg_name, distro))
    source = index.repo_to_source.get(repo_name)
    if source is not None:
        refs.append(source.version)

    for ref in refs:
        msg_path = _repo_cache_dir(cache_dir, repo_name, ref) / pkg_name / "msg" / f"{msg_name}.msg"
        if msg_path.is_file():
            return _read_msg_path(msg_path, pkg_name)
    return None


def _archive_cached_repo_dirs(cache_dir: Path) -> tuple[Path, ...]:
    if not cache_dir.exists():
        return ()
    return tuple(d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith("_"))


def _get_msg_def(
    msg_type: str,
    distro: ROS2Distro,
    extra_paths: list[Path],
) -> tuple[str, list[str]] | None:
    """Get message definition with caching and multiple search paths.

    Search priority:
    1. User-provided extra paths + AMENT_PREFIX_PATH (rglob)
    2. Per-message remote cache (_remote_msgs) at the rosdistro-derived ref
    3. Already-extracted repos in cache_dir (from archive fallback)
    4. Targeted remote resolution via rosdistro release/source metadata
    5. Optional archive fallback when PYMCAP_CLI_MSG_ARCHIVE_FALLBACK is enabled
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

    archive_repos = _archive_cached_repo_dirs(cache_dir)
    if archive_repos:
        result = _get_msg_def_disk(msg_type, archive_repos)
        if result is not None:
            return result

    result = _resolve_remote_msg_def(msg_type, distro, cache_dir)
    if result is not None:
        return result

    if not _is_archive_fallback_enabled():
        return None

    repo_dir = _resolve_and_download_repo(pkg_name, distro, cache_dir)
    if repo_dir is None:
        return None
    return _get_msg_def_disk(msg_type, (repo_dir,))


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
