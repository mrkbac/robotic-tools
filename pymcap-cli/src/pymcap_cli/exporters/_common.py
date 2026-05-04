"""Shared helpers for the per-format exporters.

Defines the default blob-schema skip-list, a topic→filename sanitiser, output
directory validation, and a topic+schema predicate factory.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from mcap_codec_support._schemas import normalize_schema_name
from small_mcap import include_topics

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from small_mcap import Channel, DecodedMessage, Schema

logger = logging.getLogger(__name__)
_TABLE_NAME_RE = re.compile(r"[^0-9a-zA-Z_]+")


def normalize_schema_names(names: Iterable[str]) -> frozenset[str]:
    """Canonicalise a collection of schema names for membership checks."""
    return frozenset(normalize_schema_name(name) for name in names)


def schema_name_in(schema: Schema | None, names: frozenset[str]) -> bool:
    """Return True when *schema* is present and canonical name is in *names*."""
    return schema is not None and normalize_schema_name(schema.name) in names


# Schemas whose payload is a raw media blob — almost always useless when
# exported as text/CSV/JSON. Skipped by default unless ``--include-blobs``.
# Stored in canonical (short) form; compare via :func:`normalize_schema_name`.
DEFAULT_BLOB_SCHEMAS: frozenset[str] = frozenset(
    {
        "sensor_msgs/Image",
        "sensor_msgs/CompressedImage",
        "foxglove_msgs/CompressedImage",
        "foxglove_msgs/CompressedVideo",
        "foxglove_msgs/RawImage",
        "audio_common_msgs/AudioData",
    }
)


class SkipSchemaMixin:
    """Shared ``accepts`` implementation for exporters with skip-lists."""

    _skipped_schemas: set[str]

    def _set_skipped_schemas(
        self,
        *,
        include_blobs: bool,
        skip_schema: Iterable[str] = (),
    ) -> None:
        skipped: set[str] = set() if include_blobs else set(DEFAULT_BLOB_SCHEMAS)
        skipped.update(normalize_schema_name(schema) for schema in skip_schema)
        self._skipped_schemas = skipped

    def accepts(self, schema: Schema | None) -> bool:
        if schema is None:
            return True
        return normalize_schema_name(schema.name) not in self._skipped_schemas


def topic_to_filename(topic: str) -> str:
    """Map a topic name (``/a/b``) to a safe filesystem component (``a_b``)."""
    name = _TABLE_NAME_RE.sub("_", topic).strip("_")
    if not name:
        name = "topic"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def unique_topic_filename(topic: str, used_filenames: set[str]) -> str:
    """Variant of :func:`topic_to_filename` that disambiguates collisions."""
    filename = topic_to_filename(topic)
    if filename not in used_filenames:
        return filename

    stem = filename
    suffix = 2
    while filename in used_filenames:
        filename = f"{stem}_{suffix}"
        suffix += 1
    return filename


def prepare_output_file(path: Path, *, force: bool) -> Path:
    """Prepare one exporter-owned file path, removing conflicts on ``--force``."""
    if force and path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def prepare_topic_dir(path: Path, *, force: bool) -> Path:
    """Prepare one exporter-owned per-topic directory."""
    if force and path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def message_timestamps_ns(msg: DecodedMessage) -> tuple[int, int]:
    """Return ``(log_time_ns, publish_time_ns)`` for a decoded MCAP message."""
    return int(msg.message.log_time), int(msg.message.publish_time)


def unique_message_path(
    directory: Path,
    log_time_ns: int,
    extension: str,
    used_counts: dict[int, int],
) -> Path:
    """Return a stable per-message path without overwriting duplicate timestamps."""
    count = used_counts.get(log_time_ns, 0)
    used_counts[log_time_ns] = count + 1
    stem = str(log_time_ns) if count == 0 else f"{log_time_ns}_{count:06d}"
    return directory / f"{stem}.{extension.lstrip('.')}"


def validate_output_dir(output: str | Path, *, force: bool) -> Path | None:
    """Resolve and validate the output directory. Returns ``None`` on error.

    On ``force=True``, the directory is left intact (callers decide which
    files to clean up — extensions vary per exporter).
    """
    out_dir = Path(output)
    if out_dir.exists() and not out_dir.is_dir():
        logger.error(f"{out_dir} exists and is not a directory.")
        return None
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        logger.error(f"{out_dir} is not empty. Use --force to overwrite.")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_should_include(
    *,
    topics: list[str] | None,
    accepts_schema: Callable[[Schema | None], bool],
) -> Callable[[Channel, Schema | None], bool]:
    """Build a ``should_include`` predicate for ``small_mcap.read_message_decoded``.

    Composes :func:`small_mcap.include_topics` (topic filter) with the
    exporter's schema acceptance test, so unsupported / blob schemas are
    rejected at chunk-decode time instead of after CDR decoding.
    """
    topic_predicate = include_topics(topics) if topics else None

    def _should_include(channel: Channel, schema: Schema | None) -> bool:
        if not accepts_schema(schema):
            return False
        if topic_predicate is None:
            return True
        return topic_predicate(channel, schema)

    return _should_include
