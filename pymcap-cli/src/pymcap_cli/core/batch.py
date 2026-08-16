"""Independent transactional jobs for recursive MCAP transforms."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast
from uuid import uuid4

from pymcap_cli.core.output_validation import validate_mcap_outputs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_ARCHIVE_NAME = ".pymcap-roscompress-archive.jsonl"


@dataclass(frozen=True, slots=True)
class BatchJobResult:
    relative_path: Path
    status: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BatchRunResult:
    jobs: list[BatchJobResult]

    @property
    def failed_count(self) -> int:
        return sum(job.status == "failed" for job in self.jobs)


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def run_batch(
    input_dir: Path,
    output_dir: Path,
    run_one: Callable[[Path, Path], int],
    *,
    recipe: JsonValue,
    continue_on_error: bool = False,
    force: bool = False,
    preserved_topic_patterns: Sequence[str] = (),
    lossy_topic_patterns: Sequence[str] = (),
) -> BatchRunResult:
    """Run sorted MCAP jobs as independent adjacent-partial transactions."""
    source_root = input_dir.resolve()
    if not source_root.is_dir():
        raise ValueError(f"batch input is not a directory: {input_dir}")
    destination_root = output_dir.resolve()
    if source_root == destination_root:
        raise ValueError("batch input and output directories must differ")
    if source_root.is_relative_to(destination_root):
        raise ValueError("batch output directory must not contain the input directory")
    destination_root.mkdir(parents=True, exist_ok=True)
    archive = destination_root / _ARCHIVE_NAME
    recipe_digest = _recipe_digest(recipe)
    archive_records = _load_archive_records(archive, recipe_digest)

    jobs: list[BatchJobResult] = []
    for source in _discover_sources(source_root, destination_root):
        relative_path = source.relative_to(source_root)
        final = destination_root / relative_path
        try:
            status = _run_job(
                source,
                relative_path,
                final,
                run_one,
                archive,
                archive_records,
                recipe_digest,
                force=force,
                preserved_topic_patterns=preserved_topic_patterns,
                lossy_topic_patterns=lossy_topic_patterns,
            )
            jobs.append(BatchJobResult(relative_path, status))
        except Exception as exc:  # noqa: BLE001 - isolate independent user transform jobs
            jobs.append(BatchJobResult(relative_path, "failed", str(exc)))
            if not continue_on_error:
                break
    return BatchRunResult(jobs)


def _discover_sources(source_root: Path, destination_root: Path) -> list[Path]:
    sources: list[Path] = []
    for candidate in source_root.rglob("*.mcap"):
        resolved = candidate.resolve()
        if resolved.is_relative_to(destination_root):
            continue
        if candidate.is_file():
            sources.append(candidate)
    return sorted(sources, key=lambda path: path.relative_to(source_root).as_posix())


def _run_job(
    source: Path,
    relative_path: Path,
    final: Path,
    run_one: Callable[[Path, Path], int],
    archive: Path,
    archive_records: dict[str, dict[str, JsonValue]],
    recipe: str,
    *,
    force: bool,
    preserved_topic_patterns: Sequence[str],
    lossy_topic_patterns: Sequence[str],
) -> str:
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(f".{final.name}.partial-{os.getpid()}-{uuid4().hex}")
    snapshot = _snapshot(source)
    record = archive_records.get(relative_path.as_posix())
    if record is not None and _can_resume(
        record,
        snapshot,
        relative_path,
        source,
        final,
        preserved_topic_patterns,
        lossy_topic_patterns,
    ):
        if _snapshot(source) != snapshot:
            raise RuntimeError("source changed during resume verification")
        return "verified-resumed"
    if final.exists() and not force:
        raise FileExistsError(f"output already exists: {final}")

    try:
        code = run_one(source, partial)
        if code:
            raise RuntimeError(f"transform exited with code {code}")
        if _snapshot(source) != snapshot:
            raise RuntimeError("source changed during transform")
        validation_error = validate_mcap_outputs(
            [source],
            [partial],
            preserved_topic_patterns=preserved_topic_patterns,
            lossy_topic_patterns=lossy_topic_patterns,
        )
        if validation_error is not None:
            raise RuntimeError(validation_error)
        output_size = partial.stat().st_size
        partial.replace(final)
        record_value = _archive_record(
            recipe,
            relative_path,
            snapshot,
            output_size,
        )
        _append_archive(archive, record_value)
        archive_records[relative_path.as_posix()] = record_value
        return "committed"
    finally:
        partial.unlink(missing_ok=True)


def _snapshot(path: Path) -> _SourceSnapshot:
    stat = path.stat()
    return _SourceSnapshot(
        device=stat.st_dev,
        inode=stat.st_ino,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
    )


def _recipe_digest(recipe: JsonValue) -> str:
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _archive_record(
    recipe: str,
    relative_path: Path,
    source_snapshot: _SourceSnapshot,
    output_size: int,
) -> dict[str, JsonValue]:
    return {
        "recipe": recipe,
        "path": relative_path.as_posix(),
        "source_size": source_snapshot.size,
        "source_mtime_ns": source_snapshot.mtime_ns,
        "output_size": output_size,
    }


def _load_archive_records(
    archive: Path,
    recipe: str,
) -> dict[str, dict[str, JsonValue]]:
    try:
        stream = archive.open()
    except FileNotFoundError:
        return {}
    records: dict[str, dict[str, JsonValue]] = {}
    with stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            record = cast("dict[str, JsonValue]", value)
            path = record.get("path")
            if record.get("recipe") == recipe and isinstance(path, str):
                records[path] = record
    return records


def _can_resume(
    record: dict[str, JsonValue],
    source_snapshot: _SourceSnapshot,
    relative_path: Path,
    source_path: Path,
    final: Path,
    preserved_topic_patterns: Sequence[str],
    lossy_topic_patterns: Sequence[str],
) -> bool:
    if (
        record.get("path") != relative_path.as_posix()
        or record.get("source_size") != source_snapshot.size
        or record.get("source_mtime_ns") != source_snapshot.mtime_ns
        or not final.is_file()
    ):
        return False
    output_size = record.get("output_size")
    if type(output_size) is not int:
        return False
    if final.stat().st_size != output_size:
        return False
    return (
        validate_mcap_outputs(
            [source_path],
            [final],
            preserved_topic_patterns=preserved_topic_patterns,
            lossy_topic_patterns=lossy_topic_patterns,
        )
        is None
    )


def _append_archive(archive: Path, record: dict[str, JsonValue]) -> None:
    with archive.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
