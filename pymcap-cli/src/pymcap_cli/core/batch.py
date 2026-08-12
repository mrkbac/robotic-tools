"""Independent transactional jobs for recursive MCAP transforms."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast
from uuid import uuid4

from pymcap_cli.doctor import examine_mcap

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_ARCHIVE_NAME = ".pymcap-roscompress-archive.jsonl"
_READ_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Transform-specific observations passed from execution to validation."""

    values: dict[str, JsonValue] = field(default_factory=dict)


class BatchTransform(Protocol):
    def recipe(self) -> JsonValue: ...

    def preflight(self, source: Path) -> None: ...

    def run(self, source: Path, partial: Path) -> TransformResult: ...

    def validate(
        self,
        source: Path,
        partial: Path,
        result: TransformResult,
    ) -> None: ...


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
    transform: BatchTransform,
    *,
    archive_path: Path | None = None,
    continue_on_error: bool = False,
    force: bool = False,
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
    archive = (
        archive_path.resolve() if archive_path is not None else destination_root / _ARCHIVE_NAME
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    recipe = _recipe_digest(transform.recipe())

    jobs: list[BatchJobResult] = []
    for source in _discover_sources(source_root, destination_root):
        relative_path = source.relative_to(source_root)
        final = destination_root / relative_path
        try:
            status = _run_job(
                source,
                relative_path,
                final,
                transform,
                archive,
                recipe,
                force=force,
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
        if ".partial-" in candidate.name:
            continue
        if candidate.is_file():
            sources.append(candidate)
    return sorted(sources, key=lambda path: path.relative_to(source_root).as_posix())


def _run_job(
    source: Path,
    relative_path: Path,
    final: Path,
    transform: BatchTransform,
    archive: Path,
    recipe: str,
    *,
    force: bool,
) -> str:
    final.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == final.resolve():
        raise ValueError("source and output resolve to the same path")
    partial = final.with_name(f".{final.name}.partial-{os.getpid()}-{uuid4().hex}")
    lock = final.with_name(f".{final.name}.batch-lock")
    with _exclusive_lock(lock):
        if final.exists() and source.samefile(final):
            raise ValueError("source and output are hard-link aliases")
        snapshot = _snapshot(source)
        source_fingerprint = _source_fingerprint(source, snapshot.size)
        with _exclusive_lock(archive.with_name(f".{archive.name}.lock")):
            record = _latest_archive_record(archive, relative_path, recipe)
        if record is not None and _authenticated_resume(
            record,
            source_fingerprint,
            snapshot.size,
            relative_path,
            final,
        ):
            if _snapshot(source) != snapshot:
                raise RuntimeError("source changed during resume verification")
            return "verified-resumed"
        if final.exists() and not force:
            raise FileExistsError(f"output already exists: {final}")

        _require_free_space(final.parent, snapshot.size)
        transform.preflight(source)
        try:
            result = transform.run(source, partial)
            transform.validate(source, partial, result)
            if _snapshot(source) != snapshot:
                raise RuntimeError("source changed during transform")
            output_digest = _validate_and_digest(partial)
            output_size = partial.stat().st_size
            _fsync_file(partial)
            partial.replace(final)
            _fsync_directory(final.parent)
            record_value = _archive_record(
                recipe,
                relative_path,
                snapshot.size,
                source_fingerprint,
                output_size,
                output_digest,
            )
            with _exclusive_lock(archive.with_name(f".{archive.name}.lock")):
                _append_archive(archive, record_value)
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


def _require_free_space(path: Path, required_bytes: int) -> None:
    free_bytes = shutil.disk_usage(path).free
    if free_bytes < required_bytes:
        raise OSError(
            f"insufficient free space in {path}: need at least {required_bytes} bytes, "
            f"have {free_bytes}"
        )


def _source_fingerprint(path: Path, size: int) -> str:
    try:
        from pymcap_cli.index.fingerprint import fingerprint_stream  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "batch mode requires the xxhash extra; run with uvx 'pymcap-cli[xxhash]'"
        ) from exc
    with path.open("rb") as stream:
        return f"xxh3_128:{fingerprint_stream(stream, size)}"


def _recipe_digest(recipe: JsonValue) -> str:
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_and_digest(path: Path) -> str:
    size = path.stat().st_size
    with path.open("rb") as stream:
        report = examine_mcap(stream, size, str(path))
    if report.error_count:
        raise RuntimeError(f"doctor found {report.error_count} output errors")
    return f"sha256:{_sha256(path)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_READ_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _archive_record(
    recipe: str,
    relative_path: Path,
    source_size: int,
    source_fingerprint: str,
    output_size: int,
    output_digest: str,
) -> dict[str, JsonValue]:
    relative = relative_path.as_posix()
    return {
        "schema_version": 1,
        "recipe": recipe,
        "source": {
            "relative_path": relative,
            "size": source_size,
            "fingerprint": source_fingerprint,
        },
        "output": {
            "relative_path": relative,
            "size": output_size,
            "sha256": output_digest,
        },
    }


def _latest_archive_record(
    archive: Path,
    relative_path: Path,
    recipe: str,
) -> dict[str, JsonValue] | None:
    try:
        stream = archive.open()
    except FileNotFoundError:
        return None
    latest: dict[str, JsonValue] | None = None
    expected_path = relative_path.as_posix()
    with stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            record = cast("dict[str, JsonValue]", value)
            source = record.get("source")
            if (
                record.get("schema_version") == 1
                and record.get("recipe") == recipe
                and isinstance(source, dict)
                and source.get("relative_path") == expected_path
            ):
                latest = record
    return latest


def _authenticated_resume(
    record: dict[str, JsonValue],
    source_fingerprint: str,
    source_size: int,
    relative_path: Path,
    final: Path,
) -> bool:
    source = record.get("source")
    output = record.get("output")
    if not isinstance(source, dict) or not isinstance(output, dict):
        return False
    expected_path = relative_path.as_posix()
    if (
        source.get("relative_path") != expected_path
        or source.get("size") != source_size
        or source.get("fingerprint") != source_fingerprint
        or output.get("relative_path") != expected_path
        or not final.is_file()
    ):
        return False
    output_size = output.get("size")
    output_digest = output.get("sha256")
    if type(output_size) is not int or not isinstance(output_digest, str):
        return False
    if final.stat().st_size != output_size or f"sha256:{_sha256(final)}" != output_digest:
        return False
    try:
        _validate_and_digest(final)
    except (OSError, RuntimeError):
        return False
    return True


def _append_archive(archive: Path, record: dict[str, JsonValue]) -> None:
    with archive.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(archive.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    if os.name != "nt":
        import fcntl  # noqa: PLC0415

        with path.open("a+b") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"batch lock is already held: {path}") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return

    # Windows fallback uses an exclusive-create lock. It is removed on every
    # normal exit; POSIX uses flock above so crashed processes cannot leave a
    # stale held lock.
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(descriptor, str(os.getpid()).encode())
        yield
    except FileExistsError as exc:
        raise RuntimeError(f"batch lock is already held: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
            path.unlink(missing_ok=True)
