from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.core import batch
from pymcap_cli.core.batch import BatchRunResult
from pymcap_cli.core.batch import run_batch as run_batch_core

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_COPY_RECIPE = {"operation": "copy", "version": 1}


@dataclass(slots=True)
class CopyTransform:
    run_count: int = 0
    fail_name: str | None = None
    mutate_source: bool = False
    interrupt: bool = False
    fail_run: bool = False

    def __call__(self, source: Path, partial: Path) -> int:
        if source.name == self.fail_name:
            raise RuntimeError("preflight failed")
        self.run_count += 1
        shutil.copyfile(source, partial)
        if self.interrupt:
            raise KeyboardInterrupt
        if self.fail_run:
            raise RuntimeError("transform failed")
        if self.mutate_source:
            source.write_bytes(source.read_bytes() + b"changed")
        return 0


def run_batch(
    input_dir: Path,
    output_dir: Path,
    run_one: CopyTransform,
    *,
    continue_on_error: bool = False,
    force: bool = False,
    preserved_topic_patterns: Sequence[str] = (),
    lossy_topic_patterns: Sequence[str] = (),
) -> BatchRunResult:
    return run_batch_core(
        input_dir,
        output_dir,
        run_one,
        recipe=_COPY_RECIPE,
        continue_on_error=continue_on_error,
        force=force,
        preserved_topic_patterns=preserved_topic_patterns,
        lossy_topic_patterns=lossy_topic_patterns,
    )


def _source_tree(tmp_path: Path, simple_mcap: Path) -> Path:
    source = tmp_path / "input"
    (source / "nested").mkdir(parents=True)
    shutil.copyfile(simple_mcap, source / "nested" / "run.mcap")
    return source


def test_run_batch_commits_output_and_records_archive(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"
    transform = CopyTransform()

    result = run_batch(source, output, transform)

    final = output / "nested" / "run.mcap"
    assert result.failed_count == 0
    assert result.jobs[0].status == "committed"
    assert final.read_bytes() == simple_mcap.read_bytes()
    archive = output / ".pymcap-roscompress-archive.jsonl"
    record = json.loads(archive.read_text().splitlines()[0])
    assert record == {
        "output_size": final.stat().st_size,
        "path": "nested/run.mcap",
        "recipe": record["recipe"],
        "source_mtime_ns": (source / "nested" / "run.mcap").stat().st_mtime_ns,
        "source_size": (source / "nested" / "run.mcap").stat().st_size,
    }
    assert list(output.rglob("*.partial-*")) == []


def test_run_batch_resumes_only_after_output_authentication(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"
    transform = CopyTransform()
    assert run_batch(source, output, transform).failed_count == 0

    resumed = run_batch(source, output, transform)

    assert transform.run_count == 1
    assert resumed.jobs[0].status == "verified-resumed"


def test_run_batch_rebuilds_tampered_archived_output(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"
    transform = CopyTransform()
    assert run_batch(source, output, transform).failed_count == 0
    final = output / "nested" / "run.mcap"
    final.write_bytes(b"tampered")

    result = run_batch(source, output, transform, force=True)

    assert result.jobs[0].status == "committed"
    assert transform.run_count == 2
    assert final.read_bytes() == simple_mcap.read_bytes()


def test_run_batch_loads_archive_once(
    tmp_path: Path,
    simple_mcap: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    shutil.copyfile(simple_mcap, source / "second.mcap")
    output = tmp_path / "output"
    original = batch._load_archive_records
    load_count = 0

    def count_loads(archive: Path, recipe: str):
        nonlocal load_count
        load_count += 1
        return original(archive, recipe)

    monkeypatch.setattr(batch, "_load_archive_records", count_loads)

    result = run_batch(source, output, CopyTransform())

    assert result.failed_count == 0
    assert load_count == 1


def test_run_batch_collision_preserves_existing_output(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"
    final = output / "nested" / "run.mcap"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"existing")

    result = run_batch(source, output, CopyTransform())

    assert result.failed_count == 1
    assert final.read_bytes() == b"existing"


def test_run_batch_continues_independent_jobs_and_cleans_partials(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    shutil.copyfile(simple_mcap, source / "good.mcap")
    output = tmp_path / "output"

    result = run_batch(
        source,
        output,
        CopyTransform(fail_name="run.mcap"),
        continue_on_error=True,
    )

    assert result.failed_count == 1
    assert (output / "good.mcap").is_file()
    assert not (output / "nested" / "run.mcap").exists()
    assert list(output.rglob("*.partial-*")) == []


def test_run_batch_rejects_source_changed_during_transform(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"

    result = run_batch(source, output, CopyTransform(mutate_source=True))

    assert result.failed_count == 1
    assert "source changed" in result.jobs[0].detail
    assert not (output / "nested" / "run.mcap").exists()
    assert list(output.rglob("*.partial-*")) == []


def test_run_batch_excludes_nested_output_tree(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = source / "compressed"
    output.mkdir()
    shutil.copyfile(simple_mcap, output / "old.mcap")
    transform = CopyTransform()

    result = run_batch(source, output, transform)

    assert result.failed_count == 0
    assert transform.run_count == 1
    assert [job.relative_path.as_posix() for job in result.jobs] == ["nested/run.mcap"]


def test_run_batch_keyboard_interrupt_cleans_owned_partial(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"

    with pytest.raises(KeyboardInterrupt):
        run_batch(source, output, CopyTransform(interrupt=True))

    assert not (output / "nested" / "run.mcap").exists()
    assert not (output / ".pymcap-roscompress-archive.jsonl").exists()
    assert list(output.rglob("*.partial-*")) == []


def test_run_batch_concurrent_jobs_use_independent_partials(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"
    barrier = threading.Barrier(2)
    partials: set[Path] = set()
    partials_lock = threading.Lock()

    def transform(input_path: Path, partial: Path) -> int:
        shutil.copyfile(input_path, partial)
        with partials_lock:
            partials.add(partial)
        barrier.wait()
        return 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_batch_core,
                source,
                output,
                transform,
                recipe=_COPY_RECIPE,
                force=True,
            )
            for _ in range(2)
        ]
        results = [future.result() for future in futures]

    assert len(partials) == 2
    assert all(result.failed_count == 0 for result in results)
    assert list(output.rglob("*.partial-*")) == []


def test_run_batch_force_preserves_existing_output_when_transform_fails(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"
    final = output / "nested" / "run.mcap"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"existing")

    result = run_batch(source, output, CopyTransform(fail_run=True), force=True)

    assert result.failed_count == 1
    assert final.read_bytes() == b"existing"
    assert list(output.rglob("*.partial-*")) == []


def test_run_batch_rejects_output_failing_lightweight_validation(
    tmp_path: Path,
    simple_mcap: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"
    monkeypatch.setattr(
        batch,
        "validate_mcap_outputs",
        lambda *_args, **_kwargs: "output failed MCAP validation",
    )

    result = run_batch(source, output, CopyTransform())

    assert result.failed_count == 1
    assert "output failed MCAP validation" in result.jobs[0].detail
    assert not (output / "nested" / "run.mcap").exists()
    assert list(output.rglob("*.partial-*")) == []


def test_run_batch_atomically_replaces_hard_link_without_changing_source(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source_root = _source_tree(tmp_path, simple_mcap)
    source = source_root / "nested" / "run.mcap"
    output = tmp_path / "output"
    final = output / "nested" / "run.mcap"
    final.parent.mkdir(parents=True)
    final.hardlink_to(source)

    result = run_batch(source_root, output, CopyTransform(), force=True)

    assert result.failed_count == 0
    assert source.read_bytes() == simple_mcap.read_bytes()
    assert final.read_bytes() == simple_mcap.read_bytes()
    assert not final.samefile(source)


def test_run_batch_ignores_truncated_archive_tail(
    tmp_path: Path,
    simple_mcap: Path,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"
    transform = CopyTransform()
    assert run_batch(source, output, transform).failed_count == 0
    archive = output / ".pymcap-roscompress-archive.jsonl"
    with archive.open("a", encoding="utf-8") as stream:
        stream.write('{"recipe":')

    result = run_batch(source, output, transform)

    assert result.jobs[0].status == "verified-resumed"
    assert transform.run_count == 1


def test_run_batch_relies_on_writes_for_available_space(
    tmp_path: Path,
    simple_mcap: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_tree(tmp_path, simple_mcap)
    output = tmp_path / "output"
    transform = CopyTransform()
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: usage._replace(free=simple_mcap.stat().st_size - 1),
    )

    result = run_batch(source, output, transform)

    assert result.failed_count == 0
    assert transform.run_count == 1
