from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pymcap_cli.core.batch import TransformResult, run_batch

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class CopyTransform:
    run_count: int = 0
    fail_name: str | None = None
    mutate_source: bool = False

    def recipe(self):
        return {"operation": "copy", "version": 1}

    def preflight(self, source: Path) -> None:
        if source.name == self.fail_name:
            raise RuntimeError("preflight failed")

    def run(self, source: Path, partial: Path) -> TransformResult:
        self.run_count += 1
        shutil.copyfile(source, partial)
        if self.mutate_source:
            source.write_bytes(source.read_bytes() + b"changed")
        return TransformResult()

    def validate(
        self,
        source: Path,
        partial: Path,
        result: TransformResult,
    ) -> None:
        assert source.exists()
        assert partial.exists()
        assert result == TransformResult()


def _source_tree(tmp_path: Path, simple_mcap: Path) -> Path:
    source = tmp_path / "input"
    (source / "nested").mkdir(parents=True)
    shutil.copyfile(simple_mcap, source / "nested" / "run.mcap")
    return source


def test_run_batch_commits_output_and_archive_atomically(
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
    assert (output / ".pymcap-roscompress-archive.jsonl").is_file()
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
