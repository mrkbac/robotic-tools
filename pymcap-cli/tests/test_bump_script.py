"""Regression tests for package selection in scripts/bump.sh."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

BUMP_SCRIPT = Path(__file__).parents[2] / "scripts" / "bump.sh"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _create_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    package = repo / "sample"
    package.mkdir(parents=True)
    (repo / "scripts").mkdir()
    shutil.copy(BUMP_SCRIPT, repo / "scripts" / "bump.sh")
    (repo / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = [\n    "sample",\n]\n')
    (package / "pyproject.toml").write_text('[project]\nname = "sample"\nversion = "1.0.0"\n')
    (package / "src" / "sample").mkdir(parents=True)
    (package / "src" / "sample" / "__init__.py").write_text("")

    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )
    _git(repo, "tag", "sample@1.0.0")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pre_commit = fake_bin / "pre-commit"
    pre_commit.write_text("#!/bin/sh\nexit 0\n")
    pre_commit.chmod(0o755)
    return repo, fake_bin


def _run_bump(repo: Path, fake_bin: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", "scripts/bump.sh"],
        cwd=repo,
        env=env,
        input="n\n",
        capture_output=True,
        text=True,
        check=False,
    )


def test_bump_ignores_tests_only_changes(tmp_path: Path) -> None:
    repo, fake_bin = _create_repo(tmp_path)
    test_file = repo / "sample" / "tests" / "test_sample.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_sample():\n    assert True\n")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "test only",
    )

    result = _run_bump(repo, fake_bin)

    assert result.returncode == 0, result.stderr
    assert "— sample: no changes, skipping" in result.stdout


def test_bump_detects_source_changes(tmp_path: Path) -> None:
    repo, fake_bin = _create_repo(tmp_path)
    (repo / "sample" / "src" / "sample" / "core.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "source change",
    )

    result = _run_bump(repo, fake_bin)

    assert result.returncode == 0, result.stderr
    assert "⬆ sample (1.0.0) has changes" in result.stdout


def test_bump_does_not_double_bump_untagged_release(tmp_path: Path) -> None:
    repo, fake_bin = _create_repo(tmp_path)
    (repo / "sample" / "src" / "sample" / "core.py").write_text("VALUE = 1\n")
    (repo / "sample" / "pyproject.toml").write_text(
        '[project]\nname = "sample"\nversion = "1.1.0"\n'
    )
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "prepare release",
    )

    result = _run_bump(repo, fake_bin)

    assert result.returncode == 0, result.stderr
    assert "— sample: version 1.1.0 is already awaiting a tag, skipping" in result.stdout
