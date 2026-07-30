"""Regression tests for package selection in scripts/bump.sh."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

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
    (repo / "uv.lock").write_text("version = 1\nrevision = 3\n")
    (package / "pyproject.toml").write_text('[project]\nname = "sample"\nversion = "1.0.0"\n')
    (package / "src" / "sample").mkdir(parents=True)
    (package / "src" / "sample" / "__init__.py").write_text("")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "tag", "sample@1.0.0")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pre_commit = fake_bin / "pre-commit"
    pre_commit.write_text("#!/bin/sh\nexit 0\n")
    pre_commit.chmod(0o755)
    return repo, fake_bin


def _create_internal_dependency_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(BUMP_SCRIPT, repo / "scripts" / "bump.sh")
    (repo / "pyproject.toml").write_text(
        "[tool.uv.workspace]\nmembers = [\n"
        '    "pureini",\n'
        '    "mcap-codec-support",\n'
        '    "pymcap-cli",\n'
        "]\n"
        "\n[tool.uv.sources]\n"
        "pureini = { workspace = true }\n"
        "mcap-codec-support = { workspace = true }\n"
    )
    packages = {
        "pureini": (
            "0.8.0",
            "",
        ),
        "mcap-codec-support": (
            "0.14.0",
            '\n[project.optional-dependencies]\npointcloud = ["pureini>=0.8.0"]\n',
        ),
        "pymcap-cli": (
            "0.26.0",
            "\n[project.optional-dependencies]\n"
            'pointcloud = ["mcap-codec-support[pointcloud]>=0.14.0"]\n',
        ),
    }
    for package_name, (version, extra_metadata) in packages.items():
        package = repo / package_name
        module_name = package_name.replace("-", "_")
        (package / "src" / module_name).mkdir(parents=True)
        (package / "src" / module_name / "__init__.py").write_text("")
        (package / "pyproject.toml").write_text(
            "[project]\n"
            f'name = "{package_name}"\n'
            f'version = "{version}"\n'
            'requires-python = ">=3.10"\n'
            f"{extra_metadata}"
        )

    subprocess.run(
        ["uv", "lock"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    for package_name, (version, _) in packages.items():
        _git(repo, "tag", f"{package_name}@{version}")

    for package_name in packages:
        module_name = package_name.replace("-", "_")
        (repo / package_name / "src" / module_name / "core.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "native codec integration")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pre_commit = fake_bin / "pre-commit"
    pre_commit.write_text("#!/bin/sh\nexit 0\n")
    pre_commit.chmod(0o755)
    return repo, fake_bin


def _run_bump(
    repo: Path,
    fake_bin: Path,
    *,
    answer: str = "n\n",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", "scripts/bump.sh"],
        cwd=repo,
        env=env,
        input=answer,
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


def test_bump_detects_readme_changes(tmp_path: Path) -> None:
    repo, fake_bin = _create_repo(tmp_path)
    (repo / "sample" / "README.md").write_text("Updated package documentation.\n")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "update readme",
    )

    result = _run_bump(repo, fake_bin)

    assert result.returncode == 0, result.stderr
    assert "⬆ sample (1.0.0) has changes" in result.stdout


def test_bump_detects_dependency_changes(tmp_path: Path) -> None:
    repo, fake_bin = _create_repo(tmp_path)
    (repo / "sample" / "pyproject.toml").write_text(
        '[project]\nname = "sample"\nversion = "1.0.0"\ndependencies = ["example>=2"]\n'
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
        "add dependency",
    )

    result = _run_bump(repo, fake_bin)

    assert result.returncode == 0, result.stderr
    assert "⬆ sample (1.0.0) has changes" in result.stdout


def test_bump_detects_extra_changes(tmp_path: Path) -> None:
    repo, fake_bin = _create_repo(tmp_path)
    (repo / "sample" / "pyproject.toml").write_text(
        '[project]\nname = "sample"\nversion = "1.0.0"\n'
        '[project.optional-dependencies]\nfeature = ["example>=2"]\n'
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
        "add extra",
    )

    result = _run_bump(repo, fake_bin)

    assert result.returncode == 0, result.stderr
    assert "⬆ sample (1.0.0) has changes" in result.stdout


def test_bump_ignores_only_project_version_line(tmp_path: Path) -> None:
    repo, fake_bin = _create_repo(tmp_path)
    (repo / "sample" / "pyproject.toml").write_text(
        '[project]\nname = "sample"\nversion = "1.0.0"\n[tool.example]\nversion = "2.0.0"\n'
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
        "update tool metadata",
    )

    result = _run_bump(repo, fake_bin)

    assert result.returncode == 0, result.stderr
    assert "⬆ sample (1.0.0) has changes" in result.stdout


def test_bump_ignores_version_only_change(tmp_path: Path) -> None:
    repo, fake_bin = _create_repo(tmp_path)
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
        "bump version",
    )

    result = _run_bump(repo, fake_bin)

    assert result.returncode == 0, result.stderr
    assert "— sample: version 1.1.0 is already awaiting a tag, skipping" in result.stdout


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


def test_bump_updates_internal_dependency_floors(tmp_path: Path) -> None:
    repo, fake_bin = _create_internal_dependency_repo(tmp_path)

    result = _run_bump(repo, fake_bin, answer="y\n")

    assert result.returncode == 0, result.stderr
    assert 'version = "0.9.0"' in (repo / "pureini" / "pyproject.toml").read_text()
    codec_pyproject = tomllib.loads((repo / "mcap-codec-support" / "pyproject.toml").read_text())
    assert codec_pyproject["project"]["version"] == "0.15.0"
    assert codec_pyproject["project"]["optional-dependencies"]["pointcloud"] == ["pureini>=0.9.0"]
    cli_pyproject = tomllib.loads((repo / "pymcap-cli" / "pyproject.toml").read_text())
    assert cli_pyproject["project"]["version"] == "0.27.0"
    assert cli_pyproject["project"]["optional-dependencies"]["pointcloud"] == [
        "mcap-codec-support[pointcloud]>=0.15.0"
    ]
    assert not _git_status(repo)


def _git_status(repo: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout
