"""Unit tests for the shared cross-argument constraint validators."""

from typing import Annotated

import cyclopts
import pytest
from cyclopts import App, Parameter
from pymcap_cli.cmd._arg_constraints import (
    at_least_one,
    conflicts,
    constraint_group,
    each_requires,
    requires,
    requires_value,
)


def _run(validators, argv: list[str]) -> None:
    group = constraint_group(*validators)
    app = App(name="t")

    @app.command
    def go(
        *,
        grep: Annotated[str | None, Parameter(name="--grep", group=group)] = None,
        ignore_case: Annotated[
            bool, Parameter(name=["-i", "--grep-ignore-case"], group=group)
        ] = False,
        query: Annotated[str | None, Parameter(name="--query", group=group)] = None,
        image_format: Annotated[str, Parameter(name="--image-format", group=group)] = "video",
        quality: Annotated[int, Parameter(name="--quality", group=group)] = 28,
        pointcloud: Annotated[bool, Parameter(name="--pointcloud", group=group)] = True,
        resolution: Annotated[float, Parameter(name="--resolution", group=group)] = 0.01,
    ) -> None:
        _ = grep, ignore_case, query, image_format, quality, pointcloud, resolution

    try:
        app(["go", *argv], exit_on_error=False)
    except SystemExit as exc:  # cyclopts exits 0 on a successful command run
        if exc.code not in (0, None):
            raise


def test_requires_errors_when_dependent_supplied_without_required():
    with pytest.raises(cyclopts.ValidationError, match="--grep-ignore-case requires --grep"):
        _run([requires("--grep-ignore-case", "--grep")], ["--grep-ignore-case"])


def test_requires_passes_when_required_present():
    _run([requires("--grep-ignore-case", "--grep")], ["--grep", "x", "--grep-ignore-case"])


def test_requires_passes_when_dependent_absent():
    _run([requires("--grep-ignore-case", "--grep")], ["--grep", "x"])


def test_requires_matches_on_all_names():
    # -i is an alias of --grep-ignore-case; the dependency must still trigger.
    with pytest.raises(cyclopts.ValidationError, match="requires --grep"):
        _run([requires("--grep-ignore-case", "--grep")], ["-i"])


def test_each_requires_reports_all_present_dependents():
    with pytest.raises(
        cyclopts.ValidationError, match="--grep, --query require --grep-ignore-case"
    ):
        _run(
            [each_requires("--grep-ignore-case", "--grep", "--query")],
            ["--grep", "x", "--query", "/t"],
        )


def test_at_least_one_errors_when_none_supplied():
    with pytest.raises(cyclopts.ValidationError, match="Specify at least one of"):
        _run([at_least_one], [])


def test_at_least_one_passes_when_one_supplied():
    _run([at_least_one], ["--query", "/t"])


def test_conflicts_errors_on_clash():
    with pytest.raises(cyclopts.ValidationError, match="--grep is incompatible with --query"):
        _run([conflicts("--grep", "--query")], ["--grep", "x", "--query", "/t"])


def test_conflicts_allows_non_clashing():
    _run([conflicts("--grep", "--query")], ["--grep", "x"])


def test_requires_value_errors_when_controller_set_to_disallowed():
    with pytest.raises(cyclopts.ValidationError, match="--quality requires --image-format video"):
        _run(
            [requires_value("--quality", "--image-format", "video", hint="--image-format video")],
            ["--image-format", "none", "--quality", "20"],
        )


def test_requires_value_allows_default_controller():
    # --image-format defaults to "video"; supplying --quality alone is fine.
    _run(
        [requires_value("--quality", "--image-format", "video", hint="--image-format video")],
        ["--quality", "20"],
    )


def test_requires_value_allows_matching_controller():
    _run(
        [requires_value("--quality", "--image-format", "video", hint="--image-format video")],
        ["--image-format", "video", "--quality", "20"],
    )


def test_requires_value_handles_bool_controller():
    with pytest.raises(cyclopts.ValidationError, match="--resolution requires --pointcloud"):
        _run(
            [requires_value("--resolution", "--pointcloud", True, hint="--pointcloud enabled")],
            ["--no-pointcloud", "--resolution", "0.05"],
        )


def test_requires_rejects_empty_required():
    with pytest.raises(ValueError, match="at least one required flag"):
        requires("--x")


def test_each_requires_rejects_empty_dependents():
    with pytest.raises(ValueError, match="at least one dependent flag"):
        each_requires("--x")


def test_conflicts_rejects_empty_others():
    with pytest.raises(ValueError, match="at least one conflicting flag"):
        conflicts("--x")
