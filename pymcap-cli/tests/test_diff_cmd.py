from __future__ import annotations

import io
from typing import TYPE_CHECKING

from pymcap_cli.cmd import diff_cmd
from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path


def test_diff_same_basename_uses_unique_labels(image_small_mcap: Path, monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(
        diff_cmd,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=180),
    )

    exit_code = diff_cmd.diff_cmd(
        [str(image_small_mcap), str(image_small_mcap)],
        skip_identical=True,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert f"{image_small_mcap.name}#2" in rendered
    assert "identical timestamps and schemas" in rendered
