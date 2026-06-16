"""split must refuse to open a segment that would truncate its input."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.cmd.split_cmd import split

if TYPE_CHECKING:
    from pathlib import Path


def test_split_rejects_output_template_equal_to_input(simple_mcap: Path, tmp_path: Path) -> None:
    """An output template that resolves to the input file must not truncate it."""
    src = tmp_path / "data.mcap"
    src.write_bytes(simple_mcap.read_bytes())
    orig = src.read_bytes()

    # Template has no index placeholder, so every segment resolves to the input path.
    rc = split(file=str(src), duration="1s", output_template=str(src), force=True)

    assert rc == 1
    assert src.read_bytes() == orig
