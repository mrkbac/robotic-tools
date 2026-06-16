"""roscompress must refuse to overwrite its own input file."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.cmd import roscompress_cmd
from pymcap_cli.cmd.roscompress_cmd import roscompress

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_roscompress_rejects_output_equal_to_input(
    simple_mcap: Path, tmp_path: Path, monkeypatch
) -> None:
    """`roscompress in.mcap -o /abs/in.mcap` must not truncate the source.

    Input is relative and output absolute; they only collide after resolution.
    """
    src = tmp_path / "data.mcap"
    src.write_bytes(simple_mcap.read_bytes())
    orig = src.read_bytes()
    monkeypatch.chdir(tmp_path)

    rc = roscompress(src.name, src.resolve(), force=True, image_format="none", pointcloud=False)

    assert rc == 1
    assert src.exists()
    assert src.read_bytes() == orig


def test_roscompress_removes_output_on_failure(
    simple_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed run must not leave a truncated/empty output file behind."""
    src = tmp_path / "data.mcap"
    src.write_bytes(simple_mcap.read_bytes())
    output = tmp_path / "out.mcap"

    monkeypatch.setattr(roscompress_cmd, "_run_compress_loop", lambda *_a, **_k: False)

    rc = roscompress(str(src), output, force=True, image_format="none", pointcloud=False)

    assert rc == 1
    assert not output.exists()
    assert src.exists()
