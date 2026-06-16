"""roscompress must refuse to overwrite its own input file."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.cmd.roscompress_cmd import roscompress

if TYPE_CHECKING:
    from pathlib import Path


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
