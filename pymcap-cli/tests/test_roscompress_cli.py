from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.cmd.roscompress_cmd import roscompress

if TYPE_CHECKING:
    from pathlib import Path


def test_roscompress_accepts_explicit_auto_backend(simple_mcap: Path, tmp_path: Path) -> None:
    output = tmp_path / "compressed.mcap"

    result = roscompress(
        str(simple_mcap),
        output,
        force=True,
        backend="auto",
        pointcloud=False,
    )

    assert result == 0
