from __future__ import annotations

import io
from typing import TYPE_CHECKING

from pymcap_cli.cmd import diff_cmd
from rich.console import Console
from small_mcap import CompressionType, McapWriter

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _write_custom_mcap(path: Path, *, timestamps: list[int]) -> None:
    with path.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=1024, compression=CompressionType.ZSTD)
        writer.start(profile="ros2", library="diff-test")
        writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
        writer.add_channel(
            channel_id=1,
            topic="/test",
            message_encoding="json",
            schema_id=1,
        )
        for index, log_time in enumerate(timestamps):
            writer.add_message(
                channel_id=1,
                log_time=log_time,
                publish_time=log_time,
                data=f'{{"i": {index}}}'.encode(),
            )
        writer.finish()


def _run_diff(
    files: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    *,
    skip_identical: bool = False,
) -> tuple[int, str]:
    output = io.StringIO()
    monkeypatch.setattr(
        diff_cmd,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=180),
    )
    exit_code = diff_cmd.diff_cmd(
        [str(file) for file in files],
        skip_identical=skip_identical,
    )
    return exit_code, output.getvalue()


def test_diff_same_basename_uses_unique_labels(
    image_small_mcap: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, rendered = _run_diff(
        [image_small_mcap, image_small_mcap],
        monkeypatch,
        skip_identical=True,
    )

    assert exit_code == 0
    assert "Smart Diff Verdict" in rendered
    assert "exact indexed duplicate" in rendered
    assert f"{image_small_mcap.name}#2" in rendered
    assert "identical timestamps and schemas" in rendered


def test_diff_reports_cutout_smart_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = tmp_path / "full.mcap"
    cutout = tmp_path / "cutout.mcap"
    _write_custom_mcap(full, timestamps=[index * 1_000_000 for index in range(10)])
    _write_custom_mcap(cutout, timestamps=[index * 1_000_000 for index in range(3, 7)])

    exit_code, rendered = _run_diff([full, cutout], monkeypatch)

    assert exit_code == 0
    assert "Smart Diff Verdict" in rendered
    assert "cutout.mcap is contained in full.mcap" in rendered
    assert "4 shared msgs" in rendered
    assert "left-only 6 msgs" in rendered
    assert "right-only 0 msgs" in rendered
    assert "/test" in rendered


def test_diff_reports_edge_overlap_smart_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left.mcap"
    right = tmp_path / "right.mcap"
    _write_custom_mcap(left, timestamps=[index * 1_000_000 for index in range(10)])
    _write_custom_mcap(right, timestamps=[index * 1_000_000 for index in range(7, 15)])

    exit_code, rendered = _run_diff([left, right], monkeypatch)

    assert exit_code == 0
    assert "edge overlap" in rendered
    assert "3 shared msgs" in rendered
    assert "left-only 7 msgs" in rendered
    assert "right-only 5 msgs" in rendered


def test_diff_compares_three_files_pairwise_to_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = tmp_path / "full.mcap"
    cutout_a = tmp_path / "cutout_a.mcap"
    cutout_b = tmp_path / "cutout_b.mcap"
    _write_custom_mcap(full, timestamps=[index * 1_000_000 for index in range(10)])
    _write_custom_mcap(cutout_a, timestamps=[index * 1_000_000 for index in range(5)])
    _write_custom_mcap(cutout_b, timestamps=[index * 1_000_000 for index in range(5, 10)])

    exit_code, rendered = _run_diff([full, cutout_a, cutout_b], monkeypatch)

    assert exit_code == 0
    assert "full.mcap <-> cutout_a.mcap" in rendered
    assert "full.mcap <-> cutout_b.mcap" in rendered
    assert "cutout_a.mcap <-> cutout_b.mcap" not in rendered
