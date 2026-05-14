from __future__ import annotations

import io
from typing import TYPE_CHECKING

from pymcap_cli.cmd import diff_cmd
from pymcap_cli.constants import NS_TO_MS
from rich.console import Console
from small_mcap import CompressionType, McapWriter, get_summary

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _write_custom_mcap(
    path: Path,
    *,
    timestamps: list[int],
    channel_id: int = 1,
    payloads: list[bytes] | None = None,
    publish_times: list[int] | None = None,
    sequences: list[int] | None = None,
    compression: CompressionType = CompressionType.ZSTD,
) -> None:
    with path.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=1024, compression=compression)
        writer.start(profile="ros2", library="diff-test")
        writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
        writer.add_channel(
            channel_id=channel_id,
            topic="/test",
            message_encoding="json",
            schema_id=1,
        )
        for index, log_time in enumerate(timestamps):
            payload = payloads[index] if payloads is not None else f'{{"i": {index}}}'.encode()
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                publish_time=publish_times[index] if publish_times is not None else log_time,
                sequence=sequences[index] if sequences is not None else 0,
                data=payload,
            )
        writer.finish()


def _corrupt_first_chunk_data(path: Path) -> None:
    with path.open("rb") as stream:
        summary = get_summary(stream)
    assert summary is not None
    chunk_index = summary.chunk_indexes[0]
    data_offset = (
        chunk_index.chunk_start_offset
        + 9
        + 8
        + 8
        + 8
        + 4
        + 4
        + len(chunk_index.compression.encode())
        + 8
    )
    with path.open("r+b") as stream:
        stream.seek(data_offset)
        original = stream.read(1)
        assert original
        stream.seek(data_offset)
        stream.write(bytes([original[0] ^ 0x01]))


def _run_diff(
    files: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    *,
    skip_identical: bool = False,
    compare_payloads: bool = False,
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
        compare_payloads=compare_payloads,
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
    _write_custom_mcap(full, timestamps=[index * NS_TO_MS for index in range(10)])
    _write_custom_mcap(cutout, timestamps=[index * NS_TO_MS for index in range(3, 7)])

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
    _write_custom_mcap(left, timestamps=[index * NS_TO_MS for index in range(10)])
    _write_custom_mcap(right, timestamps=[index * NS_TO_MS for index in range(7, 15)])

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
    _write_custom_mcap(full, timestamps=[index * NS_TO_MS for index in range(10)])
    _write_custom_mcap(cutout_a, timestamps=[index * NS_TO_MS for index in range(5)])
    _write_custom_mcap(cutout_b, timestamps=[index * NS_TO_MS for index in range(5, 10)])

    exit_code, rendered = _run_diff([full, cutout_a, cutout_b], monkeypatch)

    assert exit_code == 0
    assert "full.mcap <-> cutout_a.mcap" in rendered
    assert "full.mcap <-> cutout_b.mcap" in rendered
    assert "cutout_a.mcap <-> cutout_b.mcap" not in rendered


def test_diff_compare_payloads_detects_bit_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left.mcap"
    right = tmp_path / "right.mcap"
    timestamps = [index * NS_TO_MS for index in range(5)]
    left_payloads = [f'{{"i": {index}}}'.encode() for index in range(5)]
    right_payloads = left_payloads.copy()
    right_payloads[3] = b'{"i": 999}'
    _write_custom_mcap(left, timestamps=timestamps, payloads=left_payloads)
    _write_custom_mcap(right, timestamps=timestamps, payloads=right_payloads)

    exit_code, rendered = _run_diff([left, right], monkeypatch)

    assert exit_code == 0
    assert "exact indexed duplicate" in rendered

    exit_code, rendered = _run_diff([left, right], monkeypatch, compare_payloads=True)

    assert exit_code == 0
    assert "payload mismatch" in rendered
    assert "first mismatch" in rendered
    assert "message payloads differ" in rendered


def test_diff_compare_payloads_allows_different_channel_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left.mcap"
    right = tmp_path / "right.mcap"
    timestamps = [index * NS_TO_MS for index in range(5)]
    _write_custom_mcap(left, timestamps=timestamps, channel_id=1)
    _write_custom_mcap(right, timestamps=timestamps, channel_id=7)

    exit_code, rendered = _run_diff([left, right], monkeypatch, compare_payloads=True)

    assert exit_code == 0
    assert "payload mismatch" not in rendered
    assert "identical timestamps, schemas and payloads" in rendered


def test_diff_compare_payloads_detects_publish_time_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left.mcap"
    right = tmp_path / "right.mcap"
    timestamps = [index * NS_TO_MS for index in range(5)]
    right_publish_times = timestamps.copy()
    right_publish_times[2] += 1
    _write_custom_mcap(left, timestamps=timestamps)
    _write_custom_mcap(right, timestamps=timestamps, publish_times=right_publish_times)

    exit_code, rendered = _run_diff([left, right], monkeypatch, compare_payloads=True)

    assert exit_code == 0
    assert "payload mismatch" in rendered
    assert "publish_time" in rendered


def test_diff_compare_payloads_returns_error_on_unreadable_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left.mcap"
    right = tmp_path / "right.mcap"
    timestamps = [index * NS_TO_MS for index in range(5)]
    _write_custom_mcap(left, timestamps=timestamps, compression=CompressionType.NONE)
    _write_custom_mcap(right, timestamps=timestamps, compression=CompressionType.NONE)
    _corrupt_first_chunk_data(right)

    exit_code, _rendered = _run_diff([left, right], monkeypatch, compare_payloads=True)

    assert exit_code == 1
