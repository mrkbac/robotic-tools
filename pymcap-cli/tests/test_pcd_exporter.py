from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pymcap_cli.exporters import pcd_exporter
from pymcap_cli.exporters.pcd_exporter import write_pcd_ascii

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class _Field:
    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass(frozen=True)
class _Cloud:
    height: int
    width: int
    fields: tuple[_Field, ...]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool


def _data_rows(path: Path) -> list[str]:
    lines = path.read_text(encoding="ascii").splitlines()
    return lines[lines.index("DATA ascii") + 1 :]


def test_write_pcd_ascii_honors_organized_row_padding(tmp_path: Path) -> None:
    row_padding = b"\xff" * 4
    cloud = _Cloud(
        height=2,
        width=2,
        fields=(_Field("x", 0, 7),),
        is_bigendian=False,
        point_step=4,
        row_step=12,
        data=struct.pack("<ff", 1.0, 2.0)
        + row_padding
        + struct.pack("<ff", 3.0, 4.0)
        + row_padding,
        is_dense=True,
    )
    output = tmp_path / "organized.pcd"

    write_pcd_ascii(output, cloud)

    assert _data_rows(output) == ["1", "2", "3", "4"]


def test_write_pcd_ascii_decodes_big_endian_and_expands_field_counts(tmp_path: Path) -> None:
    cloud = _Cloud(
        height=1,
        width=1,
        fields=(_Field("normal", 0, 7, count=2), _Field("ring", 8, 4)),
        is_bigendian=True,
        point_step=10,
        row_step=10,
        data=struct.pack(">ffH", 1.25, -2.5, 513),
        is_dense=True,
    )
    output = tmp_path / "big-endian.pcd"

    write_pcd_ascii(output, cloud)

    text = output.read_text(encoding="ascii")
    assert "FIELDS normal_0 normal_1 ring\n" in text
    assert "SIZE 4 4 2\n" in text
    assert "TYPE F F U\n" in text
    assert _data_rows(output) == ["1.25 -2.5 513"]


def test_write_pcd_ascii_drops_nan_points_from_non_dense_cloud(tmp_path: Path) -> None:
    cloud = _Cloud(
        height=1,
        width=3,
        fields=(_Field("x", 0, 7), _Field("intensity", 4, 7)),
        is_bigendian=False,
        point_step=8,
        row_step=24,
        data=b"".join(
            struct.pack("<ff", x, intensity)
            for x, intensity in ((1.0, 2.0), (math.nan, 3.0), (4.0, 5.0))
        ),
        is_dense=False,
    )
    output = tmp_path / "filtered.pcd"

    write_pcd_ascii(output, cloud)

    text = output.read_text(encoding="ascii")
    assert "WIDTH 2\n" in text
    assert "POINTS 2\n" in text
    assert _data_rows(output) == ["1 2", "4 5"]


def test_write_pcd_ascii_uses_packed_row_step_when_unset(tmp_path: Path) -> None:
    cloud = _Cloud(
        height=1,
        width=3,
        fields=(_Field("x", 0, 7),),
        is_bigendian=False,
        point_step=4,
        row_step=0,
        data=struct.pack("<fff", 1.0, 2.0, 3.0),
        is_dense=True,
    )
    output = tmp_path / "unset-row-step.pcd"

    write_pcd_ascii(output, cloud)

    assert _data_rows(output) == ["1", "2", "3"]


def test_write_pcd_ascii_ignores_trailing_data(tmp_path: Path) -> None:
    cloud = _Cloud(
        height=1,
        width=3,
        fields=(_Field("x", 0, 7),),
        is_bigendian=False,
        point_step=4,
        row_step=12,
        data=struct.pack("<fff", 1.0, 2.0, 3.0) + b"\xff" * 4,
        is_dense=True,
    )
    output = tmp_path / "trailing-data.pcd"

    write_pcd_ascii(output, cloud)

    assert _data_rows(output) == ["1", "2", "3"]


def test_write_pcd_ascii_formats_integer_fields_as_decimal(tmp_path: Path) -> None:
    cloud = _Cloud(
        height=1,
        width=1,
        fields=(
            _Field("x", 0, 7),
            _Field("rgb", 4, 6),
            _Field("stamp", 8, 6),
        ),
        is_bigendian=False,
        point_step=12,
        row_step=12,
        data=struct.pack("<fII", 1.0, 0xFF8040, 1_234_567_890),
        is_dense=True,
    )
    output = tmp_path / "integer-fields.pcd"

    write_pcd_ascii(output, cloud)

    assert _data_rows(output) == ["1 16744512 1234567890"]


def test_write_pcd_ascii_iterates_non_dense_cloud_once(monkeypatch, tmp_path: Path) -> None:
    cloud = _Cloud(
        height=1,
        width=2,
        fields=(_Field("x", 0, 7),),
        is_bigendian=False,
        point_step=4,
        row_step=8,
        data=struct.pack("<ff", 1.0, 2.0),
        is_dense=False,
    )
    output = tmp_path / "single-pass.pcd"
    original_iter_rows = pcd_exporter._iter_rows
    calls = 0

    def tracked_iter_rows(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_iter_rows(*args, **kwargs)

    monkeypatch.setattr(pcd_exporter, "_iter_rows", tracked_iter_rows)

    write_pcd_ascii(output, cloud)

    assert calls == 1
    assert _data_rows(output) == ["1", "2"]
