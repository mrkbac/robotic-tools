from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from pymcap_cli.cmd import tftree_cmd

from tests.fixtures.mcap_generator import create_tf_mcap

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from rich.table import Table


def test_tftree_throttles_table_rebuilds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bag = tmp_path / "dynamic.mcap"
    bag.write_bytes(
        create_tf_mcap(
            dynamic_edges=[("base", "child", (1.0, 0.0, 0.0))],
            dynamic_samples=50,
        )
    )

    real_build_tf_table = tftree_cmd.build_tf_table
    build_count = 0

    def track_build_tf_table(*args, **kwargs) -> Table | None:
        nonlocal build_count
        build_count += 1
        return real_build_tf_table(*args, **kwargs)

    monkeypatch.setattr(tftree_cmd, "build_tf_table", track_build_tf_table)

    assert tftree_cmd.tftree(str(bag)) == 0
    capsys.readouterr()
    assert build_count <= 3


def test_tftree_prefetches_chunk_decompression(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bag = tmp_path / "static.mcap"
    bag.write_bytes(
        create_tf_mcap(
            static_edges=[("base", "child", (1.0, 0.0, 0.0))],
        )
    )

    with patch.object(
        tftree_cmd,
        "read_message_decoded",
        wraps=tftree_cmd.read_message_decoded,
    ) as read_message_decoded:
        assert tftree_cmd.tftree(str(bag)) == 0

    capsys.readouterr()
    assert read_message_decoded.call_args is not None
    assert read_message_decoded.call_args.kwargs["num_workers"] == 8
