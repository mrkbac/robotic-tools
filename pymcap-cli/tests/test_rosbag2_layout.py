"""Tests for rosbag2 split-directory discovery and aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd import info_cmd, merge_cmd
from pymcap_cli.core.input_handler import resolve_mcap_path
from pymcap_cli.core.rosbag2_layout import (
    expand_bag_paths,
    find_bag_splits,
    read_aggregated_bag_info,
)
from pymcap_cli.types.info_data import info_to_dict

from tests.fixtures.mcap_generator import create_multi_topic_mcap

if TYPE_CHECKING:
    from pathlib import Path

NS_TO_MS = 1_000_000


def _write_bag(directory: Path, name: str, split_messages: list[int]) -> Path:
    """Create <directory>/<name>/ with one .mcap split per entry in split_messages.

    Split 0 carries topic /a only; later splits carry /a and /b so channel ids
    stay stable across splits (recorder reuses ids).
    """
    bag = directory / name
    bag.mkdir()
    for index, msgs in enumerate(split_messages):
        topics = ["/a"] if index == 0 else ["/a", "/b"]
        data = create_multi_topic_mcap(topics, messages_per_topic=msgs)
        (bag / f"{name}_{index}.mcap").write_bytes(data)
    return bag


# ---------------------------------------------------------------------------
# find_bag_splits
# ---------------------------------------------------------------------------


def test_find_bag_splits_orders_by_integer_index(tmp_path: Path) -> None:
    bag = tmp_path / "rec"
    bag.mkdir()
    for index in (0, 1, 2, 10):
        (bag / f"rec_{index}.mcap").write_bytes(b"x")

    result = [p.name for p in find_bag_splits(bag)]
    assert result == ["rec_0.mcap", "rec_1.mcap", "rec_2.mcap", "rec_10.mcap"]


def test_find_bag_splits_ignores_foreign_names(tmp_path: Path) -> None:
    bag = tmp_path / "rec"
    bag.mkdir()
    (bag / "rec_0.mcap").write_bytes(b"x")
    (bag / "rec_part.mcap").write_bytes(b"x")  # non-numeric suffix
    (bag / "other.mcap").write_bytes(b"x")
    (bag / "rec_0.db3").write_bytes(b"x")

    assert [p.name for p in find_bag_splits(bag)] == ["rec_0.mcap"]


def test_find_bag_splits_falls_back_to_single_file(tmp_path: Path) -> None:
    bag = tmp_path / "rec"
    bag.mkdir()
    (bag / "rec.mcap").write_bytes(b"x")

    assert [p.name for p in find_bag_splits(bag)] == ["rec.mcap"]


def test_find_bag_splits_empty_directory(tmp_path: Path) -> None:
    bag = tmp_path / "rec"
    bag.mkdir()
    assert find_bag_splits(bag) == []


# ---------------------------------------------------------------------------
# expand_bag_paths
# ---------------------------------------------------------------------------


def test_expand_bag_paths_splices_splits_in_order(tmp_path: Path) -> None:
    bag = _write_bag(tmp_path, "rec", [5, 5])
    plain = tmp_path / "plain.mcap"
    plain.write_bytes(create_multi_topic_mcap(["/c"], messages_per_topic=3))

    result = expand_bag_paths([str(plain), str(bag), "https://x/y.mcap"])
    assert result == [
        str(plain),
        str(bag / "rec_0.mcap"),
        str(bag / "rec_1.mcap"),
        "https://x/y.mcap",
    ]


def test_expand_bag_paths_passes_plain_file_unchanged(tmp_path: Path) -> None:
    f = tmp_path / "rec.mcap"
    f.write_bytes(b"x")
    assert expand_bag_paths([str(f)]) == [str(f)]


def test_expand_bag_paths_empty_dir_raises(tmp_path: Path) -> None:
    bag = tmp_path / "rec"
    bag.mkdir()
    with pytest.raises(ValueError, match="rec"):
        expand_bag_paths([str(bag)])


# ---------------------------------------------------------------------------
# resolve_mcap_path (single-file chokepoint)
# ---------------------------------------------------------------------------


def test_resolve_mcap_path_single_split_dir(tmp_path: Path) -> None:
    bag = _write_bag(tmp_path, "rec", [5])
    assert resolve_mcap_path(str(bag)) == str(bag / "rec_0.mcap")


def test_resolve_mcap_path_multi_split_dir_raises(tmp_path: Path) -> None:
    bag = _write_bag(tmp_path, "rec", [5, 5])
    with pytest.raises(ValueError, match="multi-split"):
        resolve_mcap_path(str(bag))


def test_resolve_mcap_path_empty_dir_raises(tmp_path: Path) -> None:
    bag = tmp_path / "rec"
    bag.mkdir()
    with pytest.raises(ValueError, match="rec"):
        resolve_mcap_path(str(bag))


def test_resolve_mcap_path_plain_file_unchanged(tmp_path: Path) -> None:
    f = tmp_path / "rec.mcap"
    f.write_bytes(b"x")
    assert resolve_mcap_path(str(f)) == str(f)


# ---------------------------------------------------------------------------
# read_aggregated_bag_info
# ---------------------------------------------------------------------------


def test_read_aggregated_bag_info_sums_statistics(tmp_path: Path) -> None:
    # split0: 10 msgs on /a; split1: 15 msgs each on /a and /b (30 total).
    bag = _write_bag(tmp_path, "rec", [10, 15])
    info, total_bytes = read_aggregated_bag_info(find_bag_splits(bag))

    stats = info.summary.statistics
    assert stats is not None
    assert stats.message_count == 40
    assert stats.channel_count == 2
    assert stats.channel_message_counts == {1: 25, 2: 15}
    assert stats.message_start_time == 0
    assert stats.message_end_time == 29 * NS_TO_MS

    total_on_disk = sum(p.stat().st_size for p in find_bag_splits(bag))
    assert total_bytes == total_on_disk


def test_read_aggregated_bag_info_rebuild_rekeys_chunk_information(tmp_path: Path) -> None:
    bag = _write_bag(tmp_path, "rec", [10, 15])
    splits = find_bag_splits(bag)
    info, total_bytes = read_aggregated_bag_info(splits, rebuild=True)

    data = info_to_dict(info, str(bag), total_bytes)
    # One MessageIndex record per channel-per-chunk: split0 (1 channel) + split1
    # (2 channels) = 3. A collided re-key would drop one split's records.
    assert data["statistics"]["message_index_count"] == 3
    # Per-compression message_count joins chunk_information by chunk offset; the
    # re-key must keep both splits' counts (40 total), not drop/double-count.
    by_compression = data["chunks"]["by_compression"]
    assert sum(c["message_count"] for c in by_compression.values()) == 40


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_info_cmd_accepts_multi_split_dir(tmp_path: Path) -> None:
    bag = _write_bag(tmp_path, "rec", [5, 5])
    assert info_cmd.info([str(bag)], json_output=True) == 0


def test_info_cmd_watch_rejects_multi_split_dir(tmp_path: Path) -> None:
    bag = _write_bag(tmp_path, "rec", [5, 5])
    assert info_cmd.info([str(bag)], watch=True) == 1


def test_merge_cmd_expands_multi_split_dir(tmp_path: Path) -> None:
    bag = _write_bag(tmp_path, "rec", [5, 5])
    out = tmp_path / "merged.mcap"
    assert merge_cmd.merge([str(bag)], output=out) == 0
    assert out.is_file()
