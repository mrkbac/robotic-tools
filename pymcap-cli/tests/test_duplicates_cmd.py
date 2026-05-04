from __future__ import annotations

import io
import shutil
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd import compress_cmd, duplicates_cmd
from pymcap_cli.core.mcap_compare import discover_mcap_candidates
from rich.console import Console
from small_mcap import CompressionType, IndexType, McapWriter, rebuild_summary
from typing_extensions import Self

from tests.fixtures.mcap_generator import create_simple_mcap

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


def _run_duplicates(
    paths: list[Path | str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_all: bool = False,
    rebuild_missing: bool = False,
) -> tuple[int, str]:
    output = io.StringIO()
    monkeypatch.setattr(
        duplicates_cmd,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=180),
    )
    exit_code = duplicates_cmd.duplicates(
        [str(path) for path in paths],
        include_all=include_all,
        rebuild_missing=rebuild_missing,
    )
    return exit_code, output.getvalue()


def _write_custom_mcap(
    path: Path,
    *,
    topic: str = "/test",
    schema_data: bytes = b"{}",
    metadata: dict[str, str] | None = None,
    file_metadata: dict[str, str] | None = None,
    messages: int = 10,
    timestamps: list[int] | None = None,
    chunk_size: int = 1024,
    compression: CompressionType = CompressionType.ZSTD,
    index_types: IndexType = IndexType.ALL,
) -> None:
    message_times = (
        [index * 1_000_000 for index in range(messages)] if timestamps is None else timestamps
    )
    with path.open("wb") as stream:
        writer = McapWriter(
            stream,
            chunk_size=chunk_size,
            compression=compression,
            index_types=index_types,
        )
        writer.start(profile="ros2", library="duplicates-test")
        writer.add_schema(schema_id=1, name="test", encoding="json", data=schema_data)
        writer.add_channel(
            channel_id=1,
            topic=topic,
            message_encoding="json",
            schema_id=1,
            metadata=metadata,
        )
        for index, log_time in enumerate(message_times):
            writer.add_message(
                channel_id=1,
                log_time=log_time,
                publish_time=log_time,
                data=f'{{"i": {index}}}'.encode(),
            )
        if file_metadata is not None:
            writer.add_metadata("test", file_metadata)
        writer.finish()


def _write_channels_mcap(
    path: Path,
    topics: list[str],
    *,
    messages_per_topic: int = 10,
) -> None:
    with path.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=1024, compression=CompressionType.ZSTD)
        writer.start(profile="ros2", library="duplicates-test")
        writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
        for channel_id, topic in enumerate(topics, start=1):
            writer.add_channel(
                channel_id=channel_id,
                topic=topic,
                message_encoding="json",
                schema_id=1,
            )
        for index in range(messages_per_topic):
            log_time = index * 1_000_000
            for channel_id in range(1, len(topics) + 1):
                writer.add_message(
                    channel_id=channel_id,
                    log_time=log_time,
                    publish_time=log_time,
                    data=f'{{"i": {index}, "channel": {channel_id}}}'.encode(),
                )
        writer.finish()


def _truncate_after_data_section(path: Path) -> None:
    with path.open("r+b") as stream:
        info = rebuild_summary(
            stream,
            validate_crc=False,
            calculate_channel_sizes=False,
            exact_sizes=False,
        )
        stream.truncate(info.next_offset)


def test_discover_candidates_recurses_deduplicates_and_keeps_explicit_files(
    simple_mcap: Path, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    first = root / "first.mcap"
    second = nested / "SECOND.MCAP"
    explicit = tmp_path / "explicit.data"
    ignored = nested / "ignored.txt"
    shutil.copyfile(simple_mcap, first)
    shutil.copyfile(simple_mcap, second)
    shutil.copyfile(simple_mcap, explicit)
    ignored.write_text("not an mcap")

    candidates = discover_mcap_candidates([str(root), str(nested), str(explicit)])

    assert candidates == [str(first), str(second), str(explicit)]


def test_duplicates_groups_copied_mcaps(
    simple_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = tmp_path / "copy.mcap"
    shutil.copyfile(simple_mcap, duplicate)

    exit_code, output = _run_duplicates([simple_mcap, duplicate], monkeypatch)

    assert exit_code == 0
    assert "Duplicate MCAP Groups" in output
    assert "exact indexed duplicate" in output
    assert simple_mcap.name in output
    assert duplicate.name in output


def test_duplicates_groups_uncompressed_and_compressed_mcaps(
    uncompressed_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compressed = tmp_path / "compressed.mcap"
    quiet_output = io.StringIO()
    monkeypatch.setattr(
        compress_cmd,
        "console",
        Console(file=quiet_output, force_terminal=False, color_system=None, width=180),
    )

    assert compress_cmd.compress(str(uncompressed_mcap), compressed, force=True) == 0

    exit_code, output = _run_duplicates([uncompressed_mcap, compressed], monkeypatch)

    assert exit_code == 0
    assert uncompressed_mcap.name in output
    assert compressed.name in output


def test_duplicates_groups_different_chunk_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small_chunks = tmp_path / "small_chunks.mcap"
    large_chunks = tmp_path / "large_chunks.mcap"
    small_chunks.write_bytes(
        create_simple_mcap(
            num_messages=200,
            chunk_size=1024,
            compression=CompressionType.NONE,
        )
    )
    large_chunks.write_bytes(
        create_simple_mcap(
            num_messages=200,
            chunk_size=16 * 1024,
            compression=CompressionType.ZSTD,
        )
    )

    exit_code, output = _run_duplicates([small_chunks, large_chunks], monkeypatch)

    assert exit_code == 0
    assert small_chunks.name in output
    assert large_chunks.name in output


@pytest.mark.parametrize(
    "second_kwargs",
    [
        {"schema_data": b'{"changed": true}'},
        {"topic": "/other"},
        {"metadata": {"robot": "two"}},
    ],
)
def test_duplicates_do_not_group_different_summary_identity(
    second_kwargs: dict[str, bytes | str | dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_custom_mcap(first)
    _write_custom_mcap(second, **second_kwargs)

    exit_code, output = _run_duplicates([first, second], monkeypatch)

    assert exit_code == 0
    assert "No duplicate or partial MCAP matches found" in output


def test_duplicates_reports_partial_channel_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subset = tmp_path / "subset.mcap"
    superset = tmp_path / "superset.mcap"
    _write_channels_mcap(subset, ["/test"])
    _write_channels_mcap(superset, ["/test", "/extra"])

    exit_code, output = _run_duplicates([subset, superset], monkeypatch)

    assert exit_code == 0
    assert "Partial MCAP Matches" in output
    assert "Anchor 1:" in output
    assert "Match 1" not in output
    assert subset.name in output
    assert superset.name in output
    assert "msgs -10" in output
    assert "channels -1" in output
    assert "1 partial match" in output


def test_duplicates_promotes_full_index_overlap_to_duplicate_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_custom_mcap(first)
    _write_custom_mcap(second, file_metadata={"footer": "different"})

    exit_code, output = _run_duplicates([first, second], monkeypatch)

    assert exit_code == 0
    assert "Duplicate MCAP Groups" in output
    assert "Partial MCAP Matches" not in output
    assert "full message overlap, footer/statistics differ" in output
    assert first.name in output
    assert second.name in output
    assert "found 1 duplicate group" in output


def test_duplicates_reports_cutout_inside_larger_mcap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = tmp_path / "full.mcap"
    cutout = tmp_path / "cutout.mcap"
    _write_custom_mcap(full, timestamps=[index * 1_000_000 for index in range(10)])
    _write_custom_mcap(cutout, timestamps=[index * 1_000_000 for index in range(3, 7)])

    exit_code, output = _run_duplicates([full, cutout], monkeypatch)

    assert exit_code == 0
    assert "Partial MCAP Matches" in output
    assert "Anchor 1:" in output
    assert full.name in output
    assert cutout.name in output
    assert "msgs -6" in output
    assert "4 shared msgs" in output
    assert "/test" in output
    assert "anchor-only 6 msgs" in output
    assert "file-only 0 msgs" in output
    assert "overlap" in output


def test_duplicates_groups_multiple_partial_relations_under_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = tmp_path / "full.mcap"
    cutout_a = tmp_path / "cutout_a.mcap"
    cutout_b = tmp_path / "cutout_b.mcap"
    _write_custom_mcap(full, timestamps=[index * 1_000_000 for index in range(10)])
    _write_custom_mcap(cutout_a, timestamps=[index * 1_000_000 for index in range(5)])
    _write_custom_mcap(cutout_b, timestamps=[index * 1_000_000 for index in range(5, 10)])

    exit_code, output = _run_duplicates([full, cutout_a, cutout_b], monkeypatch)

    assert exit_code == 0
    assert "Partial MCAP Matches" in output
    assert "Anchor 1:" in output
    assert "2 relation(s)" in output
    assert full.name in output
    assert cutout_a.name in output
    assert cutout_b.name in output
    assert "2 partial match" in output


def test_duplicates_caps_topic_evidence_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = tmp_path / "full.mcap"
    cutout = tmp_path / "cutout.mcap"
    topics = [f"/topic_{index}" for index in range(5)]
    _write_channels_mcap(full, topics, messages_per_topic=10)
    _write_channels_mcap(cutout, topics, messages_per_topic=5)

    exit_code, output = _run_duplicates([full, cutout], monkeypatch)

    assert exit_code == 0
    assert "Partial MCAP Matches" in output
    assert "/topic_0" in output
    assert "/topic_1" in output
    assert "/topic_2" in output
    assert "/topic_3" not in output
    assert "/topic_4" not in output
    assert "... 2 more topic(s)" in output


def test_duplicates_reports_edge_overlap_between_mcaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left.mcap"
    right = tmp_path / "right.mcap"
    _write_custom_mcap(left, timestamps=[index * 1_000_000 for index in range(10)])
    _write_custom_mcap(right, timestamps=[index * 1_000_000 for index in range(7, 15)])

    exit_code, output = _run_duplicates([left, right], monkeypatch)

    assert exit_code == 0
    assert "Partial MCAP Matches" in output
    assert left.name in output
    assert right.name in output
    assert "msgs -2" in output
    assert "3 shared msgs" in output


def test_duplicates_ignores_same_channel_without_approximate_time_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_custom_mcap(first, timestamps=[index * 1_000_000 for index in range(10)])
    _write_custom_mcap(second, timestamps=[index * 1_000_000 for index in range(20, 30)])

    exit_code, output = _run_duplicates([first, second], monkeypatch)

    assert exit_code == 0
    assert "No duplicate or partial MCAP matches found" in output
    assert "message-index-checked 0 candidate file(s)" in output


def test_duplicates_skips_summaryless_by_default(
    simple_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summaryless = tmp_path / "summaryless.mcap"
    shutil.copyfile(simple_mcap, summaryless)
    _truncate_after_data_section(summaryless)

    exit_code, output = _run_duplicates([simple_mcap, summaryless], monkeypatch)

    assert exit_code == 0
    assert "No duplicate or partial MCAP matches found" in output
    assert "Skipped 1 file(s)" in output
    assert str(summaryless) in output


def test_duplicates_skips_summary_match_without_message_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexed = tmp_path / "indexed.mcap"
    no_message_indexes = tmp_path / "no_message_indexes.mcap"
    _write_custom_mcap(indexed)
    _write_custom_mcap(no_message_indexes, index_types=IndexType.CHUNK)

    exit_code, output = _run_duplicates([indexed, no_message_indexes], monkeypatch)

    assert exit_code == 0
    assert "No duplicate or partial MCAP matches found" in output
    assert "no complete message indexes" in output


def test_duplicates_rebuild_missing_in_memory(
    simple_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summaryless = tmp_path / "summaryless.mcap"
    shutil.copyfile(simple_mcap, summaryless)
    _truncate_after_data_section(summaryless)

    exit_code, output = _run_duplicates(
        [simple_mcap, summaryless],
        monkeypatch,
        rebuild_missing=True,
    )

    assert exit_code == 0
    assert "Duplicate MCAP Groups" in output
    assert simple_mcap.name in output
    assert summaryless.name in output


def test_duplicates_all_includes_singletons(
    simple_mcap: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, output = _run_duplicates([simple_mcap], monkeypatch, include_all=True)

    assert exit_code == 0
    assert "Duplicate MCAP Groups" in output
    assert simple_mcap.name in output


def test_duplicates_reports_progress_stages(
    simple_mcap: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = tmp_path / "copy.mcap"
    shutil.copyfile(simple_mcap, duplicate)

    class FakeProgress:
        def __init__(self) -> None:
            self.descriptions: list[str] = []
            self.updates: list[dict[str, float | int | str | None]] = []

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

        def add_task(self, description: str, *, total: int, current: str) -> int:
            self.descriptions.append(description)
            self.updates.append({"total": total, "current": current})
            return len(self.descriptions)

        def update(self, task: int, **fields: float | str | None) -> None:
            self.updates.append({"task": task, **fields})

    fake_progress = FakeProgress()
    monkeypatch.setattr(duplicates_cmd, "_create_progress", lambda: fake_progress)

    exit_code, output = _run_duplicates([simple_mcap, duplicate], monkeypatch)

    assert exit_code == 0
    assert "Duplicate MCAP Groups" in output
    assert fake_progress.descriptions == [
        "Discovering MCAP files",
        "Reading MCAP summaries",
        "Finding overlap candidates",
        "Reading message indexes",
        "Scoring indexed overlaps",
    ]
    assert any(
        simple_mcap.name in str(update.get("current", "")) for update in fake_progress.updates
    )
    assert any("index record" in str(update.get("current", "")) for update in fake_progress.updates)


def test_duplicates_returns_error_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    assert duplicates_cmd.duplicates([str(missing)]) == 1


def test_duplicates_returns_error_for_directory_without_candidates(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert duplicates_cmd.duplicates([str(empty_dir)]) == 1
