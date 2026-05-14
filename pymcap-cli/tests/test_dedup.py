"""Unit + integration tests for `merge --dedup-identical`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pymcap_cli.cmd.merge_cmd import merge
from pymcap_cli.core.processors.base import ChunkDecision
from pymcap_cli.core.processors.dedup import DedupIdenticalProcessor
from small_mcap import get_summary, read_message_decoded

from tests.helpers import chunk_context, message_context, pipeline_context

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _StubChunk:
    uncompressed_size: int = 0
    message_start_time: int = 0
    message_end_time: int = 0


@dataclass
class _StubMessage:
    channel_id: int
    log_time: int
    data: bytes
    publish_time: int = 0
    sequence: int = 0


def _on_message(p: DedupIdenticalProcessor, message: _StubMessage) -> list[_StubMessage]:
    return list(p.on_message(message_context(message), message))


def _on_chunk(
    p: DedupIdenticalProcessor,
    chunk: _StubChunk,
    *,
    stream_id: int = 0,
) -> ChunkDecision:
    return p.on_chunk(chunk_context(stream_id=stream_id), chunk)


def test_dedup_first_message_is_kept() -> None:
    p = DedupIdenticalProcessor()
    msg = _StubMessage(channel_id=1, log_time=100, data=b"abc")
    assert _on_message(p, msg) == [msg]
    assert p.dropped_count == 0


def test_dedup_repeat_is_skipped() -> None:
    p = DedupIdenticalProcessor()
    msg = _StubMessage(channel_id=1, log_time=100, data=b"abc")
    _on_message(p, msg)
    assert _on_message(p, msg) == []
    assert p.dropped_count == 1


def test_dedup_differs_by_payload() -> None:
    p = DedupIdenticalProcessor()
    _on_message(p, _StubMessage(channel_id=1, log_time=100, data=b"abc"))
    second = _StubMessage(channel_id=1, log_time=100, data=b"different")
    # Same channel + log_time but different data → keep.
    assert _on_message(p, second) == [second]
    assert p.dropped_count == 0


def test_dedup_differs_by_channel() -> None:
    p = DedupIdenticalProcessor()
    _on_message(p, _StubMessage(channel_id=1, log_time=100, data=b"abc"))
    second = _StubMessage(channel_id=2, log_time=100, data=b"abc")
    assert _on_message(p, second) == [second]


def test_dedup_differs_by_log_time() -> None:
    p = DedupIdenticalProcessor()
    _on_message(p, _StubMessage(channel_id=1, log_time=100, data=b"abc"))
    second = _StubMessage(channel_id=1, log_time=200, data=b"abc")
    assert _on_message(p, second) == [second]


def test_dedup_forces_chunk_decode_without_summaries() -> None:
    # Without chunk-index info, dedup must DECODE every chunk so it can run
    # the per-message hash check.
    p = DedupIdenticalProcessor()
    assert _on_chunk(p, _StubChunk()) is ChunkDecision.DECODE


def _summary_with_chunks(intervals: list[tuple[int, int]]):
    from small_mcap import ChunkIndex, Summary  # noqa: PLC0415

    summary = Summary()
    for start, end in intervals:
        summary.chunk_indexes.append(
            ChunkIndex(
                message_start_time=start,
                message_end_time=end,
                chunk_start_offset=0,
                chunk_length=0,
                message_index_offsets={},
                message_index_length=0,
                compression="",
                compressed_size=0,
                uncompressed_size=0,
            )
        )
    return summary


def test_dedup_fast_copies_chunk_when_no_other_input_overlaps() -> None:
    # Two inputs with disjoint time ranges → every chunk fast-copies, no DECODE.
    p = DedupIdenticalProcessor()
    p.initialize(
        pipeline_context(
            [
                _summary_with_chunks([(0, 100), (100, 200)]),
                _summary_with_chunks([(1000, 1100), (1100, 1200)]),
            ]
        )
    )
    # Stream 0's chunk is far from stream 1's chunks → CONTINUE.
    assert (
        _on_chunk(p, _StubChunk(message_start_time=0, message_end_time=100), stream_id=0)
        is ChunkDecision.CONTINUE
    )
    # And vice versa.
    assert (
        _on_chunk(
            p,
            _StubChunk(message_start_time=1000, message_end_time=1100),
            stream_id=1,
        )
        is ChunkDecision.CONTINUE
    )


def test_dedup_forces_decode_on_overlapping_chunks() -> None:
    p = DedupIdenticalProcessor()
    p.initialize(
        pipeline_context(
            [
                _summary_with_chunks([(0, 100), (200, 300)]),
                _summary_with_chunks([(50, 150), (250, 400)]),
            ]
        )
    )
    # Stream 0's [0, 100] overlaps stream 1's [50, 150] → DECODE.
    assert (
        _on_chunk(p, _StubChunk(message_start_time=0, message_end_time=100), stream_id=0)
        is ChunkDecision.DECODE
    )
    # Stream 0's [200, 300] overlaps stream 1's [250, 400] → DECODE.
    assert (
        _on_chunk(p, _StubChunk(message_start_time=200, message_end_time=300), stream_id=0)
        is ChunkDecision.DECODE
    )


def test_dedup_falls_back_to_decode_if_any_summary_missing() -> None:
    p = DedupIdenticalProcessor()
    p.initialize(pipeline_context([_summary_with_chunks([(0, 100)]), None]))
    # Can't be sure → DECODE every chunk to stay correct.
    assert (
        _on_chunk(p, _StubChunk(message_start_time=500, message_end_time=600), stream_id=0)
        is ChunkDecision.DECODE
    )


def test_dedup_promotes_to_hash_set_on_distinct_payload_at_same_time() -> None:
    # After two distinct payloads at the same (channel, log_time), a third
    # message with the first payload still dedups correctly via the hash set.
    p = DedupIdenticalProcessor()
    a = _StubMessage(channel_id=1, log_time=100, data=b"alpha")
    b = _StubMessage(channel_id=1, log_time=100, data=b"beta")
    assert _on_message(p, a) == [a]
    assert _on_message(p, b) == [b]
    # `a` again must be dropped; `b` again too.
    assert _on_message(p, a) == []
    assert _on_message(p, b) == []
    # A third distinct payload still passes through.
    gamma = _StubMessage(channel_id=1, log_time=100, data=b"gamma")
    assert _on_message(p, gamma) == [gamma]
    assert p.dropped_count == 2


def test_dedup_large_payload_uses_hash_set_on_first_sighting() -> None:
    # Payloads above the inline limit are hashed on first sighting so the
    # processor doesn't pin huge buffers in memory.
    big = b"x" * (4096 + 1)
    p = DedupIdenticalProcessor()
    first = _StubMessage(channel_id=1, log_time=100, data=big)
    assert _on_message(p, first) == [first]
    # Same large payload again → still skipped via the hash set path.
    second = _StubMessage(channel_id=1, log_time=100, data=big)
    assert _on_message(p, second) == []
    assert p.dropped_count == 1


# --- end-to-end: merge two identical MCAPs ---


def test_merge_dedup_drops_duplicates_from_identical_inputs(
    simple_mcap: Path, tmp_path: Path
) -> None:
    a = tmp_path / "a.mcap"
    b = tmp_path / "b.mcap"
    a.write_bytes(simple_mcap.read_bytes())
    b.write_bytes(simple_mcap.read_bytes())
    out = tmp_path / "merged.mcap"

    rc = merge([str(a), str(b)], out, dedup_identical=True, force=True)
    assert rc == 0

    with out.open("rb") as fh:
        summary = get_summary(fh)
    expected = sum(summary.statistics.channel_message_counts.values()) if summary else 0
    # Each input contributes the same count; dedup keeps only one copy.
    with simple_mcap.open("rb") as fh:
        single = get_summary(fh)
    assert single is not None
    single_count = sum(single.statistics.channel_message_counts.values())
    assert expected == single_count


def test_merge_without_dedup_doubles_messages(simple_mcap: Path, tmp_path: Path) -> None:
    a = tmp_path / "a.mcap"
    b = tmp_path / "b.mcap"
    a.write_bytes(simple_mcap.read_bytes())
    b.write_bytes(simple_mcap.read_bytes())
    out = tmp_path / "merged.mcap"

    rc = merge([str(a), str(b)], out, dedup_identical=False, force=True)
    assert rc == 0

    with out.open("rb") as fh:
        summary = get_summary(fh)
    merged_count = sum(summary.statistics.channel_message_counts.values()) if summary else 0
    with simple_mcap.open("rb") as fh:
        single = get_summary(fh)
    assert single is not None
    single_count = sum(single.statistics.channel_message_counts.values())
    assert merged_count == 2 * single_count


def test_merge_dedup_keeps_distinct_messages_from_overlapping_inputs(
    simple_mcap: Path, tmp_path: Path
) -> None:
    # Two copies of the same file → every message is a duplicate; merged
    # output reading via `read_message_decoded` should yield exactly the
    # message count of one file.
    a = tmp_path / "a.mcap"
    b = tmp_path / "b.mcap"
    a.write_bytes(simple_mcap.read_bytes())
    b.write_bytes(simple_mcap.read_bytes())
    out = tmp_path / "merged.mcap"

    rc = merge([str(a), str(b)], out, dedup_identical=True, force=True)
    assert rc == 0

    with out.open("rb") as fh:
        merged_messages = list(read_message_decoded(fh))
    with simple_mcap.open("rb") as fh:
        single_messages = list(read_message_decoded(fh))
    assert len(merged_messages) == len(single_messages)
