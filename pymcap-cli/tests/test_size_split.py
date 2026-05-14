"""Unit tests for the size-split processor and the size parser."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pymcap_cli.core.processors.base import ChunkDecision
from pymcap_cli.core.processors.size_split import SizeSplitProcessor
from pymcap_cli.types.size import parse_size_bytes

from tests.helpers import chunk_context, message_context


@dataclass
class _StubChunk:
    uncompressed_size: int
    message_start_time: int = 0
    message_end_time: int = 0


@dataclass
class _StubMessage:
    data: bytes
    channel_id: int = 1
    log_time: int = 0
    publish_time: int = 0
    sequence: int = 0


# --- parse_size_bytes ---


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1024", 1024),
        ("1K", 1_000),
        ("1KB", 1_000),
        ("1.5M", 1_500_000),
        ("1G", 1_000_000_000),
        ("500MB", 500_000_000),
        ("1Ki", 1024),
        ("1KiB", 1024),
        ("2GiB", 2 * 1024**3),
        ("1B", 1),
    ],
)
def test_parse_size_bytes_accepts_common_forms(value: str, expected: int) -> None:
    assert parse_size_bytes(value) == expected


@pytest.mark.parametrize("bad", ["", "abc", "1XB", "-5G", "0"])
def test_parse_size_bytes_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError, match=r"(Invalid size|must be positive)"):
        parse_size_bytes(bad)


# --- SizeSplitProcessor ---


def test_size_split_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        SizeSplitProcessor(0)


def test_size_split_output_keys_is_dynamic() -> None:
    assert SizeSplitProcessor(100).output_segments() is None


def test_size_split_always_returns_continue_so_fast_copy_works() -> None:
    # on_chunk runs during pipeline pre-classification before route_chunk has
    # updated state; returning CONTINUE keeps every chunk on the fast-copy
    # path and lets route_chunk decide segment boundaries serially.
    p = SizeSplitProcessor(1000)
    assert p.on_chunk(chunk_context(), _StubChunk(uncompressed_size=400)) is ChunkDecision.CONTINUE
    assert (
        p.on_chunk(chunk_context(), _StubChunk(uncompressed_size=10_000)) is ChunkDecision.CONTINUE
    )


def test_size_split_route_chunk_advances_segment_on_overflow() -> None:
    p = SizeSplitProcessor(1000)
    # First chunk fits.
    c1 = _StubChunk(uncompressed_size=600)
    assert list(p.route_chunk(chunk_context(), c1)) == [0]
    # Second chunk would overflow current segment → bump segment.
    c2 = _StubChunk(uncompressed_size=600)
    assert list(p.route_chunk(chunk_context(), c2)) == [1]
    # A small third chunk fits in segment 1.
    c3 = _StubChunk(uncompressed_size=100)
    assert list(p.route_chunk(chunk_context(), c3)) == [1]


def test_size_split_oversized_chunk_alone_keeps_fast_copy() -> None:
    p = SizeSplitProcessor(1000)
    big = _StubChunk(uncompressed_size=5000)
    # Fresh segment with no prior bytes; accept oversized chunk as-is.
    assert list(p.route_chunk(chunk_context(), big)) == [0]


def test_size_split_route_message_flips_segments_by_payload() -> None:
    p = SizeSplitProcessor(100)
    # 80 bytes + 31 overhead ≈ 111 — pushes past budget on second call.
    m = _StubMessage(data=b"x" * 80)
    assert list(p.route_message(message_context(m), m)) == [0]
    assert list(p.route_message(message_context(m), m)) == [1]
    assert list(p.route_message(message_context(m), m)) == [2]


def test_size_split_route_message_handles_single_oversized_message() -> None:
    p = SizeSplitProcessor(50)
    huge = _StubMessage(data=b"x" * 500)
    # First message fits "by exception" — empty segment accepts oversized.
    assert list(p.route_message(message_context(huge), huge)) == [0]
    # Next message must start a new segment.
    assert list(p.route_message(message_context(huge), huge)) == [1]
