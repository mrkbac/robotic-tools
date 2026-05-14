"""Unit tests for ``NthMessageProcessor``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pymcap_cli.core.processors.base import ChunkDecision
from pymcap_cli.core.processors.nth_message import NthMessageProcessor
from small_mcap import Channel

from tests.helpers import channel_context, chunk_context, lazy_chunk, message_context


@dataclass
class _StubMessage:
    channel_id: int
    log_time: int = 0
    publish_time: int = 0
    sequence: int = 0
    data: bytes = b""


@dataclass
class _StubMessageIndex:
    channel_id: int


def _channel(channel_id: int, topic: str) -> Channel:
    return Channel(id=channel_id, schema_id=1, topic=topic, message_encoding="raw", metadata={})


def _on_channel(p: NthMessageProcessor, channel: Channel) -> None:
    p.on_channel(channel_context(channel), channel, None)


def _emits(p: NthMessageProcessor, msg: _StubMessage) -> bool:
    """True if on_message yields the message (kept), False if dropped."""
    out = list(p.on_message(message_context(msg), msg))
    return out == [msg]


def _on_chunk(p: NthMessageProcessor, indexes: list[_StubMessageIndex]) -> ChunkDecision:
    chunk = lazy_chunk(0, 100)
    return p.on_chunk(chunk_context(indexes), chunk)


def test_keeps_first_message_and_every_nth() -> None:
    p = NthMessageProcessor({r"/lidar/.*": 5})
    _on_channel(p, _channel(1, "/lidar/top"))
    kept = [_emits(p, _StubMessage(channel_id=1)) for _ in range(11)]
    # First (0), 5, 10 → kept; everything else dropped.
    expected = [True, False, False, False, False, True, False, False, False, False, True]
    assert kept == expected


def test_uncovered_channel_passes_through() -> None:
    p = NthMessageProcessor({r"/lidar/.*": 5})
    _on_channel(p, _channel(2, "/imu/data"))
    assert _emits(p, _StubMessage(channel_id=2))
    # And again — uncovered channels never drop.
    assert _emits(p, _StubMessage(channel_id=2))


def test_n_equals_one_passes_through() -> None:
    p = NthMessageProcessor({r".*": 1})
    _on_channel(p, _channel(1, "/foo"))
    for _ in range(5):
        assert _emits(p, _StubMessage(channel_id=1))


def test_rejects_n_below_one() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        NthMessageProcessor({r"/foo": 0})


def test_per_channel_counters_are_independent() -> None:
    p = NthMessageProcessor({r"/.*": 3})
    _on_channel(p, _channel(1, "/a"))
    _on_channel(p, _channel(2, "/b"))
    # Channel 1: 0, 1, 2, 3 → keep, drop, drop, keep
    assert _emits(p, _StubMessage(channel_id=1))
    assert not _emits(p, _StubMessage(channel_id=1))
    # Channel 2 has its own counter starting at 0
    assert _emits(p, _StubMessage(channel_id=2))
    # Channel 1 continues from 2
    assert not _emits(p, _StubMessage(channel_id=1))
    assert _emits(p, _StubMessage(channel_id=1))


def test_on_chunk_decodes_when_chunk_references_covered_channel() -> None:
    p = NthMessageProcessor({r"/lidar": 5})
    _on_channel(p, _channel(1, "/lidar"))
    _on_channel(p, _channel(2, "/imu"))
    # Chunk references channel 1 (covered) — must DECODE.
    assert _on_chunk(p, [_StubMessageIndex(channel_id=1)]) is ChunkDecision.DECODE
    # Chunk references only channel 2 — fast-copy.
    assert _on_chunk(p, [_StubMessageIndex(channel_id=2)]) is ChunkDecision.CONTINUE


def test_first_match_wins_with_multiple_rules() -> None:
    # Order is preserved by dict insertion in Python 3.7+.
    p = NthMessageProcessor({r"/cam/.*": 2, r"/cam/front": 10})
    _on_channel(p, _channel(1, "/cam/front"))
    # First matching pattern (N=2) wins, so drop every other message.
    assert _emits(p, _StubMessage(channel_id=1))
    assert not _emits(p, _StubMessage(channel_id=1))
    assert _emits(p, _StubMessage(channel_id=1))
