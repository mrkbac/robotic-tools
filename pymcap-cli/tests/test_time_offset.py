"""Unit tests for ``TimeOffsetProcessor``."""

from __future__ import annotations

from dataclasses import dataclass

from pymcap_cli.core.processors.base import ChunkDecision
from pymcap_cli.core.processors.time_offset import TimeOffsetProcessor
from small_mcap import Channel

from tests.helpers import channel_context, chunk_context, lazy_chunk, message_context


@dataclass
class _StubMessage:
    channel_id: int
    log_time: int
    publish_time: int
    sequence: int = 0
    data: bytes = b""


@dataclass
class _StubMessageIndex:
    channel_id: int


def _channel(channel_id: int, topic: str) -> Channel:
    return Channel(id=channel_id, schema_id=1, topic=topic, message_encoding="raw", metadata={})


def _on_channel(p: TimeOffsetProcessor, channel: Channel) -> None:
    p.on_channel(channel_context(channel), channel, None)


def _on_message(p: TimeOffsetProcessor, message: _StubMessage) -> list[_StubMessage]:
    return list(p.on_message(message_context(message), message))


def _on_chunk(
    p: TimeOffsetProcessor,
    indexes: list[_StubMessageIndex],
) -> ChunkDecision:
    chunk = lazy_chunk(0, 100)
    return p.on_chunk(chunk_context(indexes), chunk)


def test_offset_shifts_log_time_and_publish_time() -> None:
    p = TimeOffsetProcessor({r"/gps/.*": 1_000_000_000})  # +1s
    _on_channel(p, _channel(1, "/gps/fix"))
    msg = _StubMessage(channel_id=1, log_time=100, publish_time=100)
    assert _on_message(p, msg) == [msg]
    assert msg.log_time == 100 + 1_000_000_000
    assert msg.publish_time == 100 + 1_000_000_000


def test_negative_offset_subtracts() -> None:
    p = TimeOffsetProcessor({r"/imu": -500})
    _on_channel(p, _channel(1, "/imu"))
    msg = _StubMessage(channel_id=1, log_time=1000, publish_time=1000)
    _on_message(p, msg)
    assert msg.log_time == 500
    assert msg.publish_time == 500


def test_uncovered_channel_unchanged() -> None:
    p = TimeOffsetProcessor({r"/gps": 1_000_000})
    _on_channel(p, _channel(2, "/imu"))
    msg = _StubMessage(channel_id=2, log_time=42, publish_time=42)
    _on_message(p, msg)
    assert msg.log_time == 42
    assert msg.publish_time == 42


def test_zero_offset_does_not_force_decode() -> None:
    p = TimeOffsetProcessor({r"/gps": 0})
    _on_channel(p, _channel(1, "/gps"))
    # Zero offset is a no-op; no channels are tracked, so chunk fast-copies.
    assert _on_chunk(p, [_StubMessageIndex(channel_id=1)]) is ChunkDecision.CONTINUE


def test_on_chunk_decodes_when_covered_channel_referenced() -> None:
    p = TimeOffsetProcessor({r"/gps": 1000})
    _on_channel(p, _channel(1, "/gps"))
    _on_channel(p, _channel(2, "/imu"))
    assert _on_chunk(p, [_StubMessageIndex(channel_id=1)]) is ChunkDecision.DECODE
    assert _on_chunk(p, [_StubMessageIndex(channel_id=2)]) is ChunkDecision.CONTINUE


def test_multiple_rules_first_match_wins() -> None:
    p = TimeOffsetProcessor({r"/cam/.*": 1000, r"/cam/front": 5000})
    _on_channel(p, _channel(1, "/cam/front"))
    msg = _StubMessage(channel_id=1, log_time=0, publish_time=0)
    _on_message(p, msg)
    # First pattern matched → +1000, not +5000.
    assert msg.log_time == 1000
