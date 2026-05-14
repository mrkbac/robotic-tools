"""Unit tests for ``ChannelMergeProcessor``."""

from __future__ import annotations

from dataclasses import dataclass

from pymcap_cli.core.processors.base import Action, ChunkDecision
from pymcap_cli.core.processors.channel_merge import ChannelMergeProcessor
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


def _channel(id_: int, topic: str, schema_id: int = 1) -> Channel:
    return Channel(id=id_, schema_id=schema_id, topic=topic, message_encoding="raw", metadata={})


def _on_channel(p: ChannelMergeProcessor, channel: Channel) -> Action:
    return p.on_channel(channel_context(channel), channel, None)


def _on_message(p: ChannelMergeProcessor, message: _StubMessage) -> list[_StubMessage]:
    return list(p.on_message(message_context(message), message))


def _on_chunk(p: ChannelMergeProcessor, indexes: list[_StubMessageIndex]) -> ChunkDecision:
    chunk = lazy_chunk(0, 100)
    return p.on_chunk(chunk_context(indexes), chunk)


def test_first_channel_is_kept() -> None:
    p = ChannelMergeProcessor()
    assert _on_channel(p, _channel(1, "/foo")) is Action.CONTINUE
    assert 1 not in p._redirect


def test_duplicate_channel_is_redirected() -> None:
    p = ChannelMergeProcessor()
    _on_channel(p, _channel(1, "/foo"))
    assert _on_channel(p, _channel(2, "/foo")) is Action.CONTINUE
    assert p._redirect == {2: 1}


def test_channels_with_different_topic_are_kept_separate() -> None:
    p = ChannelMergeProcessor()
    _on_channel(p, _channel(1, "/foo"))
    _on_channel(p, _channel(2, "/bar"))
    assert p._redirect == {}


def test_channels_with_different_schema_are_kept_separate() -> None:
    p = ChannelMergeProcessor()
    _on_channel(p, _channel(1, "/foo", schema_id=1))
    _on_channel(p, _channel(2, "/foo", schema_id=2))
    assert p._redirect == {}


def test_re_encounter_of_same_id_is_idempotent() -> None:
    # E.g., summary registration followed by inline-chunk Channel record.
    p = ChannelMergeProcessor()
    _on_channel(p, _channel(1, "/foo"))
    _on_channel(p, _channel(1, "/foo"))
    assert p._redirect == {}


def test_message_channel_id_rewritten_for_redirected() -> None:
    p = ChannelMergeProcessor()
    _on_channel(p, _channel(1, "/foo"))
    _on_channel(p, _channel(2, "/foo"))
    msg = _StubMessage(channel_id=2)
    out = _on_message(p, msg)
    assert msg.channel_id == 2
    assert out[0].channel_id == 1


def test_message_replacement_does_not_mutate_original() -> None:
    p = ChannelMergeProcessor()
    _on_channel(p, _channel(1, "/foo"))
    _on_channel(p, _channel(2, "/foo"))
    msg = _StubMessage(channel_id=2)
    out = _on_message(p, msg)
    assert out[0] is not msg
    assert msg.channel_id == 2
    assert out[0].channel_id == 1


def test_message_on_kept_channel_unchanged() -> None:
    p = ChannelMergeProcessor()
    _on_channel(p, _channel(1, "/foo"))
    _on_channel(p, _channel(2, "/foo"))
    msg = _StubMessage(channel_id=1)
    _on_message(p, msg)
    assert msg.channel_id == 1


def test_on_chunk_decodes_when_redirected_channel_referenced() -> None:
    p = ChannelMergeProcessor()
    _on_channel(p, _channel(1, "/foo"))
    _on_channel(p, _channel(2, "/foo"))
    assert _on_chunk(p, [_StubMessageIndex(channel_id=2)]) is ChunkDecision.DECODE
    # A chunk that only references the kept channel can fast-copy.
    assert _on_chunk(p, [_StubMessageIndex(channel_id=1)]) is ChunkDecision.CONTINUE


def test_three_way_merge_collapses_to_first() -> None:
    p = ChannelMergeProcessor()
    _on_channel(p, _channel(1, "/foo"))
    _on_channel(p, _channel(2, "/foo"))
    _on_channel(p, _channel(3, "/foo"))
    msg_a = _StubMessage(channel_id=2)
    msg_b = _StubMessage(channel_id=3)
    assert _on_message(p, msg_a)[0].channel_id == 1
    assert _on_message(p, msg_b)[0].channel_id == 1


def test_metadata_difference_keeps_channels_separate() -> None:
    p = ChannelMergeProcessor()
    a = Channel(id=1, schema_id=1, topic="/foo", message_encoding="raw", metadata={})
    b = Channel(
        id=2, schema_id=1, topic="/foo", message_encoding="raw", metadata={"qos": "best_effort"}
    )
    _on_channel(p, a)
    _on_channel(p, b)
    assert p._redirect == {}
