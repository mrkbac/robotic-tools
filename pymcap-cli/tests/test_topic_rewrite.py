"""Unit tests for ``TopicRewriteProcessor``."""

from __future__ import annotations

from dataclasses import dataclass

from pymcap_cli.core.processors.base import Action, ChunkDecision
from pymcap_cli.core.processors.topic_rewrite import TopicRewriteProcessor
from small_mcap import Channel

from tests.helpers import channel_context, chunk_context, lazy_chunk


@dataclass
class _StubMessageIndex:
    channel_id: int


def _channel(channel_id: int, topic: str) -> Channel:
    return Channel(id=channel_id, schema_id=1, topic=topic, message_encoding="raw", metadata={})


def _on_channel(p: TopicRewriteProcessor, channel: Channel) -> Action:
    return p.on_channel(channel_context(channel), channel, None)


def _on_chunk(p: TopicRewriteProcessor, indexes=()) -> ChunkDecision:
    return p.on_chunk(chunk_context(indexes), lazy_chunk(0, 100))


def test_literal_rewrite() -> None:
    p = TopicRewriteProcessor({r"^/old/foo$": "/new/foo"})
    ch = _channel(1, "/old/foo")
    assert _on_channel(p, ch) is Action.CONTINUE
    assert ch.topic == "/new/foo"


def test_regex_with_backreference() -> None:
    p = TopicRewriteProcessor({r"^/robot/(\w+)$": r"/old/\1"})
    ch = _channel(1, "/robot/imu")
    _on_channel(p, ch)
    assert ch.topic == "/old/imu"


def test_unmatched_channel_unchanged() -> None:
    p = TopicRewriteProcessor({r"^/old/.*": "/new/"})
    ch = _channel(1, "/keep/me")
    _on_channel(p, ch)
    assert ch.topic == "/keep/me"


def test_first_rule_wins() -> None:
    p = TopicRewriteProcessor({r"^/a/.*": "/first/", r"^/a/b$": "/second"})
    ch = _channel(1, "/a/b")
    _on_channel(p, ch)
    # First rule applies via re.sub; second never sees the channel.
    assert ch.topic.startswith("/first/")


def test_on_chunk_returns_decode_verify_after_a_rewrite() -> None:
    # Before any rewrite has happened, chunks fast-copy.
    p = TopicRewriteProcessor({r"^/old/.*": "/new/"})
    assert _on_chunk(p) is ChunkDecision.CONTINUE

    # After rewriting a channel, every chunk asks for DECODE_VERIFY so the
    # dispatcher can check whether the chunk has an embedded stale Channel
    # record. (We can't know that from the MessageIndex alone — some writers
    # inline every Channel into every chunk.)
    _on_channel(p, _channel(1, "/old/foo"))
    assert _on_chunk(p) is ChunkDecision.DECODE_VERIFY


def test_chunks_fast_copy_when_no_rules_match_anything() -> None:
    p = TopicRewriteProcessor({r"^/never_matches": "/x"})
    _on_channel(p, _channel(1, "/foo"))
    # Nothing was actually rewritten → fast-copy stays.
    assert _on_chunk(p, [_StubMessageIndex(channel_id=1)]) is ChunkDecision.CONTINUE
