"""Unit tests for ``TopicAliasProcessor``."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pymcap_cli.core.processors.base import ChunkDecision, InputContext
from pymcap_cli.core.processors.topic_alias import TopicAliasProcessor
from small_mcap import Channel, Summary

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


def _build_context(channels: list[Channel], *, with_summary: bool = True, stream_id: int = 0):
    """Stub InputContext with a working ``register_channel`` closure."""
    next_id = [max((c.id for c in channels), default=0) + 1]
    registered: list[Channel] = []

    def register(channel: Channel) -> Channel:
        new = replace(channel, id=next_id[0])
        next_id[0] += 1
        registered.append(new)
        return new

    summary: Summary | None
    if with_summary:
        summary = Summary()
        summary.channels = {c.id: c for c in channels}
    else:
        summary = None
    ctx = InputContext(
        stream_id=stream_id,
        summary=summary,
        statistics=summary.statistics if summary is not None else None,
        chunk_indexes=tuple(summary.chunk_indexes)
        if summary is not None and summary.chunk_indexes
        else None,
        remap_channel=lambda c: c,
        remap_message=lambda m: m,
        register_channel=register,
    )
    return ctx, registered


def _on_message(p: TopicAliasProcessor, message: _StubMessage) -> list[_StubMessage]:
    return list(p.on_message(message_context(message), message))


def _on_chunk(
    p: TopicAliasProcessor,
    indexes: list[_StubMessageIndex],
    *,
    stream_id: int = 0,
) -> ChunkDecision:
    chunk = lazy_chunk(0, 100)
    return p.on_chunk(chunk_context(indexes, stream_id=stream_id), chunk)


def _on_channel(p: TopicAliasProcessor, channel: Channel) -> None:
    p.on_channel(channel_context(channel), channel, None)


def test_alias_channel_registered_during_prepare() -> None:
    p = TopicAliasProcessor({r"^/old/(.+)$": r"/new/\1"})
    ctx, registered = _build_context([_channel(1, "/old/foo")])
    p.prepare_input(ctx)
    assert len(registered) == 1
    assert registered[0].topic == "/new/foo"
    assert p._aliases == {1: [registered[0].id]}


def test_multiple_alias_targets_per_channel() -> None:
    p = TopicAliasProcessor({r"^/raw/(.+)$": [r"/copy_a/\1", r"/copy_b/\1"]})
    ctx, registered = _build_context([_channel(1, "/raw/imu")])
    p.prepare_input(ctx)
    assert [c.topic for c in registered] == ["/copy_a/imu", "/copy_b/imu"]
    assert len(p._aliases[1]) == 2


def test_alias_skipped_when_target_equals_source() -> None:
    p = TopicAliasProcessor({r"^/foo$": "/foo"})
    ctx, registered = _build_context([_channel(1, "/foo")])
    p.prepare_input(ctx)
    # Self-alias is a no-op; no channel registered and no redirect tracked.
    assert registered == []
    assert p._aliases == {}


def test_uncovered_channel_has_no_aliases() -> None:
    p = TopicAliasProcessor({r"^/old/.+$": "/new/"})
    ctx, registered = _build_context([_channel(1, "/keep/me")])
    p.prepare_input(ctx)
    assert registered == []
    assert p._aliases == {}


def test_on_message_yields_original_plus_one_copy_per_alias() -> None:
    p = TopicAliasProcessor({r"^/raw/(.+)$": [r"/copy_a/\1", r"/copy_b/\1"]})
    ctx, registered = _build_context([_channel(1, "/raw/imu")])
    p.prepare_input(ctx)

    original = _StubMessage(channel_id=1, log_time=100, data=b"payload")
    out = _on_message(p, original)
    # First yielded is the original (same object), then one copy per alias.
    assert out[0] is original
    assert [m.channel_id for m in out[1:]] == [registered[0].id, registered[1].id]
    # Forks share the payload and log_time.
    for copy in out[1:]:
        assert copy.data == b"payload"
        assert copy.log_time == 100


def test_on_message_on_uncovered_channel_yields_only_original() -> None:
    p = TopicAliasProcessor({r"^/old/.+$": "/new/"})
    ctx, _registered = _build_context([_channel(1, "/keep/me")])
    p.prepare_input(ctx)
    msg = _StubMessage(channel_id=1)
    assert _on_message(p, msg) == [msg]


def test_on_chunk_decodes_chunks_referencing_aliased_channel() -> None:
    p = TopicAliasProcessor({r"^/old/(.+)$": r"/new/\1"})
    ctx, _ = _build_context([_channel(1, "/old/foo"), _channel(2, "/other")])
    p.prepare_input(ctx)
    assert _on_chunk(p, [_StubMessageIndex(channel_id=1)]) is ChunkDecision.DECODE
    assert _on_chunk(p, [_StubMessageIndex(channel_id=2)]) is ChunkDecision.CONTINUE


# --- No-summary / streaming-input fallback ---


def test_alias_registered_via_on_channel_when_no_summary() -> None:
    p = TopicAliasProcessor({r"^/old/(.+)$": r"/new/\1"})
    ctx, registered = _build_context([], with_summary=False)
    p.prepare_input(ctx)
    # No summary → nothing registered upfront.
    assert registered == []

    # Channel surfaces from the data section — alias gets registered now.
    _on_channel(p, _channel(1, "/old/foo"))
    assert len(registered) == 1
    assert registered[0].topic == "/new/foo"
    assert p._aliases[1] == [registered[0].id]


def test_on_chunk_pessimistic_without_summary() -> None:
    # When the input has no summary, the processor doesn't yet know which
    # channels need aliasing — force DECODE on every chunk so inline Channel
    # records can surface and aliases get registered.
    p = TopicAliasProcessor({r"^/old/.+$": "/new/"})
    ctx, _ = _build_context([], with_summary=False)
    p.prepare_input(ctx)
    assert _on_chunk(p, [_StubMessageIndex(channel_id=42)]) is ChunkDecision.DECODE


def test_on_chunk_optimistic_when_summary_present_and_unmatched() -> None:
    # With a summary present and no rule matching this channel, the chunk
    # can fast-copy.
    p = TopicAliasProcessor({r"^/old/.+$": "/new/"})
    ctx, _ = _build_context([_channel(1, "/keep/me")])
    p.prepare_input(ctx)
    assert _on_chunk(p, [_StubMessageIndex(channel_id=1)]) is ChunkDecision.CONTINUE


def test_on_chunk_remains_pessimistic_for_unsummarized_stream_after_summary_seen() -> None:
    p = TopicAliasProcessor({r"^/old/.+$": "/new/"})
    summarized_ctx, _ = _build_context([_channel(1, "/keep/me")], stream_id=0)
    p.prepare_input(summarized_ctx)
    unsummarized_ctx, _ = _build_context([], with_summary=False, stream_id=1)
    p.prepare_input(unsummarized_ctx)

    assert _on_chunk(p, [_StubMessageIndex(channel_id=42)], stream_id=1) is ChunkDecision.DECODE


def test_on_channel_idempotent_when_already_registered_via_summary() -> None:
    p = TopicAliasProcessor({r"^/old/(.+)$": r"/new/\1"})
    ctx, registered = _build_context([_channel(1, "/old/foo")])
    p.prepare_input(ctx)
    initial_count = len(registered)

    # Same channel coming through on_channel (e.g. inline in a chunk) must
    # not re-register a second alias.
    _on_channel(p, _channel(1, "/old/foo"))
    assert len(registered) == initial_count
