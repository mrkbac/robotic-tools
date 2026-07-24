"""Unit tests for decoded MessagePath filtering."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pymcap_cli.core.processors.base import ChunkDecision
from pymcap_cli.core.processors.message_predicate import MessagePredicateProcessor
from small_mcap import Channel, Schema

from tests.helpers import channel_context, chunk_context, lazy_chunk, message_context


@dataclass
class _StubMessage:
    channel_id: int
    data: bytes
    log_time: int = 0
    publish_time: int = 0
    sequence: int = 0


@dataclass
class _StubMessageIndex:
    channel_id: int


def _channel(channel_id: int, topic: str) -> Channel:
    return Channel(
        id=channel_id,
        schema_id=1,
        topic=topic,
        message_encoding="json",
        metadata={},
    )


def _schema() -> Schema:
    return Schema(id=1, name="example/msg/Event", encoding="jsonschema", data=b"{}")


def _register(processor: MessagePredicateProcessor, channel: Channel) -> None:
    processor.on_channel(channel_context(channel), channel, _schema())


def _keeps(processor: MessagePredicateProcessor, message: _StubMessage) -> bool:
    return list(processor.on_message(message_context(message), message)) == [message]


def test_repeated_predicates_on_one_topic_are_or() -> None:
    processor = MessagePredicateProcessor(
        [
            '/events{kind == "alarm"}',
            "/events{score >= 10}",
        ]
    )
    _register(processor, _channel(1, "/events"))

    assert _keeps(processor, _StubMessage(1, b'{"kind":"alarm","score":1}'))
    assert _keeps(processor, _StubMessage(1, b'{"kind":"normal","score":12}'))
    assert not _keeps(processor, _StubMessage(1, b'{"kind":"normal","score":1}'))


def test_and_stays_inside_one_message_path() -> None:
    processor = MessagePredicateProcessor(['/events{kind == "alarm" && score >= 10}'])
    _register(processor, _channel(1, "/events"))

    assert _keeps(processor, _StubMessage(1, b'{"kind":"alarm","score":12}'))
    assert not _keeps(processor, _StubMessage(1, b'{"kind":"alarm","score":1}'))
    assert not _keeps(processor, _StubMessage(1, b'{"kind":"normal","score":12}'))


def test_unrelated_topics_pass_unchanged() -> None:
    processor = MessagePredicateProcessor(["/events{score >= 10}"])
    _register(processor, _channel(1, "/events"))
    _register(processor, _channel(2, "/other"))
    message = _StubMessage(2, b'{"score":0}')

    assert _keeps(processor, message)


def test_variables_are_available_to_predicates() -> None:
    processor = MessagePredicateProcessor(
        ["/events{score >= $minimum}"],
        variables={"minimum": 10},
    )
    _register(processor, _channel(1, "/events"))

    assert _keeps(processor, _StubMessage(1, b'{"score":10}'))
    assert not _keeps(processor, _StubMessage(1, b'{"score":9}'))


def test_where_requires_a_predicate_filter() -> None:
    with pytest.raises(ValueError, match="predicate filter"):
        MessagePredicateProcessor(["/events.score"])


def test_target_chunk_is_decoded_and_unrelated_chunk_fast_copies() -> None:
    processor = MessagePredicateProcessor(["/events{score >= 10}"])
    _register(processor, _channel(1, "/events"))
    _register(processor, _channel(2, "/other"))

    target = [_StubMessageIndex(channel_id=1)]
    unrelated = [_StubMessageIndex(channel_id=2)]
    assert processor.on_chunk(chunk_context(target), lazy_chunk(0, 100)) is ChunkDecision.DECODE
    assert (
        processor.on_chunk(chunk_context(unrelated), lazy_chunk(0, 100)) is ChunkDecision.CONTINUE
    )


def test_chunk_with_unknown_channel_is_decoded_safely() -> None:
    processor = MessagePredicateProcessor(["/events{score >= 10}"])

    assert (
        processor.on_chunk(
            chunk_context([_StubMessageIndex(channel_id=99)]),
            lazy_chunk(0, 100),
        )
        is ChunkDecision.DECODE
    )


def test_missing_where_topic_is_reported_at_end() -> None:
    processor = MessagePredicateProcessor(["/missing{score >= 10}"])
    _register(processor, _channel(1, "/other"))

    with pytest.raises(ValueError, match="/missing"):
        tuple(processor.finalize())
