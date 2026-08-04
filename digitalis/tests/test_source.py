"""Tests for reader source selection and defaults."""

from collections.abc import Callable

import pytest
from digitalis import reader
from digitalis.reader.source import Source, SourceStatus
from digitalis.reader.types import MessageEvent, SourceInfo


class _FakeSource(Source):
    def __init__(
        self,
        path: str,
        *,
        on_message: Callable[[MessageEvent], None],
        on_source_info: Callable[[SourceInfo], None],
        on_time: Callable[[int], None],
        on_status: Callable[[SourceStatus], None],
    ) -> None:
        super().__init__(on_message, on_source_info, on_time, on_status)
        self.path = path

    async def initialize(self) -> SourceInfo:
        raise NotImplementedError

    def start_playback(self) -> None:
        pass

    def pause_playback(self) -> None:
        pass

    async def subscribe(self, topic: str) -> None:
        pass

    async def unsubscribe(self, topic: str) -> None:
        pass

    async def close(self) -> None:
        pass


class _FakeWebSocketSource(_FakeSource):
    pass


class _FakeMcapSource(_FakeSource):
    pass


@pytest.mark.parametrize(
    ("path", "expected_type"),
    [
        ("ws://localhost:8765", _FakeWebSocketSource),
        ("wss://example.test", _FakeWebSocketSource),
        ("recording.mcap", _FakeMcapSource),
    ],
)
def test_create_source_selects_transport(
    path: str,
    expected_type: type[_FakeSource],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reader, "WebSocketSource", _FakeWebSocketSource)
    monkeypatch.setattr(reader, "McapSource", _FakeMcapSource)

    source = reader.create_source(
        path, lambda _event: None, lambda _info: None, lambda _time: None, lambda _status: None
    )

    assert isinstance(source, expected_type)
    assert source.path == path
    assert source.get_status() is SourceStatus.READY
