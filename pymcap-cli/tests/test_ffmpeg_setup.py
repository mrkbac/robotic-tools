import pytest

from tests import _ffmpeg_setup


def test_ensure_ffmpeg_controller_initializes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_add_to_path() -> None:
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(_ffmpeg_setup, "add_to_path", fake_add_to_path)

    _ffmpeg_setup.ensure_ffmpeg(None)

    assert call_count == 1


def test_ensure_ffmpeg_worker_reuses_controller_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_add_to_path() -> None:
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(_ffmpeg_setup, "add_to_path", fake_add_to_path)

    _ffmpeg_setup.ensure_ffmpeg("gw0")

    assert call_count == 0
