"""Tests for direct-file ``pymcap-cli bridge serve`` startup."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from inspect import signature
from urllib.parse import parse_qs, urlsplit

import pymcap_cli.cmd.bridge.serve as serve_module
from pymcap_cli.cmd.bridge._playback import PlaybackChannel, PlaybackStats


@dataclass(frozen=True, slots=True)
class _Prepared:
    channels: tuple[PlaybackChannel, ...] = ()


class _Sink:
    def __init__(self) -> None:
        self.is_started = False

    async def start(self, _channels: tuple[PlaybackChannel, ...]) -> None:
        self.is_started = True


def test_foxglove_url_targets_direct_desktop_app() -> None:
    url = serve_module._foxglove_url("127.0.0.1", 9090)
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "foxglove"
    assert parsed.netloc == "open"
    assert query == {
        "ds": ["foxglove-websocket"],
        "ds.url": ["ws://127.0.0.1:9090"],
    }


def test_serve_loops_by_default() -> None:
    assert signature(serve_module.serve).parameters["loop"].default is True


def test_serve_enables_adaptive_quality_for_roscompress_video(
    monkeypatch,
) -> None:
    resolved_options: list[dict[str, object]] = []
    sink = _Sink()

    async def run_playback(*_args, **_kwargs) -> PlaybackStats:
        return PlaybackStats(state="Finished")

    def resolve_playback_transform_config(**kwargs):
        resolved_options.append(kwargs)

    monkeypatch.setattr(
        serve_module,
        "resolve_playback_transform_config",
        resolve_playback_transform_config,
    )
    monkeypatch.setattr(serve_module, "prepare_playback", lambda *_args: _Prepared())
    monkeypatch.setattr(serve_module, "create_playback_transform_plan", lambda *_args: None)
    monkeypatch.setattr(serve_module, "BridgeServerPlaybackSink", lambda *_args: sink)
    monkeypatch.setattr(serve_module, "run_playback", run_playback)
    monkeypatch.setattr(serve_module, "_launch_url", lambda _url: None)

    assert (
        serve_module.serve(
            ["recording.mcap"],
            preset="compress",
            progress=False,
            no_browser=True,
        )
        == 0
    )
    assert resolved_options[-1]["adaptive_quality"] is True

    assert (
        serve_module.serve(
            ["recording.mcap"],
            preset="compress",
            adaptive_quality=False,
            progress=False,
            no_browser=True,
        )
        == 0
    )
    assert resolved_options[-1]["adaptive_quality"] is False


def test_serve_fast_and_low_presets_scale_video(monkeypatch) -> None:
    resolved_options: list[dict[str, object]] = []
    sink = _Sink()

    async def run_playback(*_args, **_kwargs) -> PlaybackStats:
        return PlaybackStats(state="Finished")

    def resolve_playback_transform_config(**kwargs):
        resolved_options.append(kwargs)

    monkeypatch.setattr(
        serve_module,
        "resolve_playback_transform_config",
        resolve_playback_transform_config,
    )
    monkeypatch.setattr(serve_module, "prepare_playback", lambda *_args: _Prepared())
    monkeypatch.setattr(serve_module, "create_playback_transform_plan", lambda *_args: None)
    monkeypatch.setattr(serve_module, "BridgeServerPlaybackSink", lambda *_args: sink)
    monkeypatch.setattr(serve_module, "run_playback", run_playback)
    monkeypatch.setattr(serve_module, "_launch_url", lambda _url: None)

    for preset, expected_scale in (("fast", 960), ("low", 480)):
        assert (
            serve_module.serve(
                ["recording.mcap"],
                preset=preset,
                progress=False,
                no_browser=True,
            )
            == 0
        )
        options = resolved_options[-1]
        assert options["preset"] == preset
        assert options["image_format"] == "video"
        assert options["scale"] == expected_scale
        assert options["adaptive_quality"] is True

    # An explicit --scale overrides the preset default.
    assert (
        serve_module.serve(
            ["recording.mcap"],
            preset="fast",
            scale=720,
            progress=False,
            no_browser=True,
        )
        == 0
    )
    assert resolved_options[-1]["scale"] == 720


def test_serve_direct_file_launches_foxglove_unless_disabled(
    monkeypatch,
) -> None:
    sink = _Sink()
    launched: list[str] = []

    async def run_playback(*_args, **_kwargs) -> PlaybackStats:
        assert sink.is_started
        return PlaybackStats(state="Finished")

    monkeypatch.setattr(serve_module, "resolve_playback_transform_config", lambda **_kw: None)
    monkeypatch.setattr(serve_module, "prepare_playback", lambda *_args: _Prepared())
    monkeypatch.setattr(
        serve_module,
        "create_playback_transform_plan",
        lambda *_args: None,
    )
    monkeypatch.setattr(serve_module, "BridgeServerPlaybackSink", lambda *_args: sink)
    monkeypatch.setattr(serve_module, "run_playback", run_playback)
    monkeypatch.setattr(serve_module, "_launch_url", launched.append)

    assert (
        serve_module.serve(
            ["recording.mcap"],
            host="127.0.0.1",
            port=9090,
            progress=False,
        )
        == 0
    )
    assert launched == ["foxglove://open?ds=foxglove-websocket&ds.url=ws%3A%2F%2F127.0.0.1%3A9090"]

    launched.clear()
    sink.is_started = False
    assert (
        serve_module.serve(
            ["recording.mcap"],
            host="127.0.0.1",
            port=9090,
            progress=False,
            no_browser=True,
        )
        == 0
    )
    assert launched == []


def test_is_graphical_session_requires_display_on_linux(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not serve_module._is_graphical_session()

    monkeypatch.setenv("DISPLAY", ":0")
    assert serve_module._is_graphical_session()

    monkeypatch.delenv("DISPLAY")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert serve_module._is_graphical_session()

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("WAYLAND_DISPLAY")
    assert serve_module._is_graphical_session()


def test_launch_url_skips_terminal_only_sessions(monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(serve_module.webbrowser, "open", opened.append)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    serve_module._launch_url("foxglove://open?ds=foxglove-websocket")

    assert opened == []


def test_serve_direct_file_always_prints_foxglove_link(monkeypatch, capsys) -> None:
    async def run_playback(*_args, **_kwargs) -> PlaybackStats:
        return PlaybackStats()

    sink = _Sink()
    monkeypatch.setattr(serve_module, "resolve_playback_transform_config", lambda **_kw: None)
    monkeypatch.setattr(serve_module, "prepare_playback", lambda *_args: _Prepared())
    monkeypatch.setattr(
        serve_module, "create_playback_transform_plan", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(serve_module, "BridgeServerPlaybackSink", lambda *_args: sink)
    monkeypatch.setattr(serve_module, "run_playback", run_playback)
    monkeypatch.setattr(serve_module, "_launch_url", lambda _url: None)

    assert serve_module.serve(["input.mcap"], port=9090, no_browser=True) == 0
    output = capsys.readouterr().out
    assert "foxglove://open?ds=foxglove-websocket" in output
