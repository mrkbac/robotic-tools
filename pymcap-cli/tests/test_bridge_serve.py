"""Tests for ``pymcap-cli bridge serve`` startup."""

from __future__ import annotations

import sys
from inspect import signature
from typing import TYPE_CHECKING

import pymcap_cli.cmd.bridge._library as library_module
import pymcap_cli.cmd.bridge.serve as serve_module

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from pymcap_cli.cmd.bridge._playback_transforms import PlaybackTransformConfig
    from pymcap_cli.core.message_filter import MessageFilterOptions


class _LibraryServer:
    def __init__(
        self,
        library: library_module.RecordingLibrary,
        *,
        host: str,
        port: int,
        message_filter: MessageFilterOptions,
        transform_config: PlaybackTransformConfig,
        speed: float,
        loop: bool,
    ) -> None:
        self.library = library
        self.host = host
        self.port = port
        self.message_filter = message_filter
        self.transform_config = transform_config
        self.speed = speed
        self.loop = loop

    async def start(self) -> None:
        pass

    async def serve_forever(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _patch_library_server(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_LibraryServer]:
    servers: list[_LibraryServer] = []

    def create_server(
        library: library_module.RecordingLibrary,
        *,
        host: str,
        port: int,
        message_filter: MessageFilterOptions,
        transform_config: PlaybackTransformConfig,
        speed: float,
        loop: bool,
    ) -> _LibraryServer:
        server = _LibraryServer(
            library,
            host=host,
            port=port,
            message_filter=message_filter,
            transform_config=transform_config,
            speed=speed,
            loop=loop,
        )
        servers.append(server)
        return server

    monkeypatch.setattr(library_module, "RecordingLibraryServer", create_server)
    return servers


def test_serve_loops_by_default() -> None:
    assert signature(serve_module.serve).parameters["loop"].default is True


def test_serve_single_file_uses_restricted_recording_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording.mcap"
    recording.write_bytes(b"not opened during startup")
    servers = _patch_library_server(monkeypatch)
    monkeypatch.setattr(serve_module, "resolve_playback_transform_config", lambda **_kw: None)

    assert serve_module.serve([str(recording)], port=9090, no_browser=True) == 0
    assert servers[0].library.recordings() == (
        library_module.RecordingEntry(
            path="recording.mcap",
            size_bytes=recording.stat().st_size,
        ),
    )


def test_serve_enables_adaptive_quality_for_roscompress_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_options: list[dict[str, object]] = []
    recording = tmp_path / "recording.mcap"
    recording.write_bytes(b"not opened during startup")

    def resolve_playback_transform_config(**kwargs):
        resolved_options.append(kwargs)

    monkeypatch.setattr(
        serve_module,
        "resolve_playback_transform_config",
        resolve_playback_transform_config,
    )
    _patch_library_server(monkeypatch)

    assert (
        serve_module.serve(
            [str(recording)],
            preset="compress",
            no_browser=True,
        )
        == 0
    )
    assert resolved_options[-1]["adaptive_quality"] is True

    assert (
        serve_module.serve(
            [str(recording)],
            preset="compress",
            adaptive_quality=False,
            no_browser=True,
        )
        == 0
    )
    assert resolved_options[-1]["adaptive_quality"] is False


def test_serve_fast_and_low_presets_scale_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_options: list[dict[str, object]] = []
    recording = tmp_path / "recording.mcap"
    recording.write_bytes(b"not opened during startup")

    def resolve_playback_transform_config(**kwargs):
        resolved_options.append(kwargs)

    monkeypatch.setattr(
        serve_module,
        "resolve_playback_transform_config",
        resolve_playback_transform_config,
    )
    _patch_library_server(monkeypatch)

    for preset, expected_scale in (("fast", 960), ("low", 480)):
        assert (
            serve_module.serve(
                [str(recording)],
                preset=preset,
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
            [str(recording)],
            preset="fast",
            scale=720,
            no_browser=True,
        )
        == 0
    )
    assert resolved_options[-1]["scale"] == 720


def test_serve_single_file_launches_library_unless_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording.mcap"
    recording.write_bytes(b"not opened during startup")
    launched: list[str] = []

    monkeypatch.setattr(serve_module, "resolve_playback_transform_config", lambda **_kw: None)
    _patch_library_server(monkeypatch)
    monkeypatch.setattr(serve_module, "_launch_url", launched.append)

    assert (
        serve_module.serve(
            [str(recording)],
            host="127.0.0.1",
            port=9090,
        )
        == 0
    )
    assert launched == ["http://127.0.0.1:9090/"]

    launched.clear()
    assert (
        serve_module.serve(
            [str(recording)],
            host="127.0.0.1",
            port=9090,
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


def test_launch_url_skips_terminal_only_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(serve_module.webbrowser, "open", opened.append)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    serve_module._launch_url("http://127.0.0.1:9090/")

    assert opened == []


def test_serve_single_file_always_prints_library_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recording = tmp_path / "input.mcap"
    recording.write_bytes(b"not opened during startup")
    monkeypatch.setattr(serve_module, "resolve_playback_transform_config", lambda **_kw: None)
    _patch_library_server(monkeypatch)

    assert serve_module.serve([str(recording)], port=9090, no_browser=True) == 0
    output = capsys.readouterr().out
    assert "http://127.0.0.1:9090/" in output


def test_host_helpers_bind_and_display_all_interfaces() -> None:
    assert serve_module._bind_host("") == "0.0.0.0"  # noqa: S104
    assert serve_module._bind_host("127.0.0.1") == "127.0.0.1"
    assert serve_module._binds_all_interfaces("")
    assert serve_module._binds_all_interfaces("0.0.0.0")  # noqa: S104
    assert serve_module._binds_all_interfaces("::")
    assert not serve_module._binds_all_interfaces("127.0.0.1")

    assert serve_module._display_hosts("127.0.0.1") == ["127.0.0.1"]
    all_hosts = serve_module._display_hosts("")
    assert all_hosts[0] == "localhost"
    assert "0.0.0.0" not in all_hosts  # noqa: S104
    assert all(not host.startswith("127.") for host in all_hosts)
    # IPv4 only: no bracketed/colon IPv6 addresses are advertised.
    assert all(":" not in host for host in all_hosts)


def test_serve_empty_host_binds_all_and_lists_reachable_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recording = tmp_path / "input.mcap"
    recording.write_bytes(b"not opened during startup")
    monkeypatch.setattr(serve_module, "resolve_playback_transform_config", lambda **_kw: None)
    servers = _patch_library_server(monkeypatch)
    monkeypatch.setattr(serve_module, "_lan_ip_addresses", lambda: ["192.168.1.50"])

    assert serve_module.serve([str(recording)], host="", port=9090, no_browser=True) == 0
    output = capsys.readouterr().out
    # Empty --host binds every interface.
    assert servers[0].host == "0.0.0.0"  # noqa: S104
    # And advertises a reachable library link for localhost and each LAN IP.
    assert "localhost" in output
    assert "192.168.1.50" in output
