import logging
from pathlib import Path

from digitalis._runtime import DigitalisApp, configure_logging
from digitalis.screens.data import DataScreen


def test_app_initializes_ssh_behavior(monkeypatch) -> None:
    monkeypatch.setenv("SSH_CONNECTION", "client server")

    app = DigitalisApp("recording.mcap")

    assert app.file_or_url == "recording.mcap"
    assert app._disable_tooltips is True


def test_on_mount_pushes_data_screen(monkeypatch, tmp_path: Path) -> None:
    recording = tmp_path / "recording.mcap"
    recording.write_bytes(b"")
    app = DigitalisApp(str(recording))
    screens: list[DataScreen] = []
    monkeypatch.setattr(app, "push_screen", screens.append)

    app.on_mount()

    assert len(screens) == 1
    assert isinstance(screens[0], DataScreen)


def test_configure_logging_installs_textual_handler(monkeypatch) -> None:
    calls: list[dict[str, int | list[logging.Handler]]] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: calls.append(kwargs))

    configure_logging()

    assert calls[0]["level"] == "NOTSET"
    handlers = calls[0]["handlers"]
    assert isinstance(handlers, list)
    assert len(handlers) == 1
    assert handlers[0].__class__.__name__ == "TextualHandler"
