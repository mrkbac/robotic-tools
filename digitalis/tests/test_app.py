import os
import runpy
import sys
from types import ModuleType

from digitalis.app import _configure_for_ssh, main


def test_configure_for_ssh_sets_textual_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SSH_CONNECTION", "client server")
    monkeypatch.delenv("TEXTUAL_FPS", raising=False)
    monkeypatch.setenv("TEXTUAL_ANIMATIONS", "custom")

    _configure_for_ssh()

    assert os.environ["TEXTUAL_FPS"] == "5"
    assert os.environ["TEXTUAL_ANIMATIONS"] == "custom"


def test_configure_for_ssh_does_nothing_locally(monkeypatch) -> None:
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("TEXTUAL_FPS", raising=False)
    monkeypatch.delenv("TEXTUAL_ANIMATIONS", raising=False)

    _configure_for_ssh()

    assert "TEXTUAL_FPS" not in os.environ
    assert "TEXTUAL_ANIMATIONS" not in os.environ


def test_main_configures_logging_and_runs_app(monkeypatch) -> None:
    events: list[str] = []
    runtime = ModuleType("digitalis._runtime")

    class FakeApp:
        def __init__(self, file_or_url: str) -> None:
            events.append(f"init:{file_or_url}")

        def run(self) -> None:
            events.append("run")

    runtime.DigitalisApp = FakeApp  # ty: ignore[unresolved-attribute]
    runtime.configure_logging = lambda: events.append(  # ty: ignore[unresolved-attribute]
        "logging"
    )
    monkeypatch.setitem(sys.modules, "digitalis._runtime", runtime)
    monkeypatch.setattr(sys, "argv", ["digitalis", "recording.mcap"])

    main()

    assert events == ["logging", "init:recording.mcap", "run"]


def test_module_entrypoint_calls_main(monkeypatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr("digitalis.app.main", lambda: called.append(True))

    runpy.run_module("digitalis.__main__", run_name="__main__")

    assert called == [True]
