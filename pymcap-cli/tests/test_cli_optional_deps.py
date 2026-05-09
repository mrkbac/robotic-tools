"""Tests for optional command dependency handling in the top-level CLI."""

from types import SimpleNamespace

import pytest
from cyclopts import App, Parameter
from pymcap_cli import cli


def test_unavailable_command_accepts_command_args(capsys) -> None:
    app = App(default_parameter=Parameter(negative_iterable=""))
    app.command(name="video")(
        cli._unavailable_command(
            "video",
            message="Video command requires the 'video' extra.",
            install_command="uv add 'pymcap-cli[video]'",
        )
    )

    with pytest.raises(SystemExit) as exc_info:
        app(["video", "input.mcap", "--topic", "/camera/front", "-o", "out"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Video command requires the 'video' extra." in captured.err
    assert "uv add 'pymcap-cli[video]'" in captured.err


def test_load_optional_command_returns_command_when_import_succeeds(monkeypatch) -> None:
    def command() -> int:
        return 7

    def fake_import_module(module_name: str) -> SimpleNamespace:
        assert module_name == "pymcap_cli.cmd.fake_cmd"
        return SimpleNamespace(fake=command)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    loaded = cli._load_optional_command(
        "pymcap_cli.cmd.fake_cmd",
        "fake",
        expected_missing_modules=("missing_dep",),
        message="Fake command requires an extra.",
        install_command="uv add 'pymcap-cli[fake]'",
    )

    assert loaded() == 7


def test_load_optional_command_returns_stub_for_expected_missing_module(
    monkeypatch, capsys
) -> None:
    def fake_import_module(_module_name: str) -> SimpleNamespace:
        raise ModuleNotFoundError("No module named 'plotly.graph_objects'", name="plotly")

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    loaded = cli._load_optional_command(
        "pymcap_cli.cmd.plot_cmd",
        "plot",
        expected_missing_modules=("plotly",),
        message="Plot command requires the 'plot' extra.",
        install_command="uv add 'pymcap-cli[plot]'",
    )

    assert loaded("input.mcap", "/topic.value") == 1
    captured = capsys.readouterr()
    assert "Plot command requires the 'plot' extra." in captured.err
    assert "uv add 'pymcap-cli[plot]'" in captured.err


def test_load_optional_command_reraises_unexpected_missing_module(monkeypatch) -> None:
    def fake_import_module(_module_name: str) -> SimpleNamespace:
        raise ModuleNotFoundError(
            "No module named 'pymcap_cli.internal'",
            name="pymcap_cli.internal",
        )

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        cli._load_optional_command(
            "pymcap_cli.cmd.plot_cmd",
            "plot",
            expected_missing_modules=("plotly",),
            message="Plot command requires the 'plot' extra.",
            install_command="uv add 'pymcap-cli[plot]'",
        )

    assert exc_info.value.name == "pymcap_cli.internal"
