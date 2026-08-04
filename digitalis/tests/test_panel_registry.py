"""Tests for panel registration and selection."""

from typing import ClassVar

import pytest
from digitalis.ui.panels import base


def test_panel_registry_orders_specific_panels_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base, "_PANEL_SCHEMA_REGISTRY", {})
    monkeypatch.setattr(base, "_PANEL_REGISTRY", [])

    class FallbackPanel(base.BasePanel[bytes]):
        SUPPORTED_SCHEMAS: ClassVar[set[str]] = {base.SCHEMA_ANY}

    class LaterPanel(base.BasePanel[bytes]):
        SUPPORTED_SCHEMAS: ClassVar[set[str]] = {"example"}
        PRIORITY = 20

    class EarlierPanel(base.BasePanel[bytes]):
        SUPPORTED_SCHEMAS: ClassVar[set[str]] = {"example", base.SCHEMA_ANY}
        PRIORITY = 10

    assert base.get_all_panels() == [FallbackPanel, LaterPanel, EarlierPanel]
    assert base.get_available_panels("example") == [EarlierPanel, LaterPanel, FallbackPanel]
    assert base.get_default_panel("example") is EarlierPanel
    assert base.get_default_panel("missing") is FallbackPanel


def test_panel_registry_rejects_non_set_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "_PANEL_SCHEMA_REGISTRY", {})
    monkeypatch.setattr(base, "_PANEL_REGISTRY", [])

    with pytest.raises(TypeError, match="must be a set of strings"):

        class InvalidPanel(base.BasePanel[bytes]):
            SUPPORTED_SCHEMAS: ClassVar[set[str]] = ["example"]  # type: ignore[assignment]
