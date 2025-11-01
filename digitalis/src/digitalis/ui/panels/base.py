import logging
from typing import Any, ClassVar, Generic, TypeVar

from textual.reactive import reactive
from textual.widget import Widget

from digitalis.reader.types import MessageEvent

T = TypeVar("T", bound=Any)


_PANEL_SCHEMA_REGISTRY: dict[str, list[type["BasePanel[Any]"]]] = {}
_PANEL_REGISTRY: list[type["BasePanel[Any]"]] = []

SCHEMA_ANY = "ANY"


def get_all_panels() -> list[type["BasePanel[Any]"]]:
    """Get all registered panels."""
    return _PANEL_REGISTRY


def get_available_panels(schema: str) -> list[type["BasePanel[Any]"]]:
    """Get all available panels for a given schema, sorted by priority then ANY_SCHEMA."""
    panels = []

    if schema in _PANEL_SCHEMA_REGISTRY:
        panels.extend(_PANEL_SCHEMA_REGISTRY[schema])

    panels = sorted(
        panels,
        key=lambda cls: cls.PRIORITY,
    )

    # append `ANY_SCHEMA` panels if not already included
    if SCHEMA_ANY in _PANEL_SCHEMA_REGISTRY:
        for panel in _PANEL_SCHEMA_REGISTRY[SCHEMA_ANY]:
            if panel not in panels:
                panels.append(panel)

    return panels


def get_default_panel(schema: str) -> type["BasePanel[Any]"] | None:
    """Get the default panel for a schema (most specific first)."""
    panels = get_available_panels(schema)
    if not panels:
        return None

    return panels[0]


class BasePanel(Widget, Generic[T], can_focus=True):
    def __init_subclass__(
        cls,
        can_focus: bool | None = None,
        can_focus_children: bool | None = None,
        inherit_css: bool = True,
        inherit_bindings: bool = True,
    ) -> None:
        super().__init_subclass__(can_focus, can_focus_children, inherit_css, inherit_bindings)

        if not hasattr(cls, "SUPPORTED_SCHEMAS"):
            raise TypeError(
                f"Class {cls.__name__} must define a 'SUPPORTED_SCHEMAS' class variable."
            )

        if not isinstance(cls.SUPPORTED_SCHEMAS, set):
            raise TypeError(f"'SUPPORTED_SCHEMAS' in {cls.__name__} must be a set of strings.")

        for schema in cls.SUPPORTED_SCHEMAS:
            if schema not in _PANEL_SCHEMA_REGISTRY:
                _PANEL_SCHEMA_REGISTRY[schema] = []
            _PANEL_SCHEMA_REGISTRY[schema].append(cls)
            logging.info(f"Registered panel '{cls.__name__}' for schema '{schema}'")

        _PANEL_REGISTRY.append(cls)

    SUPPORTED_SCHEMAS: ClassVar[set[str]] = set()
    """Set of schemas this panel supports, `SCHEMA_ANY` for all schemas."""

    PRIORITY: ClassVar[int] = 100
    """Priority for panel selection, lower values are preferred."""

    data: reactive[MessageEvent | None] = reactive(None)

    def __init__(self) -> None:
        super().__init__()
