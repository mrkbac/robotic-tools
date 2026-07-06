"""Rich rendering of a decoded bridge service-call response."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

if TYPE_CHECKING:
    from robo_ws_bridge.ws_types import ServiceInfo

ResponseValue = str | int | float | bool | None | list["ResponseValue"] | dict[str, "ResponseValue"]


def _flatten(value: ResponseValue, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a decoded response into ``(dotted field, rendered value)`` rows."""
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(child, path))
        return rows
    return [(prefix, repr(value))]


def build_service_response_table(service: ServiceInfo, response: dict[str, ResponseValue]) -> Table:
    """Build a two-column table of the decoded response fields for ``service``."""
    table = Table(title=f"{service['name']}  ({service['type']})", title_justify="left")
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    for field, rendered in _flatten(response):
        table.add_row(field, rendered)
    return table
