"""Rich rendering of bridge parameter values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

if TYPE_CHECKING:
    from robo_ws_bridge.ws_types import Parameter


def build_parameters_table(parameters: list[Parameter]) -> Table:
    """Build a name / value / type table for a list of bridge parameters."""
    table = Table(title="Parameters", title_justify="left")
    table.add_column("name", style="cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    table.add_column("type", style="dim")
    for parameter in sorted(parameters, key=lambda p: p["name"]):
        table.add_row(parameter["name"], repr(parameter.get("value")), parameter.get("type", ""))
    return table
