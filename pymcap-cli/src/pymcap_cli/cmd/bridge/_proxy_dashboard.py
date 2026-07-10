"""Compact live terminal dashboard for `pymcap-cli bridge proxy`."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

    from pymcap_cli.cmd.bridge.proxy import BridgeProxy

_REFRESH_HZ = 2.0


class ProxyDashboard:
    """Render the proxy's counters as a single self-refreshing Rich panel."""

    def __init__(
        self, bridge: BridgeProxy, console: Console, *, refresh_hz: float = _REFRESH_HZ
    ) -> None:
        self._bridge = bridge
        self._console = console
        self._refresh_hz = refresh_hz

    def _render(self) -> Panel:
        snap = self._bridge.snapshot()
        metrics = self._bridge.metrics
        status = (
            Text("● connected", style="green")
            if snap.upstream_connected
            else Text("● disconnected", style="red")
        )
        dropped = metrics.transform_queue_drops + metrics.send_queue_drops
        errors = metrics.transform_errors + metrics.send_errors

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right", style="bold cyan")
        grid.add_column()
        grid.add_row("Upstream", Text(self._bridge.upstream_url))
        grid.add_row("Status", status)
        grid.add_row("Clients", str(snap.client_count))
        grid.add_row(
            "Channels",
            f"{snap.upstream_channels} in · {snap.transformed_channels} transformed",
        )
        grid.add_row("Received", f"{metrics.upstream_messages_received:,}")
        grid.add_row("Throttled", f"{metrics.upstream_messages_throttled:,}")
        grid.add_row(
            "Dropped",
            f"{dropped:,} (transform {metrics.transform_queue_drops:,} · "
            f"send {metrics.send_queue_drops:,})",
        )
        grid.add_row("Errors", Text(f"{errors:,}", style="red" if errors else "dim"))
        grid.add_row("Awaiting keyframe", f"{metrics.video_packets_waiting_for_keyframe:,}")
        return Panel(grid, title="pymcap-cli bridge proxy", border_style="cyan", padding=(1, 2))

    async def run(self) -> None:
        """Refresh the panel until cancelled; clears itself on exit."""
        with (
            Live(
                self._render(),
                console=self._console,
                refresh_per_second=self._refresh_hz,
                transient=True,
            ) as live,
            contextlib.suppress(asyncio.CancelledError),
        ):
            while True:
                await asyncio.sleep(1.0 / self._refresh_hz)
                live.update(self._render())


__all__ = ["ProxyDashboard"]
