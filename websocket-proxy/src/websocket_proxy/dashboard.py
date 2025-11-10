"""Rich-based dashboard for websocket proxy server."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from .metrics import MetricsCollector
    from .proxy import ProxyBridge

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:.0f}h {minutes:.0f}m"


def _format_rate(rate: float) -> str:
    """Format rate (msgs/sec) to human-readable string."""
    if rate < 0.01:
        return "0.00"
    if rate < 1:
        return f"{rate:.2f}"
    if rate < 10:
        return f"{rate:.1f}"
    return f"{rate:.0f}"


def _format_bandwidth(bytes_per_sec: float) -> str:
    """Format bandwidth to human-readable string."""
    if bytes_per_sec < 1024:
        return f"{bytes_per_sec:.0f} B/s"
    if bytes_per_sec < 1024 * 1024:
        return f"{bytes_per_sec / 1024:.1f} KB/s"
    return f"{bytes_per_sec / (1024 * 1024):.2f} MB/s"


def _format_bytes(byte_count: int) -> str:
    """Format byte count to human-readable string."""
    if byte_count < 1024:
        return f"{byte_count} B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f} KB"
    if byte_count < 1024 * 1024 * 1024:
        return f"{byte_count / (1024 * 1024):.1f} MB"
    return f"{byte_count / (1024 * 1024 * 1024):.2f} GB"


def _format_timestamp(dt: datetime | None) -> str:
    """Format timestamp to relative time string."""
    if dt is None:
        return "never"

    now = datetime.now(timezone.utc)
    diff = now - dt

    if diff < timedelta(seconds=1):
        return "just now"
    if diff < timedelta(seconds=60):
        return f"{diff.seconds}s ago"
    if diff < timedelta(minutes=60):
        return f"{diff.seconds // 60}m ago"
    return dt.strftime("%H:%M:%S")


class DashboardRenderer:
    """Renders a live dashboard for the proxy server using Rich."""

    def __init__(self, proxy: ProxyBridge, refresh_rate: float, console: Console) -> None:
        self.proxy = proxy
        self.metrics: MetricsCollector = proxy.metrics
        self.refresh_rate = refresh_rate
        self.console = console
        self._live: Live | None = None

    def _create_header_panel(self) -> Panel:
        """Create the header panel with global stats."""
        uptime = _format_duration(self.metrics.get_uptime())
        total_clients = len(self.metrics.clients)

        header_text = Text()
        header_text.append("Foxglove WebSocket Proxy Dashboard", style="bold cyan")
        header_text.append("\n\n")
        header_text.append("Uptime: ", style="bold")
        header_text.append(uptime)
        header_text.append("  |  Connected Clients: ", style="bold")
        header_text.append(str(total_clients), style="green" if total_clients > 0 else "dim")

        return Panel(header_text, border_style="blue")

    def _create_clients_table(self) -> Table:
        """Create the clients table."""
        table = Table(
            title="Connected Clients",
            title_style="bold magenta",
            show_header=True,
            header_style="bold",
            show_lines=False,
            expand=False,
        )

        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Remote Address", style="blue", no_wrap=True)
        table.add_column("Connected", style="green", no_wrap=True)
        table.add_column("Msg/s", justify="right", style="yellow")
        table.add_column("Bandwidth", justify="right", style="yellow")
        table.add_column("Msgs", justify="right", style="white")
        table.add_column("Bytes", justify="right", style="yellow")
        table.add_column("Subs", justify="center", style="cyan")
        table.add_column("Errors", justify="center", style="red")
        table.add_column("Last Msg", style="dim", no_wrap=True)

        # Sort clients by connection time (oldest first)
        sorted_clients = sorted(
            self.metrics.clients.values(),
            key=lambda c: c.connected_at,
        )

        for client in sorted_clients:
            client_id_short = client.client_id.replace("client_", "")[:12]
            duration = _format_duration(client.connected_duration)
            msg_rate = _format_rate(client.get_message_rate())
            bandwidth = _format_bandwidth(client.get_bandwidth())
            bytes_send = _format_bytes(client.bytes_sent)
            last_msg = _format_timestamp(client.last_message_at)

            # Style errors in red if > 0
            errors_str = str(client.errors)
            errors_style = "red bold" if client.errors > 0 else "dim"

            table.add_row(
                client_id_short,
                client.remote_address,
                duration,
                msg_rate,
                bandwidth,
                str(client.messages_sent),
                bytes_send,
                str(client.subscription_count),
                Text(errors_str, style=errors_style),
                last_msg,
            )

        if not sorted_clients:
            table.add_row(
                Text("No clients connected", style="dim italic"),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            )

        return table

    def _create_layout(self) -> Panel:
        """Create the full dashboard layout."""
        header = self._create_header_panel()
        clients_table = self._create_clients_table()

        # Group everything together
        layout = Group(
            header,
            "",  # Spacer
            clients_table,
        )

        return Panel(layout, border_style="bright_blue", padding=(1, 2))

    def start_sync(self) -> None:
        """Start the live dashboard (synchronous - starts the Live display)."""
        self._live = Live(
            self._create_layout(),
            console=self.console,
            refresh_per_second=1 / self.refresh_rate,
            screen=False,
            auto_refresh=True,  # Enable auto-refresh so updates are visible
        )
        self._live.start()
        # Do an initial render
        self._live.update(self._create_layout())

    async def run_updates(self) -> None:
        """Run the dashboard update loop (call after start_sync)."""
        if self._live is None:
            return

        while True:
            try:
                self._live.update(self._create_layout())
                await asyncio.sleep(self.refresh_rate)
            except asyncio.CancelledError:
                break
            except (RuntimeError, ValueError, KeyError, IndexError) as e:
                # Log errors during rendering but continue
                logger.debug("Dashboard rendering error: %s", e)
                await asyncio.sleep(self.refresh_rate)

    async def stop(self) -> None:
        """Stop the live dashboard."""
        if self._live:
            self._live.stop()
            self._live = None
