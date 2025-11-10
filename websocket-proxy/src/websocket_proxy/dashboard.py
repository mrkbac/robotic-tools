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
        total_channels = len(self.metrics.channels)
        total_msg_rate = _format_rate(self.metrics.get_total_message_rate())
        total_bandwidth = _format_bandwidth(self.metrics.get_total_bandwidth())

        header_text = Text()
        header_text.append("Foxglove WebSocket Proxy Dashboard", style="bold cyan")
        header_text.append("\n\n")
        header_text.append("Uptime: ", style="bold")
        header_text.append(uptime)
        header_text.append("  |  Clients: ", style="bold")
        header_text.append(str(total_clients), style="green" if total_clients > 0 else "dim")
        header_text.append("  |  Channels: ", style="bold")
        header_text.append(str(total_channels), style="cyan")
        header_text.append("  |  Total Rate: ", style="bold")
        header_text.append(f"{total_msg_rate} msg/s")
        header_text.append("  |  Total Bandwidth: ", style="bold")
        header_text.append(total_bandwidth)

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

        table.add_column("ID", style="cyan", no_wrap=True, width=15)
        table.add_column("Remote Address", style="blue", no_wrap=True)
        table.add_column("Connected", style="green", no_wrap=True, width=12)
        table.add_column("Msg/s", justify="right", style="yellow", width=8)
        table.add_column("Bandwidth", justify="right", style="yellow", width=12)
        table.add_column("Total Msgs", justify="right", style="white", width=12)
        table.add_column("Subs", justify="center", style="cyan", width=6)
        table.add_column("Errors", justify="center", style="red", width=8)
        table.add_column("Last Msg", style="dim", no_wrap=True, width=12)

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
            )

        return table

    def _create_channels_table(self) -> Table:
        """Create the channels/topics table."""
        table = Table(
            title="Topics / Channels",
            title_style="bold magenta",
            show_header=True,
            header_style="bold",
            show_lines=False,
            expand=False,
        )

        table.add_column("Topic", style="cyan", no_wrap=False, max_width=40)
        table.add_column("Schema", style="blue", no_wrap=True, max_width=30)
        table.add_column("Msg/s", justify="right", style="yellow", width=8)
        table.add_column("Bandwidth", justify="right", style="yellow", width=12)
        table.add_column("Total Msgs", justify="right", style="white", width=12)
        table.add_column("Subs", justify="center", style="cyan", width=6)
        table.add_column("Drops", justify="center", style="red", width=8)
        table.add_column("Transform", justify="center", style="magenta", width=12)

        # Sort channels by topic name
        sorted_channels = sorted(
            self.metrics.channels.values(),
            key=lambda c: c.topic,
        )

        for channel in sorted_channels:
            msg_rate = _format_rate(channel.get_send_rate())
            bandwidth = _format_bandwidth(channel.get_bandwidth())

            # Transform status
            if channel.is_transformed:
                transform_ratio = (
                    f"{channel.transform_successes}/{channel.transform_failures}"
                    if channel.transform_failures > 0
                    else f"✓ {channel.transform_successes}"
                )
                transform_style = "green" if channel.transform_failures == 0 else "yellow"
            else:
                transform_ratio = "-"
                transform_style = "dim"

            # Drops in red if > 0
            drops_str = str(channel.messages_dropped) if channel.messages_dropped > 0 else "-"
            drops_style = "red bold" if channel.messages_dropped > 0 else "dim"

            table.add_row(
                channel.topic,
                channel.schema_name,
                msg_rate,
                bandwidth,
                str(channel.messages_sent),
                str(channel.subscriber_count),
                Text(drops_str, style=drops_style),
                Text(transform_ratio, style=transform_style),
            )

        if not sorted_channels:
            table.add_row(
                Text("No channels advertised", style="dim italic"),
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
        channels_table = self._create_channels_table()

        # Group everything together
        layout = Group(
            header,
            "",  # Spacer
            clients_table,
            "",  # Spacer
            channels_table,
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
