"""Shared display and formatting utilities for pymcap-cli commands."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from rich.table import Table

from pymcap_cli.utils import bytes_to_human

if TYPE_CHECKING:
    from rich.console import Console

    from pymcap_cli.types import McapInfoOutput


class ChannelTableColumn(enum.IntFlag):
    """Flags for channel table columns."""

    ID = enum.auto()
    TOPIC = enum.auto()
    SCHEMA = enum.auto()
    MSGS = enum.auto()
    HZ = enum.auto()
    SIZE = enum.auto()
    PERCENT = enum.auto()
    BPS = enum.auto()
    B_PER_MSG = enum.auto()
    DISTRIBUTION = enum.auto()


def schema_to_color(schema_name: str | None) -> str:
    """Generate a deterministic color from a schema name.

    Uses Python's built-in hash function to map schema names to a palette
    of 11 colors. Same schema will always get the same color within a session.

    Args:
        schema_name: The schema name to hash (e.g., "sensor_msgs/msg/Image")

    Returns:
        Rich color name string (e.g., "cyan", "bright_green")
    """
    if not schema_name:
        return "white"

    colors = [
        "blue",
        "bright_cyan",
        "bright_green",
        "bright_magenta",
        "bright_red",
        "bright_yellow",
        "cyan",
        "green",
        "magenta",
        "red",
        "yellow",
    ]

    # Use Python's built-in hash for deterministic value
    hash_value = hash(schema_name)
    color_index = hash_value % len(colors)

    return colors[color_index]


def create_ros_docs_url(schema_name: str, distro: str = "jazzy") -> str:
    """Generate a ROS documentation URL for a schema.

    Handles both old format (package_name/MessageType) and new format
    (package_name/msg/MessageType) ROS schemas.

    Args:
        schema_name: ROS schema name (e.g., "sensor_msgs/msg/Image")
        distro: ROS distribution name (default: "jazzy")

    Returns:
        Documentation URL string, or empty string if format is invalid
    """
    parts = schema_name.split("/")
    if len(parts) == 2:
        # Old format: package_name/MessageType
        package_name, message_type = parts
        return f"https://docs.ros.org/en/{distro}/p/{package_name}/msg/{message_type}.html"
    if len(parts) == 3:
        # New format: package_name/msg/MessageType
        package_name, msg_dir, message_type = parts
        return f"https://docs.ros.org/en/{distro}/p/{package_name}/{msg_dir}/{message_type}.html"

    return ""


def format_schema_with_link(schema_name: str | None, distro: str = "jazzy") -> str:
    """Format a schema name with color and clickable ROS documentation link.

    Args:
        schema_name: Schema name to format (e.g., "sensor_msgs/msg/Image")
        distro: ROS distribution name (default: "jazzy")

    Returns:
        Rich markup string with color and link, or "[dim]unknown[/dim]" for None
    """
    if not schema_name:
        return "[dim]unknown[/dim]"

    # Skip non-ROS schemas (foxglove, malformed)
    if "/" not in schema_name:
        return schema_name

    # Create clickable link for ROS schemas
    url = create_ros_docs_url(schema_name, distro)
    color = schema_to_color(schema_name)
    return f"[{color} link={url}]{schema_name}[/]"


def display_channels_table(
    data: McapInfoOutput,
    console: Console,
    *,
    sort_key: str = "topic",
    reverse: bool = False,
    columns: ChannelTableColumn | None = None,
    responsive: bool = True,
    index_duration: bool = False,
    distribution_bar_class: type | None = None,
) -> None:
    """Display a channels table with configurable columns and sorting.

    Args:
        data: MCAP info data containing channel information
        console: Rich console instance for output
        sort_key: Field to sort by (topic, id, msgs, size, hz, bps, b_per_msg, schema)
        reverse: Sort in descending order if True
        columns: IntFlag of columns to display. If None, defaults to all columns.
        responsive: Enable responsive column hiding based on terminal width
        index_duration: Use per-channel Hz calculation instead of global duration
        distribution_bar_class: Class to use for distribution bars (must have compatible interface)
    """

    # Default columns if not specified
    if columns is None:
        columns = (
            ChannelTableColumn.ID
            | ChannelTableColumn.TOPIC
            | ChannelTableColumn.SCHEMA
            | ChannelTableColumn.MSGS
            | ChannelTableColumn.HZ
            | ChannelTableColumn.SIZE
            | ChannelTableColumn.BPS
            | ChannelTableColumn.B_PER_MSG
            | ChannelTableColumn.DISTRIBUTION
        )

    # Dictionary of sort key functions
    sort_keys = {
        "topic": lambda c: c["topic"],
        "id": lambda c: c["id"],
        "msgs": lambda c: c["message_count"],
        "size": lambda c: c.get("size_bytes") or 0,
        "hz": lambda c: (c["hz_channel"] if index_duration and c["hz_channel"] else c["hz"]),
        "bps": lambda c: c.get("bytes_per_second") or 0,
        "b_per_msg": lambda c: c.get("bytes_per_message") or 0,
        "schema": lambda c: c.get("schema_name") or "",
        "percent": lambda c: c.get("size_bytes") or 0,  # Same as size for sorting purposes
    }
    get_sort_key = sort_keys.get(sort_key, sort_keys["topic"])

    # Check if size data is available
    has_size_data = any(ch["size_bytes"] is not None for ch in data["channels"])

    # Check if distribution data is available
    has_distribution_data = any(
        ch.get("message_distribution") and len(ch["message_distribution"]) > 0
        for ch in data["channels"]
    )

    # Determine which columns to show based on responsive mode
    if responsive:
        terminal_width = console.width
        show_size = (
            terminal_width >= 80 and has_size_data and bool(columns & ChannelTableColumn.SIZE)
        )
        show_percent = (
            terminal_width >= 80 and has_size_data and bool(columns & ChannelTableColumn.PERCENT)
        )
        show_bps = (
            terminal_width >= 100 and has_size_data and bool(columns & ChannelTableColumn.BPS)
        )
        show_b_per_msg = (
            terminal_width >= 120 and has_size_data and bool(columns & ChannelTableColumn.B_PER_MSG)
        )
        show_distribution = (
            terminal_width >= 140
            and has_distribution_data
            and bool(columns & ChannelTableColumn.DISTRIBUTION)
        )
        show_schema = terminal_width >= 160 and bool(columns & ChannelTableColumn.SCHEMA)
    else:
        # Always show requested columns
        show_size = bool(columns & ChannelTableColumn.SIZE)
        show_percent = bool(columns & ChannelTableColumn.PERCENT)
        show_bps = bool(columns & ChannelTableColumn.BPS)
        show_b_per_msg = bool(columns & ChannelTableColumn.B_PER_MSG)
        show_distribution = (
            bool(columns & ChannelTableColumn.DISTRIBUTION) and has_distribution_data
        )
        show_schema = bool(columns & ChannelTableColumn.SCHEMA)

    # Calculate total size for percentage column
    total_size = (
        sum(ch["size_bytes"] for ch in data["channels"] if ch["size_bytes"]) if show_percent else 0
    )

    # Build the table
    channels_table = Table()

    # Add columns based on configuration
    if columns & ChannelTableColumn.ID:
        channels_table.add_column("ID", style="bold blue", no_wrap=True, justify="right")
    if columns & ChannelTableColumn.TOPIC:
        channels_table.add_column("Topic", overflow="fold")
    if show_schema:
        channels_table.add_column("Schema", style="blue")
    if columns & ChannelTableColumn.MSGS:
        channels_table.add_column("Msgs", justify="right", style="green")
    if columns & ChannelTableColumn.HZ:
        channels_table.add_column("Hz", justify="right", style="yellow")
    if show_size:
        channels_table.add_column("Size", justify="right", style="yellow")
    if show_percent:
        channels_table.add_column("Total %", justify="right", style="yellow")
    if show_bps:
        channels_table.add_column("B/s", justify="right", style="magenta")
    if show_b_per_msg:
        channels_table.add_column("B/msg", justify="right", style="magenta")
    if show_distribution:
        channels_table.add_column("Distribution")

    # Populate rows
    for channel in sorted(data["channels"], key=get_sort_key, reverse=reverse):
        hz = channel["hz_channel"] if index_duration and channel["hz_channel"] else channel["hz"]

        # Apply color based on schema
        topic_color = schema_to_color(channel["schema_name"])
        colored_topic = f"[{topic_color}]{channel['topic']}[/{topic_color}]"

        row = []

        # Add cells based on column configuration
        if columns & ChannelTableColumn.ID:
            row.append(str(channel["id"]))
        if columns & ChannelTableColumn.TOPIC:
            row.append(colored_topic)
        if show_schema:
            row.append(format_schema_with_link(channel["schema_name"]))
        if columns & ChannelTableColumn.MSGS:
            row.append(f"{channel['message_count']:,}")
        if columns & ChannelTableColumn.HZ:
            row.append(f"{hz:.2f}")
        if show_size:
            size = channel.get("size_bytes") or 0
            row.append(bytes_to_human(size))
        if show_percent:
            size = channel.get("size_bytes") or 0
            percentage = (size / total_size * 100) if total_size > 0 else 0
            row.append(f"{percentage:.2f}%")
        if show_bps:
            bps = channel["bytes_per_second"] if channel["bytes_per_second"] is not None else 0
            row.append(bytes_to_human(int(bps)))
        if show_b_per_msg:
            b_per_msg = (
                channel["bytes_per_message"] if channel["bytes_per_message"] is not None else 0
            )
            row.append(bytes_to_human(int(b_per_msg)))
        if show_distribution:
            distribution = channel.get("message_distribution", [])
            if distribution_bar_class:
                row.append(distribution_bar_class(distribution))
            else:
                row.append("")  # Empty cell if no distribution bar class provided

        channels_table.add_row(*row)

    console.print(channels_table)
