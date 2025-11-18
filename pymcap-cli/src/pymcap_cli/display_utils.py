"""Shared display and formatting utilities for pymcap-cli commands."""

import enum
import hashlib
from functools import lru_cache

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.jupyter import JupyterMixin
from rich.measure import Measurement
from rich.style import Style
from rich.table import Table
from rich.text import Text

from pymcap_cli.info_types import ChannelInfo, McapInfoOutput
from pymcap_cli.utils import bytes_to_human

# Common color palette used for deterministic color assignment
# Works well on both light and dark terminal backgrounds
COMMON_COLORS = [
    # Standard colors
    "blue",
    "cyan",
    "green",
    "magenta",
    "red",
    "yellow",
    # Bright variants (good contrast on dark backgrounds)
    "bright_blue",
    "bright_cyan",
    "bright_green",
    "bright_magenta",
    "bright_red",
    "bright_yellow",
    # Additional safe colors
    "purple",
    "deep_pink4",
    "orange1",
    "chartreuse3",
    "deep_sky_blue1",
    "spring_green3",
    "medium_purple",
    "gold3",
    "turquoise2",
    "salmon1",
    "light_coral",
    "medium_orchid",
    "dodger_blue2",
    "light_sea_green",
    "khaki3",
    "plum2",
]


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


class DistributionBar(JupyterMixin):
    """A width-aware distribution bar that automatically scales to fit table cells.

    This renderable creates a visual distribution histogram using Unicode block
    characters with a color gradient from blue (low) to red (high).
    """

    def __init__(
        self,
        counts: list[int],
        *,
        width: int | None = None,
        min_width: int = 10,
    ) -> None:
        """Initialize the distribution bar.

        Args:
            counts: List of message counts per bucket
            width: Fixed width in characters, or None to use available width
            min_width: Minimum width to request
        """
        self.counts = counts
        self.width = width
        self.min_width = min_width

        # Unicode block characters for vertical bars (8 levels)
        self.blocks = " ▁▂▃▄▅▆▇█"

        # Color gradient styles for each level (0-8)
        self.level_styles = [
            Style(color="gray89", bgcolor="gray89"),  # 0: Empty - light gray
            Style(color="blue", bgcolor="gray89"),  # 1: Very low - blue
            Style(color="bright_blue", bgcolor="gray89"),  # 2: Low - brighter blue
            Style(color="cyan", bgcolor="gray89"),  # 3: Medium-low - cyan
            Style(color="green", bgcolor="gray89"),  # 4: Medium - green
            Style(color="yellow", bgcolor="gray89"),  # 5: Medium-high - yellow
            Style(color="bright_yellow", bgcolor="gray89"),  # 6: High - bright yellow
            Style(color="red", bgcolor="gray89"),  # 7: Very high - red
            Style(color="bright_red", bgcolor="gray89"),  # 8: Maximum - bright red
        ]

    def _downsample_to_width(self, counts: list[int], target_width: int) -> list[int]:
        num_buckets = len(counts)
        buckets_per_char = num_buckets / target_width

        scaled = []
        for i in range(target_width):
            start_idx = int(i * buckets_per_char)
            end_idx = int((i + 1) * buckets_per_char)
            # Take max value in this range (preserves peaks)
            if start_idx < end_idx:
                scaled.append(max(counts[start_idx:end_idx]))
            else:
                scaled.append(0)

        return scaled

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Determine actual rendering width
        # Use specified width OR expand to available width
        width = min(self.width or options.max_width, options.max_width)
        width = max(width, self.min_width)  # Respect minimum

        # Handle empty data
        if not self.counts or max(self.counts) == 0:
            yield Text("no messages", style="dim")
            return

        # Scale distribution to fit available width
        num_buckets = len(self.counts)

        if num_buckets <= width:
            # Enough space: show each bucket
            scaled_counts = self.counts
        else:
            # Too many buckets: downsample by taking max in each group
            scaled_counts = self._downsample_to_width(self.counts, width)

        # Render the scaled distribution
        max_count = max(scaled_counts)
        result = Text()

        for count in scaled_counts:
            if count == 0:
                # Show subtle background for empty buckets
                result.append("░", style=self.level_styles[0])
            else:
                # Scale to 1-8 range based on this channel's max
                level = int((count / max_count) * 8) if max_count > 0 else 0
                level = min(level, 8)  # Clamp to max
                level = max(level, 1)  # Ensure non-zero counts get at least level 1
                result.append(self.blocks[level], style=self.level_styles[level])

        yield result

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        if self.width is not None:
            # Fixed width
            return Measurement(self.width, self.width)
        # Flexible: cap max width based on actual bucket count and a reasonable maximum
        # No point displaying wider than the number of buckets, and cap at 80 chars
        max_useful_width = min(len(self.counts), 80, options.max_width)
        return Measurement(self.min_width, max_useful_width)


@lru_cache(maxsize=128)
def _text_to_color(text: str | None) -> str:
    """Generate a deterministic color from a schema name (cached for performance).

    Uses MD5 hashing to map schema names to a palette of colors.
    Same schema will always get the same color across all runs.

    Args:
        schema_name: The schema name to hash (e.g., "sensor_msgs/msg/Image")

    Returns:
        Rich color name string (e.g., "cyan", "bright_green")
    """
    if not text:
        return "white"

    hash_value = int.from_bytes(hashlib.md5(text.strip().encode()).digest()[:4], "little")  # noqa: S324
    color_index = hash_value % len(COMMON_COLORS)

    return COMMON_COLORS[color_index]


def _format_parts_with_colors(topic: str) -> str:
    """Format a topic path with colors for common prefixes and optional leaf color.

    Args:
        topic: Full topic path (e.g., "/driver/vectornav/raw/imu")

    Returns:
        Rich markup string with colored path segments
    """
    parts = topic.strip("/").split("/")
    if not parts[0]:  # Empty after stripping slashes
        return topic

    result_parts: list[str] = []

    # Process each path segment
    for part in parts:
        # Determine color for this segment
        color = _text_to_color(part)

        # Apply color if available
        colored_part = f"[{color}]{part}[/{color}]" if color else part
        result_parts.append(colored_part)

    # Join with dim slashes for better readability
    ret = ""
    if topic.startswith("/"):
        ret += "[dim]/[/dim]"
    return ret + "[dim]/[/dim]".join(result_parts)


def _create_ros_docs_url(schema_name: str, distro: str = "jazzy") -> str:
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


def _format_schema_with_link(schema_name: str | None, distro: str = "jazzy") -> str:
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
    url = _create_ros_docs_url(schema_name, distro)
    formatted = _format_parts_with_colors(schema_name)
    return f"[link={url}]{formatted}[/]"


def display_channels_table(
    data: McapInfoOutput,
    console: Console,
    *,
    sort_key: str = "topic",
    reverse: bool = False,
    columns: ChannelTableColumn | None = None,
    responsive: bool = True,
    index_duration: bool = False,
    use_median: bool = False,
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
        use_median: Display median rates instead of mean rates
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
    def hz_sort_key(c: ChannelInfo) -> float:
        hz_stats = c["hz_stats"]
        if use_median and hz_stats and (median := hz_stats.get("median")) is not None:
            return median
        if index_duration and (hz_channel := c.get("hz_channel")) is not None:
            return hz_channel
        return hz_stats["average"]

    def bps_sort_key(c: ChannelInfo) -> float:
        bps_stats = c.get("bytes_per_second_stats")
        if use_median and bps_stats and (median := bps_stats.get("median")) is not None:
            return median
        return bps_stats["average"] if bps_stats else 0

    sort_keys = {
        "topic": lambda c: c["topic"],
        "id": lambda c: c["id"],
        "msgs": lambda c: c["message_count"],
        "size": lambda c: c.get("size_bytes") or 0,
        "hz": hz_sort_key,
        "bps": bps_sort_key,
        "b_per_msg": lambda c: c.get("bytes_per_message") or 0,
        "schema": lambda c: c.get("schema_name") or "",
        "percent": lambda c: c.get("size_bytes") or 0,  # Same as size for sorting purposes
    }
    get_sort_key = sort_keys.get(sort_key, sort_keys["topic"])

    # Check if size data is available
    has_size_data = any(ch.get("size_bytes") is not None for ch in data["channels"])

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
    total_size = sum(ch.get("size_bytes") or 0 for ch in data["channels"]) if show_percent else 0

    # Build the table
    channels_table = Table()
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
        colored_topic = _format_parts_with_colors(channel["topic"])

        row: list[RenderableType] = []
        if columns & ChannelTableColumn.ID:
            row.append(str(channel["id"]))
        if columns & ChannelTableColumn.TOPIC:
            row.append(colored_topic)
        if show_schema:
            row.append(_format_schema_with_link(channel.get("schema_name")))
        if columns & ChannelTableColumn.MSGS:
            row.append(f"{channel['message_count']:,}")
        if columns & ChannelTableColumn.HZ:
            # Determine Hz value to display
            hz_stats = channel.get("hz_stats")
            if use_median and hz_stats and (hz_median := hz_stats.get("median")) is not None:
                hz = hz_median
            elif index_duration and (hz_channel := channel.get("hz_channel")) is not None:
                hz = hz_channel
            else:
                hz = hz_stats["average"] if hz_stats else 0
            row.append(f"{hz:.2f}")
        if show_size:
            size = channel.get("size_bytes") or 0
            row.append(bytes_to_human(size))
        if show_percent:
            size = channel.get("size_bytes") or 0
            percentage = (size / total_size * 100) if total_size > 0 else 0
            row.append(f"{percentage:.2f}%")
        if show_bps:
            # Use median bytes per second if available and requested
            bps_stats = channel.get("bytes_per_second_stats")
            if use_median and bps_stats and (bps_median := bps_stats.get("median")) is not None:
                bps = bps_median
            elif bps_stats:
                bps = bps_stats["average"]
            else:
                bps = None
            row.append(bytes_to_human(bps))
        if show_b_per_msg:
            row.append(bytes_to_human(channel.get("bytes_per_message")))
        if show_distribution:
            distribution = channel.get("message_distribution", [])
            row.append(DistributionBar(distribution))
        channels_table.add_row(*row)

    console.print(channels_table)
