"""Shared display and formatting utilities for pymcap-cli commands."""

import enum
import hashlib
from collections.abc import Callable, Iterator
from functools import lru_cache

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.jupyter import JupyterMixin
from rich.measure import Measurement
from rich.style import Style
from rich.table import Table
from rich.text import Text

from pymcap_cli.types.info_types import ChannelInfo, McapInfoOutput, SchemaInfo
from pymcap_cli.utils import bytes_to_human


def _build_schema_map(schemas: list[SchemaInfo]) -> dict[int, str]:
    """Build a lookup from schema_id to schema name."""
    return {s["id"]: s["name"] for s in schemas}


def _hz_average(ch: ChannelInfo, global_dur_sec: float) -> float:
    """Compute average Hz from global duration."""
    return ch["message_count"] / global_dur_sec if global_dur_sec > 0 else 0


def _hz_channel(ch: ChannelInfo) -> float | None:
    """Compute Hz from per-channel duration."""
    dur = ch.get("duration_ns")
    if dur is not None and dur > 0:
        return ch["message_count"] / (dur / 1_000_000_000)
    return None


def _bytes_per_message(ch: ChannelInfo) -> float | None:
    """Compute bytes per message."""
    size = ch.get("size_bytes")
    if size is not None and ch["message_count"] > 0:
        return size / ch["message_count"]
    return None


def _bps_average(ch: ChannelInfo, global_dur_sec: float) -> float | None:
    """Compute average bytes per second from global duration."""
    size = ch.get("size_bytes")
    if size is not None and global_dur_sec > 0:
        return size / global_dur_sec
    return None


def _hz_value(
    ch: ChannelInfo,
    use_median: bool,
    index_duration: bool,
    global_dur_sec: float,
) -> float:
    """Resolve the Hz value for a channel using the preferred source."""
    hz_stats = ch.get("hz_stats")
    if use_median and hz_stats and (med := hz_stats.get("median")) is not None:
        return med
    if index_duration:
        hz_ch = _hz_channel(ch)
        if hz_ch is not None:
            return hz_ch
    return _hz_average(ch, global_dur_sec)


def _bps_value(
    ch: ChannelInfo,
    use_median: bool,
    global_dur_sec: float,
) -> float:
    """Resolve the bytes-per-second value for a channel using the preferred source."""
    hz_stats = ch.get("hz_stats")
    b_per_msg = _bytes_per_message(ch)
    if use_median and hz_stats and b_per_msg is not None:
        med = hz_stats.get("median")
        if med is not None:
            return med * b_per_msg
    bps = _bps_average(ch, global_dur_sec)
    return bps if bps is not None else 0


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


TreeNode = dict[str, tuple[ChannelInfo | None, "TreeNode"]]


def build_prefix(channels: list[ChannelInfo]) -> TreeNode:
    tree: TreeNode = {}

    for channel in channels:
        parts = channel["topic"].strip("/").split("/")
        current_node = tree

        for part in parts[:-1]:
            if part not in current_node:
                current_node[part] = (None, {})
            current_node = current_node[part][1]

        # Add the channel info at the leaf
        leaf_part = parts[-1]
        current_node[leaf_part] = (channel, {})

    return tree


def fold_tree(node: TreeNode) -> TreeNode:
    """Collapse single-child chains in a tree structure.

    Transforms chains of intermediate nodes (nodes with no channel data and exactly
    one child) into a single node with a collapsed path key (e.g., "a/b/c").

    Args:
        node: The tree node to fold

    Returns:
        A new tree with collapsed single-child chains
    """
    folded: TreeNode = {}

    for key, (leaf_value, children) in node.items():
        # If this is an intermediate node (no channel) with exactly one child,
        # start collapsing the chain
        if not leaf_value and len(children) == 1:
            # Accumulate the path as we walk single-child chains
            collapsed_path = key
            current_leaf = leaf_value
            current_children = children

            # Walk the chain while we have single children and no channel data
            while len(current_children) == 1 and current_leaf is None:
                only_key, (only_leaf, only_children) = next(iter(current_children.items()))
                collapsed_path += "/" + only_key
                current_leaf = only_leaf
                current_children = only_children

            # Now we've reached the end of the chain
            # Recursively fold the subtree and store with collapsed path
            folded[collapsed_path] = (current_leaf, fold_tree(current_children))
        else:
            # Not a collapsible chain - keep as is but fold children
            folded[key] = (leaf_value, fold_tree(children))

    return folded


def _collect_leaf_channels(node: TreeNode) -> list[ChannelInfo]:
    """Recursively collect all ChannelInfo leaves from a TreeNode subtree."""
    leaves: list[ChannelInfo] = []
    for leaf_value, children in node.values():
        if leaf_value is not None:
            leaves.append(leaf_value)
        leaves.extend(_collect_leaf_channels(children))
    return leaves


def tree_iter(
    node: TreeNode,
    prefix: str = "",
    is_root: bool = True,
) -> Iterator[tuple[str, ChannelInfo | str, str, list[ChannelInfo] | None]]:
    """Iterate over a folded tree structure, yielding display information.

    Yields (tree_prefix, channel_or_string, display_path, descendants) tuples where:
    - tree_prefix contains the tree drawing characters for proper indentation
    - display_path is the node key (may be a collapsed path like "a/b/c" or just "leaf")
    - descendants is a list of leaf ChannelInfo for folder nodes, or None for leaf nodes
    """
    # The input channels are already sorted, so the tree preserves that order
    items = list(node.items())

    for i, (key, (leaf_value, children)) in enumerate(items):
        is_last_item = i == len(items) - 1

        # Determine tree characters - no connectors at root level
        if is_root:
            connector = ""
            new_prefix = ""
        else:
            connector = "└── " if is_last_item else "├── "
            new_prefix = prefix + ("    " if is_last_item else "│   ")

        if leaf_value:
            # This node has channel data
            # Display the key as-is (which may be a collapsed path or a single segment)
            yield (prefix + connector, leaf_value, key, None)

            # If it has children, recurse into them
            if children:
                yield from tree_iter(children, new_prefix, is_root=False)
        else:
            # Folder name — collect all descendant channels for aggregation
            descendants = _collect_leaf_channels(children)
            yield (prefix + connector, key, key, descendants)
            yield from tree_iter(children, new_prefix, is_root=False)


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
    tree: bool = False,
    terminal_width: int | None = None,
) -> Table:
    """Build and return a channels table with configurable columns and sorting.

    Args:
        data: MCAP info data containing channel information
        console: Rich console instance (used for responsive width calculation)
        sort_key: Field to sort by (topic, id, msgs, size, hz, bps, b_per_msg, schema)
        reverse: Sort in descending order if True
        columns: IntFlag of columns to display. If None, defaults to all columns.
        responsive: Enable responsive column hiding based on terminal width
        index_duration: Use per-channel Hz calculation instead of global duration
        use_median: Display median rates instead of mean rates
        tree: Display channels in a tree hierarchy based on topic path

    Returns:
        Rich Table containing the formatted channels data
    """

    # Default columns if not specified
    if columns is None:
        columns = (
            ChannelTableColumn.ID
            | ChannelTableColumn.SCHEMA
            | ChannelTableColumn.MSGS
            | ChannelTableColumn.HZ
            | ChannelTableColumn.SIZE
            | ChannelTableColumn.BPS
            | ChannelTableColumn.B_PER_MSG
            | ChannelTableColumn.DISTRIBUTION
        )

    # Compute global duration for derived values
    global_dur_sec = data["statistics"]["duration_ns"] / 1_000_000_000

    # Schema lookup map
    schema_map = _build_schema_map(data["schemas"])

    # Dictionary of sort key functions
    sort_keys: dict[str, Callable[[ChannelInfo], float | str]] = {
        "topic": lambda c: c["topic"],
        "id": lambda c: c["id"],
        "msgs": lambda c: c["message_count"],
        "size": lambda c: c.get("size_bytes") or 0,
        "hz": lambda c: _hz_value(c, use_median, index_duration, global_dur_sec),
        "bps": lambda c: _bps_value(c, use_median, global_dur_sec),
        "b_per_msg": lambda c: _bytes_per_message(c) or 0,
        "schema": lambda c: schema_map.get(c["schema_id"], ""),
        "percent": lambda c: c.get("size_bytes") or 0,  # Same as size for sorting purposes
    }
    get_sort_key = sort_keys.get(sort_key, sort_keys["topic"])

    # Check if size data is available
    has_size_data = any(ch.get("size_bytes") is not None for ch in data["channels"])

    # Check if distribution data is available
    has_distribution_data = any(ch.get("message_distribution") for ch in data["channels"])

    # Determine which columns to show based on responsive mode
    effective_width = terminal_width if terminal_width is not None else console.width
    if responsive:
        show_size = (
            effective_width >= 80 and has_size_data and bool(columns & ChannelTableColumn.SIZE)
        )
        show_percent = (
            effective_width >= 80 and has_size_data and bool(columns & ChannelTableColumn.PERCENT)
        )
        show_bps = (
            effective_width >= 100 and has_size_data and bool(columns & ChannelTableColumn.BPS)
        )
        show_b_per_msg = (
            effective_width >= 120
            and has_size_data
            and bool(columns & ChannelTableColumn.B_PER_MSG)
        )
        show_distribution = (
            effective_width >= 140
            and has_distribution_data
            and bool(columns & ChannelTableColumn.DISTRIBUTION)
        )
        show_schema = effective_width >= 160 and bool(columns & ChannelTableColumn.SCHEMA)
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
    channels_table = Table(expand=terminal_width is not None, width=terminal_width)
    if columns & ChannelTableColumn.ID:
        channels_table.add_column("ID", style="bold blue", no_wrap=True, justify="right")
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
    sorted_channels = sorted(data["channels"], key=get_sort_key, reverse=reverse)
    if tree:
        # Tree mode: build hierarchy, fold single-child chains, then iterate
        node = build_prefix(sorted_channels)
        folded_node = fold_tree(node)
        channel_iter = tree_iter(folded_node)
    else:
        # List mode: sort channels and iterate directly
        channel_iter = (("", channel, channel["topic"], None) for channel in sorted_channels)

    for tree_prefix, channel, display_path, descendants in channel_iter:
        if isinstance(channel, str):
            # Intermediate node (folder) - aggregate descendant data
            hrow: list[RenderableType] = []

            # Add empty cell for ID column if it exists
            if columns & ChannelTableColumn.ID:
                hrow.append("")

            hrow.append(f"{tree_prefix}{_format_parts_with_colors(channel)}")

            if show_schema:
                hrow.append("")

            if descendants:
                agg_msgs = sum(ch["message_count"] for ch in descendants)
                agg_size = sum(ch.get("size_bytes") or 0 for ch in descendants)
                agg_hz = sum(
                    _hz_value(ch, use_median, index_duration, global_dur_sec) for ch in descendants
                )
                agg_bps = sum(_bps_value(ch, use_median, global_dur_sec) for ch in descendants)

                if columns & ChannelTableColumn.MSGS:
                    hrow.append(Text(f"{agg_msgs:,}", style="dim"))
                if columns & ChannelTableColumn.HZ:
                    hrow.append(Text(f"{agg_hz:.2f}", style="dim"))
                if show_size:
                    hrow.append(Text(bytes_to_human(agg_size), style="dim"))
                if show_percent:
                    pct = (agg_size / total_size * 100) if total_size > 0 else 0
                    hrow.append(Text(f"{pct:.2f}%", style="dim"))
                if show_bps:
                    hrow.append(Text(bytes_to_human(agg_bps or None), style="dim"))
                if show_b_per_msg:
                    b_per_msg = agg_size / agg_msgs if agg_msgs > 0 else None
                    hrow.append(Text(bytes_to_human(b_per_msg), style="dim"))
                if show_distribution:
                    # Element-wise sum of descendant distributions
                    max_len = max(
                        (len(ch.get("message_distribution", [])) for ch in descendants),
                        default=0,
                    )
                    if max_len > 0:
                        agg_dist = [0] * max_len
                        for ch in descendants:
                            for j, v in enumerate(ch.get("message_distribution", [])):
                                agg_dist[j] += v
                        bar_width = (
                            max(20, effective_width // 4) if terminal_width is not None else None
                        )
                        hrow.append(DistributionBar(agg_dist, width=bar_width))
                    else:
                        hrow.append("")

            channels_table.add_row(*hrow, end_section=False)
            continue

        if tree:
            colored_topic = tree_prefix + _format_parts_with_colors(display_path)
        else:
            colored_topic = _format_parts_with_colors(display_path)

        row: list[RenderableType] = []
        if columns & ChannelTableColumn.ID:
            row.append(str(channel["id"]))
        row.append(colored_topic)
        if show_schema:
            schema_name = schema_map.get(channel["schema_id"])
            row.append(_format_schema_with_link(schema_name))
        if columns & ChannelTableColumn.MSGS:
            row.append(f"{channel['message_count']:,}")
        if columns & ChannelTableColumn.HZ:
            hz = _hz_value(channel, use_median, index_duration, global_dur_sec)
            row.append(f"{hz:.2f}")
        if show_size:
            size = channel.get("size_bytes") or 0
            row.append(bytes_to_human(size))
        if show_percent:
            size = channel.get("size_bytes") or 0
            percentage = (size / total_size * 100) if total_size > 0 else 0
            row.append(f"{percentage:.2f}%")
        if show_bps:
            bps = _bps_value(channel, use_median, global_dur_sec)
            row.append(bytes_to_human(bps or None))
        if show_b_per_msg:
            row.append(bytes_to_human(_bytes_per_message(channel)))
        if show_distribution:
            distribution = channel.get("message_distribution", [])
            # Use fixed bar width in watch mode to prevent jitter
            bar_width = max(20, effective_width // 4) if terminal_width is not None else None
            row.append(DistributionBar(distribution, width=bar_width))
        channels_table.add_row(*row)

    return channels_table
