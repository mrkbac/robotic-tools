"""Shared display and formatting utilities for pymcap-cli commands."""

from __future__ import annotations


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
