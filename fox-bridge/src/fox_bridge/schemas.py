"""ROS2 and Foxglove message schema definitions."""


# Foxglove message schemas (from foxglove_msgs)

FOXGLOVE_COMPRESSED_VIDEO = """builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""


# Schema registry mapping schema names to definitions
SCHEMA_REGISTRY: dict[str, str] = {
    "foxglove_msgs/msg/CompressedVideo": FOXGLOVE_COMPRESSED_VIDEO,
    "foxglove_msgs/CompressedVideo": FOXGLOVE_COMPRESSED_VIDEO,
}


def get_schema(schema_name: str) -> str:
    """Get message schema definition by name.

    Args:
        schema_name: Name of the schema (e.g., 'sensor_msgs/CompressedImage')

    Returns:
        The message definition string

    Raises:
        KeyError: If schema name is not found
    """
    if schema_name in SCHEMA_REGISTRY:
        return SCHEMA_REGISTRY[schema_name]

    raise KeyError(f"Schema not found: {schema_name}")
