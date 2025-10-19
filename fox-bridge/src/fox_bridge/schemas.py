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

SENSOR_MSGS_POINTCLOUD2 = """std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: sensor_msgs/PointField
string name
uint32 offset
uint8 datatype
uint32 count"""


# Schema registry mapping schema names to definitions
SCHEMA_REGISTRY: dict[str, str] = {
    "foxglove_msgs/msg/CompressedVideo": FOXGLOVE_COMPRESSED_VIDEO,
    "foxglove_msgs/CompressedVideo": FOXGLOVE_COMPRESSED_VIDEO,
    "sensor_msgs/msg/PointCloud2": SENSOR_MSGS_POINTCLOUD2,
    "sensor_msgs/PointCloud2": SENSOR_MSGS_POINTCLOUD2,
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
