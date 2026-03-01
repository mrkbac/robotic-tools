"""ROS2 and Foxglove message schema definitions."""

from pymcap_cli.image_utils import COMPRESSED_POINTCLOUD2, FOXGLOVE_COMPRESSED_VIDEO

# Schema registry mapping output schema names to definitions.
# Only output schemas (produced by transformers) need to be registered here;
# input schemas arrive from the upstream bridge.
SCHEMA_REGISTRY: dict[str, str] = {
    "foxglove_msgs/msg/CompressedVideo": FOXGLOVE_COMPRESSED_VIDEO,
    "foxglove_msgs/CompressedVideo": FOXGLOVE_COMPRESSED_VIDEO,
    "point_cloud_interfaces/msg/CompressedPointCloud2": COMPRESSED_POINTCLOUD2,
    "point_cloud_interfaces/CompressedPointCloud2": COMPRESSED_POINTCLOUD2,
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
