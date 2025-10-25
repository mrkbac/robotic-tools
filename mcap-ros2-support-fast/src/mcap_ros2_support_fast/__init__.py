__version__ = "0.5.5"

# Export ROS2-specific types
from .writer import McapROS2WriteError, ROS2EncoderFactory

__all__ = [
    "McapROS2WriteError",
    "ROS2EncoderFactory",
    "__version__",
]
