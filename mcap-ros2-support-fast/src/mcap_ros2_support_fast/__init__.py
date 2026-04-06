__version__ = "0.5.5"

# Export ROS2-specific types
from .decoder import DecoderFactory, McapROS2DecodeError
from .writer import McapROS2WriteError, ROS2EncoderFactory

__all__ = [
    "DecoderFactory",
    "McapROS2DecodeError",
    "McapROS2WriteError",
    "ROS2EncoderFactory",
    "__version__",
]
