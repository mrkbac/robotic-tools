"""ROS message, service, and action definition parser.

Supports both ROS1 and ROS2 message formats.

For ROS2, use:
    from ros_parser.ros2_msg import parse_message_string
    from ros_parser import ros2_msg

For ROS1, use:
    from ros_parser.ros1_msg import parse_message_string
    from ros_parser import ros1_msg

For shared models:
    from ros_parser.models import MessageDefinition, Field, Type, Constant
"""

from . import ros1_msg, ros2_msg
from .message_path import ValidationError
from .models import (
    PRIMITIVE_TYPE_NAMES,
    ActionDefinition,
    Constant,
    Field,
    MessageDefinition,
    PrimitiveType,
    ServiceDefinition,
    Type,
)
from .ros2_msg.schema_parser import parse_schema_to_definitions

__all__ = [
    "PRIMITIVE_TYPE_NAMES",
    "ActionDefinition",
    "Constant",
    "Field",
    "MessageDefinition",
    "PrimitiveType",
    "ServiceDefinition",
    "Type",
    "ValidationError",
    "parse_schema_to_definitions",
    "ros1_msg",
    "ros2_msg",
]
