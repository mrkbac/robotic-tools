"""Video-related ROS schema names and definitions."""

COMPRESSED_VIDEO_SCHEMA = "foxglove_msgs/msg/CompressedVideo"

# Canonical short schema names, matching the normal form used by CLI exporters.
COMPRESSED_SCHEMAS = {"sensor_msgs/CompressedImage"}
RAW_SCHEMAS = {"sensor_msgs/Image"}
IMAGE_SCHEMAS = COMPRESSED_SCHEMAS | RAW_SCHEMAS

FOXGLOVE_COMPRESSED_VIDEO = """builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

COMPRESSED_IMAGE = """\
std_msgs/Header header
string format
uint8[] data

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

IMAGE = """\
std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""
