class Profile:
    """Well-known MCAP profiles."""

    ROS1 = "ros1"
    ROS2 = "ros2"


class SchemaEncoding:
    """Well-known encodings for schema records."""

    SelfDescribing = ""  # used for self-describing content, such as arbitrary JSON.
    Protobuf = "protobuf"
    Flatbuffer = "flatbuffer"
    ROS1 = "ros1msg"
    ROS2 = "ros2msg"
    ROS2IDL = "ros2idl"
    JSONSchema = "jsonschema"


class MessageEncoding:
    """Well-known message encodings for message records"""

    ROS1 = "ros1"
    CDR = "cdr"
    Protobuf = "protobuf"
    Flatbuffer = "flatbuffer"
    CBOR = "cbor"
    JSON = "json"
