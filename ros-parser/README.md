# ROS Parser

A Python parser for ROS message definitions and Foxglove message path syntax.

## Installation

```bash
uv add ros-parser
```

## Parsers

- **ros1_msg**: Parses ROS1 message definition files (.msg)
- **ros2_msg**: Parses ROS2 message definition files (.msg)
- **message_path**: Parses Foxglove message path syntax for data access and filtering

## Usage

### Parsing ROS2 Messages

```python
from ros_parser import ros2_msg

definition = ros2_msg.parse_message_string("""
float64 x
float64 y
float64 z
""")

for field in definition.fields:
    print(f"{field.name}: {field.type}")
```

### Parsing ROS1 Messages

```python
from ros_parser import ros1_msg

definition = ros1_msg.parse_message_string("""
Header header
float64 x
float64 y
float64 z
""")

for field in definition.fields:
    print(f"{field.name}: {field.type}")
```

### Parsing Schema with Dependencies

```python
from ros_parser import parse_schema_to_definitions

# Parse a full schema including embedded type definitions
definitions = parse_schema_to_definitions(
    "geometry_msgs/msg/Pose",
    schema_data
)
```

## Regenerating Standalone Parsers

The project uses Lark to generate standalone parsers from grammar files. If you modify any `.lark` grammar file, regenerate the parsers:

```bash
uv run _generate_standalone.py \
  src/ros_parser/ros1_msg/grammar.lark \
  src/ros_parser/ros2_msg/grammar.lark \
  src/ros_parser/message_path/grammar.lark \
  --output-dir src/ros_parser \
  --lexer contextual
```

For more details on the message path syntax, see [src/ros_parser/message_path/README.md](src/ros_parser/message_path/README.md).
