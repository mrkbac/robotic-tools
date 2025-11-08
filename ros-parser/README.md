# ROS Parser

A Python parser for ROS message definitions and Foxglove message path syntax.

This package provides two main parsers:
- **message_definition**: Parses ROS message definition files (`.msg` files)
- **message_path**: Parses Foxglove message path syntax for data access and filtering

## Regenerating Standalone Parsers

The project uses Lark to generate standalone parsers from grammar files. If you modify any `.lark` grammar file, regenerate the parsers:

```bash
uv run _generate_standalone.py \
  src/ros_parser/message_definition/grammar.lark \
  src/ros_parser/message_path/grammar.lark \
  --output-dir src/ros_parser \
  --lexer contextual
```

For more details on the message path syntax, see [src/ros_parser/message_path/README.md](src/ros_parser/message_path/README.md).
