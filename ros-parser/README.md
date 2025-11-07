# ROS Parser

Generate standalone parser:

```bash
uv run python -m lark.tools.standalone src/ros_parser/grammar.lark -o src/ros_parser/_standalone_parser.py --lexer contextual
```
