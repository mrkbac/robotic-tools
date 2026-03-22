"""Tests for ros_parser._utils — string unescaping and schema parsing utilities."""

from __future__ import annotations

from ros_parser._utils import add_msgdef_to_dict, for_each_msgdef_in_schema, unescape_string
from ros_parser.models import MessageDefinition

# ---------------------------------------------------------------------------
# unescape_string
# ---------------------------------------------------------------------------


class TestUnescapeString:
    def test_no_escapes(self):
        assert unescape_string("hello world") == "hello world"

    def test_empty(self):
        assert unescape_string("") == ""

    def test_newline(self):
        assert unescape_string("line1\\nline2") == "line1\nline2"

    def test_tab(self):
        assert unescape_string("col1\\tcol2") == "col1\tcol2"

    def test_carriage_return(self):
        assert unescape_string("a\\rb") == "a\rb"

    def test_backslash(self):
        # Note: the function processes escapes sequentially, so \\\\
        # becomes \\ first, then if followed by a known escape letter it
        # gets processed again.  Test the simple case.
        assert unescape_string("end\\\\") == "end\\"

    def test_single_quote(self):
        assert unescape_string("\\'hello\\'") == "'hello'"

    def test_double_quote(self):
        assert unescape_string('\\"hello\\"') == '"hello"'

    def test_bell_and_others(self):
        assert unescape_string("\\a\\b\\f\\v") == "\a\b\f\v"

    def test_octal_newline(self):
        # \012 = octal 12 = decimal 10 = newline
        assert unescape_string("\\012") == "\n"

    def test_octal_a(self):
        # \101 = octal 101 = decimal 65 = 'A'
        assert unescape_string("\\101") == "A"

    def test_hex(self):
        # \x41 = hex 41 = decimal 65 = 'A'
        assert unescape_string("\\x41") == "A"

    def test_hex_lowercase(self):
        assert unescape_string("\\x0a") == "\n"

    def test_unicode_4(self):
        # \u0041 = 'A'
        assert unescape_string("\\u0041") == "A"

    def test_unicode_4_non_ascii(self):
        # \u00e9 = 'é'
        assert unescape_string("\\u00e9") == "é"

    def test_unicode_8(self):
        # \U00000041 = 'A'
        assert unescape_string("\\U00000041") == "A"

    def test_unicode_8_emoji(self):
        # \U0001F600 = 😀
        assert unescape_string("\\U0001F600") == "\U0001F600"

    def test_combined(self):
        assert unescape_string("\\t\\x41\\n") == "\tA\n"


# ---------------------------------------------------------------------------
# for_each_msgdef_in_schema
# ---------------------------------------------------------------------------


def _dummy_parse(text: str, package: str | None) -> MessageDefinition:  # noqa: ARG001
    """Minimal parser that creates a MessageDefinition from raw text."""
    return MessageDefinition(name="", fields_all=[])


class TestForEachMsgdefInSchema:
    def test_single_section(self):
        results: list[tuple[str, str, MessageDefinition]] = []
        for_each_msgdef_in_schema(
            "std_msgs/msg/String",
            "string data",
            _dummy_parse,
            lambda full, short, md: results.append((full, short, md)),
        )
        assert len(results) == 1
        assert results[0][0] == "std_msgs/msg/String"
        assert results[0][1] == "std_msgs/String"

    def test_multiple_sections(self):
        schema = "string data\n===\nMSG: geometry_msgs/msg/Point\nfloat64 x\nfloat64 y\nfloat64 z"
        results: list[tuple[str, str]] = []
        for_each_msgdef_in_schema(
            "geometry_msgs/msg/Pose",
            schema,
            _dummy_parse,
            lambda full, short, _md: results.append((full, short)),
        )
        assert len(results) == 2
        assert results[0] == ("geometry_msgs/msg/Pose", "geometry_msgs/Pose")
        assert results[1] == ("geometry_msgs/msg/Point", "geometry_msgs/Point")

    def test_msg_header_parsed(self):
        schema = "float64 x\n=====\nMSG: pkg/msg/Sub\nint32 val"
        results: list[tuple[str, str]] = []
        for_each_msgdef_in_schema(
            "pkg/msg/Main",
            schema,
            _dummy_parse,
            lambda full, short, _md: results.append((full, short)),
        )
        assert results[1][0] == "pkg/msg/Sub"
        assert results[1][1] == "pkg/Sub"

    def test_ros1_style_name(self):
        """ROS1 names use package/Message without /msg/."""
        results: list[tuple[str, str]] = []
        for_each_msgdef_in_schema(
            "sensor_msgs/Image",
            "uint8[] data",
            _dummy_parse,
            lambda full, short, _md: results.append((full, short)),
        )
        assert results[0][0] == "sensor_msgs/Image"
        assert results[0][1] == "sensor_msgs/Image"

    def test_empty_lines_stripped(self):
        schema = "\n\nstring data\n\n"
        results: list[tuple[str, str]] = []
        for_each_msgdef_in_schema(
            "std_msgs/msg/String",
            schema,
            _dummy_parse,
            lambda full, short, _md: results.append((full, short)),
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# add_msgdef_to_dict
# ---------------------------------------------------------------------------


class TestAddMsgdefToDict:
    def test_adds_full_and_short_name(self):
        defs: dict[str, MessageDefinition] = {}
        md = MessageDefinition(name="test", fields_all=[])
        add_msgdef_to_dict(defs, "geometry_msgs/msg/Point", "geometry_msgs/Point", md)
        assert "geometry_msgs/msg/Point" in defs
        assert "geometry_msgs/Point" in defs
        assert "Point" in defs

    def test_same_full_and_short(self):
        defs: dict[str, MessageDefinition] = {}
        md = MessageDefinition(name="test", fields_all=[])
        add_msgdef_to_dict(defs, "sensor_msgs/Image", "sensor_msgs/Image", md)
        assert "sensor_msgs/Image" in defs
        assert "Image" in defs
        assert len(defs) == 2  # no duplicate

    def test_msg_name_only_not_overwritten(self):
        defs: dict[str, MessageDefinition] = {}
        md1 = MessageDefinition(name="first", fields_all=[])
        md2 = MessageDefinition(name="second", fields_all=[])
        add_msgdef_to_dict(defs, "pkg1/msg/Header", "pkg1/Header", md1)
        add_msgdef_to_dict(defs, "pkg2/msg/Header", "pkg2/Header", md2)
        # "Header" should still point to md1 (first added)
        assert defs["Header"].name == "first"
        assert defs["pkg2/Header"].name == "second"
