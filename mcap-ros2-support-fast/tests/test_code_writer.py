from mcap_ros2_support_fast.code_writer import CodeWriter


def test_append_none_is_ignored() -> None:
    writer = CodeWriter()

    writer.append(None)

    assert str(writer) == ""


def test_iter_returns_generated_lines() -> None:
    writer = CodeWriter()
    writer.append("first\nsecond")

    assert list(writer) == ["first", "second"]
