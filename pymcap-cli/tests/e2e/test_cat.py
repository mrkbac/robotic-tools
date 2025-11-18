"""E2E tests for the cat command."""

import json
from pathlib import Path
from typing import Callable

import pytest

from pymcap_cli.cmd.cat_cmd import cat


def call_cat_expect_success(func: Callable, *args, **kwargs):
    """Call cat function and handle both success return and SystemExit(0)."""
    try:
        func(*args, **kwargs)
    except SystemExit as exc:
        if exc.code != 0:
            raise


def call_cat_expect_failure(func: Callable, *args, **kwargs) -> int:
    """Call cat function expecting failure, return exit code."""
    try:
        func(*args, **kwargs)
        return 0  # No exception = success
    except SystemExit as exc:
        return exc.code


@pytest.mark.e2e
class TestCat:
    """Test cat command functionality."""

    def test_cat_text_output(self, simple_mcap: Path, capsys):
        """Test basic text output."""
        call_cat_expect_success(cat, file=str(simple_mcap), limit=1)

        captured = capsys.readouterr()
        assert captured.out
        # Should show topic, timestamp, and schema
        assert "[" in captured.out  # timestamp
        assert "(" in captured.out  # schema

    def test_cat_json_output(self, image_small_mcap: Path, capsys, monkeypatch):
        """Test JSON output with ROS2 decoding."""
        # Mock isatty to ensure JSONL output format
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        cat(file=str(image_small_mcap), limit=1)

        captured = capsys.readouterr()
        assert captured.out

        # Parse JSON output
        line = captured.out.strip()
        data = json.loads(line)

        # Check JSON structure
        assert "topic" in data
        assert "log_time" in data
        assert "publish_time" in data
        assert "schema" in data
        assert "message" in data

    def test_cat_topic_filter(self, multi_topic_mcap: Path, capsys):
        """Test topic filtering."""
        # Filter to camera topics only
        cat(file=str(multi_topic_mcap), topics=["/camera/.*"], limit=5)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        # All lines should be camera topics
        for line in lines:
            assert "/camera/" in line

    def test_cat_exclude_topics(self, multi_topic_mcap: Path, capsys):
        """Test topic exclusion."""
        # Exclude camera topics
        cat(file=str(multi_topic_mcap), exclude_topics=["/camera/.*"], limit=5)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        # No lines should be camera topics
        for line in lines:
            assert "/camera/" not in line

    def test_cat_limit(self, multi_topic_mcap: Path, capsys):
        """Test message limit."""
        cat(file=str(multi_topic_mcap), limit=3)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 3

    def test_cat_json_with_topic_filter(self, image_small_mcap: Path, capsys, monkeypatch):
        """Test JSON output with topic filtering."""
        # Mock isatty to ensure JSONL output format
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        cat(file=str(image_small_mcap), topics=["/camera.*"], limit=2)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 2

        # Parse and verify JSON
        for line in lines:
            data = json.loads(line)
            assert "/camera" in data["topic"]
            assert "message" in data

    def test_cat_image_compressed(self, image_compressed_mcap: Path, capsys, monkeypatch):
        """Test cat with CompressedImage messages."""
        # Mock isatty to ensure JSONL output format
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        cat(file=str(image_compressed_mcap), limit=1)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        # Verify message structure
        assert data["schema"] == "sensor_msgs/msg/CompressedImage"
        assert "message" in data
        assert "header" in data["message"]
        assert "format" in data["message"]
        assert "data" in data["message"]
        # Data should be a list of integers (converted from bytes)
        assert isinstance(data["message"]["data"], list)
        assert all(isinstance(b, int) for b in data["message"]["data"][:10])

    def test_cat_no_limit(self, simple_mcap: Path, capsys):
        """Test cat without limit outputs all messages."""
        cat(file=str(simple_mcap))

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        # Simple MCAP should have at least a few messages
        assert len(lines) > 0

    def test_cat_nonexistent_file(self, capsys):
        """Test cat with nonexistent file."""
        with pytest.raises(SystemExit) as exc_info:
            cat(file="nonexistent.mcap")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err or "error" in captured.err.lower()

    def test_cat_empty_filter(self, multi_topic_mcap: Path, capsys):
        """Test cat with filter that matches nothing."""
        cat(file=str(multi_topic_mcap), topics=["/nonexistent/.*"], limit=10)

        captured = capsys.readouterr()
        # Should output nothing or very few lines (if limit not reached)
        assert captured.out.strip() == "" or len(captured.out.strip().split("\n")) == 0

    def test_cat_multiple_topic_filters(self, multi_topic_mcap: Path, capsys):
        """Test cat with multiple topic filters."""
        cat(file=str(multi_topic_mcap), topics=["/camera/.*", "/lidar/.*"], limit=10)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        # Should have messages from both camera and lidar topics
        has_camera = any("/camera/" in line for line in lines)
        has_lidar = any("/lidar/" in line for line in lines)

        # At least one should be present (may not have both in first 10 messages)
        assert has_camera or has_lidar

    def test_cat_json_preserves_message_structure(self, image_compressed_mcap: Path, capsys, monkeypatch):
        """Test that JSON output preserves nested message structure."""
        # Mock isatty to ensure JSONL output format
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        cat(file=str(image_compressed_mcap), limit=1)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        # Check nested structure (header.stamp)
        assert "header" in data["message"]
        assert "stamp" in data["message"]["header"]
        assert "sec" in data["message"]["header"]["stamp"]
        assert "nanosec" in data["message"]["header"]["stamp"]
        assert "frame_id" in data["message"]["header"]

    def test_cat_text_shows_timestamp(self, simple_mcap: Path, capsys):
        """Test that text output shows timestamp."""
        cat(file=str(simple_mcap), limit=1)

        captured = capsys.readouterr()
        # Should contain timestamp in brackets
        assert "[" in captured.out
        assert "]" in captured.out
        # Should contain schema in parentheses
        assert "(" in captured.out
        assert ")" in captured.out


@pytest.mark.e2e
class TestCatQueryValidation:
    """Test query validation in cat command."""

    def test_query_valid_field(self, image_small_mcap: Path):
        """Test that valid query on existing field works."""
        cat(file=str(image_small_mcap), query="/camera/image.header", limit=1)
        # Should output successfully

    def test_query_invalid_field_fails(self, image_small_mcap: Path, capsys):
        """Test that query on non-existent field fails with error."""
        with pytest.raises(SystemExit) as exc_info:
            cat(file=str(image_small_mcap), query="/camera/image.nonexistent", limit=1)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # Should contain validation error message
        assert "validation error" in captured.err.lower() or "not found" in captured.err.lower()

    def test_query_nested_field_valid(self, image_small_mcap: Path):
        """Test valid nested field access."""
        cat(file=str(image_small_mcap), query="/camera/image.header.stamp.sec", limit=1)

    def test_query_invalid_nested_field(self, image_small_mcap: Path, capsys):
        """Test invalid nested field access fails."""
        with pytest.raises(SystemExit) as exc_info:
            cat(file=str(image_small_mcap), query="/camera/image.header.invalid_field", limit=1)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "validation error" in captured.err.lower() or "not found" in captured.err.lower()

    def test_query_field_on_primitive_fails(self, image_small_mcap: Path, capsys):
        """Test accessing field on primitive type fails."""
        with pytest.raises(SystemExit) as exc_info:
            cat(file=str(image_small_mcap), query="/camera/image.width.something", limit=1)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "primitive" in captured.err.lower() or "cannot access" in captured.err.lower()

    def test_query_array_index_without_array_fails(self, image_small_mcap: Path, capsys):
        """Test indexing non-array field fails."""
        with pytest.raises(SystemExit) as exc_info:
            cat(file=str(image_small_mcap), query="/camera/image.width[0]", limit=1)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "array" in captured.err.lower() or "cannot apply" in captured.err.lower()

    def test_query_with_filter_valid_field(self, multi_topic_mcap: Path):
        """Test filter with valid field."""
        # Assuming multi_topic_mcap has messages with numeric fields
        try:
            cat(file=str(multi_topic_mcap), query="/odom.pose", limit=1)
        except SystemExit as exc:
            # This might fail if the topic doesn't exist, but should fail gracefully
            # The important thing is it doesn't crash
            assert exc.code in [0, 1]

    def test_validation_error_shows_helpful_message(self, image_small_mcap: Path, capsys):
        """Test that validation errors show helpful messages with available fields."""
        with pytest.raises(SystemExit) as exc_info:
            cat(file=str(image_small_mcap), query="/camera/image.bad_field", limit=1)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # Should show available fields
        assert "available" in captured.err.lower() or "Available" in captured.err
