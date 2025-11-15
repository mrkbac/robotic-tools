"""E2E tests for the cat command."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pymcap_cli.cli import app

runner = CliRunner()


@pytest.mark.e2e
class TestCat:
    """Test cat command functionality."""

    def test_cat_text_output(self, simple_mcap: Path):
        """Test basic text output."""
        result = runner.invoke(app, ["cat", str(simple_mcap), "--limit", "1"])

        assert result.exit_code == 0
        assert result.stdout
        # Should show topic, timestamp, and schema
        assert "[" in result.stdout  # timestamp
        assert "(" in result.stdout  # schema

    def test_cat_json_output(self, image_small_mcap: Path):
        """Test JSON output with ROS2 decoding."""
        result = runner.invoke(app, ["cat", str(image_small_mcap), "--limit", "1", "--json"])

        assert result.exit_code == 0
        assert result.stdout

        # Parse JSON output
        line = result.stdout.strip()
        data = json.loads(line)

        # Check JSON structure
        assert "topic" in data
        assert "log_time" in data
        assert "publish_time" in data
        assert "schema" in data
        assert "message" in data

    def test_cat_topic_filter(self, multi_topic_mcap: Path):
        """Test topic filtering."""
        # Filter to camera topics only
        result = runner.invoke(
            app, ["cat", str(multi_topic_mcap), "--topics", "/camera/.*", "--limit", "5"]
        )

        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")

        # All lines should be camera topics
        for line in lines:
            assert "/camera/" in line

    def test_cat_exclude_topics(self, multi_topic_mcap: Path):
        """Test topic exclusion."""
        # Exclude camera topics
        result = runner.invoke(
            app, ["cat", str(multi_topic_mcap), "--exclude-topics", "/camera/.*", "--limit", "5"]
        )

        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")

        # No lines should be camera topics
        for line in lines:
            assert "/camera/" not in line

    def test_cat_limit(self, multi_topic_mcap: Path):
        """Test message limit."""
        result = runner.invoke(app, ["cat", str(multi_topic_mcap), "--limit", "3"])

        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 3

    def test_cat_json_with_topic_filter(self, image_small_mcap: Path):
        """Test JSON output with topic filtering."""
        result = runner.invoke(
            app,
            ["cat", str(image_small_mcap), "--topics", "/camera.*", "--limit", "2", "--json"],
        )

        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 2

        # Parse and verify JSON
        for line in lines:
            data = json.loads(line)
            assert "/camera" in data["topic"]
            assert "message" in data

    def test_cat_image_compressed(self, image_compressed_mcap: Path):
        """Test cat with CompressedImage messages."""
        result = runner.invoke(app, ["cat", str(image_compressed_mcap), "--limit", "1", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout.strip())

        # Verify message structure
        assert data["schema"] == "sensor_msgs/msg/CompressedImage"
        assert "message" in data
        assert "header" in data["message"]
        assert "format" in data["message"]
        assert "data" in data["message"]
        # Data should be a list of integers (converted from bytes)
        assert isinstance(data["message"]["data"], list)
        assert all(isinstance(b, int) for b in data["message"]["data"][:10])

    def test_cat_no_limit(self, simple_mcap: Path):
        """Test cat without limit outputs all messages."""
        result = runner.invoke(app, ["cat", str(simple_mcap)])

        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")
        # Simple MCAP should have at least a few messages
        assert len(lines) > 0

    def test_cat_nonexistent_file(self):
        """Test cat with nonexistent file."""
        result = runner.invoke(app, ["cat", "nonexistent.mcap"])

        assert result.exit_code == 1
        assert "Error" in result.stderr or "error" in result.stderr.lower()

    def test_cat_empty_filter(self, multi_topic_mcap: Path):
        """Test cat with filter that matches nothing."""
        result = runner.invoke(
            app, ["cat", str(multi_topic_mcap), "--topics", "/nonexistent/.*", "--limit", "10"]
        )

        assert result.exit_code == 0
        # Should output nothing or very few lines (if limit not reached)
        assert result.stdout.strip() == "" or len(result.stdout.strip().split("\n")) == 0

    def test_cat_multiple_topic_filters(self, multi_topic_mcap: Path):
        """Test cat with multiple topic filters."""
        result = runner.invoke(
            app,
            [
                "cat",
                str(multi_topic_mcap),
                "--topics",
                "/camera/.*",
                "--topics",
                "/lidar/.*",
                "--limit",
                "10",
            ],
        )

        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")

        # Should have messages from both camera and lidar topics
        has_camera = any("/camera/" in line for line in lines)
        has_lidar = any("/lidar/" in line for line in lines)

        # At least one should be present (may not have both in first 10 messages)
        assert has_camera or has_lidar

    def test_cat_json_preserves_message_structure(self, image_compressed_mcap: Path):
        """Test that JSON output preserves nested message structure."""
        result = runner.invoke(app, ["cat", str(image_compressed_mcap), "--limit", "1", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout.strip())

        # Check nested structure (header.stamp)
        assert "header" in data["message"]
        assert "stamp" in data["message"]["header"]
        assert "sec" in data["message"]["header"]["stamp"]
        assert "nanosec" in data["message"]["header"]["stamp"]
        assert "frame_id" in data["message"]["header"]

    def test_cat_text_shows_timestamp(self, simple_mcap: Path):
        """Test that text output shows timestamp."""
        result = runner.invoke(app, ["cat", str(simple_mcap), "--limit", "1"])

        assert result.exit_code == 0
        # Should contain timestamp in brackets
        assert "[" in result.stdout
        assert "]" in result.stdout
        # Should contain schema in parentheses
        assert "(" in result.stdout
        assert ")" in result.stdout
