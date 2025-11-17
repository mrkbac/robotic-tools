"""E2E tests for the info and info-json commands."""

import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.e2e
class TestInfo:
    """Test info command functionality."""

    def test_info_single_file(self, image_small_mcap: Path):
        """Test basic info output with single file."""
        result = subprocess.run(
            ["pymcap-cli", "info", str(image_small_mcap)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "File:" in result.stdout
        assert "Messages:" in result.stdout
        assert "Chunks:" in result.stdout
        assert "Channels:" in result.stdout

    def test_info_multiple_files(self, image_small_mcap: Path, image_rgb_mcap: Path):
        """Test info with multiple files shows each file separately."""
        result = subprocess.run(
            ["pymcap-cli", "info", str(image_small_mcap), str(image_rgb_mcap)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        # Should have separator between files
        assert "=" * 80 in result.stdout
        # Should have stats for both files
        assert result.stdout.count("File:") == 2
        assert result.stdout.count("Messages:") == 2
        assert result.stdout.count("Chunks:") == 2

    def test_info_no_files(self):
        """Test info with no files shows error."""
        result = subprocess.run(
            ["pymcap-cli", "info"],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_info_nonexistent_file(self):
        """Test info with nonexistent file fails."""
        result = subprocess.run(
            ["pymcap-cli", "info", "nonexistent.mcap"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1

    def test_info_with_rebuild(self, image_small_mcap: Path):
        """Test info with --rebuild flag."""
        result = subprocess.run(
            ["pymcap-cli", "info", str(image_small_mcap), "--rebuild"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "File:" in result.stdout
        assert "Messages:" in result.stdout


@pytest.mark.e2e
class TestInfoJson:
    """Test info-json command functionality."""

    def test_info_json_single_file(self, image_small_mcap: Path):
        """Test info-json with single file returns object."""
        result = subprocess.run(
            ["pymcap-cli", "info-json", str(image_small_mcap)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)

        # Should be a dict, not a list
        assert isinstance(data, dict)
        assert "file" in data
        assert "header" in data
        assert "statistics" in data
        assert "channels" in data
        assert data["file"]["path"] == str(image_small_mcap)

    def test_info_json_multiple_files(self, image_small_mcap: Path, image_rgb_mcap: Path):
        """Test info-json with multiple files returns array."""
        result = subprocess.run(
            ["pymcap-cli", "info-json", str(image_small_mcap), str(image_rgb_mcap)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)

        # Should be a list
        assert isinstance(data, list)
        assert len(data) == 2

        # Each item should be a complete info object
        for item in data:
            assert "file" in item
            assert "header" in item
            assert "statistics" in item
            assert "channels" in item

        # Should have both file paths
        paths = [item["file"]["path"] for item in data]
        assert str(image_small_mcap) in paths
        assert str(image_rgb_mcap) in paths

    def test_info_json_no_files(self):
        """Test info-json with no files shows error."""
        result = subprocess.run(
            ["pymcap-cli", "info-json"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1

    def test_info_json_with_rebuild(self, image_small_mcap: Path):
        """Test info-json with --rebuild flag."""
        result = subprocess.run(
            ["pymcap-cli", "info-json", str(image_small_mcap), "--rebuild"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, dict)
        assert "file" in data

    def test_info_json_compressed(self, image_small_mcap: Path):
        """Test info-json with --compress flag."""
        result = subprocess.run(
            ["pymcap-cli", "info-json", str(image_small_mcap), "--compress"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        # Output should be base64 encoded
        assert result.stdout.strip()
        # Should not be valid JSON directly
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stdout)

    def test_info_json_structure(self, image_small_mcap: Path):
        """Test info-json output has expected structure."""
        result = subprocess.run(
            ["pymcap-cli", "info-json", str(image_small_mcap)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)

        # Check main structure
        assert "file" in data
        assert "header" in data
        assert "statistics" in data
        assert "chunks" in data
        assert "channels" in data
        assert "schemas" in data
        assert "message_distribution" in data

        # Check file info
        assert "path" in data["file"]
        assert "size_bytes" in data["file"]

        # Check header
        assert "library" in data["header"]
        assert "profile" in data["header"]

        # Check statistics
        stats = data["statistics"]
        assert "message_count" in stats
        assert "chunk_count" in stats
        assert "channel_count" in stats
        assert "duration_ns" in stats

        # Check channels
        assert isinstance(data["channels"], list)
        if data["channels"]:
            channel = data["channels"][0]
            assert "id" in channel
            assert "topic" in channel
            assert "message_count" in channel
