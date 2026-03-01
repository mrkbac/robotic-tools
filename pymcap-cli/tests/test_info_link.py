"""Tests for info_link URL generation."""

import base64
import json
import subprocess
import zlib
from pathlib import Path

import pytest
from pymcap_cli.cmd.info_link import generate_link


def _decompress_from_base64url(b64: str) -> str:
    """Reverse the compress pipeline to inspect the payload."""
    padded = b64 + "=" * (-len(b64) % 4)
    compressed = base64.urlsafe_b64decode(padded)
    return zlib.decompress(compressed, wbits=-15).decode("utf-8")


def _get_info_json(mcap_path: Path) -> dict:
    result = subprocess.run(
        ["pymcap-cli", "info-json", str(mcap_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _generate_and_decode(mcap_path: Path) -> dict:
    data = _get_info_json(mcap_path)
    url = generate_link(
        data,
        file_path=str(mcap_path),
        file_size=mcap_path.stat().st_size,
        mode="summary",
    )
    hash_part = url.split("#", 1)[1]
    return json.loads(_decompress_from_base64url(hash_part))


@pytest.mark.e2e
class TestInfoLink:
    """Test URL generation from the info command."""

    def test_nuscenes_url_under_limit(self, nuscenes_mcap: Path):
        """Generated URL for nuScenes file fits within terminal URL limits."""
        data = _get_info_json(nuscenes_mcap)
        url = generate_link(
            data,
            file_path=str(nuscenes_mcap),
            file_size=nuscenes_mcap.stat().st_size,
            mode="summary",
        )

        assert url.startswith("https://mrkbac.github.io/robotic-tools/view#")
        hash_part = url.split("#", 1)[1]
        assert len(hash_part) <= 2000, f"Hash is {len(hash_part)} chars, must be <= 2000"

    def test_nuscenes_url_has_chunks_overlaps(self, nuscenes_mcap: Path):
        """Stripped URL payload includes chunks.overlaps but not by_compression."""
        payload = _generate_and_decode(nuscenes_mcap)
        url_data = payload["data"]

        # chunks.overlaps must be present
        assert "chunks" in url_data
        assert "overlaps" in url_data["chunks"]
        assert "max_concurrent" in url_data["chunks"]["overlaps"]
        assert "max_concurrent_bytes" in url_data["chunks"]["overlaps"]

        # by_compression must be empty (stripped)
        assert url_data["chunks"]["by_compression"] == {}

    def test_nuscenes_url_strips_heavy_fields(self, nuscenes_mcap: Path):
        """Heavy fields (metadata, attachments, thumbnail) are stripped."""
        payload = _generate_and_decode(nuscenes_mcap)
        url_data = payload["data"]

        # Stripped fields
        assert "metadata" not in url_data
        assert "attachments" not in url_data
        assert "thumbnail" not in url_data

        # Kept fields
        assert "file" in url_data
        assert "header" in url_data
        assert "statistics" in url_data
        assert "schemas" in url_data
        assert "channels" in url_data
        assert "message_distribution" in url_data

    def test_nuscenes_url_slim_channels(self, nuscenes_mcap: Path):
        """Channel entries keep base fields, heavy/derived fields stripped."""
        payload = _generate_and_decode(nuscenes_mcap)
        url_data = payload["data"]

        kept_keys = {
            "id",
            "topic",
            "schema_id",
            "message_count",
            "size_bytes",
            "duration_ns",
            "estimated_sizes",
        }
        stripped_keys = {
            "schema_name",
            "hz_channel",
            "bytes_per_second_stats",
            "bytes_per_message",
            "message_distribution",
            "jitter_cv",
        }

        assert len(url_data["channels"]) > 0
        for ch in url_data["channels"]:
            # All essential fields present
            for key in kept_keys:
                assert key in ch, f"Missing expected key: {key}"
            # Derived/heavy fields stripped
            for key in stripped_keys:
                assert key not in ch, f"Unexpected key present: {key}"
            # If hz_stats present, it has only min/max/median (no average)
            if "hz_stats" in ch and ch["hz_stats"] is not None:
                assert set(ch["hz_stats"].keys()) == {"minimum", "maximum", "median"}, (
                    f"hz_stats keys should be min/max/median only, got {set(ch['hz_stats'].keys())}"
                )
            # Derived fields should not be present
            assert "schema_name" not in ch
            assert "hz_channel" not in ch
            assert "bytes_per_message" not in ch
            assert "bytes_per_second_stats" not in ch
            assert "jitter_cv" not in ch
