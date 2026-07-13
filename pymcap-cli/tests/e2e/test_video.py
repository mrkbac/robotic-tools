"""E2E tests for the video command (post-Exporter-pipeline refactor)."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest
from mcap_codec_support.video import EncoderBackend, EncoderMode, VideoCodec
from pymcap_cli.cmd.video_cmd import QualityPreset, video

if TYPE_CHECKING:
    from pathlib import Path


def _ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@pytest.mark.e2e
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not available")
class TestVideoCommand:
    """Test video command functionality."""

    def test_video_from_compressed_image(self, image_compressed_mcap: Path, tmp_path: Path):
        """CompressedImage messages are encoded to a per-topic MP4."""
        out_dir = tmp_path / "compressed_out"

        rc = video(
            file=str(image_compressed_mcap),
            topic=["/camera/image_compressed"],
            output=out_dir,
            codec=VideoCodec.H264,
            encoder=EncoderBackend.SOFTWARE,
            quality=QualityPreset.MEDIUM,
        )

        assert rc == 0
        mp4s = list(out_dir.glob("*.mp4"))
        assert len(mp4s) == 1
        assert mp4s[0].stat().st_size > 0

    def test_video_from_raw_image(self, image_rgb_mcap: Path, tmp_path: Path):
        """Raw Image messages are encoded to a per-topic MP4."""
        out_dir = tmp_path / "raw_out"

        rc = video(
            file=str(image_rgb_mcap),
            topic=["/camera/image_raw"],
            output=out_dir,
            codec=VideoCodec.H264,
            encoder=EncoderBackend.SOFTWARE,
            quality=QualityPreset.MEDIUM,
        )

        assert rc == 0
        mp4s = list(out_dir.glob("*.mp4"))
        assert len(mp4s) == 1
        assert mp4s[0].stat().st_size > 0

    def test_video_small_file(self, image_small_mcap: Path, tmp_path: Path):
        """Tiny MCAP with few frames still produces a valid MP4."""
        out_dir = tmp_path / "small_out"

        rc = video(
            file=str(image_small_mcap),
            topic=["/camera/image_compressed"],
            output=out_dir,
            codec=VideoCodec.H264,
            encoder=EncoderBackend.SOFTWARE,
            quality=QualityPreset.MEDIUM,
        )

        assert rc == 0
        mp4s = list(out_dir.glob("*.mp4"))
        assert len(mp4s) == 1
        assert mp4s[0].stat().st_size > 0

    def test_video_topic_not_found(self, image_compressed_mcap: Path, tmp_path: Path):
        """Nonexistent topic exits with non-zero status (no MP4 written)."""
        out_dir = tmp_path / "miss_out"

        rc = video(
            file=str(image_compressed_mcap),
            topic=["/nonexistent/topic"],
            output=out_dir,
            codec=VideoCodec.H264,
            encoder=EncoderBackend.SOFTWARE,
            quality=QualityPreset.MEDIUM,
        )

        assert rc != 0
        assert list(out_dir.glob("*.mp4")) == []

    def test_video_unsupported_schema(self, simple_mcap: Path, tmp_path: Path):
        """Non-image topic is skipped; no MP4 is produced."""
        out_dir = tmp_path / "skip_out"

        rc = video(
            file=str(simple_mcap),
            topic=["/test"],
            output=out_dir,
            codec=VideoCodec.H264,
            encoder=EncoderBackend.SOFTWARE,
            quality=QualityPreset.MEDIUM,
        )

        assert rc != 0
        assert list(out_dir.glob("*.mp4")) == []

    def test_video_ffmpeg_cli_mode(self, image_compressed_mcap: Path, tmp_path: Path):
        """`--mode ffmpeg-cli` produces an MP4 via the ffmpeg subprocess."""
        out_dir = tmp_path / "ffmpeg_cli_out"

        rc = video(
            file=str(image_compressed_mcap),
            topic=["/camera/image_compressed"],
            output=out_dir,
            codec=VideoCodec.H264,
            encoder=EncoderBackend.SOFTWARE,
            quality=QualityPreset.MEDIUM,
            mode=EncoderMode.FFMPEG_CLI,
        )

        assert rc == 0
        mp4s = list(out_dir.glob("*.mp4"))
        assert len(mp4s) == 1
        assert mp4s[0].stat().st_size > 0
        # MP4 magic: ISO base media file.
        head = mp4s[0].read_bytes()[:12]
        assert b"ftyp" in head

    def test_video_force_overwrites_existing_dir(self, image_compressed_mcap: Path, tmp_path: Path):
        """--force allows overwriting MP4s in an existing non-empty directory."""
        out_dir = tmp_path / "force_out"
        out_dir.mkdir()
        (out_dir / "stale.mp4").write_bytes(b"placeholder")

        rc = video(
            file=str(image_compressed_mcap),
            topic=["/camera/image_compressed"],
            output=out_dir,
            codec=VideoCodec.H264,
            encoder=EncoderBackend.SOFTWARE,
            quality=QualityPreset.MEDIUM,
            force=True,
        )

        assert rc == 0
        mp4s = list(out_dir.glob("*.mp4"))
        # At least the new MP4 exists and is larger than the placeholder.
        new_mp4s = [p for p in mp4s if p.stat().st_size > len(b"placeholder")]
        assert new_mp4s
