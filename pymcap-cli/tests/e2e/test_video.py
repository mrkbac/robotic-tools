"""E2E tests for the video command."""

import shutil
from pathlib import Path

import pytest
from pymcap_cli.cmd.video_cmd import VideoCodec, VideoEncoderError, encode_video


@pytest.mark.e2e
class TestVideoCommand:
    """Test video command functionality."""

    def test_video_from_compressed_image(self, image_compressed_mcap: Path, tmp_path: Path):
        """Test creating video from CompressedImage messages."""
        # Skip if ffmpeg not available
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            pytest.skip("ffmpeg not available")

        output_file = tmp_path / "compressed_output.mp4"

        encode_video(
            mcap_path=image_compressed_mcap,
            topics=["/camera/image_compressed"],
            output_path=output_file,
            codec=VideoCodec.H264,
            encoder_preference="software",
            quality=23,
            watermark=False,
        )

        # Verify video was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_video_from_raw_image(self, image_rgb_mcap: Path, tmp_path: Path):
        """Test creating video from raw Image messages."""
        # Skip if ffmpeg not available
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            pytest.skip("ffmpeg not available")

        output_file = tmp_path / "raw_output.mp4"

        encode_video(
            mcap_path=image_rgb_mcap,
            topics=["/camera/image_raw"],
            output_path=output_file,
            codec=VideoCodec.H264,
            encoder_preference="software",
            quality=23,
            watermark=False,
        )

        # Verify video was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_video_small_file(self, image_small_mcap: Path, tmp_path: Path):
        """Test creating video from small MCAP with few frames."""
        # Skip if ffmpeg not available
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            pytest.skip("ffmpeg not available")

        output_file = tmp_path / "small_output.mp4"

        encode_video(
            mcap_path=image_small_mcap,
            topics=["/camera/image_compressed"],
            output_path=output_file,
            codec=VideoCodec.H264,
            encoder_preference="software",
            quality=23,
            watermark=False,
        )

        # Verify video was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_video_topic_not_found(self, image_compressed_mcap: Path, tmp_path: Path):
        """Test error handling when topic doesn't exist."""
        # Skip if ffmpeg not available
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            pytest.skip("ffmpeg not available")

        output_file = tmp_path / "output.mp4"

        with pytest.raises(VideoEncoderError, match="not found or is not an image topic"):
            encode_video(
                mcap_path=image_compressed_mcap,
                topics=["/nonexistent/topic"],
                output_path=output_file,
                codec=VideoCodec.H264,
                encoder_preference="software",
                quality=23,
                watermark=False,
            )

    def test_video_unsupported_schema(self, simple_mcap: Path, tmp_path: Path):
        """Test error handling when schema is not an image type."""
        # Skip if ffmpeg not available
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            pytest.skip("ffmpeg not available")

        output_file = tmp_path / "output.mp4"

        with pytest.raises(VideoEncoderError, match="not found or is not an image topic"):
            encode_video(
                mcap_path=simple_mcap,
                topics=["/test"],
                output_path=output_file,
                codec=VideoCodec.H264,
                encoder_preference="software",
                quality=23,
                watermark=False,
            )

    def test_video_with_watermark(self, image_compressed_mcap: Path, tmp_path: Path):
        """Test creating video with watermark enabled (default)."""
        # Skip if ffmpeg not available
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            pytest.skip("ffmpeg not available")

        output_file = tmp_path / "watermark_output.mp4"

        encode_video(
            mcap_path=image_compressed_mcap,
            topics=["/camera/image_compressed"],
            output_path=output_file,
            codec=VideoCodec.H264,
            encoder_preference="software",
            quality=23,
            watermark=True,
        )

        # Verify video was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_video_without_watermark(self, image_compressed_mcap: Path, tmp_path: Path):
        """Test creating video with watermark disabled."""
        # Skip if ffmpeg not available
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            pytest.skip("ffmpeg not available")

        output_file = tmp_path / "no_watermark_output.mp4"

        encode_video(
            mcap_path=image_compressed_mcap,
            topics=["/camera/image_compressed"],
            output_path=output_file,
            codec=VideoCodec.H264,
            encoder_preference="software",
            quality=23,
            watermark=False,
        )

        # Verify video was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0
