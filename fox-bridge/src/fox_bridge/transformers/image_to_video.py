"""Transform JPEG CompressedImage to H.264 CompressedVideo using ffmpeg."""

import logging
import platform
import subprocess
from io import BytesIO
from typing import Any

from PIL import Image
from portable_ffmpeg import get_ffmpeg

from . import Transformer, TransformError

logger = logging.getLogger(__name__)


class ImageToVideoTransformer(Transformer):
    """Transform sensor_msgs/CompressedImage (JPEG) to foxglove_msgs/CompressedVideo (H.264)."""

    def __init__(
        self,
        codec: str = "h264",
        quality: int = 23,  # CRF value for libx264 (lower = better quality)
        preset: str = "fast",
        use_hardware: bool = True,
        max_dimension: int = 480,  # Maximum dimension (width or height) for downscaling
    ) -> None:
        """Initialize the transformer.

        Args:
            codec: Video codec to use ('h264', 'h265', 'vp9', 'av1')
            quality: Encoding quality (CRF for x264: 0-51, lower is better)
            preset: Encoding preset for libx264 ('ultrafast', 'fast', 'medium', 'slow')
            use_hardware: Whether to try hardware acceleration
            max_dimension: Maximum dimension (width or height) in pixels. Images larger
                          than this will be downscaled proportionally (default: 720 for HD)
        """
        self.codec = codec
        self.quality = quality
        self.preset = preset
        self.use_hardware = use_hardware
        self.max_dimension = max_dimension

        # Detect best encoder
        self.encoder = self._detect_encoder()
        logger.info(f"Using encoder: {self.encoder}")

    def get_input_schema(self) -> str:
        """Get the input message schema name."""
        return "sensor_msgs/msg/CompressedImage"

    def get_output_schema(self) -> str:
        """Get the output message schema name."""
        return "foxglove_msgs/CompressedVideo"

    def _detect_encoder(self) -> str:
        """Detect the best available encoder.

        Returns:
            The encoder name to use with ffmpeg
        """
        if not self.use_hardware:
            return self._get_software_encoder()

        system = platform.system()

        if system == "Darwin":
            # macOS - try VideoToolbox
            if self._test_encoder("h264_videotoolbox"):
                return "h264_videotoolbox"
        elif system == "Linux":
            # Try NVENC for NVIDIA GPUs
            if self._test_encoder("h264_nvenc"):
                return "h264_nvenc"
            # Try VAAPI for Intel/AMD
            if self._test_encoder("h264_vaapi"):
                return "h264_vaapi"

        # Fallback to software encoder
        return self._get_software_encoder()

    def _get_software_encoder(self) -> str:
        """Get the software encoder for the selected codec."""
        codec_map = {
            "h264": "libx264",
            "h265": "libx265",
            "vp9": "libvpx-vp9",
            "av1": "libaom-av1",
        }
        return codec_map.get(self.codec, "libx264")

    def _test_encoder(self, encoder: str) -> bool:
        """Test if an encoder is available.

        Args:
            encoder: The encoder name to test

        Returns:
            True if the encoder is available
        """
        ffmpeg, _ = get_ffmpeg()
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return encoder in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _calculate_downscale_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Calculate target dimensions for downscaling while maintaining aspect ratio.

        Args:
            width: Original image width
            height: Original image height

        Returns:
            Tuple of (target_width, target_height). Returns original dimensions
            if no downscaling is needed.
        """

        def ensure_even(value: int) -> int:
            if value % 2 == 0:
                return value
            # Prefer shrinking by one pixel; if that would hit zero, bump to 2 instead
            adjusted = value - 1
            return adjusted if adjusted >= 2 else 2

        # If both dimensions are within max_dimension, no scaling needed besides ensuring even size
        if width <= self.max_dimension and height <= self.max_dimension:
            return ensure_even(width), ensure_even(height)

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Determine which dimension to constrain
        if width > height:
            # Width is the limiting factor
            new_width = self.max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = self.max_dimension
            new_width = int(new_height * aspect_ratio)

        # Ensure dimensions are even (required for most video codecs)
        new_width = ensure_even(new_width)
        new_height = ensure_even(new_height)

        return new_width, new_height

    def transform(self, message: Any) -> dict[str, Any]:
        """Transform CompressedImage to CompressedVideo.

        Args:
            message: Decoded CompressedImage message

        Returns:
            CompressedVideo message dict

        Raises:
            TransformError: If transformation fails
        """
        try:
            if not message.data:
                raise TransformError(f"Empty image data, {message}")

            # Verify it's JPEG (for now we only support JPEG)
            if message.format and "jpeg" not in message.format.lower():
                raise TransformError(f"Unsupported format: {message.format}")

            # Get image dimensions using PIL (faster than ffprobe)
            try:
                img = Image.open(BytesIO(message.data))
                width, height = img.size
            except Exception as e:
                raise TransformError(f"Failed to read image: {e}") from e

            # Calculate target dimensions (may downscale if too large)
            target_width, target_height = self._calculate_downscale_dimensions(width, height)

            if (target_width, target_height) != (width, height):
                logger.debug(f"Downscaling from {width}x{height} to {target_width}x{target_height}")

            # Encode JPEG to H.264
            h264_data = self._encode_to_h264(
                message.data, width, height, target_width, target_height
            )

            # Build output message
            output_message = {
                "timestamp": {
                    "sec": message.header.stamp.sec,
                    "nanosec": message.header.stamp.nanosec,
                },
                "frame_id": message.header.frame_id,
                "data": list(h264_data),  # Convert bytes to list of uint8
                "format": self.codec,
            }

            return output_message

        except TransformError:
            raise
        except Exception as e:
            raise TransformError(f"Transform failed: {e}") from e

    def _encode_to_h264(
        self,
        jpeg_data: bytes,
        width: int,
        height: int,
        target_width: int,
        target_height: int,
    ) -> bytes:
        """Encode JPEG data to H.264 video frame.

        Args:
            jpeg_data: JPEG image bytes
            width: Original image width
            height: Original image height
            target_width: Target width after scaling
            target_height: Target height after scaling

        Returns:
            H.264 encoded data (Annex B format with NAL units)

        Raises:
            TransformError: If encoding fails
        """
        # Build ffmpeg command
        # Input: JPEG from stdin
        # Output: H.264 Annex B format to stdout
        ffmpeg_path, _ = get_ffmpeg()
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            # Input
            "-i",
            "pipe:0",  # Read JPEG from stdin
        ]

        # Add scaling filter if dimensions changed
        needs_scaling = (target_width != width) or (target_height != height)
        if needs_scaling:
            cmd.extend(
                [
                    "-vf",
                    f"scale={target_width}:{target_height}:flags=lanczos",
                ]
            )

        # Add encoder
        cmd.extend(["-c:v", self.encoder])

        # Add encoder-specific options
        if self.encoder == "h264_videotoolbox":
            # VideoToolbox options
            cmd.extend(
                [
                    "-q:v",
                    str(self.quality),  # Quality (lower = better)
                ]
            )
        elif self.encoder in ("h264_nvenc", "h265_nvenc"):
            # NVENC options
            cmd.extend(
                [
                    "-preset",
                    self.preset,
                    "-cq",
                    str(self.quality),
                ]
            )
        elif self.encoder == "h264_vaapi":
            # VAAPI options
            cmd.extend(
                [
                    "-qp",
                    str(self.quality),
                ]
            )
        elif self.encoder in ("h264_v4l2m2m", "hevc_v4l2m2m"):
            # V4L2 M2M options (Jetson and other embedded devices)
            # Note: V4L2 M2M doesn't support all quality/preset options
            # We use bitrate control instead
            cmd.extend(
                [
                    "-b:v",
                    "2M",  # Target bitrate (adjust as needed)
                ]
            )
        else:
            # Software encoder options (libx264, libx265)
            cmd.extend(
                [
                    "-preset",
                    self.preset,
                    "-crf",
                    str(self.quality),
                ]
            )

        # Force keyframe (every frame is a keyframe for now)
        cmd.extend(
            [
                "-g",
                "1",  # GOP size = 1 (all keyframes)
                "-keyint_min",
                "1",
            ]
        )

        # Output format
        cmd.extend(
            [
                "-f",
                "h264",  # Raw H.264 (Annex B)
                "-an",  # No audio
                "pipe:1",  # Write to stdout
            ]
        )

        try:
            # Run ffmpeg
            result = subprocess.run(
                cmd,
                input=jpeg_data,
                capture_output=True,
                timeout=30,  # 30 second timeout
                check=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode("utf-8", errors="ignore")
                raise TransformError(f"FFmpeg encoding failed: {error_msg}")

            h264_data = result.stdout

            if not h264_data:
                raise TransformError("FFmpeg produced empty output")

            # Verify it starts with H.264 Annex B start code
            if not (
                len(h264_data) >= 4
                and (h264_data[:4] == b"\x00\x00\x00\x01" or h264_data[:3] == b"\x00\x00\x01")
            ):
                logger.warning("H.264 data may not be in Annex B format")

            return h264_data

        except subprocess.TimeoutExpired as e:
            raise TransformError("FFmpeg encoding timed out") from e
        except FileNotFoundError as e:
            raise TransformError(
                "FFmpeg not found. Install portable-ffmpeg: pip install portable-ffmpeg"
            ) from e
        except Exception as e:
            raise TransformError(f"Encoding error: {e}") from e


__all__ = ["ImageToVideoTransformer"]
