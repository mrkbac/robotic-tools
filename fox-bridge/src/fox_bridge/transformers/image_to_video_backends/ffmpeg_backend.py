"""FFmpeg-backed implementation for JPEG to video encoding."""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Final

from portable_ffmpeg import get_ffmpeg

from .base import EncodingConfig, ImageToVideoBackend, ImageToVideoBackendError

logger = logging.getLogger(__name__)


def is_ffmpeg_available() -> bool:
    """Return True if portable-ffmpeg can locate an ffmpeg binary."""
    try:
        get_ffmpeg()
        return True
    except FileNotFoundError:
        return False


class FFmpegBackend(ImageToVideoBackend):
    """Encode frames using the ffmpeg CLI."""

    _NV_ENCODERS: Final[tuple[str, ...]] = ("h264_nvenc", "h265_nvenc")
    _V4L2_ENCODERS: Final[tuple[str, ...]] = ("h264_v4l2m2m", "hevc_v4l2m2m")

    def __init__(self, config: EncodingConfig) -> None:
        super().__init__(config)
        self._encoder = self._detect_encoder()
        logger.info("ImageToVideoTransformer using FFmpeg encoder: %s", self._encoder)

    def encode(
        self,
        jpeg_data: bytes,
        input_width: int,
        input_height: int,
        target_width: int,
        target_height: int,
    ) -> bytes:
        ffmpeg_path, _ = get_ffmpeg()
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
        ]

        needs_scaling = (target_width != input_width) or (target_height != input_height)
        if needs_scaling:
            cmd.extend(
                [
                    "-vf",
                    f"scale={target_width}:{target_height}:flags=lanczos",
                ]
            )

        cmd.extend(["-c:v", self._encoder])
        cmd.extend(self._build_encoder_options())

        cmd.extend(
            [
                "-g",
                "1",
                "-keyint_min",
                "1",
                "-f",
                self._derive_muxer(),
                "-an",
                "pipe:1",
            ]
        )

        try:
            result = subprocess.run(
                cmd,
                input=jpeg_data,
                capture_output=True,
                timeout=self.config.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as err:
            raise ImageToVideoBackendError("FFmpeg encoding timed out") from err
        except FileNotFoundError as err:
            raise ImageToVideoBackendError(
                "FFmpeg binary not found. Install portable-ffmpeg: pip install portable-ffmpeg"
            ) from err

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")
            raise ImageToVideoBackendError(f"FFmpeg encoding failed: {stderr}")

        output = result.stdout
        if not output:
            raise ImageToVideoBackendError("FFmpeg produced empty output")

        if not (
            len(output) >= 4
            and (output[:4] == b"\x00\x00\x00\x01" or output[:3] == b"\x00\x00\x01")
        ):
            logger.warning("Encoded data may not be in Annex B format")

        return output

    def _detect_encoder(self) -> str:
        if not self.config.use_hardware:
            return self._software_encoder_for_codec()

        system = platform.system()
        if system == "Darwin" and self._encoder_exists("h264_videotoolbox"):
            return "h264_videotoolbox"

        if system == "Linux":
            if self._encoder_exists("h264_nvenc") and self.config.codec == "h264":
                return "h264_nvenc"
            if self._encoder_exists("h265_nvenc") and self.config.codec == "h265":
                return "h265_nvenc"
            if self._encoder_exists("h264_vaapi") and self.config.codec == "h264":
                return "h264_vaapi"

        return self._software_encoder_for_codec()

    def _software_encoder_for_codec(self) -> str:
        return {
            "h264": "libx264",
            "h265": "libx265",
            "vp9": "libvpx-vp9",
            "av1": "libaom-av1",
        }.get(self.config.codec, "libx264")

    def _encoder_exists(self, encoder: str) -> bool:
        ffmpeg_path, _ = get_ffmpeg()
        try:
            result = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        return encoder in result.stdout

    def _build_encoder_options(self) -> list[str]:
        if self._encoder == "h264_videotoolbox":
            return ["-q:v", str(self.config.quality)]

        if self._encoder in self._NV_ENCODERS:
            return [
                "-preset",
                self.config.preset,
                "-cq",
                str(self.config.quality),
            ]

        if self._encoder == "h264_vaapi":
            return ["-qp", str(self.config.quality)]

        if self._encoder in self._V4L2_ENCODERS:
            return ["-b:v", "2M"]

        # Software encoders
        return [
            "-preset",
            self.config.preset,
            "-crf",
            str(self.config.quality),
        ]

    def _derive_muxer(self) -> str:
        if self.config.codec in ("h264", "h265"):
            return self.config.codec
        if self.config.codec == "vp9":
            return "ivf"
        if self.config.codec == "av1":
            return "ivf"
        return "h264"

