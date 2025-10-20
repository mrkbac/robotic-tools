"""Minimal GStreamer backend executed via gst-launch-1.0."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Final

from .base import EncodingConfig, ImageToVideoBackend, ImageToVideoBackendError

logger = logging.getLogger(__name__)

_DEFAULT_FRAMERATE: Final[str] = "30/1"


def is_gstreamer_available() -> bool:
    """Return True if gst-launch-1.0 is on PATH."""
    return shutil.which("gst-launch-1.0") is not None


class GStreamerBackend(ImageToVideoBackend):
    """Very small wrapper that pipes JPEG into gst-launch-1.0 + x264enc."""

    def __init__(self, config: EncodingConfig) -> None:
        if config.codec != "h264":
            raise ImageToVideoBackendError(
                "Minimal GStreamer backend only supports H.264 output"
            )
        super().__init__(config)

        self._gst_launch = shutil.which("gst-launch-1.0")
        if not self._gst_launch:
            raise ImageToVideoBackendError("gst-launch-1.0 executable not found on PATH")

        self._gst_inspect = shutil.which("gst-inspect-1.0")
        if self._gst_inspect:
            if not self._element_exists("x264enc"):
                raise ImageToVideoBackendError(
                    "GStreamer x264enc plugin is not available (gst-inspect-x264enc failed)"
                )
            self._supports_nv = all(
                self._element_exists(name)
                for name in ("nvjpegdec", "nvvidconv", "nvv4l2h264enc")
            )
        else:
            logger.debug(
                "gst-inspect-1.0 not found; skipping plugin preflight. Pipeline may fail at runtime."
            )
            self._supports_nv = False

        self._use_nvidia = self.config.use_hardware and self._supports_nv
        if self._use_nvidia:
            logger.info("GStreamer backend will use NVIDIA accelerated pipeline")
        elif self.config.use_hardware:
            logger.info("NVIDIA GStreamer plugins not found; falling back to software x264 pipeline")

    def encode(
        self,
        jpeg_data: bytes,
        input_width: int,
        input_height: int,
        target_width: int,
        target_height: int,
    ) -> bytes:
        pipeline = self._build_pipeline(
            len(jpeg_data), input_width, input_height, target_width, target_height
        )

        try:
            result = subprocess.run(
                pipeline,
                input=jpeg_data,
                capture_output=True,
                timeout=self.config.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as err:
            raise ImageToVideoBackendError("GStreamer encoding timed out") from err
        except FileNotFoundError as err:
            raise ImageToVideoBackendError("gst-launch-1.0 executable not found") from err

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")
            raise ImageToVideoBackendError(f"GStreamer pipeline failed: {stderr}")

        output = result.stdout
        if not output:
            raise ImageToVideoBackendError("GStreamer pipeline produced empty output")

        if not (
            len(output) >= 4
            and (output[:4] == b"\x00\x00\x00\x01" or output[:3] == b"\x00\x00\x01")
        ):
            logger.warning("GStreamer output may not be Annex B formatted")

        return output

    def _build_pipeline(
        self,
        buffer_size: int,
        input_width: int,
        input_height: int,
        target_width: int,
        target_height: int,
    ) -> list[str]:
        base = [
            self._gst_launch or "gst-launch-1.0",
            "-q",
            "-e",
            "fdsrc",
            "fd=0",
            f"blocksize={buffer_size}",
            "!",
            "capsfilter",
            f"caps=image/jpeg,width={input_width},height={input_height},framerate={_DEFAULT_FRAMERATE}",
            "!",
        ]
        if self._use_nvidia:
            return base + self._nvidia_pipeline_tail(target_width, target_height)
        return base + self._software_pipeline_tail(target_width, target_height)

    def _nvidia_pipeline_tail(self, width: int, height: int) -> list[str]:
        bitrate = self._quality_to_bitrate(self.config.quality)
        return [
            "jpegparse",
            "!",
            "nvjpegdec",
            "!",
            "nvvidconv",
            "!",
            f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height},framerate={_DEFAULT_FRAMERATE}",
            "!",
            "nvv4l2h264enc",
            "iframeinterval=1",
            "insert-sps-pps=1",
            "control-rate=1",
            f"bitrate={bitrate}",
            "profile=high",
            "!",
            "h264parse",
            "config-interval=-1",
            "!",
            "video/x-h264,stream-format=byte-stream,alignment=au",
            "!",
            "filesink",
            "location=/dev/stdout",
        ]

    def _software_pipeline_tail(self, width: int, height: int) -> list[str]:
        # Keep the software path intentionally minimal: jpegdec -> videoconvert -> videoscale -> x264enc.
        return [
            "jpegdec",
            "!",
            "videoconvert",
            "!",
            "videoscale",
            "!",
            f"video/x-raw,format=I420,width={width},height={height},framerate={_DEFAULT_FRAMERATE}",
            "!",
            "x264enc",
            "tune=zerolatency",
            "byte-stream=true",
            "key-int-max=1",
            "bframes=0",
            "speed-preset=fast",
            "rc-lookahead=0",
            "!",
            "h264parse",
            "config-interval=-1",
            "!",
            "video/x-h264,stream-format=byte-stream,alignment=au",
            "!",
            "filesink",
            "location=/dev/stdout",
        ]

    def _element_exists(self, element: str) -> bool:
        if not self._gst_inspect:
            return False
        try:
            result = subprocess.run(
                [self._gst_inspect, element],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def _quality_to_bitrate(self, quality: int) -> int:
        """Map quality (CRF-like 0-51) to a conservative bitrate in bits per second."""
        clamped = max(0, min(51, quality))
        high = 18_000_000
        low = 2_000_000
        scale = (51 - clamped) / 51
        return int(low + (high - low) * scale)
