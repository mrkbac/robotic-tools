"""GStreamer CLI backend that avoids PyGObject dependencies."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Final

from .base import EncodingConfig, ImageToVideoBackend, ImageToVideoBackendError

logger = logging.getLogger(__name__)


def is_gstreamer_available() -> bool:
    """Detect whether gst-launch-1.0 is present on PATH."""
    return shutil.which("gst-launch-1.0") is not None


class GStreamerBackend(ImageToVideoBackend):
    """Encode frames by invoking gst-launch-1.0 pipelines."""

    _DEFAULT_FRAMERATE: Final[str] = "30/1"

    def __init__(self, config: EncodingConfig) -> None:
        if config.codec != "h264":
            raise ImageToVideoBackendError(
                "GStreamer backend currently supports only H.264 encoding"
            )
        super().__init__(config)

        self._gst_launch = shutil.which("gst-launch-1.0")
        if not self._gst_launch:
            raise ImageToVideoBackendError("gst-launch-1.0 executable not found in PATH")

        self._gst_inspect = shutil.which("gst-inspect-1.0")
        if not self._gst_inspect:
            raise ImageToVideoBackendError("gst-inspect-1.0 executable not found in PATH")

        self._supports_nvjpeg = self._element_exists("nvjpegdec")
        self._supports_nvconv = self._element_exists("nvvidconv")
        self._supports_nvh264 = self._element_exists("nvv4l2h264enc")
        self._supports_x264 = self._element_exists("x264enc")
        self._supports_jpegdec = self._element_exists("jpegdec")
        self._supports_videoscale = self._element_exists("videoscale")

        self._use_nvidia = (
            self.config.use_hardware
            and self._supports_nvjpeg
            and self._supports_nvconv
            and self._supports_nvh264
        )

        if self._use_nvidia:
            logger.info("Using NVIDIA accelerated GStreamer pipeline")
        elif not (self._supports_x264 and self._supports_jpegdec and self._supports_videoscale):
            raise ImageToVideoBackendError(
                "Required software GStreamer plugins (x264enc/jpegdec/videoscale) are unavailable"
            )
        else:
            logger.info("Using software GStreamer pipeline with x264enc")

    def encode(
        self,
        jpeg_data: bytes,
        input_width: int,
        input_height: int,
        target_width: int,
        target_height: int,
    ) -> bytes:
        pipeline_args = self._build_pipeline(
            len(jpeg_data), input_width, input_height, target_width, target_height
        )

        try:
            result = subprocess.run(
                pipeline_args,
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
            f"fd=0",
            f"blocksize={buffer_size}",
            "!",
            f"image/jpeg,width={input_width},height={input_height},framerate={self._DEFAULT_FRAMERATE}",
            "!",
            "jpegparse",
            "!",
        ]

        if self._use_nvidia:
            pipeline_tail = self._build_nvidia_pipeline(target_width, target_height)
        else:
            pipeline_tail = self._build_software_pipeline(target_width, target_height)

        return base + pipeline_tail

    def _build_nvidia_pipeline(self, width: int, height: int) -> list[str]:
        preset_level = self._map_nvidia_preset(self.config.preset)
        qp = max(0, min(self.config.quality, 51))
        return [
            "nvjpegdec",
            "!",
            "nvvidconv",
            "!",
            f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height},framerate={self._DEFAULT_FRAMERATE}",
            "!",
            "nvv4l2h264enc",
            "iframeinterval=1",
            "idrinterval=1",
            "insert-sps-pps=1",
            "control-rate=0",
            f"preset-level={preset_level}",
            f"qp={qp}",
            "!",
            "h264parse",
            "config-interval=-1",
            "!",
            "video/x-h264,stream-format=byte-stream,alignment=au",
            "!",
            "filesink",
            "location=/dev/stdout",
        ]

    def _build_software_pipeline(self, width: int, height: int) -> list[str]:
        quantizer = max(0, min(self.config.quality, 51))
        speed_preset = self._validate_x264_preset(self.config.preset)
        return [
            "jpegdec",
            "!",
            "videoconvert",
            "!",
            "videoscale",
            "!",
            f"video/x-raw,format=I420,width={width},height={height},framerate={self._DEFAULT_FRAMERATE}",
            "!",
            "x264enc",
            "tune=zerolatency",
            "byte-stream=true",
            "key-int-max=1",
            "bframes=0",
            f"speed-preset={speed_preset}",
            f"quantizer={quantizer}",
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

    def _map_nvidia_preset(self, preset: str) -> int:
        mapping = {
            "ultrafast": 0,
            "superfast": 1,
            "veryfast": 2,
            "faster": 3,
            "fast": 4,
            "medium": 5,
            "slow": 6,
            "slower": 7,
            "veryslow": 7,
        }
        return mapping.get(preset.lower(), 4)

    def _validate_x264_preset(self, preset: str) -> str:
        valid = {
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
            "placebo",
        }
        lower = preset.lower()
        if lower in valid:
            return lower
        logger.warning("Unsupported x264 preset '%s', defaulting to 'fast'", preset)
        return "fast"

