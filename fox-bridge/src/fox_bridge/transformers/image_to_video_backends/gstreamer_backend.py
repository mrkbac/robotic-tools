"""GStreamer-backed implementation for JPEG to video encoding."""

from __future__ import annotations

import logging
from typing import ClassVar

from .base import EncodingConfig, ImageToVideoBackend, ImageToVideoBackendError

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except (ImportError, ValueError):  # pragma: no cover - GStreamer optional
    Gst = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_FRAME_RATE_NUMERATOR = 30
_FRAME_RATE_DENOMINATOR = 1


def is_gstreamer_available() -> bool:
    """Return True if GStreamer bindings are importable."""
    if Gst is None:
        return False
    _GStreamerSession.ensure_initialized()
    return True


class _GStreamerSession:
    """Utility to lazily initialise the GStreamer runtime once."""

    _initialised: ClassVar[bool] = False

    @classmethod
    def ensure_initialized(cls) -> None:
        if cls._initialised:
            return
        if Gst is None:
            raise ImageToVideoBackendError("GStreamer Python bindings not available")
        Gst.init(None)
        cls._initialised = True

    @classmethod
    def element_factory_exists(cls, name: str) -> bool:
        cls.ensure_initialized()
        if Gst is None:
            return False
        return Gst.ElementFactory.find(name) is not None


class GStreamerBackend(ImageToVideoBackend):
    """Encode frames using a GStreamer in-process pipeline."""

    def __init__(self, config: EncodingConfig) -> None:
        if Gst is None:
            raise ImageToVideoBackendError("GStreamer Python bindings not installed")

        super().__init__(config)

        if self.config.codec != "h264":
            raise ImageToVideoBackendError("GStreamer backend currently supports only H.264 encoding")
        _GStreamerSession.ensure_initialized()

        self._supports_nvjpeg = _GStreamerSession.element_factory_exists("nvjpegdec")
        self._supports_nvconv = _GStreamerSession.element_factory_exists("nvvidconv")
        self._supports_nvh264 = _GStreamerSession.element_factory_exists("nvv4l2h264enc")
        self._supports_x264 = _GStreamerSession.element_factory_exists("x264enc")

        self._use_nvidia = (
            self.config.use_hardware
            and self.config.codec == "h264"
            and self._supports_nvjpeg
            and self._supports_nvconv
            and self._supports_nvh264
        )

        if not self._supports_x264 and not self._use_nvidia:
            raise ImageToVideoBackendError(
                "Neither x264enc nor NVIDIA accelerated encoders are available in GStreamer"
            )

        if self._use_nvidia:
            logger.info("ImageToVideoTransformer using GStreamer NVIDIA accelerated encoder")
        else:
            logger.info("ImageToVideoTransformer using GStreamer x264 encoder")

    def encode(
        self,
        jpeg_data: bytes,
        input_width: int,
        input_height: int,
        target_width: int,
        target_height: int,
    ) -> bytes:
        pipeline_description = self._build_pipeline_description(target_width, target_height)

        try:
            pipeline = Gst.parse_launch(pipeline_description)
        except Exception as err:  # pragma: no cover - Gst throws generic errors
            raise ImageToVideoBackendError(f"Failed to create GStreamer pipeline: {err}") from err

        appsrc = pipeline.get_by_name("src")
        appsink = pipeline.get_by_name("sink")
        if appsrc is None or appsink is None:
            pipeline.set_state(Gst.State.NULL)
            raise ImageToVideoBackendError("Failed to retrieve appsrc/appsink from pipeline")

        gst_buffer = Gst.Buffer.new_allocate(None, len(jpeg_data), None)
        if gst_buffer is None:
            pipeline.set_state(Gst.State.NULL)
            raise ImageToVideoBackendError("Failed to allocate GStreamer buffer")

        gst_buffer.fill(0, jpeg_data)
        duration = Gst.util_uint64_scale(
            1, Gst.SECOND * _FRAME_RATE_DENOMINATOR, _FRAME_RATE_NUMERATOR
        )
        gst_buffer.pts = 0
        gst_buffer.dts = 0
        gst_buffer.duration = duration

        timeout_ns = int(self.config.timeout_s * Gst.SECOND)
        bus = pipeline.get_bus()

        pipeline.set_state(Gst.State.PLAYING)

        try:
            push_result = appsrc.emit("push-buffer", gst_buffer)
            if push_result != Gst.FlowReturn.OK:
                raise ImageToVideoBackendError(f"push-buffer returned {push_result!s}")
            appsrc.emit("end-of-stream")

            sample = self._await_sample(bus, appsink, timeout_ns)
            if sample is None:
                raise ImageToVideoBackendError("GStreamer pipeline produced no sample")

            try:
                gst_buffer = sample.get_buffer()
                if gst_buffer is None:
                    raise ImageToVideoBackendError("Appsink sample missing buffer")
                success, map_info = gst_buffer.map(Gst.MapFlags.READ)
                if not success:
                    raise ImageToVideoBackendError("Failed to map GStreamer buffer for reading")
                try:
                    return bytes(map_info.data)
                finally:
                    gst_buffer.unmap(map_info)
            finally:
                try:
                    sample.unref()
                except AttributeError:
                    pass
        finally:
            pipeline.set_state(Gst.State.NULL)

    def _await_sample(
        self, bus: "Gst.Bus", appsink: "Gst.AppSink", timeout_ns: int
    ) -> "Gst.Sample | None":
        """Block until EOS or error, then pull sample."""
        while True:
            msg = bus.timed_pop_filtered(
                timeout_ns,
                Gst.MessageType.ERROR
                | Gst.MessageType.EOS
                | Gst.MessageType.STATE_CHANGED
                | Gst.MessageType.WARNING,
            )
            if msg is None:
                raise ImageToVideoBackendError("Timed out waiting for GStreamer pipeline")
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                debug_text = f" ({debug})" if debug else ""
                raise ImageToVideoBackendError(f"GStreamer error: {err}{debug_text}")
            if msg.type == Gst.MessageType.WARNING:
                warn, debug = msg.parse_warning()
                debug_text = f" ({debug})" if debug else ""
                logger.warning("GStreamer warning: %s%s", warn, debug_text)
            if msg.type == Gst.MessageType.EOS:
                return appsink.emit("pull-sample")

    def _build_pipeline_description(self, width: int, height: int) -> str:
        common_caps = (
            f"caps=image/jpeg,framerate={_FRAME_RATE_NUMERATOR}/{_FRAME_RATE_DENOMINATOR}"
        )
        scale_caps = f"video/x-raw,format=I420,width={width},height={height}"

        if self._use_nvidia:
            # Use accelerated pipeline (requires NV elements).
            preset_level = self._map_nvidia_preset(self.config.preset)
            qp = max(0, min(self.config.quality, 51))
            return (
                "appsrc name=src is-live=false do-timestamp=false format=time "
                f"{common_caps} ! nvjpegdec ! nvvidconv ! "
                f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height} ! "
                "nvv4l2h264enc iframeinterval=1 idrinterval=1 insert-sps-pps=1 "
                f"preset-level={preset_level} control-rate=0 qp={qp} ! "
                "h264parse config-interval=-1 ! "
                "video/x-h264,stream-format=byte-stream,alignment=au ! "
                "appsink name=sink sync=false emit-signals=false max-buffers=1 drop=false wait-on-eos=true"
            )

        # Fallback to software x264 encoder.
        quantizer = max(0, min(self.config.quality, 51))
        speed_preset = self._validate_x264_preset(self.config.preset)
        return (
            "appsrc name=src is-live=false do-timestamp=false format=time "
            f"{common_caps} ! jpegdec ! videoconvert ! videoscale ! "
            f"{scale_caps} ! "
            "x264enc tune=zerolatency byte-stream=true key-int-max=1 bframes=0 "
            f"speed-preset={speed_preset} quantizer={quantizer} ! "
            "h264parse config-interval=-1 ! "
            "video/x-h264,stream-format=byte-stream,alignment=au ! "
            "appsink name=sink sync=false emit-signals=false max-buffers=1 drop=false wait-on-eos=true"
        )

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
